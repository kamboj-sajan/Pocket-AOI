"""Stage 3: compare each golden part with the same spot in the aligned test photo, tuned for recall.

SAM's masks say where things are, and the netlist says what they are, so matching uses position
only. Whatever SAM found, a part is also compared by appearance, so a mask SAM happens to miss
does not raise a fault by itself.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

from pocket_aoi.align import normalise_lighting
from pocket_aoi.schemas import FaultType, GoldenProfile, Part
from pocket_aoi.segment import PartMask, rle_to_mask

Box = tuple[int, int, int, int]

_NEVER_MATCH = 1e6
_MIN_CROP_SIDE = 16


@dataclass(frozen=True)
class Finding:
    """One thing Stage 3 found, before its composite image is saved."""

    golden_ref: str | None
    cv_finding: FaultType
    bbox: Box
    part_bbox: Box
    evidence: dict[str, float]


@dataclass(frozen=True)
class ComparisonImage:
    """An image with its lighting normalised, for comparison only."""

    rgb: np.ndarray
    gray: np.ndarray


def comparison_image(image: np.ndarray) -> ComparisonImage:
    """Normalise lighting, and keep both colour and grayscale versions."""
    rgb = normalise_lighting(image)
    return ComparisonImage(rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


@dataclass(frozen=True)
class Appearance:
    """How alike one spot looks in the two photos."""

    ssim: float             # 1 is identical; follows structure, so flat areas of different colours can score high
    low_fraction: float     # share of the part's pixels with a clearly low SSIM, for small local changes
    diff_fraction: float    # share of the part's pixels whose colour clearly changed, for flat areas


def expand_box(box: Sequence[float], margin: float, width: int, height: int) -> Box:
    """Grow a box by a share of its own size on each side, keeping it inside the image."""
    x0, y0, x1, y1 = box
    grow_x, grow_y = (x1 - x0) * margin, (y1 - y0) * margin
    return (
        max(int(x0 - grow_x), 0),
        max(int(y0 - grow_y), 0),
        min(int(x1 + grow_x) + 1, width),
        min(int(y1 + grow_y) + 1, height),
    )


def mask_centroid(rle: Mapping[str, Any]) -> tuple[float, float]:
    """Return the centre of mass of a compressed mask."""
    rows, columns = np.nonzero(rle_to_mask(rle))
    return (float(columns.mean()), float(rows.mean())) if rows.size else (0.0, 0.0)


def match_parts(
    golden_parts: Sequence[Part],
    test_parts: Sequence[PartMask],
    compare_cfg: Mapping[str, Any],
) -> dict[int, tuple[int, float]]:
    """Match test masks to golden parts with the Hungarian algorithm, on position alone.

    Returns golden index -> (test index, IoU). The cost is (1 - IoU) plus a share of the centroid
    distance over the part diagonal, and pairs below the minimum IoU never match.
    """
    if not golden_parts or not test_parts:
        return {}
    ious = np.asarray(
        mask_utils.iou([p.rle for p in test_parts], [p.mask_rle for p in golden_parts], [0] * len(golden_parts))
    ).reshape(len(test_parts), len(golden_parts))
    test_centroids = np.array([mask_centroid(p.rle) for p in test_parts])

    cost = np.full((len(golden_parts), len(test_parts)), _NEVER_MATCH)
    for g, part in enumerate(golden_parts):
        x0, y0, x1, y1 = part.bbox
        diagonal = max(float(np.hypot(x1 - x0, y1 - y0)), 1.0)
        # NumPy broadcasting: one golden centroid minus every test centroid gives all distances at once.
        distance = np.linalg.norm(test_centroids - np.array(part.centroid), axis=1)
        allowed = ious[:, g] >= compare_cfg["min_match_iou"]
        cost[g, allowed] = (1 - ious[allowed, g]) + compare_cfg["centroid_cost_weight"] * distance[allowed] / diagonal

    golden_indices, test_indices = linear_sum_assignment(cost)
    return {
        int(g): (int(t), float(ious[t, g]))
        for g, t in zip(golden_indices, test_indices)
        if cost[g, t] < _NEVER_MATCH
    }


def best_shift(golden_crop: np.ndarray, test_gray: np.ndarray, box: Box, search_px: int) -> tuple[int, int]:
    """Find the small shift that best lines up a part, since tall parts lean with camera angle."""
    x0, y0, x1, y1 = box
    height, width = test_gray.shape
    search = test_gray[max(y0 - search_px, 0) : min(y1 + search_px, height), max(x0 - search_px, 0) : min(x1 + search_px, width)]
    if search.shape[0] < golden_crop.shape[0] or search.shape[1] < golden_crop.shape[1] or golden_crop.std() < 1e-3:
        return 0, 0
    scores = cv2.matchTemplate(search, golden_crop, cv2.TM_CCOEFF_NORMED)
    _, _, _, (best_x, best_y) = cv2.minMaxLoc(scores)
    return best_x - (x0 - max(x0 - search_px, 0)), best_y - (y0 - max(y0 - search_px, 0))


def _ssim_map(first: np.ndarray, second: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the mean SSIM of two same-sized grayscale crops, and its per-pixel map."""
    win_size = min(7, min(first.shape) // 2 * 2 - 1)
    score, full = structural_similarity(first, second, data_range=255, win_size=max(win_size, 3), full=True)
    return float(score), full


def _shrink(image: np.ndarray, scale: float) -> np.ndarray:
    """Shrink by averaging pixels, which removes fine texture. A scale of 1 or more changes nothing."""
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else image


def appearance(
    golden: ComparisonImage,
    test: ComparisonImage,
    box: Box,
    part_mask: np.ndarray | None,
    compare_cfg: Mapping[str, Any],
) -> Appearance:
    """Compare a crop in both images, after correcting a small shift, at a scale set by the crop's size.

    A part that is gone or replaced changes the crop at a coarse scale, whereas glints, mesh moire and
    a pixel of misalignment only change it at a fine scale. So every crop is shrunk to the same size
    first: on clean real photos, full-size SSIM fell as low as 0.4 for a part that had not changed.

    SSIM follows structure, and hardly reacts when one flat colour replaces another, so a bare
    colour difference is measured too. Both catch what the other misses.
    """
    x0, y0, x1, y1 = box
    golden_crop = golden.gray[y0:y1, x0:x1]
    if min(golden_crop.shape) < _MIN_CROP_SIDE:
        return Appearance(1.0, 0.0, 0.0)
    search_px = max(int(compare_cfg["shift_search_ratio"] * min(golden_crop.shape)), 2)
    dx, dy = best_shift(golden_crop, test.gray, box, search_px)
    height, width = test.gray.shape
    if y0 + dy < 0 or x0 + dx < 0 or y1 + dy > height or x1 + dx > width:
        dx, dy = 0, 0

    scale = compare_cfg["compare_side_px"] / min(golden_crop.shape)
    golden_small = _shrink(golden_crop, scale)
    test_small = _shrink(test.gray[y0 + dy : y1 + dy, x0 + dx : x1 + dx], scale)
    score, full = _ssim_map(golden_small, test_small)

    # int16 because the difference of two unsigned bytes would wrap around instead of going negative.
    golden_rgb = _shrink(golden.rgb[y0:y1, x0:x1], scale).astype(np.int16)
    test_rgb = _shrink(test.rgb[y0 + dy : y1 + dy, x0 + dx : x1 + dx], scale).astype(np.int16)
    changed = np.abs(golden_rgb - test_rgb).max(axis=2) > compare_cfg["diff_value"]

    if part_mask is None:
        return Appearance(score, 0.0, float(changed.mean()))
    # dsize is (width, height), the reverse of a NumPy shape, so the slice flips the order.
    inside = cv2.resize(
        part_mask[y0:y1, x0:x1].astype(np.uint8), golden_small.shape[::-1], interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    if not inside.any():
        return Appearance(score, 0.0, float(changed.mean()))
    return Appearance(
        score,
        float((full[inside] < compare_cfg["low_ssim_value"]).mean()),
        float(changed[inside].mean()),
    )


def _edge_magnitude(image: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centred, second_centred = first - first.mean(), second - second.mean()
    scale = float(np.linalg.norm(first_centred) * np.linalg.norm(second_centred))
    return float((first_centred * second_centred).sum() / scale) if scale > 1e-6 else 0.0


def rotation_test(
    golden_gray: np.ndarray,
    test_gray: np.ndarray,
    box: Sequence[float],
    angles: Sequence[int],
    compare_cfg: Mapping[str, Any],
) -> tuple[int, float]:
    """Compare the test part with the golden part turned by each angle, on SSIM plus edge correlation.

    Returns the best angle, and how far it beats 0 degrees. A non-zero angle winning clearly
    means the part sits turned. The crop is square and tight, so turning it keeps the part in frame.
    """
    x0, y0, x1, y1 = box
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    half = int(max(x1 - x0, y1 - y0) * (0.5 + compare_cfg["rotation_crop_margin"])) + 1
    height, width = golden_gray.shape
    left, top = int(centre_x) - half, int(centre_y) - half
    right, bottom = int(centre_x) + half, int(centre_y) + half
    if left < 0 or top < 0 or right > width or bottom > height or half * 2 < _MIN_CROP_SIDE:
        return 0, 0.0

    golden_crop = golden_gray[top:bottom, left:right]
    test_crop = test_gray[top:bottom, left:right]
    test_edges = _edge_magnitude(test_crop)
    scores: dict[int, float] = {}
    for angle in angles:
        # rot90 turns counter-clockwise in quarter turns, so k is the angle over 90.
        turned = np.rot90(golden_crop, k=angle // 90)
        ssim_score, _ = _ssim_map(turned, test_crop)
        scores[angle] = 0.5 * ssim_score + 0.5 * _correlation(_edge_magnitude(turned), test_edges)
    best = max(scores, key=lambda angle: scores[angle])
    return best, scores[best] - scores[0]


def rotation_angles(part: Part) -> tuple[int, ...]:
    """Return the turns a part could physically be in. A DIP or two-lead part cannot sit at 90 degrees."""
    if part.cls == "ic" and (part.package or "").startswith("DIP"):
        return (0, 180)
    if part.cls in ("diode", "electrolytic_capacitor", "led"):
        return (0, 180)
    return (0, 90, 180, 270)


def classify_part(
    part: Part,
    matched: tuple[int, float] | None,
    offset_ratio: float,
    look: Appearance,
    best_angle: int,
    rotation_margin: float,
    compare_cfg: Mapping[str, Any],
) -> FaultType | None:
    """Apply the decision rules of section 3, Stage 3, step 3, returning None for an OK part."""
    if matched is None:
        gone = look.ssim < compare_cfg["missing_ssim"] or look.diff_fraction > compare_cfg["missing_diff_fraction"]
        return "MISSING" if gone else None

    if part.polarized and best_angle != 0 and rotation_margin > compare_cfg["rotation_margin"]:
        return "REVERSED" if best_angle == 180 else "ROTATED"

    _, iou = matched
    looks_changed = (
        look.ssim < compare_cfg["same_appearance_ssim"] or look.diff_fraction > compare_cfg["same_appearance_diff_fraction"]
    )
    if looks_changed and (offset_ratio > compare_cfg["shifted_offset_ratio"] or iou < compare_cfg["shifted_min_iou"]):
        return "SHIFTED"
    far_off = (
        look.ssim < compare_cfg["wrong_part_ssim"]
        or look.low_fraction > compare_cfg["wrong_part_low_fraction"]
        or look.diff_fraction > compare_cfg["wrong_part_diff_fraction"]
    )
    return "WRONG_PART" if far_off else None


def compare_boards(
    golden: GoldenProfile,
    golden_image: np.ndarray,
    test_image: np.ndarray,
    test_parts: Sequence[PartMask],
    compare_cfg: Mapping[str, Any],
) -> list[Finding]:
    """Check every inspected golden part at its own spot, then look for parts the golden board lacks."""
    golden_view = comparison_image(golden_image)
    test_view = comparison_image(test_image)
    height, width = golden_view.gray.shape
    margin = compare_cfg["crop_margin"]

    matches = match_parts(golden.parts, test_parts, compare_cfg)
    findings: list[Finding] = []
    for index, part in enumerate(golden.parts):
        if part.ref is None:  # A known object with no reference, kept only so it does not read as EXTRA.
            continue
        crop = expand_box(part.bbox, margin, width, height)
        look = appearance(golden_view, test_view, crop, rle_to_mask(part.mask_rle), compare_cfg)

        matched = matches.get(index)
        iou = matched[1] if matched else 0.0
        offset_ratio = 0.0
        if matched:
            test_centroid = np.array(mask_centroid(test_parts[matched[0]].rle))
            shorter_side = max(min(part.bbox[2] - part.bbox[0], part.bbox[3] - part.bbox[1]), 1)
            offset_ratio = float(np.linalg.norm(test_centroid - np.array(part.centroid)) / shorter_side)

        best_angle, rotation_margin = 0, 0.0
        if part.polarized and matched:
            best_angle, rotation_margin = rotation_test(
                golden_view.gray, test_view.gray, part.bbox, rotation_angles(part), compare_cfg
            )

        finding = classify_part(part, matched, offset_ratio, look, best_angle, rotation_margin, compare_cfg)
        if finding is not None:
            findings.append(
                Finding(
                    golden_ref=part.ref,
                    cv_finding=finding,
                    bbox=crop,
                    part_bbox=part.bbox,
                    evidence={
                        "ssim": look.ssim,
                        "iou": iou,
                        "offset_ratio": offset_ratio,
                        "best_rotation_deg": float(best_angle),
                        "rotation_margin": rotation_margin,
                        "low_fraction": look.low_fraction,
                        "diff_fraction": look.diff_fraction,
                    },
                )
            )

    matched_test = {test_index for test_index, _ in matches.values()}
    findings.extend(_find_extras(golden, golden_view, test_view, test_parts, matched_test, compare_cfg))
    return findings


def _find_extras(
    golden: GoldenProfile,
    golden_view: ComparisonImage,
    test_view: ComparisonImage,
    test_parts: Sequence[PartMask],
    matched_test: set[int],
    compare_cfg: Mapping[str, Any],
) -> list[Finding]:
    """Return test masks that match no golden mask, overlap none, and sit where the photo really differs."""
    height, width = golden_view.gray.shape
    golden_rles = [part.mask_rle for part in golden.parts]
    extras: list[Finding] = []
    for index, test_part in enumerate(test_parts):
        if index in matched_test:
            continue
        area = float(mask_utils.area(test_part.rle))
        if golden_rles and area > 0:
            overlaps = [
                float(mask_utils.area(mask_utils.merge([test_part.rle, rle], intersect=True))) / area
                for rle in golden_rles
            ]
            if max(overlaps) > compare_cfg["extra_max_golden_overlap"]:
                continue
        box = tuple(int(value) for value in test_part.box)
        crop = expand_box(box, compare_cfg["crop_margin"], width, height)
        look = appearance(golden_view, test_view, crop, rle_to_mask(test_part.rle), compare_cfg)
        if look.ssim < compare_cfg["extra_ssim"] or look.diff_fraction > compare_cfg["extra_diff_fraction"]:
            extras.append(
                Finding(
                    golden_ref=None,
                    cv_finding="EXTRA",
                    bbox=crop,
                    part_bbox=box,
                    evidence={
                        "ssim": look.ssim,
                        "iou": 0.0,
                        "offset_ratio": 0.0,
                        "best_rotation_deg": 0.0,
                        "rotation_margin": 0.0,
                        "low_fraction": look.low_fraction,
                        "diff_fraction": look.diff_fraction,
                    },
                )
            )
    return extras
