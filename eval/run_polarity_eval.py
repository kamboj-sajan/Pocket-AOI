"""Measure the orientation check on real reversed parts, separately from the main evaluation.

VisA has no reversed parts. The PCBA set on Roboflow (CC BY 4.0) does: phone photos of one board design,
with through-hole diodes and electrolytic capacitors really fitted backwards. Its photos are 640 x 640
close-ups, so most never show the whole board, and Stages 1 and 2 cannot run on them. This script
therefore tests only the rest of the path. Each photo is aligned to a correct photo of the same part with
the pipeline's own SIFT code, and the part is judged by the same Stage 3 functions and frozen thresholds.
Its boxes come from the dataset's labels rather than from SAM, and the README must say so.

Each labelled reversed part is a positive. The same part in another shot, with no label on it, is a
control that counts false alarms. Only one copy of each original shot is used.

Run from the repo root, after eval/fetch_roboflow_pcba.py:
    python -m eval.run_polarity_eval
"""

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pocket_aoi.align import refine_with_sift, warp_image, warp_mask
from pocket_aoi.compare import (
    Box,
    appearance,
    classify_part,
    comparison_image,
    expand_box,
    rotation_angles,
    rotation_test,
)
from pocket_aoi.composite import build_composite, encode_png
from pocket_aoi.config import REPO_ROOT, load_config
from pocket_aoi.llm import LLMClient, LLMSettings
from pocket_aoi.schemas import GoldenProfile, Part, Suspect
from pocket_aoi.verdict import judge_suspects

DATA_DIR = REPO_ROOT / "data" / "raw" / "roboflow_pcba"
OUT_DIR = REPO_ROOT / "runs" / "eval" / "polarity"

REVERSED_LABELS = {"Diode Opposite polarity", "Electrolytic capacitor Opposite polarity"}
MISSING_LABELS = {"Diode-missed-welding", "Electrolytic capacitor missed welding"}
SPLIT_ORDER = {"test": 0, "valid": 1, "train": 2}  # Roboflow only augments the train split.

# One shot per part, chosen by eye, where the part's own silkscreen ref is readable and only it has this label.
ANCHORS: dict[str, tuple[str, str]] = {
    "D1": ("IMG_20230702_111217.jpg", "Diode Opposite polarity"),
    "D2": ("IMG_20230703_190444.jpg", "Diode Opposite polarity"),
    "E1": ("IMG_20230702_111711.jpg", "Electrolytic capacitor Opposite polarity"),
    "E2": ("IMG_20230703_143936.jpg", "Electrolytic capacitor Opposite polarity"),
    "E4": ("IMG_20230703_193145.jpg", "Electrolytic capacitor Opposite polarity"),
}

# Where each part's marker sits in its golden crop, written by hand from the golden photo, as Stage 0 setup would.
MARKER_HINTS: dict[str, str] = {
    "D1": "the wider light grey band is at the bottom end of the diode body",
    "D2": "the light band is at the right end of the diode body",
    "E1": "the black crescent on the can's top face is on the upper side",
    "E2": "the black crescent on the can's top face is on the upper side",
    "E4": "the black crescent on the can's top face is on the lower side",
}

MIN_LABEL_IOU = 0.3
MIN_GOLDEN_INLIERS = 100
MIN_VISIBLE_SHARE = 0.95


@dataclass(frozen=True)
class Row:
    """One part in one shot, with its label and what Stage 3 made of it."""

    ref: str
    shot: str
    split: str
    truth: str
    inliers: int
    median_error_px: float
    ssim: float
    diff_fraction: float
    best_angle: int
    rotation_margin: float
    finding: str | None
    box: tuple[int, int, int, int]
    composite: str


def label_box(box: Mapping[str, float]) -> tuple[float, float, float, float]:
    """Turn Roboflow's centre, width and height into corners."""
    half_width, half_height = box["width"] / 2, box["height"] / 2
    return (box["x"] - half_width, box["y"] - half_height, box["x"] + half_width, box["y"] + half_height)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    width = min(first[2], second[2]) - max(first[0], second[0])
    height = min(first[3], second[3]) - max(first[1], second[1])
    if width <= 0 or height <= 0:
        return 0.0
    overlap = width * height
    area = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1])
    return overlap / (area - overlap)


def project_box(box: Sequence[float], homography: np.ndarray) -> tuple[float, float, float, float]:
    """Return the bounding box of a box's corners after a homography."""
    x0, y0, x1, y1 = box
    corners = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
    points = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    return (float(points[:, 0].min()), float(points[:, 1].min()), float(points[:, 0].max()), float(points[:, 1].max()))


def pick_shots(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Keep one copy of each original shot, preferring a copy Roboflow did not augment."""
    chosen: dict[str, dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: (SPLIT_ORDER.get(item["split"], 3), item["id"])):
        chosen.setdefault(record["name"], record)
    return chosen


def load_shot(record: Mapping[str, Any]) -> np.ndarray:
    with Image.open(DATA_DIR / "source" / f"{record['id']}.jpg") as image:
        return np.asarray(image.convert("RGB"))


def align_pair(test: np.ndarray, golden: np.ndarray, align_cfg: Mapping[str, Any]) -> tuple[np.ndarray, int, float] | None:
    """Find the homography from one photo to another with the pipeline's two SIFT passes, or None."""
    height, width = golden.shape[:2]
    everything = np.ones((height, width), bool)
    first = refine_with_sift(test, golden, everything, everything, align_cfg)
    if first is None or first.inliers < align_cfg["min_inliers_to_iterate"]:
        return None
    warped = warp_image(test, first.homography, (width, height))
    visible = warp_mask(everything, first.homography, (width, height))
    second = refine_with_sift(warped, golden, visible, everything, align_cfg)
    if second is None or not second.accepted:
        return None
    homography = second.homography @ first.homography
    # A mirrored copy would have a negative determinant, and no real camera can produce one.
    if np.linalg.det(homography[:2, :2]) <= 0:
        return None
    return homography, second.inliers, second.median_error_px


def part_for(ref: str, box: Box) -> Part:
    """Describe the part as the pipeline would. The mask is unused here, because the box comes from a label."""
    x0, y0, x1, y1 = box
    cls = "diode" if ref.startswith("D") else "electrolytic_capacitor"
    return Part(
        ref=ref,
        cls=cls,
        bbox=box,
        centroid=((x0 + x1) / 2, (y0 + y1) / 2),
        mask_rle={},
        score=1.0,
        package=None,
        polarized=True,
        marker_hint=MARKER_HINTS.get(ref),
    )


def status_in_shot(record: Mapping[str, Any], box: Sequence[float], ref: str) -> str:
    """Read a part's state in one shot from the labels of its own kind that overlap it."""
    wanted = "Diode" if ref.startswith("D") else "Electrolytic"
    overlapping = [item["label"] for item in record["boxes"] if item["label"].startswith(wanted) and box_iou(label_box(item), box) >= MIN_LABEL_IOU]
    if any(label in REVERSED_LABELS for label in overlapping):
        return "REVERSED"
    if any(label in MISSING_LABELS for label in overlapping):
        return "MISSING"
    return "OK"


def locate_part(
    ref: str,
    shots: Mapping[str, Mapping[str, Any]],
    images: Mapping[str, np.ndarray],
    align_cfg: Mapping[str, Any],
) -> tuple[tuple[float, float, float, float], dict[str, dict[str, Any]]]:
    """Map a part's anchor box into every shot that shows it, and read its state there."""
    anchor_name, anchor_label = ANCHORS[ref]
    labelled = [item for item in shots[anchor_name]["boxes"] if item["label"] == anchor_label]
    if len(labelled) != 1:
        raise ValueError(f"{anchor_name} should hold exactly one {anchor_label}, but holds {len(labelled)}")
    anchor_box = label_box(labelled[0])

    located: dict[str, dict[str, Any]] = {anchor_name: {"box": anchor_box, "status": "REVERSED", "inliers": 10_000}}
    for name, record in shots.items():
        if name == anchor_name:
            continue
        aligned = align_pair(images[anchor_name], images[name], align_cfg)
        if aligned is None:
            continue
        homography, inliers, _ = aligned
        box = project_box(anchor_box, homography)
        height, width = images[name].shape[:2]
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            continue
        located[name] = {"box": box, "status": status_in_shot(record, box, ref), "inliers": inliers}
    return anchor_box, located


def choose_golden(located: Mapping[str, Mapping[str, Any]]) -> str:
    """Pick the correct shot that shows the part largest, among those matching its anchor well."""
    candidates = [name for name, item in located.items() if item["status"] == "OK" and item["inliers"] >= MIN_GOLDEN_INLIERS]
    if not candidates:
        raise ValueError("no correct shot of this part aligns well enough to act as golden")

    def area(name: str) -> float:
        x0, y0, x1, y1 = located[name]["box"]
        return (x1 - x0) * (y1 - y0)

    return max(candidates, key=area)


def build_alignments(
    parts: Sequence[str],
    shots: Mapping[str, Mapping[str, Any]],
    images: Mapping[str, np.ndarray],
    align_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Do the slow SIFT work once: find each part in every shot, choose its golden shot, and align to it."""
    alignments: dict[str, Any] = {"parts": {}, "unmapped": []}
    claimed: dict[str, list[tuple[float, float, float, float]]] = {}
    for ref in parts:
        _, located = locate_part(ref, shots, images, align_cfg)
        golden_name = choose_golden(located)
        golden_box = [int(round(value)) for value in located[golden_name]["box"]]
        entries: dict[str, Any] = {}
        for name, item in located.items():
            if item["status"] in ("REVERSED", "OK"):
                claimed.setdefault(name, []).append(item["box"])
            if name == golden_name or item["status"] == "MISSING":
                continue
            aligned = align_pair(images[name], images[golden_name], align_cfg)
            entry: dict[str, Any] = {"status": item["status"]}
            if aligned is not None:
                entry.update(homography=aligned[0].tolist(), inliers=aligned[1], median_error_px=aligned[2])
            entries[name] = entry
        states = Counter(item["status"] for item in located.values())
        print(f"{ref}: located in {len(located)} shots {dict(states)}; golden {golden_name} box {golden_box}", flush=True)
        alignments["parts"][ref] = {"golden": golden_name, "box": golden_box, "shots": entries}

    for name, record in shots.items():
        for item in record["boxes"]:
            if item["label"] in REVERSED_LABELS and not any(box_iou(label_box(item), box) >= MIN_LABEL_IOU for box in claimed.get(name, [])):
                alignments["unmapped"].append([name, item["label"]])
    return alignments


def judge_part(
    ref: str,
    name: str,
    status: str,
    record: Mapping[str, Any],
    golden: np.ndarray,
    test: np.ndarray,
    box: Box,
    aligned: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[Row | None, str]:
    """Run Stage 3's checks on the part in one aligned shot. Returns a reason when skipped."""
    if "homography" not in aligned:
        return None, "did not align to golden"
    homography = np.array(aligned["homography"])
    inliers, median_error = aligned["inliers"], aligned["median_error_px"]
    height, width = golden.shape[:2]
    warped = warp_image(test, homography, (width, height))
    visible = warp_mask(np.ones(test.shape[:2], bool), homography, (width, height))
    compare_cfg = cfg["compare"]
    crop = expand_box(box, compare_cfg["crop_margin"], width, height)
    if visible[crop[1] : crop[3], crop[0] : crop[2]].mean() < MIN_VISIBLE_SHARE:
        return None, "part not fully in view"

    golden_view, test_view = comparison_image(golden), comparison_image(warped)
    part = part_for(ref, box)
    look = appearance(golden_view, test_view, crop, None, compare_cfg)
    best_angle, margin = rotation_test(golden_view.gray, test_view.gray, box, rotation_angles(part), compare_cfg)
    finding = classify_part(part, (0, 1.0), 0.0, look, best_angle, margin, compare_cfg)

    composite = build_composite(
        golden, warped, crop, box, cfg["verdict"]["composite_min_part_px"], cfg["verdict"]["composite_max_side_px"]
    )
    composite_path = OUT_DIR / "composites" / f"{ref}_{status}_{Path(name).stem}.png"
    composite_path.write_bytes(encode_png(composite))
    row = Row(
        ref=ref,
        shot=name,
        split=str(record["split"]),
        truth=status,
        inliers=inliers,
        median_error_px=median_error,
        ssim=look.ssim,
        diff_fraction=look.diff_fraction,
        best_angle=best_angle,
        rotation_margin=margin,
        finding=finding,
        box=box,
        composite=str(composite_path),
    )
    return row, ""


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})" if denominator else "none"


def format_table(rows: Sequence[Row]) -> str:
    """Show, per part, how many reversed parts Stage 3 caught and how many correct parts it flagged."""
    lines = [
        "| Part | Reversed: called REVERSED | Reversed: flagged as anything | Correct: called REVERSED | Correct: flagged as anything |",
        "| --- | --- | --- | --- | --- |",
    ]
    for ref in [*sorted({row.ref for row in rows}), "ALL"]:
        members = [row for row in rows if ref in ("ALL", row.ref)]
        positives = [row for row in members if row.truth == "REVERSED"]
        controls = [row for row in members if row.truth == "OK"]
        lines.append(
            f"| {ref} | {_rate(sum(r.finding == 'REVERSED' for r in positives), len(positives))} "
            f"| {_rate(sum(r.finding is not None for r in positives), len(positives))} "
            f"| {_rate(sum(r.finding == 'REVERSED' for r in controls), len(controls))} "
            f"| {_rate(sum(r.finding is not None for r in controls), len(controls))} |"
        )
    return "\n".join(lines)


def format_margin_sweep(rows: Sequence[Row]) -> str:
    """Show what each possible rotation margin would catch and wrongly flag, for choosing one on dev shots."""
    positives = [row for row in rows if row.truth == "REVERSED"]
    controls = [row for row in rows if row.truth == "OK"]
    lines = ["| Margin | Reversed caught | Correct wrongly called turned |", "| --- | --- | --- |"]
    for margin in (0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4):
        caught = sum(row.best_angle == 180 and row.rotation_margin > margin for row in positives)
        wrong = sum(row.best_angle == 180 and row.rotation_margin > margin for row in controls)
        lines.append(f"| {margin:.2f} | {_rate(caught, len(positives))} | {_rate(wrong, len(controls))} |")
    return "\n".join(lines)


def judge_with_glimmer(rows: Sequence[Row], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Run Stage 4 on what Stage 3 flagged, as the pipeline does. An unflagged part never reaches the model."""
    client = LLMClient(LLMSettings.from_config(cfg))
    profile = GoldenProfile(
        board_id="roboflow-pcba",
        canonical_size=(640, 640),
        parts=[part_for(ref, next(row.box for row in rows if row.ref == ref)) for ref in sorted({row.ref for row in rows})],
    )
    flagged = [row for row in rows if row.finding is not None]
    suspects = [
        Suspect(
            golden_ref=row.ref,
            cv_finding=row.finding,
            bbox=row.box,
            evidence={"ssim": row.ssim, "best_rotation_deg": float(row.best_angle), "offset_ratio": 0.0},
            composite_path=row.composite,
        )
        for row in flagged
    ]
    judged = dict(zip((row.composite for row in flagged), judge_suspects(client, suspects, profile, None, cfg)))

    results = []
    for row in rows:
        item = judged.get(row.composite)
        fault = item.fault if item else None
        results.append(
            {
                **asdict(row),
                "verdict": item.verdict.verdict if item else None,
                "confidence": item.verdict.confidence if item else None,
                "evidence": item.verdict.evidence if item else None,
                "status": fault.status if fault else None,
                "final_type": fault.fault_type if fault else None,
            }
        )
    return results


def format_fused_table(results: Sequence[Mapping[str, Any]]) -> str:
    """Show the end result per part: what a user would be told, after Muse Glimmer's verdict."""
    lines = [
        "| Part | Reversed: confirmed REVERSED | Reversed: confirmed, any type | Reversed: needs review | Unlabelled: confirmed fault | Unlabelled: needs review |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for ref in [*sorted({item["ref"] for item in results}), "ALL"]:
        members = [item for item in results if ref in ("ALL", item["ref"])]
        positives = [item for item in members if item["truth"] == "REVERSED"]
        controls = [item for item in members if item["truth"] == "OK"]
        confirmed = [item for item in positives if item["status"] == "confirmed"]
        lines.append(
            f"| {ref} | {_rate(sum(item['final_type'] in ('REVERSED', 'ROTATED') for item in confirmed), len(positives))} "
            f"| {_rate(len(confirmed), len(positives))} "
            f"| {_rate(sum(item['status'] == 'needs_review' for item in positives), len(positives))} "
            f"| {_rate(sum(item['status'] == 'confirmed' for item in controls), len(controls))} "
            f"| {_rate(sum(item['status'] == 'needs_review' for item in controls), len(controls))} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parts", nargs="*", default=sorted(ANCHORS), help="default is every part with an anchor")
    parser.add_argument("--holdout", action="store_true", help="also score the test and valid shots; do this once, after freezing the margin")
    parser.add_argument("--search-ratio", type=float, help="try another rotation_search_ratio on dev shots")
    args = parser.parse_args()

    cfg = load_config()
    if args.search_ratio is not None:
        cfg["compare"]["rotation_search_ratio"] = args.search_ratio
    (OUT_DIR / "composites").mkdir(parents=True, exist_ok=True)
    records = json.loads((DATA_DIR / "records.json").read_text(encoding="utf-8"))
    shots = pick_shots(records)
    images = {name: load_shot(record) for name, record in shots.items()}
    print(f"{len(shots)} distinct shots", flush=True)

    alignments_path = OUT_DIR / "alignments.json"
    if alignments_path.exists():
        alignments = json.loads(alignments_path.read_text(encoding="utf-8"))
    else:
        alignments = build_alignments(args.parts, shots, images, cfg["align"])
        alignments_path.write_text(json.dumps(alignments), encoding="utf-8")

    rows: list[Row] = []
    for ref, part in alignments["parts"].items():
        skipped: Counter[str] = Counter()
        box = tuple(part["box"])
        for name, aligned in part["shots"].items():
            row, reason = judge_part(ref, name, aligned["status"], shots[name], images[part["golden"]], images[name], box, aligned, cfg)
            if row is None:
                skipped[f"{aligned['status']}: {reason}"] += 1
            else:
                rows.append(row)
        print(f"{ref}: golden {part['golden']}, judged {sum(row.ref == ref for row in rows)}, skipped {dict(skipped)}", flush=True)

    total_labels = sum(item["label"] in REVERSED_LABELS for record in shots.values() for item in record["boxes"])
    print(f"reversed labels on distinct shots: {total_labels}; not mapped to any part above: {len(alignments['unmapped'])}")

    (OUT_DIR / "results_cv.json").write_text(json.dumps([asdict(row) for row in rows], indent=1), encoding="utf-8")
    dev = [row for row in rows if row.split == "train"]
    print(f"\nDev shots (train split), search ratio {cfg['compare']['rotation_search_ratio']}:\n" + format_table(dev))
    print("\n" + format_margin_sweep(dev))
    if args.holdout:
        held_out = [row for row in rows if row.split != "train"]
        table = format_table(held_out)
        (OUT_DIR / "summary_cv.md").write_text(table + "\n", encoding="utf-8")
        print(f"\nHeld-out shots (test and valid splits), margin {cfg['compare']['rotation_margin']}:\n" + table, flush=True)
        results = judge_with_glimmer(held_out, cfg)
        (OUT_DIR / "results_fused.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        fused_table = format_fused_table(results)
        (OUT_DIR / "summary_fused.md").write_text(fused_table + "\n", encoding="utf-8")
        print("\nHeld-out shots, after Muse Glimmer:\n" + fused_table)


if __name__ == "__main__":
    main()
