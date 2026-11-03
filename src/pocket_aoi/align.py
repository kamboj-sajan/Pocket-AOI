"""Stage 1: board quad, homography into the canonical frame, SIFT refinement, quality gate.

Images here are RGB numpy arrays, and masks are H x W bool arrays. The board mask comes from
segment.py, because model calls stay out of this module.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class GoldenFrame:
    """The golden photo warped into the canonical frame that every test photo is compared in."""

    image: np.ndarray
    mask: np.ndarray
    homography: np.ndarray    # original photo pixels -> canonical pixels
    size: tuple[int, int]     # width, height


@dataclass(frozen=True)
class Refinement:
    """The result of matching SIFT features between two canonical images."""

    homography: np.ndarray
    inliers: int
    median_error_px: float
    accepted: bool


@dataclass(frozen=True)
class AlignResult:
    """A test photo in the golden frame, or the reason it must be retaken."""

    ok: bool
    message: str
    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    homography: np.ndarray | None = None
    board_iou: float = 0.0
    refinement: Refinement | None = None
    tilt_ratio: float = 1.0


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return four corners clockwise on screen, starting from the top-left."""
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    # argsort returns the indices that would sort the array; y points down, so rising angle is clockwise.
    clockwise = points[np.argsort(angles)]
    start = int(np.argmin(clockwise.sum(axis=1)))
    return np.roll(clockwise, -start, axis=0)


def _fit_board_lines(points: np.ndarray, align_cfg: Mapping[str, Any]) -> list[np.ndarray]:
    """Find the four straightest runs of a contour by RANSAC, as homogeneous lines.

    A board's edges are its longest straight runs, even when a connector splits one in two or a
    chamfer shortens it. Each found line's points are removed before searching for the next.
    """
    rng = np.random.default_rng(0)
    tolerance = align_cfg["edge_tolerance_ratio"] * float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    remaining = points
    lines: list[np.ndarray] = []
    for _ in range(4):
        if len(remaining) < 2:
            break
        best_on_line: np.ndarray | None = None
        for _ in range(align_cfg["edge_ransac_iterations"]):
            first, second = remaining[rng.choice(len(remaining), size=2, replace=False)]
            direction = second - first
            length = float(np.linalg.norm(direction))
            if length < 1.0:
                continue
            # A unit normal gives each point's perpendicular distance as a dot product, across the whole array.
            normal = np.array([-direction[1], direction[0]]) / length
            on_line = np.abs((remaining - first) @ normal) <= tolerance
            if best_on_line is None or on_line.sum() > best_on_line.sum():
                best_on_line = on_line
        if best_on_line is None:
            break
        lines.append(_least_squares_line(remaining[best_on_line]))
        remaining = remaining[~best_on_line]
    return lines


def _least_squares_line(points: np.ndarray) -> np.ndarray:
    """Fit a homogeneous line through points, minimising perpendicular distance."""
    vx, vy, x0, y0 = cv2.fitLine(points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([-vy, vx, vy * x0 - vx * y0], dtype=np.float64)


def _intersect(first: np.ndarray, second: np.ndarray) -> np.ndarray | None:
    """Intersect two homogeneous lines; the cross product of two lines is their shared point."""
    x, y, w = np.cross(first, second)
    return None if abs(w) < 1e-9 else np.array([x / w, y / w])


def _quad_from_lines(lines: list[np.ndarray]) -> np.ndarray | None:
    """Pair four lines into two roughly parallel pairs and return their four intersections."""
    if len(lines) != 4:
        return None
    # Line directions repeat every 180 degrees, so sort them on that circle and try both pairings.
    angles = [float(np.degrees(np.arctan2(line[1], line[0])) % 180) for line in lines]
    order = np.argsort(angles)
    first, second, third, fourth = (lines[i] for i in order)
    gap_a = angles[order[1]] - angles[order[0]]
    gap_b = angles[order[3]] - angles[order[2]]
    if max(gap_a, gap_b) > 30:
        # A side near 0 degrees can sit at both ends of the sorted list.
        first, second, third, fourth = second, third, fourth, first
        gap_a = angles[order[2]] - angles[order[1]]
        gap_b = (angles[order[0]] + 180 - angles[order[3]]) % 180
        if max(gap_a, gap_b) > 30:
            return None
    corners = [_intersect(first, third), _intersect(first, fourth), _intersect(second, fourth), _intersect(second, third)]
    if any(corner is None for corner in corners):
        return None
    return np.array(corners)


def _contour_area(points: np.ndarray) -> float:
    return float(cv2.contourArea(points.astype(np.float32)))


def fit_quad(mask: np.ndarray, align_cfg: Mapping[str, Any]) -> np.ndarray:
    """Fit four board corners to a mask, ordered clockwise from the top-left.

    cv2.approxPolyDP settles it when the outline is a clean quadrilateral. Overhanging parts and
    chamfered corners add vertices, and then the four straightest edges of the contour decide.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("the board mask is empty")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    epsilon = align_cfg["polygon_epsilon_ratio"] * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float64)

    if len(polygon) == 4:
        quad: np.ndarray | None = polygon
    else:
        quad = _quad_from_lines(_fit_board_lines(contour.reshape(-1, 2).astype(np.float64), align_cfg))

    if quad is not None:
        ordered = order_corners(quad)
        if abs(_contour_area(ordered) - area) <= align_cfg["max_quad_area_error"] * area:
            return ordered
    return order_corners(cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float64))


def canonical_frame(quad: np.ndarray, long_side: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reorder a quad so its long side runs along x, and return the canonical size for it."""
    edges = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)  # top, right, bottom, left
    along_x = (edges[0] + edges[2]) / 2
    along_y = (edges[1] + edges[3]) / 2
    if along_y > along_x:
        quad = np.roll(quad, -1, axis=0)
        along_x, along_y = along_y, along_x
    return quad, (long_side, round(long_side * along_y / along_x))


def quad_homography(quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return the homography taking four quad corners to the corners of the canonical rectangle."""
    width, height = size
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad.astype(np.float32), target)


def warp_image(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp into the canonical frame, blurring first when shrinking so fine detail does not alias."""
    # The square root of the Jacobian determinant of the linear part is the local scale factor.
    shrink = 1.0 / max(np.sqrt(abs(np.linalg.det(homography[:2, :2]))), 1e-6)
    if shrink > 1.2:
        image = cv2.GaussianBlur(image, (0, 0), sigmaX=0.4 * shrink)
    return cv2.warpPerspective(image, homography, size, flags=cv2.INTER_LINEAR)


def warp_mask(mask: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Warp a bool mask, with nearest-neighbour sampling so it stays binary."""
    return cv2.warpPerspective(mask.astype(np.uint8), homography, size, flags=cv2.INTER_NEAREST).astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return intersection over union of two bool masks."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def clahe_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB to contrast-equalised grayscale, which keeps SIFT stable across lighting."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def normalise_lighting(image: np.ndarray) -> np.ndarray:
    """Balance colour with gray-world and even out brightness with CLAHE. For comparison only."""
    pixels = image.astype(np.float32)
    # NumPy broadcasting: the 3-element gain vector applies to every pixel's three channels.
    gains = pixels.reshape(-1, 3).mean(axis=0).mean() / np.maximum(pixels.reshape(-1, 3).mean(axis=0), 1.0)
    balanced = np.clip(pixels * gains, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(balanced, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def refine_with_sift(
    test: np.ndarray,
    golden: np.ndarray,
    test_mask: np.ndarray,
    golden_mask: np.ndarray,
    align_cfg: Mapping[str, Any],
) -> Refinement | None:
    """Estimate the homography from canonical test to canonical golden, or None without enough matches."""
    sift = cv2.SIFT_create()
    test_points, test_features = sift.detectAndCompute(clahe_gray(test), test_mask.astype(np.uint8) * 255)
    golden_points, golden_features = sift.detectAndCompute(clahe_gray(golden), golden_mask.astype(np.uint8) * 255)
    if test_features is None or golden_features is None or len(test_points) < 2 or len(golden_points) < 2:
        return None

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(test_features, golden_features, k=2)
    # Lowe's ratio test keeps a match only when it clearly beats the runner-up.
    good = [
        pair[0]
        for pair in pairs
        if len(pair) == 2 and pair[0].distance < align_cfg["lowe_ratio"] * pair[1].distance
    ]
    if len(good) < 4:
        return None

    source = np.float32([test_points[match.queryIdx].pt for match in good]).reshape(-1, 1, 2)
    target = np.float32([golden_points[match.trainIdx].pt for match in good]).reshape(-1, 1, 2)
    homography, inlier_flags = cv2.findHomography(source, target, cv2.RANSAC, align_cfg["ransac_px"])
    if homography is None:
        return None

    inliers = inlier_flags.ravel().astype(bool)
    projected = cv2.perspectiveTransform(source[inliers], homography)
    errors = np.linalg.norm(projected - target[inliers], axis=2)
    median_error = float(np.median(errors)) if errors.size else float("inf")
    count = int(inliers.sum())
    accepted = count >= align_cfg["min_inliers"] and median_error <= align_cfg["max_median_reproj_px"]
    return Refinement(homography, count, median_error, accepted)


def filled_hull(mask: np.ndarray) -> np.ndarray:
    """Fill a mask's convex hull, so holes where parts or missing parts confuse SAM do not count."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    if contours:
        cv2.fillConvexPoly(filled, cv2.convexHull(np.vstack(contours)), 1)
    return filled.astype(bool)


def camera_tilt_ratio(homography: np.ndarray, size: tuple[int, int]) -> float:
    """Return how much longer one board edge looks than its opposite edge in the photo.

    The canonical corners are mapped back through the final homography, which comes from board
    features when SIFT succeeds, so overhanging parts do not disturb it. A straight-down photo gives 1.
    """
    width, height = size
    corners = np.float32([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]]])
    quad = cv2.perspectiveTransform(corners, np.linalg.inv(homography)).reshape(-1, 2)
    top, right, bottom, left = np.linalg.norm(np.roll(quad, -1, axis=0) - quad, axis=1)
    return float(max(max(top, bottom) / max(min(top, bottom), 1.0), max(left, right) / max(min(left, right), 1.0)))


def diagnose_retake(
    mask: np.ndarray,
    aligned: np.ndarray,
    golden: GoldenFrame,
    board_iou: float,
    tilt_ratio: float,
    outline_ok: bool,
    align_cfg: Mapping[str, Any],
) -> str:
    """Name the likely reason a photo failed the quality gate, and what to do about it."""
    retake = align_cfg["retake"]

    # Each pair is the mask's outermost row or column and its name; .mean() of bools is the True fraction.
    border = {"top": mask[0, :], "bottom": mask[-1, :], "left": mask[:, 0], "right": mask[:, -1]}
    cut_off = [name for name, line in border.items() if line.mean() > retake["border_touch_ratio"]]
    if cut_off:
        return (
            f"Retake: the board is cut off at the {' and '.join(cut_off)} of the photo. "
            "Move the phone back so the whole board is in frame."
        )

    # A covered board gives a skewed quad, so tilt only counts as the cause when the outline is sound.
    if outline_ok and tilt_ratio > retake["max_side_ratio"]:
        return (
            f"Retake: the camera is at a steep angle (opposite board edges differ by {tilt_ratio:.2f} times). "
            "Hold the phone parallel to the board."
        )

    # Measured against the golden board, because white parts and white boards are not glare.
    def bright_fraction(image: np.ndarray) -> float:
        return float((image.min(axis=2) >= retake["glare_value"])[golden.mask].mean())

    extra_glare = bright_fraction(aligned) - bright_fraction(golden.image)
    if extra_glare > retake["max_glare_fraction"]:
        return f"Retake: glare covers {extra_glare:.0%} more of the board than in the golden photo. Use diffuse light with no flash."

    return (
        f"Retake: the board outline matches the golden board by only {board_iou:.3f}. "
        "Check that nothing covers the board and that it is the same design."
    )


def canonicalise_golden(image: np.ndarray, mask: np.ndarray, align_cfg: Mapping[str, Any]) -> GoldenFrame:
    """Warp the golden photo into the canonical frame, which sets the frame's size and aspect ratio."""
    quad, size = canonical_frame(fit_quad(mask, align_cfg), align_cfg["canonical_long_side"])
    homography = quad_homography(quad, size)
    return GoldenFrame(
        warp_image(image, homography, size),
        warp_mask(mask, homography, size),
        homography,
        size,
    )


def align_to_golden(
    image: np.ndarray,
    mask: np.ndarray | None,
    golden: GoldenFrame,
    align_cfg: Mapping[str, Any],
) -> AlignResult:
    """Warp a test photo into the golden frame and apply the quality gate."""
    if mask is None or not mask.any():
        return AlignResult(False, "Retake: no board found in the photo. Use a plain background that contrasts with it.")

    quad, _ = canonical_frame(fit_quad(mask, align_cfg), golden.size[0])
    quad_only = quad_homography(quad, golden.size)
    homography = quad_only

    # A first pass corrects a quad skewed by overhanging parts, and then the second pass, starting
    # from a close warp, finds many more matches. Only the final pass faces the acceptance rule.
    refinement: Refinement | None = None
    for _ in range(align_cfg["sift_passes"]):
        attempt = refine_with_sift(
            warp_image(image, homography, golden.size),
            golden.image,
            warp_mask(mask, homography, golden.size),
            golden.mask,
            align_cfg,
        )
        if attempt is None or attempt.inliers < align_cfg["min_inliers_to_iterate"]:
            break
        # Applying the refinement after the warp so far is a matrix product, right to left.
        homography = attempt.homography @ homography
        refinement = attempt

    refined = refinement is not None and refinement.accepted
    if not refined:
        homography = quad_only

    aligned_mask = warp_mask(mask, homography, golden.size)
    aligned = warp_image(image, homography, golden.size)
    tilt_ratio = camera_tilt_ratio(homography, golden.size)

    # Accepted SIFT is direct evidence that the board lines up, so its outline only has to agree
    # roughly, with holes filled, because SAM's mask moves with overhanging parts. Without SIFT the
    # raw mask is all there is: a quad-only warp fills the rectangle by construction, so a hull
    # would score near 1 whatever the photo showed.
    if refined:
        board_iou = mask_iou(filled_hull(aligned_mask), filled_hull(golden.mask))
        min_iou = align_cfg["min_board_iou_with_sift"]
    else:
        board_iou = mask_iou(aligned_mask, golden.mask)
        min_iou = align_cfg["min_board_iou"]

    outline_ok = board_iou >= min_iou
    passed = outline_ok and tilt_ratio <= align_cfg["retake"]["max_side_ratio"]
    message = "Aligned." if passed else diagnose_retake(mask, aligned, golden, board_iou, tilt_ratio, outline_ok, align_cfg)
    return AlignResult(passed, message, aligned, aligned_mask, homography, board_iou, refinement, tilt_ratio)


def draw_quad(image: np.ndarray, homography: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Draw the canonical frame's outline on the original photo, for the debug folder."""
    width, height = size
    corners = np.float32([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]]])
    quad = cv2.perspectiveTransform(corners, np.linalg.inv(homography)).astype(np.int32)
    thickness = max(image.shape[1] // 400, 2)
    return cv2.polylines(image.copy(), [quad], isClosed=True, color=(255, 0, 255), thickness=thickness)


def checkerboard_blend(first: np.ndarray, second: np.ndarray, squares_along_x: int = 10) -> np.ndarray:
    """Interleave two same-sized images in squares, so any misalignment breaks lines at the seams."""
    height, width = first.shape[:2]
    cell = max(width // squares_along_x, 1)
    # np.indices gives each pixel's row and column, and // and % work element-wise across the whole array.
    rows, columns = np.indices((height, width))
    use_first = ((rows // cell + columns // cell) % 2 == 0)[:, :, np.newaxis]
    return np.where(use_first, first, second)
