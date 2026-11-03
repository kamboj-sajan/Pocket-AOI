"""Stage 1 against synthetic photos, where the true homography is known exactly."""

import cv2
import numpy as np
import pytest

from pocket_aoi.align import (
    align_to_golden,
    camera_tilt_ratio,
    canonical_frame,
    canonicalise_golden,
    checkerboard_blend,
    diagnose_retake,
    filled_hull,
    fit_quad,
    mask_iou,
    normalise_lighting,
    order_corners,
)
from pocket_aoi.config import load_config

ALIGN_CFG = load_config()["align"]
BOARD_SIZE = (900, 600)  # width, height of the drawn board
PHOTO_SIZE = (3000, 2300)  # width, height of the simulated phone photo


def _draw_board(seed: int = 3, blank: bool = False) -> np.ndarray:
    """Draw a green board with random traces, parts and labels for SIFT to lock onto."""
    width, height = BOARD_SIZE
    board = np.full((height, width, 3), (30, 110, 60), dtype=np.uint8)
    if blank:
        return board
    rng = np.random.default_rng(seed)
    for _ in range(90):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        colour = tuple(int(c) for c in rng.integers(0, 256, size=3))
        kind = rng.integers(0, 3)
        if kind == 0:
            cv2.rectangle(board, (x, y), (x + int(rng.integers(12, 70)), y + int(rng.integers(12, 40))), colour, -1)
        elif kind == 1:
            cv2.circle(board, (x, y), int(rng.integers(6, 30)), colour, -1)
        else:
            cv2.putText(board, f"R{rng.integers(1, 99)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return board


def _take_photo(
    board: np.ndarray, corners: list[tuple[float, float]], seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project the board onto a noisy background. Returns the photo, board mask and true homography."""
    width, height = BOARD_SIZE
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    truth = cv2.getPerspectiveTransform(source, np.float32(corners))
    photo_width, photo_height = PHOTO_SIZE

    rng = np.random.default_rng(seed)
    photo = rng.integers(120, 160, size=(photo_height, photo_width, 3), dtype=np.uint8)
    warped = cv2.warpPerspective(board, truth, PHOTO_SIZE, flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((height, width), np.uint8), truth, PHOTO_SIZE, flags=cv2.INTER_NEAREST).astype(bool)
    photo[mask] = warped[mask]
    return photo, mask, truth


GOLDEN_CORNERS = [(500.0, 350.0), (2550.0, 400.0), (2500.0, 1750.0), (450.0, 1700.0)]


@pytest.mark.parametrize(
    "corners",
    [
        pytest.param([(600, 420), (2400, 360), (2520, 1640), (520, 1700)], id="mild perspective"),
        pytest.param([(700, 300), (2500, 600), (2300, 1800), (500, 1500)], id="tilted"),
        pytest.param([(2500, 1700), (500, 1720), (520, 400), (2480, 380)], id="turned 180 degrees"),
    ],
)
def test_synthetic_homography_recovered_within_2px(corners: list[tuple[float, float]]) -> None:
    board = _draw_board()
    golden_photo, golden_mask, golden_truth = _take_photo(board, GOLDEN_CORNERS, seed=1)
    test_photo, test_mask, test_truth = _take_photo(board, corners, seed=2)

    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    result = align_to_golden(test_photo, test_mask, golden, ALIGN_CFG)

    assert result.ok, result.message
    assert result.refinement is not None and result.refinement.accepted

    # Every board point must land on the same canonical pixel through either photo.
    width, height = BOARD_SIZE
    board_corners = np.float32([[[0, 0]], [[width - 1, 0]], [[width - 1, height - 1]], [[0, height - 1]]])
    via_golden = cv2.perspectiveTransform(board_corners, golden.homography @ golden_truth)
    via_test = cv2.perspectiveTransform(board_corners, result.homography @ test_truth)
    assert np.linalg.norm(via_golden - via_test, axis=2).max() <= 2.0


def test_golden_frame_has_the_long_side_and_the_board_aspect_ratio() -> None:
    golden_photo, golden_mask, _ = _take_photo(_draw_board(), GOLDEN_CORNERS)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    width, height = golden.size
    assert width == ALIGN_CFG["canonical_long_side"]
    assert height == pytest.approx(width * 1350 / 2050, rel=0.05)


def test_blank_board_keeps_the_quad_warp_when_sift_has_nothing_to_match() -> None:
    board = _draw_board(blank=True)
    golden_photo, golden_mask, _ = _take_photo(board, GOLDEN_CORNERS, seed=1)
    test_photo, test_mask, _ = _take_photo(board, [(600, 420), (2400, 360), (2520, 1640), (520, 1700)], seed=2)

    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    result = align_to_golden(test_photo, test_mask, golden, ALIGN_CFG)

    assert result.refinement is None or not result.refinement.accepted
    assert result.ok, result.message


def test_gate_names_the_cut_off_edge() -> None:
    board = _draw_board()
    golden_photo, golden_mask, _ = _take_photo(board, GOLDEN_CORNERS, seed=1)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)

    test_photo, test_mask, _ = _take_photo(board, GOLDEN_CORNERS, seed=2)
    cut = 900  # Crop away the left part of the board.
    result = align_to_golden(test_photo[:, cut:], test_mask[:, cut:], golden, ALIGN_CFG)

    assert not result.ok
    assert "cut off at the left" in result.message


def test_gate_flags_a_steep_angle() -> None:
    board = _draw_board()
    golden_photo, golden_mask, _ = _take_photo(board, GOLDEN_CORNERS, seed=1)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)

    # The far edge looks well under half the length of the near edge, yet SIFT can still align it.
    steep_photo, steep_mask, _ = _take_photo(board, [(1050, 500), (1950, 500), (2700, 1800), (300, 1800)], seed=2)
    result = align_to_golden(steep_photo, steep_mask, golden, ALIGN_CFG)

    assert result.tilt_ratio > ALIGN_CFG["retake"]["max_side_ratio"]
    assert not result.ok
    assert "steep angle" in result.message


def test_white_parts_in_the_golden_photo_are_not_called_glare() -> None:
    board = _draw_board()
    board[100:500, 100:800] = 255  # A large white area in both photos.
    golden_photo, golden_mask, _ = _take_photo(board, GOLDEN_CORNERS, seed=1)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    message = diagnose_retake(golden_mask, golden.image, golden, 0.5, 1.0, False, ALIGN_CFG)
    assert "glare" not in message


def test_new_bright_area_is_called_glare() -> None:
    golden_photo, golden_mask, _ = _take_photo(_draw_board(), GOLDEN_CORNERS, seed=1)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    aligned = golden.image.copy()
    height, width = aligned.shape[:2]
    aligned[: height // 2, : width // 2] = 255
    message = diagnose_retake(golden_mask, aligned, golden, 0.5, 1.0, False, ALIGN_CFG)
    assert "glare" in message


def test_a_covered_board_is_not_blamed_on_the_camera_angle() -> None:
    """A partly covered board gives a skewed quad and a large tilt, but the angle is not the cause."""
    golden_photo, golden_mask, _ = _take_photo(_draw_board(), GOLDEN_CORNERS, seed=1)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    message = diagnose_retake(golden_mask, golden.image, golden, 0.6, 3.2, False, ALIGN_CFG)
    assert "steep" not in message
    assert "covers the board" in message


def test_fit_quad_ignores_a_connector_overhanging_one_edge() -> None:
    """A part sticking out past the edge must not skew the corners, as a convex hull would."""
    mask = np.zeros((800, 1200), np.uint8)
    cv2.rectangle(mask, (200, 150), (1000, 650), 1, -1)
    cv2.rectangle(mask, (120, 300), (220, 480), 1, -1)  # A USB connector past the left edge.
    quad = fit_quad(mask.astype(bool), ALIGN_CFG)
    expected = np.array([[200, 150], [1000, 150], [1000, 650], [200, 650]])
    assert np.linalg.norm(quad - expected, axis=1).max() <= 3.0


def test_fit_quad_ignores_header_pins_splitting_one_edge() -> None:
    mask = np.zeros((800, 1200), np.uint8)
    cv2.rectangle(mask, (200, 200), (1000, 650), 1, -1)
    for x in (500, 560, 620, 680):  # Pins standing up past the top edge.
        cv2.rectangle(mask, (x, 90), (x + 16, 210), 1, -1)
    quad = fit_quad(mask.astype(bool), ALIGN_CFG)
    expected = np.array([[200, 200], [1000, 200], [1000, 650], [200, 650]])
    assert np.linalg.norm(quad - expected, axis=1).max() <= 3.0


def test_filled_hull_ignores_holes_in_a_mask() -> None:
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    holed = mask.copy()
    holed[30:70, 30:70] = False
    assert mask_iou(holed, mask) < 0.8
    assert mask_iou(filled_hull(holed), filled_hull(mask)) > 0.95


def test_straight_down_photo_has_no_tilt() -> None:
    golden_photo, golden_mask, _ = _take_photo(_draw_board(), [(500, 300), (2500, 300), (2500, 1633), (500, 1633)])
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    assert camera_tilt_ratio(golden.homography, golden.size) == pytest.approx(1.0, abs=0.01)


def test_no_mask_asks_for_a_retake() -> None:
    golden_photo, golden_mask, _ = _take_photo(_draw_board(), GOLDEN_CORNERS, seed=1)
    golden = canonicalise_golden(golden_photo, golden_mask, ALIGN_CFG)
    result = align_to_golden(golden_photo, None, golden, ALIGN_CFG)
    assert not result.ok
    assert "no board found" in result.message


def test_fit_quad_ignores_chamfered_corners() -> None:
    """A board with its corners cut off still gives the corners of its enclosing rectangle."""
    mask = np.zeros((800, 1200), np.uint8)
    chamfered = np.array(
        [[180, 100], [960, 100], [1040, 180], [1040, 620], [960, 700], [180, 700], [100, 620], [100, 180]]
    )
    cv2.fillPoly(mask, [chamfered], 1)
    quad = fit_quad(mask.astype(bool), ALIGN_CFG)
    expected = np.array([[100, 100], [1040, 100], [1040, 700], [100, 700]])
    assert np.linalg.norm(quad - expected, axis=1).max() <= 3.0


def test_order_corners_handles_any_input_order() -> None:
    corners = np.array([[900.0, 700.0], [100.0, 650.0], [120.0, 90.0], [950.0, 60.0]])
    assert order_corners(corners).tolist() == [[120.0, 90.0], [950.0, 60.0], [900.0, 700.0], [100.0, 650.0]]


def test_portrait_quad_turns_its_long_side_onto_x() -> None:
    portrait = np.array([[100.0, 100.0], [500.0, 100.0], [500.0, 900.0], [100.0, 900.0]])
    reordered, size = canonical_frame(portrait, 2000)
    assert size == (2000, 1000)
    assert np.linalg.norm(reordered[1] - reordered[0]) == pytest.approx(800.0)


def test_normalise_lighting_reduces_a_colour_cast() -> None:
    board = _draw_board()
    warm = np.clip(board * np.array([1.3, 1.0, 0.7]), 0, 255).astype(np.uint8)
    before = np.abs(warm.astype(float) - board).mean()
    after = np.abs(normalise_lighting(warm).astype(float) - normalise_lighting(board)).mean()
    assert after < before


def test_checkerboard_blend_alternates_between_the_two_images() -> None:
    first = np.zeros((40, 40, 3), np.uint8)
    second = np.full((40, 40, 3), 255, np.uint8)
    blend = checkerboard_blend(first, second, squares_along_x=2)
    assert blend[5, 5, 0] == 0 and blend[5, 25, 0] == 255 and blend[25, 25, 0] == 0


def test_mask_iou_of_half_overlapping_masks() -> None:
    a = np.zeros((10, 10), bool)
    b = np.zeros((10, 10), bool)
    a[:, :6] = True
    b[:, 3:] = True
    assert mask_iou(a, b) == pytest.approx(3 / 10)
