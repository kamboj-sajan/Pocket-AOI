"""Stage 3 on a drawn board with synthetic defects, as engineering rule 9 asks."""

import cv2
import numpy as np
import pytest

from eval.metrics import PhotoResult, box_covers_mask, same_type, summarise
from pocket_aoi.compare import compare_boards, expand_box, match_parts, rotation_test
from pocket_aoi.composite import build_composite, composite_scale, encode_png
from pocket_aoi.config import load_config
from pocket_aoi.pipeline import build_golden_parts
from pocket_aoi.schemas import Component, GoldenProfile, Net, Netlist, Part, Supply
from pocket_aoi.segment import PartMask, mask_to_rle

CFG = load_config()
COMPARE = CFG["compare"]
SIZE = (1000, 700)  # width, height


def _board_background() -> np.ndarray:
    """Draw a textured green board, so SSIM sees structure everywhere."""
    width, height = SIZE
    image = np.full((height, width, 3), (30, 110, 60), dtype=np.uint8)
    rng = np.random.default_rng(1)
    for _ in range(50):
        y = int(rng.integers(0, height))
        cv2.line(image, (0, y), (width, y + int(rng.integers(-30, 30))), (50, 140, 80), 2)
    return image


def _draw_capacitor(image: np.ndarray, centre: tuple[int, int], stripe_left: bool = True) -> None:
    """Draw an electrolytic can from above, with a polarity stripe on one side."""
    cv2.circle(image, centre, 50, (35, 35, 35), -1)
    x, y = centre
    x0, x1 = (x - 48, x - 18) if stripe_left else (x + 18, x + 48)
    cv2.rectangle(image, (x0, y - 28), (x1, y + 28), (225, 225, 225), -1)


def _draw_resistor(image: np.ndarray, x: int, y: int) -> None:
    """Draw an axial resistor with coloured bands."""
    cv2.line(image, (x - 90, y), (x + 90, y), (190, 190, 190), 4)
    cv2.rectangle(image, (x - 55, y - 20), (x + 55, y + 20), (210, 180, 130), -1)
    for offset, colour in ((-35, (160, 40, 40)), (-12, (30, 30, 30)), (12, (40, 90, 170))):
        cv2.rectangle(image, (x + offset, y - 20), (x + offset + 12, y + 20), colour, -1)


def _golden_scene() -> tuple[np.ndarray, list[Part]]:
    image = _board_background()
    _draw_capacitor(image, (250, 200))
    _draw_resistor(image, 650, 200)
    _draw_capacitor(image, (250, 500))
    _draw_resistor(image, 650, 500)
    parts = [
        _part("C1", "electrolytic_capacitor", (200, 150, 300, 250), True),
        _part("R1", "resistor", (560, 180, 740, 220), False),
        _part("C2", "electrolytic_capacitor", (200, 450, 300, 550), True),
        _part("R2", "resistor", (560, 480, 740, 520), False),
    ]
    return image, parts


def _mask_of(box: tuple[int, int, int, int]) -> np.ndarray:
    width, height = SIZE
    mask = np.zeros((height, width), dtype=bool)
    mask[box[1] : box[3], box[0] : box[2]] = True
    return mask


def _part(ref: str | None, cls: str, box: tuple[int, int, int, int], polarized: bool) -> Part:
    return Part(
        ref=ref,
        cls=cls,
        bbox=box,
        centroid=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        mask_rle=mask_to_rle(_mask_of(box)),
        score=0.9,
        package=None,
        polarized=polarized,
    )


def _test_mask(box: tuple[int, int, int, int]) -> PartMask:
    return PartMask(mask_to_rle(_mask_of(box)), tuple(float(v) for v in box), 0.9, "other", "electronic component")


def _compare(golden_image: np.ndarray, parts: list[Part], test_image: np.ndarray, masks: list[PartMask]):
    profile = GoldenProfile(board_id="demo", canonical_size=SIZE, parts=parts)
    return compare_boards(profile, golden_image, test_image, masks, COMPARE)


def _all_masks(parts: list[Part]) -> list[PartMask]:
    return [_test_mask(part.bbox) for part in parts]


def test_identical_board_raises_nothing() -> None:
    image, parts = _golden_scene()
    assert _compare(image, parts, image.copy(), _all_masks(parts)) == []


def test_part_painted_out_with_inpaint_is_missing() -> None:
    image, parts = _golden_scene()
    test = image.copy()
    inpaint_mask = _mask_of((190, 140, 310, 260)).astype(np.uint8) * 255
    test = cv2.inpaint(test, inpaint_mask, 5, cv2.INPAINT_TELEA)
    masks = [m for m, p in zip(_all_masks(parts), parts) if p.ref != "C1"]

    findings = _compare(image, parts, test, masks)

    assert [(f.golden_ref, f.cv_finding) for f in findings] == [("C1", "MISSING")]


def test_part_rotated_180_degrees_is_reversed() -> None:
    image, parts = _golden_scene()
    test = image.copy()
    x0, y0, x1, y1 = parts[0].bbox
    test[y0:y1, x0:x1] = np.rot90(image[y0:y1, x0:x1], 2)

    findings = _compare(image, parts, test, _all_masks(parts))

    assert [(f.golden_ref, f.cv_finding) for f in findings] == [("C1", "REVERSED")]
    assert findings[0].evidence["best_rotation_deg"] == 180


def test_rotation_test_names_180_for_a_turned_part() -> None:
    image, parts = _golden_scene()
    test = image.copy()
    x0, y0, x1, y1 = parts[0].bbox
    test[y0:y1, x0:x1] = np.rot90(image[y0:y1, x0:x1], 2)
    golden_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
    angle, margin = rotation_test(golden_gray, test_gray, parts[0].bbox, (0, 180), COMPARE)
    assert angle == 180 and margin > COMPARE["rotation_margin"]


def test_unpolarized_part_is_not_tested_for_rotation() -> None:
    image, parts = _golden_scene()
    test = image.copy()
    x0, y0, x1, y1 = parts[1].bbox
    test[y0:y1, x0:x1] = np.rot90(image[y0:y1, x0:x1], 2)  # A resistor has no polarity.
    assert all(f.cv_finding not in ("REVERSED", "ROTATED") for f in _compare(image, parts, test, _all_masks(parts)))


def test_part_that_moved_is_shifted() -> None:
    image, parts = _golden_scene()
    test = _board_background()
    _draw_capacitor(test, (330, 200))  # C1 moved right by most of its own width.
    _draw_resistor(test, 650, 200)
    _draw_capacitor(test, (250, 500))
    _draw_resistor(test, 650, 500)
    masks = [_test_mask((280, 150, 380, 250)), *_all_masks(parts)[1:]]

    findings = _compare(image, parts, test, masks)

    assert ("C1", "SHIFTED") in [(f.golden_ref, f.cv_finding) for f in findings]


def test_new_part_is_extra() -> None:
    image, parts = _golden_scene()
    test = image.copy()
    cv2.circle(test, (450, 350), 40, (200, 40, 40), -1)
    masks = [*_all_masks(parts), _test_mask((410, 310, 490, 390))]

    findings = _compare(image, parts, test, masks)

    assert [(f.golden_ref, f.cv_finding) for f in findings] == [(None, "EXTRA")]


def test_mask_appearing_where_nothing_changed_is_not_extra() -> None:
    image, parts = _golden_scene()
    masks = [*_all_masks(parts), _test_mask((410, 310, 490, 390))]  # SAM imagines something on bare board.
    assert _compare(image, parts, image.copy(), masks) == []


def test_mask_sam_missed_is_fine_when_the_part_is_still_there() -> None:
    image, parts = _golden_scene()
    masks = _all_masks(parts)[1:]  # No mask for C1, though C1 is in the photo.
    assert _compare(image, parts, image.copy(), masks) == []


def test_known_object_without_a_ref_is_not_inspected_but_blocks_extra() -> None:
    image, parts = _golden_scene()
    cv2.circle(image, (450, 350), 40, (200, 200, 40), -1)
    parts.append(_part(None, "other", (410, 310, 490, 390), False))
    test = image.copy()
    test[310:390, 410:490] = (30, 110, 60)  # The known object is gone; no ref, so no finding.
    assert _compare(image, parts, test, _all_masks(parts)[:-1]) == []


def test_matching_is_one_to_one_and_ignores_class() -> None:
    _, parts = _golden_scene()
    masks = [_test_mask(parts[0].bbox), _test_mask((205, 155, 305, 255))]  # Two masks on C1.
    matches = match_parts(parts, masks, COMPARE)
    assert list(matches) == [0]
    assert matches[0][0] == 0  # The closer of the two.


def test_pairs_below_the_minimum_iou_never_match() -> None:
    _, parts = _golden_scene()
    assert match_parts(parts[:1], [_test_mask((800, 600, 900, 690))], COMPARE) == {}


def test_expand_box_stays_inside_the_image() -> None:
    assert expand_box((0, 0, 100, 100), 0.25, 110, 110) == (0, 0, 110, 110)


def test_composite_labels_both_halves_and_keeps_the_same_scale() -> None:
    image, parts = _golden_scene()
    crop = expand_box(parts[0].bbox, 0.25, *SIZE)
    composite = build_composite(image, image, crop, parts[0].bbox, 384, 1024)
    height, width = composite.shape[:2]
    half = (width - 8) // 2
    assert half / (crop[2] - crop[0]) == pytest.approx(height / (crop[3] - crop[1]), rel=0.02)
    assert half >= 384
    assert encode_png(composite).startswith(b"\x89PNG")


def test_composite_scale_makes_the_part_span_the_minimum_within_the_cap() -> None:
    assert composite_scale((0, 0, 100, 100), (150, 150), 384, 1024) == pytest.approx(3.84)
    assert composite_scale((0, 0, 20, 20), (30, 30), 384, 300) == pytest.approx(10.0)  # The cap wins.
    assert composite_scale((0, 0, 800, 800), (1000, 1000), 384, 1024) == 1.0  # Never shrinks.


def _located(box: tuple[int, int, int, int]) -> PartMask:
    return _test_mask(box)


def test_golden_parts_get_refs_from_labels_and_classes_from_the_netlist() -> None:
    netlist = Netlist(
        board_id="demo",
        supply=Supply(net="VCC", volts=5),
        components=[Component(ref="C1", value="100uF", type="electrolytic_capacitor", package="radial")],
        nets=[Net(name="VCC", nodes=["C1.+"])],
    )
    labels = [{"ref": "C1", "cls": "other", "box": [200, 150, 300, 250]}]
    located = [_located((202, 152, 298, 248)), _located((600, 300, 700, 400))]

    parts = build_golden_parts(located, labels, SIZE, netlist, CFG)

    assert (parts[0].ref, parts[0].cls, parts[0].package, parts[0].polarized) == ("C1", "electrolytic_capacitor", "radial", True)
    assert parts[1].ref is None  # A real object outside the inspection list.


def test_golden_label_passes_on_its_marker_hint() -> None:
    labels = [
        {"ref": "D1", "cls": "diode", "box": [10, 20, 50, 40], "marker_hint": "band at the bottom end"},
        {"ref": "R1", "cls": "resistor", "box": [60, 20, 100, 40]},
    ]
    hinted, plain = build_golden_parts([], labels, SIZE, None, CFG)
    assert hinted.marker_hint == "band at the bottom end"
    assert plain.marker_hint is None


def test_golden_label_with_no_mask_uses_its_box() -> None:
    parts = build_golden_parts([], [{"ref": "R1", "cls": "resistor", "box": [10, 20, 50, 40]}], SIZE, None, CFG)
    assert parts[0].bbox == (10, 20, 50, 40) and parts[0].score == 0.0


def test_board_without_labels_gets_numbered_refs_in_reading_order() -> None:
    located = [_located((100, 300, 150, 350)), _located((100, 100, 150, 150))]
    parts = build_golden_parts(located, [], SIZE, None, CFG)
    assert [(p.ref, p.bbox[1]) for p in parts] == [("P01", 100), ("P02", 300)]


def test_reversed_and_rotated_count_as_one_type() -> None:
    assert same_type("ROTATED", "REVERSED") and same_type("REVERSED", "ROTATED")
    assert not same_type("MISSING", "REVERSED")
    assert same_type("EXTRA", "SHIFTED", ["EXTRA"])


def test_box_covers_mask_uses_the_smaller_area() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:50, 40:50] = True
    assert box_covers_mask((30, 30, 60, 60), mask, 0.3)   # The box holds the whole defect.
    assert not box_covers_mask((70, 70, 90, 90), mask, 0.3)
    assert not box_covers_mask((0, 0, 10, 10), np.zeros((100, 100), dtype=bool), 0.3)


def test_summary_counts_false_alarms_on_clean_photos_only() -> None:
    results = [
        PhotoResult("a", "b1", "missing", "test", True, planted=1, found_any=1, found_right_type=1, extra_findings=3),
        PhotoResult("c", "b1", "normal", "test", True, extra_findings=2),
        PhotoResult("d", "b1", "normal", "test", True),
        PhotoResult("e", "b1", "normal", "test", False, message="retake"),
    ]
    row = summarise(results)["b1"]
    assert (row["planted"], row["found_right_type"], row["rejected_by_gate"]) == (1, 1, 1)
    assert row["clean_with_false_alarm"] == 1
    assert row["false_alarms_per_clean_photo"] == 1.0  # Two alarms over two aligned clean photos.
    assert row["extra_per_defect_photo"] == 3.0
