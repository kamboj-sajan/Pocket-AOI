"""Check the chunked post-processing against the library's own result, with no model download."""

from types import SimpleNamespace

import numpy as np
import torch
from transformers import Sam3ImageProcessor

from pocket_aoi.config import load_config
from pocket_aoi.segment import (
    Detection,
    PartMask,
    _to_part_mask,
    keep_plausible,
    mask_to_rle,
    merge_duplicates,
    post_process_in_chunks,
    resolve_class_conflicts,
    rle_to_mask,
    select_parts,
    tile_boxes,
    touches_inner_edge,
)
from pocket_aoi.spike import (
    best_scores_on_labels,
    box_matches_label,
    kit_prompts,
    score_against_labels,
    suggest_from_labels,
    suggest_threshold,
)

NUM_QUERIES = 40
TARGET_SIZES = [[60, 80]]
SPECIFICITY = ["electrolytic_capacitor", "resistor"]


def _part(x0: int, y0: int, x1: int, y1: int, score: float, cls: str = "resistor") -> PartMask:
    """Build a part whose mask is a filled rectangle in a 100 x 100 image."""
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return PartMask(mask_to_rle(mask), (x0, y0, x1, y1), score, cls, cls)


def test_rle_round_trip_keeps_the_mask_and_is_json_ready() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:15, 10:25] = True
    rle = mask_to_rle(mask)
    assert isinstance(rle["counts"], str)
    assert np.array_equal(rle_to_mask(rle), mask)


def test_tiles_cover_the_image_and_overlap_by_the_requested_fraction() -> None:
    tiles = tile_boxes(1800, 900, grid=2, overlap=0.2)
    assert len(tiles) == 4
    assert tiles[0] == (0, 0, 1000, 500)
    assert tiles[-1] == (800, 400, 1800, 900)
    assert (tiles[0][2] - tiles[1][0]) / (tiles[0][2] - tiles[0][0]) == 0.2


def test_mask_cut_by_an_inner_tile_edge_is_flagged_but_an_image_edge_is_not() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:20, 40:50] = True  # Reaches the tile's right edge.
    assert touches_inner_edge(mask, (0, 0, 50, 50), width=100, height=100)
    assert not touches_inner_edge(mask, (50, 0, 100, 50), width=100, height=100)


def test_tile_detection_maps_back_to_full_image_coordinates() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:20, 5:15] = True
    part = _to_part_mask(Detection(mask, (5, 10, 15, 20), 0.9), (60, 50, 100, 90), 100, 100, "resistor", "resistor")
    assert part.box == (65, 60, 75, 70)
    expected = np.zeros((100, 100), dtype=bool)
    expected[60:70, 65:75] = True
    assert np.array_equal(rle_to_mask(part.rle), expected)


def test_duplicates_merge_to_the_higher_score_within_a_class() -> None:
    kept = merge_duplicates([_part(10, 10, 30, 30, 0.5), _part(11, 11, 31, 31, 0.8), _part(60, 60, 80, 80, 0.4)], 0.5)
    assert [part.score for part in kept] == [0.8, 0.4]


def test_same_place_in_different_classes_is_not_a_same_class_duplicate() -> None:
    parts = [_part(10, 10, 30, 30, 0.5, "resistor"), _part(10, 10, 30, 30, 0.6, "electrolytic_capacitor")]
    assert len(merge_duplicates(parts, 0.5)) == 2


def test_close_scores_keep_the_more_specific_class() -> None:
    parts = [_part(10, 10, 30, 30, 0.60, "resistor"), _part(10, 10, 30, 30, 0.55, "electrolytic_capacitor")]
    kept = resolve_class_conflicts(parts, 0.5, 0.1, SPECIFICITY)
    assert [part.cls for part in kept] == ["electrolytic_capacitor"]


def test_a_clear_score_gap_keeps_the_higher_score() -> None:
    parts = [_part(10, 10, 30, 30, 0.80, "resistor"), _part(10, 10, 30, 30, 0.55, "electrolytic_capacitor")]
    kept = resolve_class_conflicts(parts, 0.5, 0.1, SPECIFICITY)
    assert [part.cls for part in kept] == ["resistor"]


def test_plausibility_drops_tiny_huge_and_off_board_masks() -> None:
    board = np.zeros((100, 100), dtype=bool)
    board[:, :50] = True  # The board fills the left half.
    parts = [
        _part(10, 10, 20, 20, 0.9),    # fine
        _part(1, 1, 2, 2, 0.9),        # 1 px, too small
        _part(0, 0, 50, 50, 0.9),      # half the board, too large
        _part(60, 60, 70, 70, 0.9),    # off the board
    ]
    kept = keep_plausible(parts, mask_to_rle(board), 0.001, 0.25, 0.9)
    assert [part.box for part in kept] == [(10, 10, 20, 20)]


def test_select_parts_applies_a_per_class_threshold() -> None:
    cfg = {**load_config()["segment"], "min_part_area_ratio": 0.0, "class_score_thresholds": {"resistor": 0.7}}
    parts = [_part(10, 10, 30, 30, 0.65), _part(60, 60, 80, 80, 0.75)]
    assert [part.score for part in select_parts(parts, cfg)] == [0.75]


def test_suggested_threshold_sits_just_under_the_last_expected_part() -> None:
    assert suggest_threshold([0.9, 0.85, 0.8, 0.4, 0.3], 3, 0.4, 0.02) == 0.78


def test_suggested_threshold_keeps_everything_when_too_few_parts_were_found() -> None:
    assert suggest_threshold([0.6, 0.5], 5, 0.4, 0.02) == 0.48


def test_board_without_a_class_keeps_the_default_threshold() -> None:
    assert suggest_threshold([0.9, 0.8], 0, 0.4, 0.02) == 0.4


def test_label_threshold_sits_under_the_weakest_labelled_part() -> None:
    assert suggest_from_labels([0.9, 0.7, 0.8], 0.4, 0.02) == 0.68
    assert suggest_from_labels([], 0.4, 0.02) == 0.4


def test_threshold_uses_each_parts_best_view_not_its_weakest() -> None:
    labels = [{"ref": "R1", "cls": "resistor", "box": (10, 10, 30, 30)}]
    parts = [_part(10, 10, 30, 30, 0.9), _part(10, 10, 30, 30, 0.3)]  # A tile saw it badly; the full view did not.
    assert best_scores_on_labels(parts, labels, "resistor", 0.25) == [0.9]


def test_detection_must_sit_on_the_label_to_match_it() -> None:
    label = (100, 100, 200, 200)
    assert box_matches_label((110, 110, 190, 190), label, 0.25)
    assert not box_matches_label((300, 300, 400, 400), label, 0.25)  # Elsewhere on the board.
    assert not box_matches_label((0, 0, 400, 400), label, 0.25)      # Centre inside, but far too large.


def test_scoring_separates_finding_a_part_from_naming_it() -> None:
    labels = [
        {"ref": "R1", "cls": "resistor", "box": (10, 10, 30, 30)},
        {"ref": "C1", "cls": "ceramic_capacitor", "box": (60, 60, 80, 80)},
    ]
    parts = [
        _part(10, 10, 30, 30, 0.9, "resistor"),            # Right place, right label.
        _part(60, 60, 80, 80, 0.9, "resistor"),            # Right place, wrong label.
    ]
    found, missed, strays = score_against_labels(parts, labels, 0.25, check_class=False)
    assert found == {"resistor": (1, 1), "ceramic_capacitor": (1, 1)} and missed == [] and strays == 0

    found, missed, strays = score_against_labels(parts, labels, 0.25, check_class=True)
    assert found == {"resistor": (1, 1), "ceramic_capacitor": (0, 1)} and missed == ["C1"] and strays == 1


def test_a_missed_other_part_does_not_count_as_a_missed_part() -> None:
    labels = [{"ref": "S1", "cls": "other", "box": (10, 10, 30, 30)}]
    found, missed, _ = score_against_labels([], labels, 0.25, check_class=False)
    assert found == {"other": (0, 1)} and missed == []


def test_kit_gets_no_prompt_for_a_class_it_does_not_have() -> None:
    defaults = {"resistor": "resistor", "transistor": "transistor", "diode": "diode"}
    assert kit_prompts({"counts": {"resistor": 8, "transistor": 0}}, defaults) == {"resistor": "resistor", "diode": "diode"}


def test_kit_prompts_override_the_defaults() -> None:
    kit = {"counts": {"led": 7}, "class_prompts": {"red LED": "led"}}
    assert kit_prompts(kit, {"LED": "led", "resistor": "resistor"}) == {"red LED": "led"}


def _fake_outputs() -> SimpleNamespace:
    """Build model outputs with a mix of confident and rejected query slots."""
    generator = torch.Generator().manual_seed(0)
    return SimpleNamespace(
        pred_logits=torch.randn(1, NUM_QUERIES, generator=generator) * 3,
        pred_boxes=torch.rand(1, NUM_QUERIES, 4, generator=generator),
        pred_masks=torch.randn(1, NUM_QUERIES, 12, 16, generator=generator),
        presence_logits=torch.tensor([[4.0]]),
    )


def test_chunked_result_matches_the_library_called_on_everything() -> None:
    processor = Sam3ImageProcessor()
    outputs = _fake_outputs()

    expected = processor.post_process_instance_segmentation(
        outputs, threshold=0.4, mask_threshold=0.5, target_sizes=TARGET_SIZES
    )[0]
    detections = post_process_in_chunks(processor, outputs, 0.4, 0.5, TARGET_SIZES)

    assert 0 < len(detections) < NUM_QUERIES  # The fake data must exercise both outcomes.
    assert len(detections) == len(expected["scores"])
    for detection, mask, box, score in zip(
        detections, expected["masks"], expected["boxes"], expected["scores"]
    ):
        assert detection.mask.dtype == bool
        assert detection.mask.shape == (60, 80)
        assert np.array_equal(detection.mask, mask.numpy().astype(bool))
        assert np.allclose(detection.box, box.numpy())
        assert np.isclose(detection.score, score.item())


def test_no_instances_above_the_threshold_gives_an_empty_list() -> None:
    detections = post_process_in_chunks(Sam3ImageProcessor(), _fake_outputs(), 1.1, 0.5, TARGET_SIZES)
    assert detections == []
