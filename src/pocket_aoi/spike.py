"""Block 1 calibration: run Stage 2 on each golden photo, count parts per class, suggest thresholds.

Reads data/raw/<kit>/golden.jpg, an optional boards/<kit>/board.yaml of hand counts and class prompts,
and an optional refs.yaml of labelled part positions, and writes overlays and raw detections to runs/spike/<kit>/.
A rerun reuses the saved detections unless told otherwise, so thresholds and merge rules can be
retuned without running SAM again. Change the prompts and that reuse must be turned off.
"""

import json
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from pocket_aoi.align import draw_quad, filled_hull
from pocket_aoi.io import load_rgb
from pocket_aoi.pipeline import load_golden_frame
from pocket_aoi.segment import (
    PartMask,
    Segmenter,
    draw_parts,
    keep_plausible,
    mask_to_rle,
    merge_duplicates,
    rle_to_mask,
    select_parts,
)


def suggest_threshold(scores: Sequence[float], hand_count: int | None, default: float, margin: float) -> float:
    """Return a threshold just under the score of the last expected part, since recall comes first."""
    if not hand_count:
        return default  # A board without this class cannot calibrate it.
    ranked = sorted(scores, reverse=True)
    if len(ranked) < hand_count:
        return round(max(min(ranked, default=default) - margin, 0.0), 2)
    return round(max(ranked[hand_count - 1] - margin, 0.0), 2)


def merged_scores(
    parts: Sequence[PartMask],
    cls: str,
    board_rle: dict[str, Any],
    segment_cfg: Mapping[str, Any],
) -> list[float]:
    """Return one class's scores after plausibility and same-class merging."""
    plausible = keep_plausible(
        [part for part in parts if part.cls == cls],
        board_rle,
        segment_cfg["min_part_area_ratio"],
        segment_cfg["max_part_area_ratio"],
        segment_cfg["min_inside_board"],
    )
    return [part.score for part in merge_duplicates(plausible, segment_cfg["merge_iou"])]


def box_matches_label(box: Sequence[float], label_box: Sequence[float], min_iou: float) -> bool:
    """Return True if a detection sits on a labelled part: its centre inside, and the boxes overlapping."""
    centre_x, centre_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    if not (label_box[0] <= centre_x <= label_box[2] and label_box[1] <= centre_y <= label_box[3]):
        return False
    overlap_x = max(0.0, min(box[2], label_box[2]) - max(box[0], label_box[0]))
    overlap_y = max(0.0, min(box[3], label_box[3]) - max(box[1], label_box[1]))
    overlap = overlap_x * overlap_y
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    label_area = (label_box[2] - label_box[0]) * (label_box[3] - label_box[1])
    return overlap / max(box_area + label_area - overlap, 1e-6) >= min_iou


def best_scores_on_labels(
    parts: Sequence[PartMask], labels: Sequence[Mapping[str, Any]], cls: str, min_iou: float
) -> list[float]:
    """Return each labelled part's best score for its own class, across every view that saw it."""
    best = []
    for label in labels:
        if label["cls"] != cls:
            continue
        # max with default= returns the default for an empty sequence instead of raising.
        score = max(
            (part.score for part in parts if part.cls == cls and box_matches_label(part.box, label["box"], min_iou)),
            default=None,
        )
        if score is not None:
            best.append(score)
    return best


def suggest_from_labels(scores_on_parts: Sequence[float], default: float, margin: float) -> float:
    """Return a threshold just under the weakest labelled part's best score."""
    return round(max(min(scores_on_parts) - margin, 0.0), 2) if scores_on_parts else default


def score_against_labels(
    parts: Sequence[PartMask],
    labels: Sequence[Mapping[str, Any]],
    min_iou: float,
    check_class: bool,
) -> tuple[dict[str, tuple[int, int]], list[str], int]:
    """Compare detections with labelled parts.

    Returns (found, total) per class, the refs that were missed, and how many detections sit on no
    labelled part of the right kind. With check_class off, any detection on a part counts, which
    measures where the parts are rather than what they are called.
    """
    totals: Counter[str] = Counter()
    found: Counter[str] = Counter()
    missed: list[str] = []
    for label in labels:
        totals[label["cls"]] += 1
        hit = any(
            box_matches_label(part.box, label["box"], min_iou) and (not check_class or part.cls == label["cls"])
            for part in parts
        )
        if hit:
            found[label["cls"]] += 1
        elif label["cls"] != "other":
            missed.append(label["ref"])

    strays = sum(
        not any(
            box_matches_label(part.box, label["box"], min_iou) and (not check_class or part.cls == label["cls"])
            for label in labels
        )
        for part in parts
    )
    return {cls: (found[cls], totals[cls]) for cls in totals}, missed, strays


def _print_label_score(title: str, score: tuple[dict[str, tuple[int, int]], list[str], int]) -> None:
    per_class, missed, strays = score
    summary = ", ".join(f"{cls} {found}/{total}" for cls, (found, total) in per_class.items())
    print(f"  {title}: {summary}")
    print(f"    missed: {', '.join(missed) if missed else 'none'}; detections on nothing labelled: {strays}")


def kit_prompts(kit_config: Mapping[str, Any], default_prompts: Mapping[str, str]) -> dict[str, str]:
    """Return the prompts for one kit: its own list if it has one, else the defaults for its classes.

    A class the kit does not have gets no prompt, because SAM happily finds "transistor" on a
    board with none. Section 3, Stage 2 takes class prompts from the board config for this reason.
    """
    if "class_prompts" in kit_config:
        return dict(kit_config["class_prompts"])
    counts = kit_config.get("counts", {})
    return {prompt: cls for prompt, cls in default_prompts.items() if counts.get(cls, 1) > 0}


def _load_or_run(
    kit_dir: Path,
    out_dir: Path,
    prompts: Mapping[str, str],
    make_segmenter: Callable[[], Segmenter],
    cfg: Mapping[str, Any],
    fresh: bool,
) -> tuple[Image.Image, dict[str, Any], list[PartMask]]:
    """Return the canonical golden image, its board mask and the raw detections."""
    saved = out_dir / "parts.json"
    canonical_path = out_dir / "canonical.png"
    if saved.exists() and canonical_path.exists() and not fresh:
        data = json.loads(saved.read_text(encoding="utf-8"))
        parts = [PartMask(p["rle"], tuple(p["box"]), p["score"], p["cls"], p["prompt"]) for p in data["parts"]]
        print(f"  reusing {saved}")
        return Image.open(canonical_path).convert("RGB"), data["board_rle"], parts

    segmenter = make_segmenter()
    segment_cfg = cfg["segment"]
    photo = load_rgb(kit_dir / "golden.jpg")
    golden = load_golden_frame(segmenter, photo, cfg)
    canonical = Image.fromarray(golden.image)
    canonical.save(canonical_path)
    Image.fromarray(draw_quad(np.asarray(photo), golden.homography, golden.size)).save(out_dir / "board_quad.png")
    board_rle = mask_to_rle(golden.mask)

    segmenter.seconds = {"encode": 0.0, "decode": 0.0}
    started = time.perf_counter()
    parts = segmenter.segment_parts(
        canonical,
        prompts,
        threshold=segment_cfg["spike_floor"],
        mask_threshold=segment_cfg["mask_threshold"],
        tile_grid=segment_cfg["tile_grid"],
        tile_overlap=segment_cfg["tile_overlap"],
    )
    views = segment_cfg["tile_grid"] ** 2 + 1
    print(
        f"  {views} views x {len(prompts)} prompts: {time.perf_counter() - started:.1f} s, "
        f"{len(parts)} raw detections"
    )
    saved.write_text(
        json.dumps({"board_rle": board_rle, "parts": [asdict(part) for part in parts]}),
        encoding="utf-8",
    )
    return canonical, board_rle, parts


def run_kit(
    kit_dir: Path,
    boards_dir: Path,
    out_root: Path,
    make_segmenter: Callable[[], Segmenter],
    cfg: Mapping[str, Any],
    fresh: bool = False,
) -> dict[str, float]:
    """Print one kit's count table, save its overlays, and return its suggested thresholds."""
    segment_cfg = cfg["segment"]
    out_dir = out_root / kit_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nKit {kit_dir.name}")

    board_path = boards_dir / kit_dir.name / "board.yaml"
    kit_config = yaml.safe_load(board_path.read_text(encoding="utf-8")) if board_path.exists() else {}
    hand_counts = kit_config.get("counts", {})
    refs_path = boards_dir / kit_dir.name / "refs.yaml"
    labels = yaml.safe_load(refs_path.read_text(encoding="utf-8")).get("parts", []) if refs_path.exists() else []
    locate_prompt = segment_cfg["locate_prompt"]
    class_prompts = kit_prompts(kit_config, segment_cfg["class_prompts"])
    prompts = {locate_prompt: "other", **class_prompts}
    image, raw_board_rle, parts = _load_or_run(kit_dir, out_dir, prompts, make_segmenter, cfg, fresh)
    # SAM's board mask can leave holes where the parts sit, so "on the board" means inside its outline.
    board_rle = mask_to_rle(filled_hull(rle_to_mask(raw_board_rle)))
    print(f"  canonical size {image.width} x {image.height}")
    print(f"  class prompts: {', '.join(class_prompts)}")

    located_raw = [part for part in parts if part.prompt == locate_prompt]
    classed_raw = [part for part in parts if part.prompt != locate_prompt]

    sweep = segment_cfg["spike_sweep"]
    default = segment_cfg["score_threshold"]
    margin = segment_cfg["suggested_threshold_margin"]
    min_iou = segment_cfg["label_match_iou"]
    header = "".join(f"{threshold:>6.2f}" for threshold in sweep)
    print(f"  {'class':<24}{'hand':>5} |{header} | suggested")

    suggested: dict[str, float] = {}
    # Keeps the prompts' order but drops repeats, since several prompts can share one class.
    classes = list(dict.fromkeys(class_prompts.values()))
    for cls in classes:
        scores = merged_scores(classed_raw, cls, board_rle, segment_cfg)
        hand = hand_counts.get(cls)
        suggested[cls] = (
            suggest_from_labels(best_scores_on_labels(classed_raw, labels, cls, min_iou), default, margin)
            if labels
            else suggest_threshold(scores, hand, default, margin)
        )
        row = "".join(f"{sum(score >= threshold for score in scores):>6d}" for threshold in sweep)
        shown_hand = "?" if hand is None else str(hand)
        print(f"  {cls:<24}{shown_hand:>5} |{row} | {suggested[cls]:.2f}")

    located = select_parts(located_raw, segment_cfg, board_rle, {"other": segment_cfg["locate_threshold"]})
    classed = select_parts(classed_raw, segment_cfg, board_rle, suggested)
    print(f"  {locate_prompt!r} at {segment_cfg['locate_threshold']}: {len(located)} masks")
    if labels:
        _print_label_score("where the parts are, any label", score_against_labels(located, labels, min_iou, False))
        _print_label_score("what they are, right label", score_against_labels(classed, labels, min_iou, True))

    draw_parts(image, located).save(out_dir / "overlay_located.png")
    draw_parts(image, classed).save(out_dir / "overlay_classed.png")
    print(f"  overlays in {out_dir}")
    return suggested


def run_spike(
    kits_dir: Path,
    boards_dir: Path,
    out_root: Path,
    make_segmenter: Callable[[], Segmenter],
    cfg: Mapping[str, Any],
    fresh: bool = False,
) -> None:
    """Run every kit under kits_dir that has a golden.jpg, then print thresholds for all of them."""
    kit_dirs = sorted(path.parent for path in kits_dir.glob("*/golden.jpg"))
    if not kit_dirs:
        raise FileNotFoundError(f"no <kit>/golden.jpg found under {kits_dir}")

    per_kit = [run_kit(kit_dir, boards_dir, out_root, make_segmenter, cfg, fresh) for kit_dir in kit_dirs]
    print("\nSuggested thresholds, the lowest across kits so no kit loses a part:")
    classes = list(dict.fromkeys(cls for suggested in per_kit for cls in suggested))
    for cls in classes:
        values = [suggested[cls] for suggested in per_kit if cls in suggested]
        print(f"  {cls:<24}{min(values):.2f}")
