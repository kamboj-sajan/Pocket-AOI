"""Draw every missed defect and false alarm as a golden and test crop pair, for the README.

Run from the repo root, after eval has finished:
    python -m eval.make_result_sheets --tag final

SAM does not run again, because eval saved each aligned photo under runs/cache/.
"""

import argparse
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from eval.run_eval import Claim, load_defect_mask, load_ground_truth, on_defect, photo_name, share_in_frame
from pocket_aoi.align import warp_mask
from pocket_aoi.compare import expand_box
from pocket_aoi.config import REPO_ROOT, load_config
from pocket_aoi.pipeline import GoldenBoard, PreparedPhoto, load_golden_board, prepare_photo
from pocket_aoi.schemas import GroundTruth
from pocket_aoi.segment import Segmenter

CROP_MARGIN = 0.25
CELL_PX = 150
SEPARATOR_PX = 4
CAPTION_PX = 24
GAP_PX = 6
COLUMNS = 8
BACKGROUND = 24


def _fit(crop: np.ndarray, side: int) -> np.ndarray:
    """Scale a crop to fit a square cell, centred on a dark background."""
    height, width = crop.shape[:2]
    scale = side / max(height, width)
    new_width, new_height = max(round(width * scale), 1), max(round(height * scale), 1)
    resized = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    cell = np.full((side, side, 3), BACKGROUND, dtype=np.uint8)
    top, left = (side - new_height) // 2, (side - new_width) // 2
    cell[top : top + new_height, left : left + new_width] = resized
    return cell


def make_tile(golden: np.ndarray, test: np.ndarray, box: Sequence[int], caption: str) -> np.ndarray:
    """Return one tile: the golden crop on the left, the test crop on the right, and a caption below."""
    height, width = golden.shape[:2]
    x0, y0, x1, y1 = expand_box(box, CROP_MARGIN, width, height)
    separator = np.full((CELL_PX, SEPARATOR_PX, 3), 255, dtype=np.uint8)
    pair = np.hstack([_fit(golden[y0:y1, x0:x1], CELL_PX), separator, _fit(test[y0:y1, x0:x1], CELL_PX)])
    strip = np.full((CAPTION_PX, pair.shape[1], 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(strip, caption, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([pair, strip])


def make_sheet(tiles: Sequence[np.ndarray], columns: int = COLUMNS) -> np.ndarray:
    """Lay equal-sized tiles out in rows."""
    tile_height, tile_width = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    sheet = np.full((rows * (tile_height + GAP_PX), columns * (tile_width + GAP_PX), 3), BACKGROUND, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        # divmod returns the quotient and the remainder together.
        row, column = divmod(index, columns)
        y, x = row * (tile_height + GAP_PX), column * (tile_width + GAP_PX)
        sheet[y : y + tile_height, x : x + tile_width] = tile
    return sheet


def missed_defects(
    truth: GroundTruth,
    result: Mapping[str, Any],
    prepared: PreparedPhoto,
    board: GoldenBoard,
    eval_cfg: Mapping[str, Any],
) -> Iterator[tuple[str, tuple[int, int, int, int]]]:
    """Yield the type and canonical box of each defect that no finding covers, skipping those off the board."""
    claims = [Claim(f["ref"], f["type"], tuple(f["bbox"]), f["evidence"]) for f in result["findings"]]
    for defect in truth.defects:
        if defect.mask is None:
            continue
        mask = load_defect_mask(defect.mask)
        if share_in_frame(mask, prepared.homography, board.frame.size) < eval_cfg["min_in_frame_share"]:
            continue
        canonical = warp_mask(mask, prepared.homography, board.frame.size)
        if any(on_defect(claim, defect.ref, canonical, eval_cfg["min_defect_overlap"]) for claim in claims):
            continue
        rows, columns = np.nonzero(canonical)
        yield defect.type, (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1)


def _no_segmenter() -> Segmenter:
    raise RuntimeError("A photo has no saved SAM result. Run eval first, with the same config.")


def build_tiles(
    results: Sequence[Mapping[str, Any]], records: Mapping[str, GroundTruth], cfg: Mapping[str, Any]
) -> dict[str, list[np.ndarray]]:
    """Draw one tile per miss and per false alarm, with false alarms on clean and defect photos kept apart."""
    tiles: dict[str, list[np.ndarray]] = {"misses": [], "false_alarms_clean": [], "false_alarms_defect_photos": []}
    boards: dict[str, GoldenBoard] = {}
    for result in results:
        wrong = [finding for finding in result["findings"] if not finding["on_defect"]]
        if result["planted"] <= result["found_any"] and not wrong:
            continue
        truth = records[result["photo"]]
        if truth.board_id not in boards:
            boards[truth.board_id] = load_golden_board(truth.board_id, cfg)
        board = boards[truth.board_id]
        prepared = prepare_photo(REPO_ROOT / truth.photo, board, _no_segmenter, cfg)
        name = photo_name(truth).removeprefix("visa-")

        for defect_type, box in missed_defects(truth, result, prepared, board, cfg["eval"]):
            tiles["misses"].append(make_tile(board.image, prepared.image, box, f"{name} {defect_type}"))
        group = "false_alarms_defect_photos" if truth.defects else "false_alarms_clean"
        for finding in wrong:
            caption = f"{name} {finding['ref'] or '?'} {finding['type']}"
            tiles[group].append(make_tile(board.image, prepared.image, finding["bbox"], caption))
    return tiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="final", help="read results from runs/eval/<tag>/")
    parser.add_argument("--gt", type=Path, default=REPO_ROOT / "eval" / "ground_truth")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "results")
    args = parser.parse_args()

    run_dir = REPO_ROOT / "runs" / "eval" / args.tag
    # With Muse Glimmer's results present they are the final answer; a CV-only run has only the other file.
    results_path = run_dir / "results_fused.json"
    if not results_path.exists():
        results_path = run_dir / "results_cv.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    records = {record.photo: record for record in load_ground_truth(args.gt, None, None)}

    tiles = build_tiles(results, records, load_config())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, group in tiles.items():
        if group:
            Image.fromarray(make_sheet(group)).save(args.out / f"{name}.jpg", quality=85)
        print(f"{name}: {len(group)}")
    print(f"From {results_path.name}: {sum(r['planted'] - r['found_any'] for r in results)} misses, "
          f"{sum(r['extra_findings'] for r in results)} false alarms")


if __name__ == "__main__":
    main()
