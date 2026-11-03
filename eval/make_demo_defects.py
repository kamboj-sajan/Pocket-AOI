"""Plant synthetic defects on a real photo of a real board, to demonstrate the report (rule 9).

Nothing here reaches the results table. These photos are for the demo and for testing Stages 4 to 6
on a board with a netlist, and every output name says "synthetic".

Each part is edited in the golden frame and pasted back through the inverse warp, so the photo keeps
its own camera angle and lighting. When the only clean photo of a board is its golden photo, --second-shot
also moves the camera slightly and dims the light, so the two do not match pixel for pixel.

Run from the repo root:
    python -m eval.make_demo_defects --board-id ardutouch --photo data/raw/ardutouch/web_02.jpg \
        --reverse C3 U2 --remove R1 C4 --second-shot --out data/demo/ardutouch_synthetic.jpg
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pocket_aoi.align import filled_hull
from pocket_aoi.config import load_config
from pocket_aoi.io import load_rgb
from pocket_aoi.pipeline import GoldenBoard, load_golden_board, prepare_photo
from pocket_aoi.schemas import Part
from pocket_aoi.segment import Segmenter, resolve_model_id, rle_to_mask

_FEATHER_PX = 5
_SHOT_JITTER_RATIO = 0.008  # How far each photo corner moves, as a share of the photo's width.
_SHOT_GAIN = 0.94
_SHOT_NOISE = 2.0


def _part(board: GoldenBoard, ref: str) -> Part:
    # next() takes the first match from a generator, and raises StopIteration without a default.
    part = next((item for item in board.profile.parts if item.ref == ref), None)
    if part is None:
        raise ValueError(f"{ref} is not on {board.board_id}")
    return part


def _footprint(part: Part, shape: tuple[int, int]) -> np.ndarray:
    """Return the part's mask with holes filled, or its box for a part that has only a hand-drawn box."""
    mask = rle_to_mask(part.mask_rle)
    if mask.any():
        return filled_hull(mask)
    footprint = np.zeros(shape, bool)
    x0, y0, x1, y1 = part.bbox
    footprint[y0:y1, x0:x1] = True
    return footprint


def reverse_part(canonical: np.ndarray, part: Part) -> tuple[np.ndarray, np.ndarray]:
    """Turn a part 180 degrees where it sits. Returns the edited image and the changed area as a mask."""
    x0, y0, x1, y1 = part.bbox
    edited = canonical.copy()
    # A [::-1, ::-1] slice flips both axes, which is the same as a half turn.
    edited[y0:y1, x0:x1] = canonical[y0:y1, x0:x1][::-1, ::-1]
    changed = np.zeros(canonical.shape[:2], bool)
    changed[y0:y1, x0:x1] = True
    return edited, changed


def remove_part(canonical: np.ndarray, part: Part) -> tuple[np.ndarray, np.ndarray]:
    """Paint a part out with cv2.inpaint, as rule 9 describes."""
    changed = cv2.dilate(_footprint(part, canonical.shape[:2]).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    edited = cv2.inpaint(canonical, changed.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return edited, changed


def paste_back(photo: np.ndarray, edited: np.ndarray, changed: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Warp the edited pixels back onto the photo, with a soft edge so no hard seam shows."""
    height, width = photo.shape[:2]
    inverse = np.linalg.inv(homography)
    warped = cv2.warpPerspective(edited, inverse, (width, height), flags=cv2.INTER_LINEAR)
    weight = cv2.warpPerspective(changed.astype(np.float32), inverse, (width, height), flags=cv2.INTER_LINEAR)
    weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=_FEATHER_PX)[:, :, np.newaxis]
    # NumPy broadcasting: the H x W x 1 weight applies to all three colour channels.
    return (photo * (1 - weight) + warped * weight).astype(np.uint8)


def simulate_second_shot(photo: np.ndarray, seed: int = 7) -> np.ndarray:
    """Imitate another shot of the same board: a small change of camera position, light and sensor noise."""
    rng = np.random.default_rng(seed)
    height, width = photo.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    moved = corners + rng.uniform(-1, 1, (4, 2)).astype(np.float32) * _SHOT_JITTER_RATIO * width
    warp = cv2.getPerspectiveTransform(corners, moved)
    shifted = cv2.warpPerspective(photo, warp, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.clip(shifted * _SHOT_GAIN + rng.normal(0, _SHOT_NOISE, shifted.shape), 0, 255).astype(np.uint8)


def plant(photo_path: Path, board: GoldenBoard, reverse: list[str], remove: list[str], cfg: dict) -> np.ndarray:
    """Return the photo with the named parts edited."""
    prepared = prepare_photo(photo_path, board, lambda: Segmenter(resolve_model_id(cfg)), cfg)
    if not prepared.ok or prepared.image is None or prepared.homography is None:
        raise ValueError(f"the photo failed the quality gate: {prepared.message}")

    photo = np.asarray(load_rgb(photo_path))
    canonical = prepared.image
    for ref in reverse:
        edited, changed = reverse_part(canonical, _part(board, ref))
        photo = paste_back(photo, edited, changed, prepared.homography)
    for ref in remove:
        edited, changed = remove_part(canonical, _part(board, ref))
        photo = paste_back(photo, edited, changed, prepared.homography)
    return photo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--board-id", required=True)
    parser.add_argument("--photo", type=Path, required=True)
    parser.add_argument("--reverse", nargs="*", default=[], help="refs to turn by 180 degrees")
    parser.add_argument("--remove", nargs="*", default=[], help="refs to paint out")
    parser.add_argument("--second-shot", action="store_true", help="also imitate a slightly different camera position and light")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config()
    board = load_golden_board(args.board_id, cfg)
    photo = plant(args.photo, board, args.reverse, args.remove, cfg)
    if args.second_shot:
        photo = simulate_second_shot(photo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(photo).save(args.out, quality=95)
    print(f"Saved {args.out}. Turned: {args.reverse or 'none'}. Removed: {args.remove or 'none'}.")


if __name__ == "__main__":
    main()
