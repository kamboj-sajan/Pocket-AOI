"""Write one ground-truth JSON per VisA photo used, with a fixed dev and test split.

VisA holds real photos of real defects on four PCBs, with a pixel mask per defect. Only defects
that are assembly mistakes are used: missing, extra, misplaced and bent parts. Melted, scratched,
dirty and burnt boards are not assembly mistakes and are left out, and clean photos come from the
dataset's normal folder. The golden photo is never evaluated.

Run once from the repo root: python eval/make_visa_ground_truth.py
"""

import csv
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VISA_DIR = REPO_ROOT / "data" / "raw" / "visa"
OUT_DIR = REPO_ROOT / "eval" / "ground_truth" / "visa"
BOARDS = ("pcb1", "pcb2", "pcb3", "pcb4")
GOLDEN_NAME = "0000.JPG"
CLEAN_DEV_PER_BOARD = 20
CLEAN_TEST_PER_BOARD = 40
SEED = 20261103

# Dataset label -> (fault type, other finding types that also count), most important first.
# A misplaced part leaves a gap at its old spot and a new part at its new one, so either counts.
STRUCTURAL = {
    "missing": ("MISSING", []),
    "extra": ("EXTRA", []),
    "wrong place": ("SHIFTED", ["MISSING", "EXTRA", "WRONG_PART"]),
    "bent": ("SHIFTED", ["WRONG_PART"]),
}


def relative(path: str) -> str:
    return str(Path("data/raw/visa") / path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.json"):
        old.unlink()

    rng = random.Random(SEED)
    written = 0
    for board in BOARDS:
        with (VISA_DIR / board / "image_anno.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        # Group defect photos by their fault, so dev and test each get a fair share of every kind.
        by_kind: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            labels = [label.strip() for label in row["label"].split(",")]
            kind = next((name for name in STRUCTURAL if name in labels), None)
            if kind is not None:
                by_kind.setdefault(kind, []).append(row)

        for kind, kind_rows in by_kind.items():
            rng.shuffle(kind_rows)
            middle = len(kind_rows) // 2
            for index, row in enumerate(kind_rows):
                fault, accept = STRUCTURAL[kind]
                record = {
                    "photo": relative(row["image"]),
                    "board_id": f"visa-{board}",
                    "defects": [{"type": fault, "mask": relative(row["mask"]), "accept": accept}],
                    "label": row["label"],
                    "split": "dev" if index < middle else "test",
                }
                name = f"{board}_{kind.replace(' ', '')}_{Path(row['image']).stem}.json"
                (OUT_DIR / name).write_text(json.dumps(record, indent=1), encoding="utf-8")
                written += 1

        clean = [row for row in rows if row["label"] == "normal" and not row["image"].endswith(GOLDEN_NAME)]
        rng.shuffle(clean)
        for index, row in enumerate(clean[: CLEAN_DEV_PER_BOARD + CLEAN_TEST_PER_BOARD]):
            record = {
                "photo": relative(row["image"]),
                "board_id": f"visa-{board}",
                "defects": [],
                "label": "normal",
                "split": "dev" if index < CLEAN_DEV_PER_BOARD else "test",
            }
            (OUT_DIR / f"{board}_clean_{Path(row['image']).stem}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
            written += 1
    print(f"wrote {written} ground-truth files to {OUT_DIR}")


if __name__ == "__main__":
    main()
