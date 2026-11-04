"""Gradio page with two tabs, "Set up board" and "Inspect".

Run from the repo root: python app/gradio_app.py        (add --share for a public link)
"""

import argparse
import time
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import yaml
from openai import APIError
from PIL import Image

from pocket_aoi.config import REPO_ROOT, load_config
from pocket_aoi.llm import LLMClient, LLMSettings
from pocket_aoi.pipeline import (
    BOARDS_DIR,
    inspect_photo,
    load_golden_board,
    read_yaml,
    relabel_board,
    setup_board,
)
from pocket_aoi.report import CARD_STYLE, banner_text, card_html
from pocket_aoi.schemas import GoldenProfile
from pocket_aoi.segment import PartMask, Segmenter, draw_parts, resolve_model_id
from pocket_aoi.silkscreen import read_refs

CFG = load_config()
REF_COLUMNS = ["ref", "class", "x0", "y0", "x1", "y1", "SAM score"]
PART_CLASSES = ["electrolytic_capacitor", "ceramic_capacitor", "resistor", "diode", "led", "ic", "transistor", "other"]

CAPTURE_GUIDE = """
**Taking the photo.** Hold the phone 25–35 cm above and roughly parallel to the board, with the board
filling most of the frame. Use diffuse light with no flash, on a plain background that contrasts with the board.
"""

# The error's own text stays out of this message, because a --share page is public.
BACKEND_DOWN = (
    "Muse Glimmer did not answer, so this photo was not fully inspected. If it runs on Kaggle, "
    "rerun the notebook, paste its new LLM_ lines into .env, and restart this page."
)


# @cache keeps the first result, so SAM and the client are each built once, and only when first used.
@cache
def get_segmenter() -> Segmenter:
    return Segmenter(resolve_model_id(CFG))


@cache
def get_client() -> LLMClient:
    return LLMClient(LLMSettings.from_config(CFG))


def list_boards() -> list[str]:
    """Return every board that has finished setup."""
    return sorted(path.parent.name for path in BOARDS_DIR.glob("*/golden_profile.json"))


def _rows(profile: GoldenProfile) -> list[list[Any]]:
    return [[part.ref or "", part.cls, *part.bbox, round(part.score, 2)] for part in profile.parts]


def _overlay(board_id: str, profile: GoldenProfile) -> Image.Image:
    canonical = Image.open(BOARDS_DIR / board_id / "golden_canonical.png").convert("RGB")
    masks = [
        PartMask(part.mask_rle, tuple(float(v) for v in part.bbox), part.score, part.cls, part.ref or "")
        for part in profile.parts
    ]
    return draw_parts(canonical, masks)


def find_parts(board_id: str, golden: str | None, netlist: str | None) -> tuple[Any, list[list[Any]], str]:
    """Run Stage 0, then prefill the table with whatever Prompt B can read from the silkscreen."""
    if not board_id or not golden:
        raise gr.Error("Enter a board id and add a golden photo.")
    profile = setup_board(board_id, Path(golden), Path(netlist) if netlist else None, get_segmenter, CFG)
    board = load_golden_board(board_id, CFG)
    rows = _rows(profile)
    note = f"Found {len(rows)} parts. Check the table, type any missing refs, and save."
    if board.netlist is None:
        note += " With no netlist, consequence cards can only use a template."
    else:
        try:
            refs = read_refs(get_client(), board.image, profile.parts, board.netlist, CFG)
        except ValueError as error:  # The .env file has no LLM key, so the refs stay for typing by hand.
            refs, note = [None] * len(rows), f"{note} Silkscreen reading was skipped: {error}"
        except APIError:  # The backend is down, which also leaves the refs for typing by hand.
            refs, note = [None] * len(rows), f"{note} Silkscreen reading was skipped: Muse Glimmer did not answer."
        for row, ref in zip(rows, refs):
            row[0] = ref or ""
    return _overlay(board_id, profile), rows, note


def save_refs(board_id: str, table: Any) -> tuple[Any, str, Any]:
    """Save the table to refs.yaml and rebuild the golden profile, without running SAM again."""
    board_dir = BOARDS_DIR / board_id
    if not (board_dir / "golden_profile.json").exists():
        raise gr.Error("Run Find parts first.")
    # A Gradio Dataframe can arrive as a pandas table; values.tolist() gives plain lists of rows.
    rows: Sequence[Sequence[Any]] = table.values.tolist() if hasattr(table, "values") else table

    # The table has no column for marker hints, so carry over any already typed into refs.yaml.
    hints = {label["ref"]: label["marker_hint"] for label in read_yaml(board_dir / "refs.yaml").get("parts", []) if label.get("marker_hint")}

    labels, unnamed = [], 0
    for row in rows:
        box = [int(float(value)) for value in row[2:6]]
        ref = str(row[0]).strip()
        if ref:
            label = {"ref": ref, "cls": str(row[1]), "box": box}
            if ref in hints:
                label["marker_hint"] = hints[ref]
            labels.append(label)
        else:
            unnamed += 1
    (board_dir / "refs.yaml").write_text(yaml.safe_dump({"parts": labels}, default_flow_style=None), encoding="utf-8")

    profile = relabel_board(board_id, CFG)

    note = f"Saved {len(labels)} parts to inspect."
    if unnamed:
        note += f" {unnamed} rows had no ref, so they are kept as known objects and not inspected."
    return _overlay(board_id, profile), note, gr.update(choices=list_boards(), value=board_id)


def inspect(board_id: str | None, photo: str | None) -> tuple[str, np.ndarray | None, str]:
    """Run the whole pipeline on one uploaded photo."""
    if not board_id or not photo:
        raise gr.Error("Choose a board and add a photo.")
    try:
        client = get_client()
    except ValueError as error:
        raise gr.Error(str(error)) from error

    board = load_golden_board(board_id, CFG)
    run_dir = REPO_ROOT / "runs" / "app" / f"{board_id}_{int(time.time())}"
    try:
        result = inspect_photo(Path(photo), board, get_segmenter, client, CFG, run_dir)
    except APIError as error:
        raise gr.Error(BACKEND_DOWN) from error
    if not result.ok or result.report is None:
        return f'<div class="banner check">{result.message}</div>', None, ""

    report = result.report
    text, style = banner_text(report)
    timings = ", ".join(f"{stage} {seconds:.1f} s" for stage, seconds in report.timings_s.items())
    overlay = np.asarray(Image.open(result.paths["overlay.png"]).convert("RGB"))
    cards = "".join(card_html(fault) for fault in report.faults) or "<p>No differences from the golden board.</p>"
    return f'<div class="banner {style}">{text}</div>', overlay, f'{cards}<p class="timing">{timings}</p>'


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Pocket AOI") as app:
        gr.HTML(f"<style>{CARD_STYLE}</style>")
        gr.Markdown("# Pocket AOI\nFrom one phone photo, find assembly mistakes on a through-hole board, and see what each one will break.")

        with gr.Tab("Inspect"):
            boards = gr.Dropdown(list_boards(), label="Board")
            gr.Markdown(CAPTURE_GUIDE)
            photo = gr.Image(type="filepath", label="Photo of the board under test", height=320)
            run = gr.Button("Inspect", variant="primary")
            banner = gr.HTML()
            overlay = gr.Image(label="Faults on your photo", interactive=False)
            cards = gr.HTML()
            run.click(inspect, [boards, photo], [banner, overlay, cards])

        with gr.Tab("Set up board"):
            gr.Markdown("One photo of a correctly assembled board, once per design, plus its netlist if you have one.")
            board_id = gr.Textbox(label="Board id", placeholder="ne555-flasher")
            golden = gr.Image(type="filepath", label="Golden photo", height=320)
            netlist = gr.File(label="netlist.json (optional)", file_types=[".json"], type="filepath")
            find = gr.Button("Find parts", variant="primary")
            found = gr.Image(label="Parts found", interactive=False)
            note = gr.Markdown()
            table = gr.Dataframe(headers=REF_COLUMNS, label="Parts: type each ref, leave blank to skip", interactive=True, wrap=True)
            save = gr.Button("Save refs")
            find.click(find_parts, [board_id, golden, netlist], [found, table, note])
            save.click(save_refs, [board_id, table], [found, note, boards])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--share", action="store_true", help="also serve a public link through Gradio")
    args = parser.parse_args()
    build_app().launch(share=args.share)


if __name__ == "__main__":
    main()
