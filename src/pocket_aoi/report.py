"""Stage 6: overlay on the original photo, report.json, report.html."""

import base64
import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from pocket_aoi.consequence import SEVERITY_RANK
from pocket_aoi.schemas import Fault, Report

# RGB, because images here are RGB arrays.
_SEVERITY_COLOURS = {"CRITICAL": (220, 30, 30), "MAJOR": (245, 130, 20), "MINOR": (245, 205, 30)}
_REVIEW_COLOUR = _SEVERITY_COLOURS["MINOR"]


def fault_label(fault: Fault) -> str:
    """Name a fault the way the overlay shows it, such as "C7 MISSING"."""
    return f"{fault.ref or 'Unknown'} {fault.fault_type}"


def fault_colour(fault: Fault) -> tuple[int, int, int]:
    """Red for CRITICAL, orange for MAJOR, yellow for MINOR or needs review."""
    if fault.status == "needs_review" or fault.consequence is None:
        return _REVIEW_COLOUR
    return _SEVERITY_COLOURS[fault.consequence.severity]


def sort_faults(faults: Sequence[Fault]) -> list[Fault]:
    """Sort by severity, worst first, then by ref. Faults needing review come after confirmed ones."""

    def key(fault: Fault) -> tuple[int, int, str]:
        rank = SEVERITY_RANK[fault.consequence.severity] if fault.consequence else -1
        # Negating the rank sorts the worst first while the ref still sorts A to Z.
        return (0 if fault.status == "confirmed" else 1, -rank, fault.ref or "~")

    return sorted(faults, key=key)


def is_safe_to_power(faults: Sequence[Fault]) -> bool:
    """A board is unsafe when any confirmed fault is CRITICAL or its card says not to power it."""
    return not any(
        fault.status == "confirmed"
        and fault.consequence is not None
        and (fault.consequence.severity == "CRITICAL" or not fault.consequence.safe_to_power)
        for fault in faults
    )


def draw_report_overlay(
    photo: np.ndarray,
    faults: Sequence[Fault],
    homography: np.ndarray,
    long_side: int,
) -> np.ndarray:
    """Draw each fault's box on the original photo, through the inverse of the canonical warp."""
    inverse = np.linalg.inv(homography)
    scale = min(long_side / max(photo.shape[:2]), 1.0)
    canvas = cv2.resize(photo, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else photo.copy()
    thickness = max(canvas.shape[1] // 500, 2)
    font_scale = max(canvas.shape[1] / 1600, 0.6)

    for fault in faults:
        x0, y0, x1, y1 = fault.suspect.bbox
        corners = np.float32([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]])
        polygon = (cv2.perspectiveTransform(corners, inverse) * scale).astype(np.int32)
        colour = fault_colour(fault)
        cv2.polylines(canvas, [polygon], isClosed=True, color=colour, thickness=thickness, lineType=cv2.LINE_AA)

        label = fault_label(fault)
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # A label above the box, kept on the canvas by clamping its corner.
        left = int(np.clip(polygon[:, 0, 0].min(), 0, max(canvas.shape[1] - width - 8, 0)))
        top = int(np.clip(polygon[:, 0, 1].min() - height - baseline - 8, 0, max(canvas.shape[0] - height - baseline - 8, 0)))
        cv2.rectangle(canvas, (left, top), (left + width + 8, top + height + baseline + 8), colour, thickness=-1)
        cv2.putText(canvas, label, (left + 4, top + height + 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    return canvas


def _image_uri(path: str) -> str:
    """Embed a composite in the page as JPEG, so report.html stays a single file that can be shared."""
    image = cv2.imread(path)
    if image is None:
        return ""
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.tobytes()).decode('ascii')}" if ok else ""


def _field(title: str, text: str) -> str:
    return f'<p class="field"><b>{html.escape(title)}:</b> {html.escape(text)}</p>' if text else ""


def card_html(fault: Fault) -> str:
    """Render one fault as a card: both crops, then the headline, consequence, fix and check-first."""
    card = fault.consequence
    severity = card.severity if card else "REVIEW"
    chip = "NEEDS REVIEW" if fault.status == "needs_review" else severity
    chip_class = "review" if fault.status == "needs_review" else severity.lower()
    headline = card.headline if card else fault_label(fault)
    verdict = fault.verdict
    cv_vote = f"CV finding {fault.suspect.cv_finding}, SSIM {fault.suspect.evidence.get('ssim', 0.0):.2f}"
    model_vote = f"Muse Glimmer {verdict.verdict} at {verdict.confidence:.2f}: {verdict.evidence}" if verdict else ""
    body = [
        f'<article class="card"><div class="head"><span class="chip {chip_class}">{chip}</span>',
        f"<h2>{html.escape(headline)}</h2></div>",
        f'<img src="{_image_uri(fault.suspect.composite_path)}" alt="Golden and test crops of {html.escape(fault_label(fault))}">',
    ]
    if card:
        body += [
            f"<p>{html.escape(card.consequence)}</p>",
            _field("Fix", card.fix),
            _field("Check first if already powered", card.check_first),
            _field("Not certain", card.uncertain),
        ]
    body += [f'<p class="votes">{html.escape(cv_vote)}<br>{html.escape(model_vote)}</p>', "</article>"]
    return "".join(body)


# Used by report.html and by the Gradio page, so a card looks the same in both.
CARD_STYLE = """
:root { --bg: #f6f7f8; --panel: #ffffff; --ink: #16191d; --muted: #5b6470; --line: #d9dde2; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #14171a; --panel: #1d2125; --ink: #eceff2; --muted: #a2abb5; --line: #333a41; }
}
.card h2 { font-size: 1.05rem; margin: 0; }
.card img { display: block; width: 100%; height: auto; border-radius: 6px; margin: 8px 0; }
.banner { padding: 12px 16px; border-radius: 8px; font-weight: 700; margin-bottom: 16px; color: #fff; }
.banner.no { background: #b3261e; } .banner.yes { background: #1b7f3b; } .banner.check { background: #a85b00; }
.card { background: var(--panel); color: var(--ink); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; margin: 14px 0; }
.head { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.chip { font-size: .75rem; font-weight: 700; padding: 2px 8px; border-radius: 99px; color: #000; }
.chip.critical { background: #dc1e1e; color: #fff; } .chip.major { background: #f58214; }
.chip.minor, .chip.review { background: #f5cd1e; }
.field { margin: 6px 0; } .votes { color: var(--muted); font-size: .85rem; margin: 8px 0 0; }
.timing { color: var(--muted); font-size: .85rem; }
"""

_PAGE_STYLE = CARD_STYLE + """
body { margin: 0 auto; max-width: 960px; padding: 16px; background: var(--bg); color: var(--ink);
       font: 16px/1.5 system-ui, sans-serif; }
h1 { font-size: 1.4rem; margin: 0 0 12px; }
.overlay { display: block; width: 100%; height: auto; border-radius: 6px; margin: 8px 0; }
"""


def banner_text(report: Report) -> tuple[str, str]:
    """Return the banner's text and style. It only says yes when nothing at all was flagged."""
    if not report.safe_to_power:
        return "Safe to power: NO", "no"
    if report.faults:
        return f"Safe to power: no critical fault, but {len(report.faults)} to check first", "check"
    return "Safe to power: YES, no differences from the golden board", "yes"


def render_html(report: Report, overlay_path: Path) -> str:
    """Build report.html as one self-contained page."""
    text, style = banner_text(report)
    ok, overlay = cv2.imencode(".jpg", cv2.imread(str(overlay_path)), [cv2.IMWRITE_JPEG_QUALITY, 85])
    overlay_uri = f"data:image/jpeg;base64,{base64.b64encode(overlay.tobytes()).decode('ascii')}" if ok else ""
    timings = ", ".join(f"{stage} {seconds:.1f} s" for stage, seconds in report.timings_s.items())
    cards = "".join(card_html(fault) for fault in report.faults)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Pocket AOI report</title><style>{_PAGE_STYLE}</style></head><body>"
        f"<h1>Pocket AOI: {html.escape(report.board_id)}</h1>"
        f'<div class="banner {style}">{html.escape(text)}</div>'
        f'<img class="overlay" src="{overlay_uri}" alt="Test photo with each fault outlined">'
        f"{cards}"
        f'<p class="timing">{html.escape(timings)}</p>'
        "</body></html>"
    )


def save_report(report: Report, photo: np.ndarray, homography: np.ndarray, run_dir: Path, cfg: Mapping[str, Any]) -> dict[str, Path]:
    """Write overlay.png, report.json and report.html, and return their paths."""
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: run_dir / name for name in ("overlay.png", "report.json", "report.html")}
    overlay = draw_report_overlay(photo, report.faults, homography, cfg["report"]["overlay_long_side"])
    Image.fromarray(overlay).save(paths["overlay.png"])
    paths["report.json"].write_text(report.model_dump_json(indent=1), encoding="utf-8")
    paths["report.html"].write_text(render_html(report, paths["overlay.png"]), encoding="utf-8")
    return paths
