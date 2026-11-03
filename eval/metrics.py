"""Detection rate, false positives per photo, runtime by stage, and the ablation (section 7)."""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

ORIENTATION = {"REVERSED", "ROTATED"}


def same_type(found: str, expected: str, accept: Sequence[str] = ()) -> bool:
    """REVERSED and ROTATED count as one orientation type, and a defect may accept other types."""
    if found == expected or found in accept:
        return True
    return found in ORIENTATION and expected in ORIENTATION


def box_covers_mask(box: Sequence[int], mask: np.ndarray, min_overlap: float) -> bool:
    """Return True if a finding's box and a defect's mask overlap by this share of the smaller one."""
    x0, y0, x1, y1 = box
    mask_area = int(mask.sum())
    box_area = max((x1 - x0) * (y1 - y0), 1)
    if mask_area == 0:
        return False
    overlap = int(mask[y0:y1, x0:x1].sum())
    return overlap / min(mask_area, box_area) >= min_overlap


@dataclass
class PhotoResult:
    """One photo's outcome. Defects outside the board outline are kept apart from `planted`."""

    photo: str
    board_id: str
    label: str
    split: str
    aligned: bool
    message: str = ""
    planted: int = 0
    found_any: int = 0
    found_right_type: int = 0
    found_in_review: int = 0
    extra_findings: int = 0
    needs_review: int = 0
    off_board: int = 0
    off_board_found: int = 0
    missed: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    seconds: dict[str, float] = field(default_factory=dict)


def _rate(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})" if denominator else "none"


def _short_label(result: PhotoResult) -> str:
    """Group a photo by its fault, ignoring other labels the dataset gives the same photo."""
    labels = [label.strip() for label in result.label.split(",")]
    for name in ("missing", "extra", "wrong place", "bent"):
        if name in labels:
            return name
    return "clean"


def _row(members: Sequence[PhotoResult]) -> dict[str, Any]:
    """Add up one group of photos."""
    defect_photos = [r for r in members if r.planted or r.off_board]
    clean_photos = [r for r in members if not r.planted and not r.off_board]
    clean_aligned = [r for r in clean_photos if r.aligned]
    return {
        "photos": len(members),
        "rejected_by_gate": sum(not r.aligned for r in members),
        "planted": sum(r.planted for r in defect_photos),
        "found_right_type": sum(r.found_right_type for r in defect_photos),
        "found_any_type": sum(r.found_any for r in defect_photos),
        "found_in_review": sum(r.found_in_review for r in defect_photos),
        "off_board": sum(r.off_board for r in defect_photos),
        "off_board_found": sum(r.off_board_found for r in defect_photos),
        "clean_photos": len(clean_photos),
        "clean_with_false_alarm": sum(r.extra_findings > 0 for r in clean_aligned),
        "false_alarms_per_clean_photo": (
            sum(r.extra_findings for r in clean_aligned) / len(clean_aligned) if clean_aligned else 0.0
        ),
        "extra_per_defect_photo": (
            sum(r.extra_findings for r in defect_photos) / len(defect_photos) if defect_photos else 0.0
        ),
        "needs_review_per_photo": sum(r.needs_review for r in members) / len(members) if members else 0.0,
    }


def _grouped(results: Sequence[PhotoResult], key: Callable[[PhotoResult], str]) -> dict[str, dict[str, Any]]:
    """Add up the results per group, with a final row for everything."""
    groups: dict[str, list[PhotoResult]] = defaultdict(list)
    for result in results:
        groups[key(result)].append(result)
    summary = {name: _row(members) for name, members in sorted(groups.items())}
    summary["ALL"] = _row(results)
    return summary


def summarise(results: Sequence[PhotoResult]) -> dict[str, dict[str, Any]]:
    """Group results per board into detection and false-alarm numbers."""
    return _grouped(results, lambda result: result.board_id)


def summarise_by_label(results: Sequence[PhotoResult]) -> dict[str, dict[str, Any]]:
    """Group results per kind of defect, so each kind's detection rate can be read separately."""
    return _grouped(results, _short_label)


def format_summary(summary: Mapping[str, Mapping[str, Any]], first_column: str = "Board") -> str:
    """Render a summary as a Markdown table."""
    lines = [
        f"| {first_column} | Photos | Rejected by gate | Found, right type | Found, any type | Seen only in review "
        "| Off the board outline (found) | Clean photos with a false alarm | False alarms per clean photo "
        "| Extra findings per defect photo | Needs review per photo |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {row['photos']} | {row['rejected_by_gate']} "
            f"| {_rate(row['found_right_type'], row['planted'])} | {_rate(row['found_any_type'], row['planted'])} "
            f"| {row['found_in_review']} | {row['off_board']} ({row['off_board_found']}) "
            f"| {_rate(row['clean_with_false_alarm'], row['clean_photos'])} | {row['false_alarms_per_clean_photo']:.2f} "
            f"| {row['extra_per_defect_photo']:.2f} | {row['needs_review_per_photo']:.2f} |"
        )
    return "\n".join(lines)


def format_ablation(cv_only: Sequence[PhotoResult], fused: Sequence[PhotoResult]) -> str:
    """Compare CV alone with CV plus Muse Glimmer: false alarms removed against recall lost."""
    cv_row, fused_row = _row(cv_only), _row(fused)
    lines = [
        "| Pipeline | Found, right type | Found, any type | Seen only in review | False alarms per clean photo | Extra findings per defect photo | Needs review per photo |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in (("CV only", cv_row), ("CV + Muse Glimmer", fused_row)):
        lines.append(
            f"| {name} | {_rate(row['found_right_type'], row['planted'])} | {_rate(row['found_any_type'], row['planted'])} "
            f"| {row['found_in_review']} | {row['false_alarms_per_clean_photo']:.2f} | {row['extra_per_defect_photo']:.2f} "
            f"| {row['needs_review_per_photo']:.2f} |"
        )
    return "\n".join(lines)


def format_runtime(results: Sequence[PhotoResult]) -> str:
    """Report the median seconds per photo for each stage."""
    stages: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for stage, seconds in result.seconds.items():
            stages[stage].append(seconds)
    lines = ["| Stage | Median seconds per photo |", "| --- | --- |"]
    lines.extend(f"| {stage} | {float(np.median(values)):.1f} |" for stage, values in stages.items())
    return "\n".join(lines)


def format_misses(results: Sequence[PhotoResult]) -> str:
    """List every missed defect and every photo with a false alarm, for the README."""
    lines = ["## Missed defects", ""]
    for result in results:
        if result.planted > result.found_any:
            state = "" if result.aligned else f" (rejected: {result.message})"
            lines.append(f"- {result.board_id} `{result.photo}`: {result.label}{state}")
    lines += ["", "## Clean photos with a false alarm", ""]
    for result in results:
        if not result.planted and not result.off_board and result.extra_findings:
            names = [f"{finding['ref'] or 'unknown'} {finding['type']}" for finding in result.findings]
            lines.append(f"- {result.board_id} `{result.photo}`: {', '.join(names)}")
    return "\n".join(lines)
