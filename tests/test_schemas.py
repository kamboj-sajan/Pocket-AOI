"""Every data contract must survive a trip through JSON unchanged."""

import pytest
from pydantic import ValidationError

from pocket_aoi.schemas import Consequence, Fault, GoldenProfile, Part, Report, Suspect, Verdict


def _report() -> Report:
    suspect = Suspect(
        golden_ref="C1",
        cv_finding="REVERSED",
        bbox=(10, 20, 110, 220),
        evidence={"ssim": 0.41, "best_rotation_deg": 180.0},
        composite_path="runs/x/debug/C1.png",
    )
    verdict = Verdict(
        golden_observation="stripe on the left",
        test_observation="stripe on the right",
        verdict="REVERSED",
        confidence=0.92,
        evidence="the stripe moved to the opposite side",
    )
    consequence = Consequence(
        headline="C1 reversed: electrolytic can fail on power-up",
        consequence="C1 sees reverse voltage across the VCC and GND nets.",
        severity="CRITICAL",
        safe_to_power=False,
        fix="Turn C1 so its stripe faces the GND pad.",
        check_first="Check C1 for a bulging top.",
        facts_used=["C1.1 on net VCC"],
        uncertain="",
    )
    fault = Fault(
        ref="C1",
        fault_type="REVERSED",
        status="confirmed",
        suspect=suspect,
        verdict=verdict,
        consequence=consequence,
    )
    return Report(
        board_id="ne555-flasher",
        test_image="data/eval/555/555_c03_s1.jpg",
        safe_to_power=False,
        faults=[fault],
        timings_s={"align": 1.2},
        versions={"config": "abc123"},
    )


def test_report_round_trip() -> None:
    report = _report()
    assert Report.model_validate_json(report.model_dump_json()) == report


def test_golden_profile_round_trip_restores_tuples() -> None:
    part = Part(
        ref="U1",
        cls="ic",
        bbox=(1, 2, 3, 4),
        centroid=(2.0, 3.0),
        mask_rle={"size": [10, 10], "counts": "abc"},
        score=0.9,
        package="DIP-8",
        polarized=True,
    )
    profile = GoldenProfile(board_id="b", canonical_size=(2000, 1500), parts=[part])
    restored = GoldenProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert isinstance(restored.parts[0].bbox, tuple)  # JSON has no tuples; pydantic rebuilds them.


def test_verdict_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        Verdict(golden_observation="", test_observation="", verdict="OK", confidence=1.5, evidence="")


def test_verdict_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        Verdict(golden_observation="", test_observation="", verdict="BROKEN", confidence=0.5, evidence="")
