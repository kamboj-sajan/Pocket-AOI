"""Stage 4: Prompt A per suspect, and the fusion rule that combines it with the CV finding."""

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pocket_aoi.llm import LLMClient, load_prompt
from pocket_aoi.netlist import find_component
from pocket_aoi.schemas import Fault, GoldenProfile, Netlist, Part, Suspect, Verdict

NO_REPLY = "no valid reply"


@dataclass(frozen=True)
class Judged:
    """One suspect with both votes, kept even when dropped, so the ablation can compare them."""

    suspect: Suspect
    verdict: Verdict
    fault: Fault | None  # None when Glimmer rejected the suspect


def unsure_verdict(reason: str) -> Verdict:
    """Stand in for a reply that never validated, so one bad call cannot crash a run."""
    return Verdict(golden_observation="", test_observation="", verdict="UNSURE", confidence=0.0, evidence=reason)


def prompt_values(
    suspect: Suspect,
    part: Part | None,
    netlist: Netlist | None,
    cfg: Mapping[str, Any],
) -> dict[str, object]:
    """Fill Prompt A's placeholders. EXTRA suspects have no golden part, so those read as unknown."""
    component = find_component(netlist, part.ref) if netlist and part and part.ref else None
    evidence = suspect.evidence
    return {
        "ref": suspect.golden_ref or "unknown part",
        "part_class": part.cls if part else "unknown",
        "package": (part.package if part else None) or "unknown",
        "value": (component.value if component else "") or "unknown",
        "polarity_hint": cfg["polarity_hints"].get(part.cls if part else "", "none"),
        "golden_marker_hint": "unknown",
        "cv_finding": suspect.cv_finding,
        "ssim": float(evidence.get("ssim", 0.0)),
        "best_rotation_deg": int(evidence.get("best_rotation_deg", 0)),
        "offset_ratio": float(evidence.get("offset_ratio", 0.0)),
    }


def fuse(suspect: Suspect, verdict: Verdict, verdict_cfg: Mapping[str, Any]) -> Fault | None:
    """Apply section 3, Stage 4, step 3. Glimmer can reject a suspect, but cannot invent a new one."""
    if verdict.verdict == "OK" and verdict.confidence >= verdict_cfg["ok_drop_confidence"]:
        return None
    if verdict.verdict in ("OK", "UNSURE"):
        return Fault(
            ref=suspect.golden_ref,
            fault_type=suspect.cv_finding,
            status="needs_review",
            suspect=suspect,
            verdict=verdict,
            consequence=None,
        )
    trusted = verdict.confidence >= verdict_cfg["fault_type_confidence"]
    return Fault(
        ref=suspect.golden_ref,
        fault_type=verdict.verdict if trusted else suspect.cv_finding,
        status="confirmed",
        suspect=suspect,
        verdict=verdict,
        consequence=None,
    )


def judge_suspects(
    client: LLMClient,
    suspects: Sequence[Suspect],
    profile: GoldenProfile,
    netlist: Netlist | None,
    cfg: Mapping[str, Any],
) -> list[Judged]:
    """Ask Muse Glimmer about every suspect at once, keeping the suspects' order."""
    prompt = load_prompt("verdict", cfg["llm"]["prompt_versions"]["verdict"])
    parts = {part.ref: part for part in profile.parts if part.ref}

    def judge(suspect: Suspect) -> Judged:
        values = prompt_values(suspect, parts.get(suspect.golden_ref), netlist, cfg)
        image = Path(suspect.composite_path).read_bytes()
        verdict = client.complete_json(Verdict, prompt, values, image=image) or unsure_verdict(NO_REPLY)
        return Judged(suspect, verdict, fuse(suspect, verdict, cfg["verdict"]))

    # A thread pool suits this because each call waits on the network, not the CPU. pool.map keeps input order.
    with ThreadPoolExecutor(max_workers=cfg["verdict"]["max_workers"]) as pool:
        return list(pool.map(judge, suspects))
