"""Stage 5: netlist facts, severity floors, Prompt D, and ref and net validation."""

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pocket_aoi.llm import LLMClient, load_prompt
from pocket_aoi.netlist import (
    component_refs,
    find_component,
    neighbour_table,
    net_names,
    pad_to_net,
    pin_net_table,
    pin_swap_table,
)
from pocket_aoi.schemas import Consequence, Fault, FaultType, Netlist

SEVERITY_RANK = {"MINOR": 0, "MAJOR": 1, "CRITICAL": 2}
ORIENTATION_FAULTS = ("REVERSED", "ROTATED")

# Looks like a reference designator or a chip name: letters then digits, such as C7, LED3 or LM386.
_REF_LIKE = re.compile(r"\b[A-Z]{1,4}\d+[A-Z]?\b")
# A pad on a part, such as U2.8, C3.+ or Eagle's R1.P$2
_NODE_LIKE = re.compile(r"\b([A-Z]{1,4}\d+)\.([A-Za-z0-9$+\-]+)")
# The word "net" followed by a name, which must hold a capital or a digit so prose such as "net effect" passes.
_NET_LIKE = re.compile(r"\bnets?\s+([A-Za-z0-9_$+\-/]*[A-Z0-9][A-Za-z0-9_$+\-/]*)")


def severity_floor(fault_type: str, part_type: str) -> str:
    """Return the lowest severity the rules allow. The model may raise it, but never lower it."""
    if fault_type in ORIENTATION_FAULTS:
        if part_type in ("electrolytic_capacitor", "ic"):
            return "CRITICAL"
        if part_type in ("diode", "led", "transistor"):
            return "MAJOR"
    return "MINOR"


def prompt_values(fault: Fault, netlist: Netlist, confidence: float) -> dict[str, object]:
    """Gather the exact netlist facts Prompt D may use."""
    ref = fault.ref or ""
    component = find_component(netlist, ref)
    part_type = component.type if component else "other"
    return {
        "board_description": netlist.description or netlist.board_id,
        "supply_volts": netlist.supply.volts,
        "supply_net": netlist.supply.net,
        "ref": ref,
        "fault_type": fault.fault_type,
        "confidence": confidence,
        "value": (component.value if component else "") or "unknown",
        "part_type": part_type,
        "package": (component.package if component else None) or "unknown",
        "pin_net_table": pin_net_table(netlist, ref),
        "neighbour_table": neighbour_table(netlist, ref),
        "pin_swap_table": pin_swap_table(netlist, ref) if fault.fault_type in ORIENTATION_FAULTS else "not an orientation fault",
        "severity_floor": severity_floor(fault.fault_type, part_type),
    }


def _known_tokens(netlist: Netlist) -> set[str]:
    """Collect every ref-like word the netlist itself contains, including chip names and pin functions."""
    texts: list[str] = [*component_refs(netlist), *net_names(netlist)]
    for component in netlist.components:
        texts.extend([component.value, component.package or "", *component.pins.values()])
    # A set comprehension with two for clauses reads like nested loops: for each text, for each match.
    return {token for text in texts for token in _REF_LIKE.findall(text)}


def check_card(card: Consequence, ref: str, netlist: Netlist) -> str | None:
    """Return what is wrong with a card, or None. Every cited part, pad and net must really exist."""
    if not re.match(rf"{re.escape(ref)}\b", card.headline):
        return f'The headline must start with the reference "{ref}".'

    prose = " ".join([card.headline, card.consequence, card.fix, card.check_first, *card.facts_used])
    unknown_refs = sorted(set(_REF_LIKE.findall(prose)) - _known_tokens(netlist))
    if unknown_refs:
        return f"These references are not on this board: {', '.join(unknown_refs)}. Cite only parts in the facts given."

    refs = component_refs(netlist)
    known_nodes = {node for net in netlist.nets for node in net.nodes}
    bad_nodes = sorted(
        f"{part}.{pad}"
        for part, pad in _NODE_LIKE.findall(prose)
        if part in refs and f"{part}.{pad}" not in known_nodes
    )
    if bad_nodes:
        return f"These pads are not on any net: {', '.join(bad_nodes)}. Cite only pads in the facts given."

    bad_nets = sorted(set(_NET_LIKE.findall(prose)) - net_names(netlist))
    if bad_nets:
        return f"These nets do not exist: {', '.join(bad_nets)}. Cite only nets in the facts given."
    return None


def apply_floor(card: Consequence, floor: str) -> Consequence:
    """Hold a card to the rule floor, and make its power advice agree with its severity."""
    severity = card.severity if SEVERITY_RANK[card.severity] >= SEVERITY_RANK[floor] else floor
    safe = card.safe_to_power and severity != "CRITICAL"
    # model_copy returns a changed copy, because this code treats model output as immutable.
    return card.model_copy(update={"severity": severity, "safe_to_power": safe})


def template_card(ref: str | None, fault_type: FaultType, floor: str, netlist: Netlist | None, reason: str) -> Consequence:
    """Write a plain card from the facts alone, when no netlist exists or the model's replies failed."""
    name = ref or "Unknown part"
    words = fault_type.lower().replace("_", " ")
    nets = sorted(set(pad_to_net(netlist, ref).values())) if netlist and ref else []
    reach = f" It connects to nets {', '.join(nets)}." if nets else ""
    fixes: dict[str, str] = {
        "MISSING": f"Fit {name} in its marked position.",
        "REVERSED": f"Remove {name} and fit it the other way round.",
        "ROTATED": f"Remove {name} and fit it in the orientation shown in the golden photo.",
        "SHIFTED": f"Reseat {name} so it sits flat in its marked position.",
        "WRONG_PART": f"Compare {name} with the parts list and replace it if it differs.",
        "EXTRA": "Remove the extra part, and check that nothing is bridging the pads under it.",
    }
    return Consequence(
        headline=f"{name} {words}: check before powering",
        consequence=f"{name} does not match the golden board ({words}).{reach}",
        severity=floor,
        safe_to_power=floor != "CRITICAL",
        fix=fixes[fault_type],
        check_first=f"Inspect {name} and the pads around it by eye before applying power.",
        facts_used=[],
        uncertain=reason,
    )


def write_card(client: LLMClient, fault: Fault, netlist: Netlist | None, cfg: Mapping[str, Any]) -> Consequence:
    """Return the card for one fault: from Prompt D when possible, otherwise from the template."""
    component = find_component(netlist, fault.ref) if netlist and fault.ref else None
    floor = severity_floor(fault.fault_type, component.type if component else "other")

    if fault.ref is None:
        # An unknown object could bridge pads, and no netlist fact can rule that out.
        return template_card(None, fault.fault_type, "MAJOR", netlist, "This part is not in the golden board, so its effect is unknown.")
    if netlist is None or component is None:
        return template_card(fault.ref, fault.fault_type, floor, netlist, "This board has no netlist entry for the part, so its effect is unknown.")

    prompt = load_prompt("consequence", cfg["llm"]["prompt_versions"]["consequence"])
    confidence = float(fault.verdict.confidence) if fault.verdict else 0.0
    card = client.complete_json(
        Consequence,
        prompt,
        prompt_values(fault, netlist, confidence),
        check=lambda reply: check_card(reply, fault.ref or "", netlist),
    )
    if card is None:
        card = template_card(fault.ref, fault.fault_type, floor, netlist, "The model's replies failed validation, so this card uses only the netlist.")
    return apply_floor(card, floor)


def add_consequences(client: LLMClient, faults: Sequence[Fault], netlist: Netlist | None, cfg: Mapping[str, Any]) -> list[Fault]:
    """Write a card for every fault, confirmed or needing review, keeping their order."""

    def with_card(fault: Fault) -> Fault:
        return fault.model_copy(update={"consequence": write_card(client, fault, netlist, cfg)})

    with ThreadPoolExecutor(max_workers=cfg["verdict"]["max_workers"]) as pool:
        return list(pool.map(with_card, faults))
