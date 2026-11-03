"""Netlists: parse KiCad and Eagle exports, list the facts about a part, and build the pin-swap table.

Everything here is exact, because the consequence card may only cite what these functions return.
"""

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pocket_aoi.schemas import Component, Net, Netlist, Supply


def load_netlist(path: Path) -> Netlist:
    """Read a netlist.json file."""
    return Netlist.model_validate_json(path.read_text(encoding="utf-8"))


def classify_part(ref: str, value: str, package: str | None) -> str:
    """Guess a part's class from its reference designator and package name."""
    # re.match anchors at the start of the string; group(1) is the bracketed part of the pattern.
    match = re.match(r"([A-Za-z]+)", ref)
    prefix = match.group(1).upper() if match else ""
    package_name = (package or "").upper()
    if prefix == "LED":
        return "led"
    if prefix in ("U", "IC"):
        return "ic"
    if prefix == "Q":
        return "transistor"
    if prefix == "D":
        return "diode"
    if prefix == "R":
        return "other" if "TRIM" in package_name or "POT" in package_name else "resistor"
    if prefix == "C":
        return "electrolytic_capacitor" if "POL" in package_name or "ELEC" in package_name else "ceramic_capacitor"
    return "other"


def normalise_package(package: str | None) -> str | None:
    """Turn library names such as DIL28-3 and DIP08 into DIP-28 and DIP-8, and name radial parts."""
    if not package:
        return package
    upper = package.upper()
    # (?:...) groups without capturing, so group(1) is still the pin count.
    dip = re.match(r"(?:DIL|DIP)-?0*(\d+)", upper)
    if dip:
        return f"DIP-{int(dip.group(1))}"
    if "TO-92" in upper or "TO92" in upper:
        return "TO-92"
    if "RADIAL" in upper or "CPOL" in upper:
        return "radial"
    if "AXIAL" in upper:
        return "axial"
    return package


def netlist_from_eagle(
    schematic: Path,
    board: Path,
    board_id: str,
    description: str,
    supply_net: str,
    supply_volts: float,
) -> Netlist:
    """Build a netlist from an Eagle .sch and .brd pair: nets from the board, pin names from the schematic."""
    schematic_root = ET.parse(schematic).getroot()
    board_root = ET.parse(board).getroot()

    # A dict comprehension builds a lookup table, like filling a std::map in a loop.
    libraries = {library.get("name"): library for library in schematic_root.findall(".//libraries/library")}
    pin_names: dict[str, dict[str, str]] = {}
    for part in schematic_root.findall(".//parts/part"):
        library = libraries.get(part.get("library"))
        if library is None:
            continue
        for device_set in library.findall(".//deviceset"):
            if device_set.get("name") != part.get("deviceset"):
                continue
            for device in device_set.findall(".//device"):
                if (device.get("name") or "") != (part.get("device") or ""):
                    continue
                pads: dict[str, str] = {}
                for connect in device.findall(".//connect"):
                    # Eagle marks repeated pins as GND@1, and one pin can span several pads.
                    pin = (connect.get("pin") or "").split("@")[0]
                    for pad in (connect.get("pad") or "").split():
                        pads[pad] = pin
                pin_names[part.get("name") or ""] = pads

    components = []
    for element in board_root.findall(".//elements/element"):
        ref = element.get("name") or ""
        if ref.startswith("U$"):  # Logos placed on the board, not electronic parts.
            continue
        package = normalise_package(element.get("package"))
        value = element.get("value") or ""
        components.append(
            Component(
                ref=ref,
                value=value,
                type=classify_part(ref, value, element.get("package")),
                package=package,
                pins=pin_names.get(ref, {}),
            )
        )

    nets = [
        Net(
            name=signal.get("name") or "",
            nodes=[f"{contact.get('element')}.{contact.get('pad')}" for contact in signal.findall("contactref")],
        )
        for signal in board_root.findall(".//signals/signal")
    ]
    nets = [net for net in nets if net.nodes]
    return Netlist(
        board_id=board_id,
        description=description,
        supply=Supply(net=supply_net, volts=supply_volts),
        components=components,
        nets=nets,
    )


def netlist_from_kicad_xml(path: Path, board_id: str, description: str, supply_net: str, supply_volts: float) -> Netlist:
    """Build a netlist from `kicad-cli sch export netlist --format kicadxml`."""
    root = ET.parse(path).getroot()
    pin_names: dict[str, dict[str, str]] = {}
    for net in root.findall(".//nets/net"):
        for node in net.findall("node"):
            function = node.get("pinfunction") or node.get("pin") or ""
            pin_names.setdefault(node.get("ref") or "", {})[node.get("pin") or ""] = function

    components = []
    for comp in root.findall(".//components/comp"):
        ref = comp.get("ref") or ""
        value = comp.findtext("value") or ""
        footprint = comp.findtext("footprint") or ""
        # KiCad footprints read "Library:Name", and only the name describes the package.
        package = normalise_package(footprint.split(":")[-1]) if footprint else None
        components.append(
            Component(
                ref=ref,
                value=value,
                type=classify_part(ref, value, footprint),
                package=package,
                pins=pin_names.get(ref, {}),
            )
        )

    nets = [
        Net(
            name=net.get("name") or "",
            nodes=[f"{node.get('ref')}.{node.get('pin')}" for node in net.findall("node")],
        )
        for net in root.findall(".//nets/net")
    ]
    return Netlist(
        board_id=board_id,
        description=description,
        supply=Supply(net=supply_net, volts=supply_volts),
        components=components,
        nets=nets,
    )


def find_component(netlist: Netlist, ref: str) -> Component | None:
    """Return the component with this reference, or None."""
    # next() takes the first item from a generator, or the default when there is none.
    return next((component for component in netlist.components if component.ref == ref), None)


def pad_to_net(netlist: Netlist, ref: str) -> dict[str, str]:
    """Return each connected pad of a part and the net it sits on."""
    prefix = f"{ref}."
    pads: dict[str, str] = {}
    for net in netlist.nets:
        for node in net.nodes:
            if node.startswith(prefix):
                pads[node[len(prefix) :]] = net.name
    return pads


def net_names(netlist: Netlist) -> set[str]:
    return {net.name for net in netlist.nets}


def component_refs(netlist: Netlist) -> set[str]:
    return {component.ref for component in netlist.components}


def _reversed_pad_map(component: Component, pads: list[str]) -> dict[str, str] | None:
    """Return which pad each chip pin sits on after a 180 degree turn, or None if the rule is unknown."""
    package = component.package or ""
    dip = re.fullmatch(r"DIP-(\d+)", package)
    if dip:
        count = int(dip.group(1))
        # Pins number around the package, so half a turn moves pin k on by half the pin count.
        return {str(k): str(((k - 1 + count // 2) % count) + 1) for k in range(1, count + 1)}
    if package == "TO-92":
        return {"1": "3", "2": "2", "3": "1"}
    if len(pads) == 2:
        return {pads[0]: pads[1], pads[1]: pads[0]}
    return None


@dataclass(frozen=True)
class PinSwap:
    """One chip pin after a reversal: the pad it now sits on, and that pad's net."""

    pin: str
    function: str
    pad: str
    net: str
    own_net: str


def pin_swap(netlist: Netlist, ref: str) -> list[PinSwap]:
    """Return the exact pin-swap table for a part turned 180 degrees."""
    component = find_component(netlist, ref)
    if component is None:
        return []
    nets = pad_to_net(netlist, ref)
    pads = sorted(component.pins or nets, key=_pad_sort_key)
    mapping = _reversed_pad_map(component, pads)
    if mapping is None:
        return []
    return [
        PinSwap(
            pin=pin,
            function=component.pins.get(pin, pin),
            pad=pad,
            net=nets.get(pad, "not connected"),
            own_net=nets.get(pin, "not connected"),
        )
        for pin, pad in mapping.items()
    ]


def _pad_sort_key(pad: str) -> tuple[int, str]:
    """Sort numbered pads numerically and named pads by name, with numbers first."""
    return (int(pad), "") if pad.isdigit() else (10**6, pad)


def pin_swap_table(netlist: Netlist, ref: str) -> str:
    """Format the pin-swap table as Prompt D expects it, one line per chip pin."""
    component = find_component(netlist, ref)
    rows = pin_swap(netlist, ref)
    if component is None or not rows:
        return "none"
    width = max(len(row.function) for row in rows)
    header = f"{ref} {component.value} reversed ({component.package}). Chip pin -> pad it now sits on -> net of that pad"
    lines = [f"pin {row.pin} {row.function:<{width}} -> pad {row.pad} -> {row.net}" for row in rows]
    return "\n".join([header, *lines])


def pin_net_table(netlist: Netlist, ref: str) -> str:
    """List a part's pads, their functions and their nets for Prompt D."""
    component = find_component(netlist, ref)
    if component is None:
        return "unknown"
    nets = pad_to_net(netlist, ref)
    pads = sorted(set(component.pins) | set(nets), key=_pad_sort_key)
    return "; ".join(f"{ref}.{pad} ({component.pins.get(pad, pad)}) on {nets.get(pad, 'not connected')}" for pad in pads)


def neighbour_table(netlist: Netlist, ref: str, max_per_net: int = 12) -> str:
    """List the other parts on each of this part's nets, with their values and pin functions."""
    by_ref: Mapping[str, Component] = {component.ref: component for component in netlist.components}
    lines = []
    for net in netlist.nets:
        if not any(node.startswith(f"{ref}.") for node in net.nodes):
            continue
        others = []
        for node in net.nodes:
            other_ref, _, pad = node.partition(".")
            if other_ref == ref:
                continue
            other = by_ref.get(other_ref)
            value = other.value if other else ""
            function = other.pins.get(pad, pad) if other else pad
            others.append(f"{other_ref}.{pad} ({value or 'no value'}, {function})")
        shown = others[:max_per_net]
        suffix = f" and {len(others) - len(shown)} more" if len(others) > len(shown) else ""
        lines.append(f"{net.name}: {', '.join(shown) if shown else 'nothing else'}{suffix}")
    return "\n".join(lines) if lines else "none"
