"""The pin-swap table and netlist facts must be exact, because consequence cards cite them."""

from pathlib import Path

import pytest

from pocket_aoi.netlist import (
    classify_part,
    load_netlist,
    neighbour_table,
    netlist_from_eagle,
    netlist_from_kicad_xml,
    normalise_package,
    pad_to_net,
    pin_net_table,
    pin_swap,
    pin_swap_table,
)
from pocket_aoi.schemas import Component, Net, Netlist, Supply


def _ne555_board() -> Netlist:
    """The 555 flasher from AGENTS.md section 4, with RESET tied to VCC."""
    return Netlist(
        board_id="ne555-flasher",
        description="555 astable LED flasher",
        supply=Supply(net="VCC", volts=9),
        components=[
            Component(
                ref="U1", value="NE555", type="ic", package="DIP-8",
                pins={"1": "GND", "2": "TRIG", "3": "OUT", "4": "RESET", "5": "CTRL", "6": "THR", "7": "DIS", "8": "VCC"},
            ),
            Component(ref="C1", value="100uF", type="electrolytic_capacitor", package="radial", pins={"+": "+", "-": "-"}),
            Component(ref="Q1", value="BC547", type="transistor", package="TO-92", pins={"1": "E", "2": "B", "3": "C"}),
        ],
        nets=[
            Net(name="VCC", nodes=["U1.8", "U1.4", "C1.+"]),
            Net(name="GND", nodes=["U1.1", "C1.-", "Q1.1"]),
            Net(name="N_CTRL", nodes=["U1.5"]),
            Net(name="OUT", nodes=["U1.3", "Q1.2"]),
            Net(name="LOAD", nodes=["Q1.3"]),
        ],
    )


def test_dip8_reversal_maps_pins_as_the_spec_says() -> None:
    mapping = {row.pin: row.pad for row in pin_swap(_ne555_board(), "U1")}
    assert (mapping["1"], mapping["4"], mapping["5"], mapping["8"]) == ("5", "8", "1", "4")
    assert sorted(mapping.values(), key=int) == [str(k) for k in range(1, 9)]  # Every pad used exactly once.


def test_dip_reversal_is_its_own_inverse() -> None:
    mapping = {row.pin: row.pad for row in pin_swap(_ne555_board(), "U1")}
    assert all(mapping[mapping[pin]] == pin for pin in mapping)


def test_to92_reversal_swaps_pins_1_and_3_and_keeps_2() -> None:
    mapping = {row.pin: row.pad for row in pin_swap(_ne555_board(), "Q1")}
    assert mapping == {"1": "3", "2": "2", "3": "1"}


def test_two_pin_part_swaps_its_nets() -> None:
    rows = {row.pin: row for row in pin_swap(_ne555_board(), "C1")}
    assert rows["+"].pad == "-" and rows["+"].net == "GND" and rows["+"].own_net == "VCC"
    assert rows["-"].pad == "+" and rows["-"].net == "VCC" and rows["-"].own_net == "GND"


def test_pin_swap_table_matches_the_spec_example() -> None:
    table = pin_swap_table(_ne555_board(), "U1")
    assert table.splitlines()[0] == "U1 NE555 reversed (DIP-8). Chip pin -> pad it now sits on -> net of that pad"
    assert "pin 1 GND   -> pad 5 -> N_CTRL" in table
    assert "pin 4 RESET -> pad 8 -> VCC" in table
    assert "pin 5 CTRL  -> pad 1 -> GND" in table
    assert "pin 8 VCC   -> pad 4 -> VCC" in table


def test_unknown_part_has_no_swap_table() -> None:
    assert pin_swap_table(_ne555_board(), "Z9") == "none"


def test_pad_to_net_lists_only_this_part() -> None:
    assert pad_to_net(_ne555_board(), "C1") == {"+": "VCC", "-": "GND"}


def test_pin_net_table_names_function_and_net() -> None:
    table = pin_net_table(_ne555_board(), "U1")
    assert "U1.8 (VCC) on VCC" in table
    assert "U1.2 (TRIG) on not connected" in table


def test_neighbour_table_lists_the_other_parts_on_each_net() -> None:
    table = neighbour_table(_ne555_board(), "C1")
    assert "VCC: U1.8 (NE555, VCC), U1.4 (NE555, RESET)" in table
    assert "GND: U1.1 (NE555, GND), Q1.1 (BC547, E)" in table


def test_ref_with_shared_prefix_is_not_confused() -> None:
    board = _ne555_board()
    board.nets.append(Net(name="X", nodes=["C10.1", "C1.+"]))
    assert "1" not in pad_to_net(board, "C1")


@pytest.mark.parametrize(
    ("ref", "package", "expected"),
    [
        ("R1", "AXIAL-0.4", "resistor"),
        ("R10", "TRIMPOT-3319P-1", "other"),
        ("C3", "CPOL-RADIAL-10UF-25V", "electrolytic_capacitor"),
        ("C1", "CAP-PTH-SMALL", "ceramic_capacitor"),
        ("LED1", "LED5MM", "led"),
        ("U1", "DIL28-3", "ic"),
        ("Q1", "TO-92", "transistor"),
        ("D1", "DO-41", "diode"),
        ("S1", "TACTILE", "other"),
    ],
)
def test_classify_part_from_ref_and_package(ref: str, package: str, expected: str) -> None:
    assert classify_part(ref, "", package) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("DIL28-3", "DIP-28"), ("DIP08", "DIP-8"), ("TO-92", "TO-92"), ("CPOL-RADIAL-10UF-25V", "radial"), ("AXIAL-0.4", "axial"), (None, None)],
)
def test_package_names_normalise(raw: str | None, expected: str | None) -> None:
    assert normalise_package(raw) == expected


def test_netlist_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "netlist.json"
    path.write_text(_ne555_board().model_dump_json(), encoding="utf-8")
    assert load_netlist(path) == _ne555_board()


def test_kicad_xml_parser(tmp_path: Path) -> None:
    path = tmp_path / "netlist.xml"
    path.write_text(
        """<export version="E"><components>
        <comp ref="U1"><value>NE555</value><footprint>Package_DIP:DIP-8_W7.62mm</footprint></comp>
        <comp ref="C1"><value>100uF</value><footprint>Capacitor_THT:CP_Radial_D5.0mm</footprint></comp>
        </components><nets>
        <net code="1" name="VCC"><node ref="U1" pin="8" pinfunction="VCC"/><node ref="C1" pin="1"/></net>
        <net code="2" name="GND"><node ref="U1" pin="1" pinfunction="GND"/><node ref="C1" pin="2"/></net>
        </nets></export>""",
        encoding="utf-8",
    )
    netlist = netlist_from_kicad_xml(path, "demo", "demo", "VCC", 9)
    assert netlist.components[0].package == "DIP-8"
    assert netlist.components[0].pins["8"] == "VCC"
    assert netlist.components[1].type == "ceramic_capacitor" or netlist.components[1].package == "radial"
    assert netlist.nets[0].nodes == ["U1.8", "C1.1"]


def test_eagle_parser_takes_nets_from_the_board_and_pin_names_from_the_schematic(tmp_path: Path) -> None:
    schematic = tmp_path / "board.sch"
    board = tmp_path / "board.brd"
    schematic.write_text(
        """<eagle><drawing><schematic>
        <libraries><library name="lib"><devicesets><deviceset name="LM386"><devices>
          <device name="" package="DIP08"><connects>
            <connect gate="G$1" pin="V+" pad="6"/><connect gate="G$1" pin="GAIN@1" pad="1"/>
          </connects></device></devices></deviceset></devicesets></library></libraries>
        <parts><part name="U2" library="lib" deviceset="LM386" device=""/></parts>
        </schematic></drawing></eagle>""",
        encoding="utf-8",
    )
    board.write_text(
        """<eagle><drawing><board>
        <elements>
          <element name="U2" library="lib" package="DIP08" value="LM386" x="1" y="2"/>
          <element name="U$1" library="lib" package="OSH-LOGO" value="" x="1" y="2"/>
        </elements>
        <signals><signal name="VCC"><contactref element="U2" pad="6"/></signal>
                 <signal name="EMPTY"></signal></signals>
        </board></drawing></eagle>""",
        encoding="utf-8",
    )
    netlist = netlist_from_eagle(schematic, board, "demo", "demo board", "VCC", 4.5)
    assert [component.ref for component in netlist.components] == ["U2"]  # The logo is not a part.
    assert netlist.components[0].package == "DIP-8"
    assert netlist.components[0].pins == {"6": "V+", "1": "GAIN"}
    assert [net.name for net in netlist.nets] == ["VCC"]
    assert netlist.supply.volts == 4.5
