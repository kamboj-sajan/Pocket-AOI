"""Data contracts from AGENTS.md section 4. Every stage reads and writes these as JSON."""

from typing import Literal

from pydantic import BaseModel, Field

PartClass = Literal[
    "electrolytic_capacitor", "ceramic_capacitor", "resistor", "diode",
    "led", "ic", "transistor", "other",
]
FaultType = Literal["MISSING", "REVERSED", "ROTATED", "SHIFTED", "WRONG_PART", "EXTRA"]


class Part(BaseModel):
    ref: str | None                     # "C7"; None until confirmed
    cls: PartClass
    bbox: tuple[int, int, int, int]     # x0, y0, x1, y1 in canonical pixels
    centroid: tuple[float, float]
    mask_rle: dict                      # COCO RLE via pycocotools
    score: float
    package: str | None                 # "DIP-8", "TO-92", "radial"
    polarized: bool
    marker_hint: str | None = None      # where the polarity marker sits on the golden part, e.g. "band at the bottom end"


class GoldenProfile(BaseModel):
    board_id: str
    canonical_size: tuple[int, int]
    parts: list[Part]


class Suspect(BaseModel):
    golden_ref: str | None              # None for EXTRA
    cv_finding: FaultType
    bbox: tuple[int, int, int, int]     # the part's own box; its composite shows a wider crop
    evidence: dict[str, float]          # ssim, iou, offset_ratio, best_rotation_deg, rotation_margin
    composite_path: str


class Verdict(BaseModel):               # Prompt A output
    golden_observation: str
    test_observation: str
    verdict: Literal[
        "OK", "MISSING", "REVERSED", "ROTATED", "SHIFTED",
        "WRONG_PART", "EXTRA", "UNSURE",
    ]
    confidence: float = Field(ge=0, le=1)
    evidence: str


class Consequence(BaseModel):           # Prompt D output
    headline: str
    consequence: str
    severity: Literal["CRITICAL", "MAJOR", "MINOR"]
    safe_to_power: bool
    fix: str
    check_first: str
    facts_used: list[str]
    uncertain: str


class Fault(BaseModel):
    ref: str | None
    fault_type: FaultType
    status: Literal["confirmed", "needs_review"]
    suspect: Suspect
    verdict: Verdict | None
    consequence: Consequence | None


class Report(BaseModel):
    board_id: str
    test_image: str
    safe_to_power: bool
    faults: list[Fault]
    timings_s: dict[str, float]
    versions: dict[str, str]            # model ids, prompt versions, config hash


class Supply(BaseModel):
    net: str
    volts: float


class Component(BaseModel):
    ref: str
    value: str = ""
    type: str = "other"
    package: str | None = None
    pins: dict[str, str] = Field(default_factory=dict)  # pad name -> pin function


class Net(BaseModel):
    name: str
    nodes: list[str]                    # "U1.8" is pad 8 of U1


class Netlist(BaseModel):
    board_id: str
    description: str = ""
    supply: Supply
    components: list[Component]
    nets: list[Net]


class GroundTruthDefect(BaseModel):
    ref: str | None = None              # None when only a defect mask is known
    type: FaultType
    mask: str | None = None             # pixel mask of the defect, in photo coordinates
    accept: list[FaultType] = Field(default_factory=list)  # other finding types that also count


class GroundTruth(BaseModel):
    photo: str
    board_id: str
    defects: list[GroundTruthDefect]
    label: str = ""                     # the source dataset's own label, for reporting
    split: str = "test"                 # "dev" photos may tune thresholds, "test" photos may not
