# Pocket AOI

From one phone photo, find assembly mistakes on a circuit board by comparing it with a known-good golden board, and say what each mistake will break.

Built for the Meta Global AI Developer Hackathon (manufacturing). The full spec is in [AGENTS.md](AGENTS.md).

## The problem

In student labs, assembly mistakes cost more debugging time than design errors: a reversed electrolytic, an IC seated backwards, a missing resistor. Factory optical inspection machines cost tens of thousands of dollars, so students, makers and small shops inspect by eye.

Pocket AOI needs a phone, one photo of a correctly assembled board, and optionally the board's netlist. It returns:

- the test photo with each fault outlined
- a safe-to-power banner
- one card per fault, for example "C7 missing: U2 loses its decoupling cap"

## How it works

Every decision starts from the golden board. Each golden part is checked at its own spot in the aligned test photo, and Muse Glimmer only confirms or rejects what classical vision raises. It can never add a fault of its own.

| Stage | What happens | Uses |
| --- | --- | --- |
| 0. Board setup | Find the parts on the golden photo, give each its ref, and save a profile. Once per design. | SAM 3.1, a person checking refs, optionally Muse Glimmer reading the silkscreen |
| 1. Align | Warp the test photo into the golden board's frame, or ask for a retake and say why | SAM 3.1, SIFT |
| 2. Segment | Find where every part is, with one prompt on the full image and overlapping tiles | SAM 3.1 |
| 3. Compare | Match parts by position, compare each crop at a coarse scale, and test polarized parts for rotation. Tuned to miss nothing, so false alarms are expected here. | OpenCV, scikit-image |
| 4. Verdicts | Show Muse Glimmer the golden and test crops side by side, and keep, drop or flag each suspect | Muse Glimmer 30B |
| 5. Consequences | Write a card from netlist facts and an exact pin-swap table, with severity floors set by rules. Every cited part, pad and net is checked against the netlist. | Muse Glimmer 30B |
| 6. Report | Draw the overlay on the original photo, and write `report.html`, `report.json` and `overlay.png` | OpenCV |

## Models

| Role | Model |
| --- | --- |
| Board outline and part masks | SAM 3.1 by Meta, loaded from [unsloth/sam3.1](https://huggingface.co/unsloth/sam3.1), a transformers-format conversion of Meta's weights under the SAM License. It is not an official Meta release. |
| Crop verdicts, silkscreen reading and consequence text | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) by Meta, Apache 2.0 |

Nothing is trained or fine-tuned. Muse Glimmer sits behind one OpenAI-compatible client, so switching backends only changes `.env`.

## Results

<!-- RESULTS -->

## Limitations

- **Solder joints** are not inspected, because they sit on the underside of a through-hole board.
- **Wrong values in identical bodies**, such as 1 kΩ against 10 kΩ, cannot be told apart.
- **Fine-pitch SMD boards** are out of scope.
- **Anything outside the board's outline**, such as a bent header pin hanging off the edge, is not seen, because the comparison frame is cropped to the board.
- **Reversed parts have not been measured on real photos.** The public dataset used here contains none, so that path has only run on synthetic defects.

## Setup

Needs Python 3.11 or newer. These commands use [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env    # then set LLM_API_KEY
.venv/bin/pytest        # no network or GPU needed
```

SAM 3.1 runs on an NVIDIA GPU or on Apple silicon through MPS. On an 8 GB M2 it takes about 25 s per photo. Muse Glimmer can run on a hosted backend, which the default `.env.example` uses, or on your own server.

## Using it

```bash
# Once per design: find the parts, then edit boards/<id>/refs.yaml and rebuild without running SAM again
.venv/bin/pocket-aoi setup-board --board-id ardutouch --golden golden.jpg --netlist netlist.json
.venv/bin/pocket-aoi relabel --board-id ardutouch

# Inspect one photo. --cv-only stops after Stage 3, without calling Muse Glimmer
.venv/bin/pocket-aoi inspect --board-id ardutouch --photo test.jpg

# Or use the web page, with --share for a public link
.venv/bin/python app/gradio_app.py
```

`inspect` writes to `runs/<board>_<photo>/`: `report.html`, `report.json`, `overlay.png`, `votes.json` with both votes per suspect, the composites Muse Glimmer saw, and debug images for each stage.

To score the pipeline against ground truth, with both CV only and CV plus Muse Glimmer from one run:

```bash
.venv/bin/pocket-aoi eval --gt eval/ground_truth/ --split test --tag final    # results in runs/eval/final/
```

## Taking photos

Hold the phone 25–35 cm above and roughly parallel to the board, with the board filling most of the frame. Use diffuse light with no flash, on a plain background that contrasts with the board. Save photos as JPEG; on an iPhone, set Settings > Camera > Formats to Most Compatible.

## Data and credits

None of these photos are stored in this repo.

- [VisA dataset](https://github.com/amazon-science/spot-diff) by Zou et al., CC BY 4.0: real photos of real defects, used for the results above
- ArduTouch kit by [Mitch Altman](https://commons.wikimedia.org/wiki/File:ArduTouch_music_synthesizer_Rev_C_Assembled,_by_Cornfield_Electronics_(2023-03-19_12.20.37_by_Mitch_Altman_@Flickr_52930161269).jpg), CC BY-SA 2.0, with its [design files](https://github.com/maltman23/ArduTouch) as the source of the demo board's netlist
- Electronic die kit by Raimond Spekking ([top-down](https://commons.wikimedia.org/wiki/File:Assembled_Kit_for_electronic_cube-2461.jpg), [angled](https://commons.wikimedia.org/wiki/File:Assembled_Kit_for_electronic_cube-2460.jpg)), CC BY-SA 4.0
