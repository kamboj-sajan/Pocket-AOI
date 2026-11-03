# Pocket AOI

From one phone photo, find assembly mistakes on a through-hole circuit board by comparing it with a known-good golden board, and say what each mistake will break.

Built for the Meta Global AI Developer Hackathon (manufacturing). The full spec is in [AGENTS.md](AGENTS.md).

**Status:** Block 1, segment and align. Both models run, and photos align into one canonical frame. The fault comparison is not built yet, so there are no inspection results to report.

## Models

| Role | Model |
| --- | --- |
| Board outline and part masks | SAM 3.1 by Meta, loaded from [unsloth/sam3.1](https://huggingface.co/unsloth/sam3.1), a transformers-format conversion of Meta's weights under the SAM License. It is not an official Meta release. |
| Crop verdicts and consequence text | [Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) by Meta, Apache 2.0 |

## Setup

Needs Python 3.11 or newer. These commands use [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
cp .env.example .env    # then set LLM_API_KEY
```

## Smoke tests

```bash
.venv/bin/pytest                                                  # no network or GPU needed
.venv/bin/pocket-aoi smoke-llm                                    # generated test board to Muse Glimmer
.venv/bin/pocket-aoi smoke-llm --image photo.jpg                  # your own photo
.venv/bin/pocket-aoi smoke-sam --image photo.jpg                  # finds "printed circuit board"
.venv/bin/pocket-aoi smoke-sam --image photo.jpg --prompt resistor
```

`smoke-sam` saves its overlay to `runs/smoke/sam_overlay.png`.

## Calibrate and align

```bash
.venv/bin/pocket-aoi spike                                        # every data/raw/<kit>/golden.jpg
.venv/bin/pocket-aoi align --golden golden.jpg --photo test.jpg
```

`spike` reads prompts, hand counts and labelled part positions from `boards/<kit>/board.yaml`, and saves overlays to `runs/spike/<kit>/`. `align` saves the aligned photo and a checkerboard blend to `runs/align/<photo name>/`, and exits non-zero with a retake message if the photo fails the quality gate. Misalignment shows as broken lines at the checkerboard's seams.

## Calibration so far

These numbers tune the pipeline on web photos. They are not inspection results, and no defect detection has been measured yet.

SAM 3.1 on two golden photos, against part positions labelled by hand from each board's silkscreen:

| Kit | Parts found by "electronic component" | Right label from class prompts |
| --- | --- | --- |
| ArduTouch | 25 of 25 | 25 of 25, with prompts chosen for this board |
| Electronic die | 21 of 21 | 15 of 21 |

Alignment on real photo pairs of one board:

| Pair | SIFT inliers | Median error | Result |
| --- | --- | --- | --- |
| Die, steep angle against top-down | 1295 | 0.67 px | Aligned |
| VisA `pcb1`, turned 180° | 68 | 1.07 px | Aligned |
| VisA `pcb2`, turned 180° | 256 | 1.35 px | Aligned |
| VisA `pcb3` | 107 | 1.43 px | Aligned |
| VisA `pcb4` | 49, under the 50 needed | 1.38 px | Aligned on the board outline alone |
| ArduTouch, battery pack covering the board | 13 | 1.05 px | Retake message, correctly |

Two photos pass that should not, which is still open: one with bubble wrap over the board, and the steep die photo, where tall parts lean a long way.

### Photos used

None of these are stored in this repo.

- ArduTouch kit by [Mitch Altman](https://commons.wikimedia.org/wiki/File:ArduTouch_music_synthesizer_Rev_C_Assembled,_by_Cornfield_Electronics_(2023-03-19_12.20.37_by_Mitch_Altman_@Flickr_52930161269).jpg), CC BY-SA 2.0
- Electronic die kit by Raimond Spekking ([top-down](https://commons.wikimedia.org/wiki/File:Assembled_Kit_for_electronic_cube-2461.jpg), [angled](https://commons.wikimedia.org/wiki/File:Assembled_Kit_for_electronic_cube-2460.jpg)), CC BY-SA 4.0
- [VisA dataset](https://github.com/amazon-science/spot-diff) by Zou et al., CC BY 4.0

## Taking photos

Hold the phone 25–35 cm above and roughly parallel to the board, with the board filling most of the frame. Use diffuse light with no flash, on a plain background that contrasts with the board. Save photos as JPEG; on an iPhone, set Settings > Camera > Formats to Most Compatible.
