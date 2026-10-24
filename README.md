# Continuous Angular Control in Text-to-Image Generation via Geometric Token Encoding

Inference code for our WACV 2027 paper. A target angular quantity — an analog clock time or a
circular fraction — is encoded with a periodic (sin/cos) encoding, mapped by a small network to
geometric tokens, and inserted into the prompt of a LoRA-adapted Stable Diffusion 2.1. No bounding
boxes are needed at inference.

This repository contains only what is needed to generate images with the trained models.

## Setup

```bash
pip install -r requirements.txt
```

The base model (`sd2-community/stable-diffusion-2-1`, a mirror of Stable Diffusion 2.1) is
downloaded from the Hugging Face Hub on first use. Pass `--base_model <path>` to use a local copy.

## Weights

Place the trained weights in `weights/`:

| File | Model |
|---|---|
| `weights/clock.pt` | clock time (hour and minute hand tokens) |
| `weights/fraction.pt` | circular fraction (start and end boundary tokens) |

<!-- TODO: add download link -->

## Usage

Write `PLACEHOLDER` in the prompt where the geometric tokens should go.

**Clock times** (`HH:MM` or `HH:MM:SS`):

```bash
python scripts/generate.py --task clock --checkpoint weights/clock.pt \
    --prompt "PLACEHOLDER, a black analog clock" \
    --targets 03:37 07:15 10:10 --out outputs/clock
```

**Circular fractions** (a value in `[0, 1]` or a ratio such as `1/3`; the sector starts at
12 o'clock and runs clockwise):

```bash
python scripts/generate.py --task fraction --checkpoint weights/fraction.pt \
    --prompt "a top-down photo of a round pepperoni pizza with PLACEHOLDER on a clean white ceramic plate" \
    --targets 0.25 1/3 0.75 --out outputs/fraction
```

Image *i* uses seed `--seed + i`; add `--same_seed` to reuse the same initial noise for every target.
Defaults match the paper: 50 DDIM steps, guidance scale 7.5, empty negative prompt.

From Python:

```python
from angular_control import AngularControlPipeline

pipe = AngularControlPipeline.from_pretrained("weights/clock.pt", task="clock")
image = pipe("PLACEHOLDER on a warm oak wood analog wall clock, cream dial", "03:37", seed=0)
image.save("clock_0337.png")
```

## Notes

- For fractions, the start angle is fixed at 0, so the periodic code of a full object (`1.0`) equals
  that of an empty sector (`0.0`). Targets very close to either end are not reliably controlled
  (see the limitations section of the paper).
- Hour and minute tokens are produced by the same network; their role comes from their position in
  the prompt. Occasional hour/minute hand swaps are the main failure mode for clocks.

## Citation

```bibtex
@inproceedings{angularcontrol2027,
  title     = {Continuous Angular Control in Text-to-Image Generation via Geometric Token Encoding},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2027}
}
```

## Acknowledgements

The token-injection approach builds on Compass Control (Parihar et al., CVPR 2025). The LoRA layers are adapted from
[cloneofsimo/lora](https://github.com/cloneofsimo/lora) (Apache-2.0); see `THIRD_PARTY_NOTICES.md`.
