"""Strip a training state down to the weights needed for inference.

The training state also stores optimizer states; the exported file keeps only the angle
encoder and the UNet LoRA weights (a few MB for rank 4).

  python scripts/export_weights.py --training_state training_state_25000.pth --out weights/clock.pt
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angular_control.pipeline import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    encoder_state, lora_state, _ = load_checkpoint(args.training_state)
    encoder_state = {k: v for k, v in encoder_state.items() if k.startswith("linear")}
    rank = next(v.shape[0] for k, v in lora_state.items() if k.endswith("lora_down.weight"))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder_state, "unet_lora": lora_state, "config": {"lora_rank": rank}}, args.out)
    size = sum(v.numel() for v in list(encoder_state.values()) + list(lora_state.values()))
    print(f"wrote {args.out}: {size / 1e6:.2f}M parameters, LoRA rank {rank}")


if __name__ == "__main__":
    main()
