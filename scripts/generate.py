"""Generate clocks at given times or circular fractions at given values.

Examples
  python scripts/generate.py --task clock --checkpoint weights/clock.pt \
      --prompt "PLACEHOLDER, a black analog clock" --targets 03:37 07:15 10:10 --out outputs/clock

  python scripts/generate.py --task fraction --checkpoint weights/fraction.pt \
      --prompt "a top-down photo of a round pepperoni pizza with PLACEHOLDER on a clean white ceramic plate" \
      --targets 0.25 1/3 0.75 --out outputs/fraction

PLACEHOLDER marks where the geometric tokens go. By default image i uses seed + i, as in the
paper's evaluation; pass --same_seed to use the same initial noise for every target.
"""
import argparse
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from angular_control import AngularControlPipeline, GenerationConfig  # noqa: E402
from angular_control.pipeline import DEFAULT_BASE_MODEL  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["clock", "fraction"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True, help="prompt template containing PLACEHOLDER")
    ap.add_argument("--targets", nargs="+", required=True,
                    help="clock: HH:MM[:SS]; fraction: value in [0, 1] or a ratio like 1/3")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--same_seed", action="store_true")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--local_files_only", action="store_true")
    args = ap.parse_args()

    pipe = AngularControlPipeline.from_pretrained(args.checkpoint, args.task, base_model=args.base_model,
                                                  local_files_only=args.local_files_only)
    config = GenerationConfig(args.steps, args.guidance_scale, args.negative_prompt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, target in enumerate(args.targets):
        value = target if args.task == "clock" else float(Fraction(target))
        seed = args.seed if args.same_seed else args.seed + i
        image = pipe(args.prompt, value, seed=seed, config=config)
        name = target.replace(":", "-").replace("/", "of")
        image.save(out / f"{args.task}_{name}_seed{seed}.png")
        print(f"{target} -> {out / f'{args.task}_{name}_seed{seed}.png'}")


if __name__ == "__main__":
    main()
