#!/usr/bin/env python
"""Run the rotation-operator round-trip quality diagnostic."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.experiments import rotation_operator_quality  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default=os.path.join("results", "rotation_operator_quality"))
    ap.add_argument("--angle-step", type=float, default=5.0)
    ap.add_argument("--noise-sigma", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_steps = int(round(360.0 / args.angle_step))
    angles = [i * args.angle_step for i in range(n_steps + 1)]
    out = rotation_operator_quality.run(
        angles=angles,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    df = out["results"]
    print("\n=== Rotation operator quality ===")
    print(f"  save_dir: {out['save_dir']}")
    print("  figures:")
    for figure in out["figures"]:
        print(f"    {figure}")
    print("\nMean PSNR by signal/operator:")
    summary = df.groupby(["signal", "operator"])["psnr"].mean().reset_index()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
