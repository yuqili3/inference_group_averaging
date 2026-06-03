#!/usr/bin/env python
"""Unified runner for Q1/Q2/Q3 experiments.

Usage:
    python scripts/run_experiment.py --config configs/q2_orbit_averaging.yaml
    python scripts/run_experiment.py --config configs/q1_equivariance.yaml
    python scripts/run_experiment.py --config configs/q3_degradation.yaml
    python scripts/run_experiment.py --config configs/smoke.yaml

The experiment type is read from the config's ``experiment`` field (q1|q2|q3).
Paths/datasets/device come from configs/base.yaml.
"""
import argparse
import os
import sys

# allow running from a source checkout without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, build_denoiser, dataset_path  # noqa: E402
from groupavg.experiments import q1_equivariance, q2_orbit_averaging, q3_degradation  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--max-images", type=int, default=None)
    args = ap.parse_args()

    base = load_config(args.base)
    exp = load_config(args.config)
    save_dir = args.save_dir or base.get("save_dir", "results")
    max_images = args.max_images if args.max_images is not None else exp.get("max_images")

    kind = exp.get("experiment", "q2")
    denoiser, dname = build_denoiser(exp["denoiser"], base)

    if kind == "q2":
        g = exp["group"]
        out = q2_orbit_averaging.run(
            dataset_dir=dataset_path(base, exp["dataset"]),
            denoiser=denoiser, denoiser_name=dname,
            group_name=g["name"], averaging=g["averaging"], expand=g.get("expand", True),
            upsample=exp.get("upsample", 1.0), noise_sigma=exp.get("noise_sigma", 15.0),
            num_noise=exp.get("num_noise", 8), noise_mask=exp.get("noise_mask", "circle"),
            se_mask=exp.get("se_mask", "circle"), max_images=max_images,
            seed=exp.get("seed", 0), save_dir=save_dir,
        )
        print("\n=== Q2 summary ===")
        for k, v in out["summary"].items():
            print(f"  {k}: {v:.6f}")
        s = out["summary"]
        print(f"\n  PSNR improvement (averaged - direct): "
              f"{s['E_x_SEavg_x_psnr'] - s['E_x_EhSE_hx_psnr']:+.3f} dB")
        print(f"  identity check  E[EhSE-SEavg]={s['E_x_EhSE_minus_SEavg']:.6f}  vs  E[e1]={s['E_x_e1']:.6f}")

    elif kind == "q1":
        g = exp["group"]
        out = q1_equivariance.run(
            dataset_dir=dataset_path(base, exp["dataset"]),
            denoiser=denoiser, denoiser_name=dname,
            group_name=g["name"], averaging=g["averaging"], expand=g.get("expand", True),
            upsample=exp.get("upsample", 1.0), noise_sigma=exp.get("noise_sigma", 15.0),
            num_noise=exp.get("num_noise", 4), noise_mask=exp.get("noise_mask", "none"),
            se_mask=exp.get("se_mask", "content"), clean_mode=exp.get("clean_mode", False),
            max_images=max_images, seed=exp.get("seed", 0), save_dir=save_dir,
        )
        pi = out["per_image"]
        print("\n=== Q1 equivariance error ===")
        print(f"  mean rel. equivariance error: {pi['mean_rel_err'].mean():.6e}")
        print(f"  mean e1 (averageable part):   {pi['e1'].mean():.6e}")

    elif kind == "q3":
        datasets = {k: dataset_path(base, k) for k in exp["datasets"]}
        df = q3_degradation.run(
            datasets=datasets, denoiser=denoiser, denoiser_name=dname,
            groups=exp.get("groups", ["fourier_rotation"]),
            averagings=exp.get("averagings", [4, 8, 16]),
            noise_masks=exp.get("noise_masks", ["circle"]),
            noise_sigma=exp.get("noise_sigma", 15.0), num_noise=exp.get("num_noise", 4),
            se_mask=exp.get("se_mask", "content"), upsample=exp.get("upsample", 1.0),
            max_images=max_images, seed=exp.get("seed", 0), save_dir=save_dir,
        )
        print("\n=== Q3 degradation grid ===")
        print(df.to_string(index=False))
    else:
        raise SystemExit(f"Unknown experiment kind: {kind}")


if __name__ == "__main__":
    main()
