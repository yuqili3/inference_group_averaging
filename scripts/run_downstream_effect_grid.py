#!/usr/bin/env python
"""Run downstream PnP/RED/diffusion-style effect experiments.

This script only defines the grid. It is intentionally not started by default.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import downstream_effect  # noqa: E402


RESTORMER_AUG_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)

MODELS = ["restormer", "restormer-aug", "wavelet", "tv", "nlm"]
SIGMAS = [15.0, 25.0, 50.0]
ALGORITHMS = ["pnp_hqs", "red_gd", "diffusion_style"]
PROBLEMS = ["blur", "inpaint"]


def _require_cuda(device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA is required, but device is {device!r}.")


def _make_model(label, base, device, sigma):
    if label == "restormer":
        weights = restormer_weights(base, sigma=sigma, color=False)
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        return make_denoiser("restormer", weights=weights, color=False, device=device)
    if label == "restormer-aug":
        if sigma != 15.0:
            raise ValueError("restormer-aug is only configured for sigma=15")
        return make_denoiser("restormer", weights=RESTORMER_AUG_WEIGHTS, color=False, device=device)
    return make_denoiser(label)


def _model_available(label, base, sigma):
    if label == "restormer":
        return os.path.exists(restormer_weights(base, sigma=sigma, color=False))
    if label == "restormer-aug":
        return sigma == 15.0 and os.path.exists(RESTORMER_AUG_WEIGHTS)
    return True


def _rebuild_grid_summary(save_dir, grid_summary_path):
    frames = []
    for root, _, files in os.walk(save_dir):
        for name in files:
            if not name.endswith("_summary.csv"):
                continue
            if name == os.path.basename(grid_summary_path):
                continue
            path = os.path.join(root, name)
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            df = df.copy()
            df["save_dir"] = root
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(
            ["denoiser_sigma", "denoiser", "problem", "algorithm", "denoiser_mode", "angle_deg"],
            kind="stable",
        )
    out.to_csv(grid_summary_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/downstream_effect_grid")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dataset", default="val_images")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    ap.add_argument("--problems", nargs="+", default=PROBLEMS)
    ap.add_argument("--angles", type=float, nargs="+", default=None)
    ap.add_argument("--angle-start", type=float, default=0.0)
    ap.add_argument("--angle-stop", type=float, default=180.0)
    ap.add_argument("--angle-step", type=float, default=15.0)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--measurement-sigma", type=float, default=2.0)
    ap.add_argument("--num-iter", type=int, default=12)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--data-step", type=float, default=0.4)
    ap.add_argument("--prior-step", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-skip-complete", action="store_true")
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    _require_cuda(device)

    angles = args.angles
    if angles is None:
        angles = np.arange(args.angle_start, args.angle_stop, args.angle_step, dtype=np.float32)

    os.makedirs(args.save_dir, exist_ok=True)
    grid_summary_path = os.path.join(args.save_dir, "downstream_effect_grid_summary.csv")
    failures_path = os.path.join(args.save_dir, "downstream_effect_grid_failures.csv")
    failures = []

    for sigma in args.sigmas:
        for model_label in args.models:
            if not _model_available(model_label, base, sigma):
                print(f"\n[Downstream grid] skip model={model_label} sigma={sigma}: required weights unavailable")
                continue
            combo_dir = os.path.join(args.save_dir, f"sigma{int(sigma)}", args.dataset, model_label)
            os.makedirs(combo_dir, exist_ok=True)
            summary_name = (
                f"downstream_dataset-{args.dataset}_denoiser-{model_label}_sigma-{sigma}_summary.csv"
            )
            summary_path = os.path.join(combo_dir, summary_name)
            if not args.no_skip_complete and os.path.exists(summary_path) and os.path.getsize(summary_path) > 0:
                print(f"\n[Downstream grid] sigma={sigma} model={model_label} complete; skipping")
                _rebuild_grid_summary(args.save_dir, grid_summary_path)
                continue

            print(f"\n[Downstream grid] sigma={sigma} dataset={args.dataset} model={model_label}")
            try:
                denoiser = _make_model(model_label, base, device, sigma)
                downstream_effect.run(
                    dataset_dir=dataset_path(base, args.dataset),
                    denoiser=denoiser,
                    denoiser_name=model_label,
                    denoiser_sigma=sigma,
                    algorithms=args.algorithms,
                    problems=args.problems,
                    denoiser_modes=("vanilla", "group_avg"),
                    group_name=args.group_name,
                    group_size=args.group_size,
                    angles_deg=angles,
                    max_images=args.max_images,
                    seed=args.seed,
                    measurement_sigma=args.measurement_sigma,
                    num_iter=args.num_iter,
                    rho=args.rho,
                    step=args.step,
                    lam=args.lam,
                    data_step=args.data_step,
                    prior_step=args.prior_step,
                    se_mask="content",
                    expand=True,
                    save_dir=combo_dir,
                    save_csv=True,
                    verbose=True,
                )
                _rebuild_grid_summary(args.save_dir, grid_summary_path)
            except Exception as exc:
                failures.append({
                    "sigma": sigma,
                    "dataset": args.dataset,
                    "model": model_label,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                pd.DataFrame(failures).to_csv(failures_path, index=False)
                print(f"[Downstream grid] FAILED: {type(exc).__name__}: {exc}")
                raise

    _rebuild_grid_summary(args.save_dir, grid_summary_path)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)


if __name__ == "__main__":
    main()
