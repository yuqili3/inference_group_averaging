#!/usr/bin/env python
"""Run Figure 5 PSNR-vs-rotation-angle grids."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import figure5_angle_psnr  # noqa: E402


RESTORMER_AUG_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)

DATASETS = [("val_images", "none", "none")]
MODELS = ["wavelet", "tv", "nlm", "restormer", "restormer-aug"]
SIGMAS = [15.0, 25.0, 50.0]


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
        return make_denoiser(
            "restormer",
            weights=weights,
            color=False,
            device=device,
        )
    if label == "restormer-aug":
        return make_denoiser(
            "restormer",
            weights=RESTORMER_AUG_WEIGHTS,
            color=False,
            device=device,
        )
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
    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(
            ["noise_sigma", "denoiser", "dataset", "angle_deg", "estimator", "group_size"],
            kind="stable",
        )
    else:
        out = pd.DataFrame()
    out.to_csv(grid_summary_path, index=False)
    return out


def _combo_complete(path, expected_summary_rows, expected_detail_rows):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        summary = pd.read_csv(path)
    except Exception:
        return False
    if len(summary) != expected_summary_rows:
        return False
    detail_path = path.replace("_summary.csv", "_detail.csv")
    if not os.path.exists(detail_path) or os.path.getsize(detail_path) == 0:
        return False
    try:
        detail = pd.read_csv(detail_path, usecols=["psnr"])
    except Exception:
        return False
    return len(detail) == expected_detail_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/figure5_angle_psnr_grid")
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="+", default=[d[0] for d in DATASETS])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=32)
    ap.add_argument("--eval-group-sizes", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--angle-start", type=float, default=0.0)
    ap.add_argument("--angle-stop", type=float, default=180.0)
    ap.add_argument("--angle-step", type=float, default=5.0)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--upsample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--orbit-expand", action="store_true")
    ap.add_argument("--no-skip-complete", action="store_true")
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    _require_cuda(device)

    os.makedirs(args.save_dir, exist_ok=True)
    angles = np.arange(args.angle_start, args.angle_stop, args.angle_step, dtype=np.float32)
    expected_summary_rows = len(angles) * (1 + len(args.eval_group_sizes))
    expected_detail_rows = args.max_images * args.num_noise * expected_summary_rows
    grid_summary_path = os.path.join(args.save_dir, "figure5_angle_psnr_grid_summary.csv")
    failures_path = os.path.join(args.save_dir, "figure5_angle_psnr_grid_failures.csv")
    failures = []
    if os.path.exists(failures_path):
        os.remove(failures_path)

    dataset_cfg = {name: ("circle", "circle") if name.endswith("circle") else ("none", "none") for name in args.datasets}

    for sigma in args.sigmas:
        for model_label in args.models:
            if model_label == "restormer-aug" and sigma != 15.0:
                print(f"\n[Figure5 grid] skip model={model_label} sigma={sigma}: augmented weights only configured for sigma=15")
                continue
            if not _model_available(model_label, base, sigma):
                print(f"\n[Figure5 grid] skip model={model_label} sigma={sigma}: required weights are not available")
                continue
            denoiser = _make_model(model_label, base, device, sigma)
            for dataset_name in args.datasets:
                noise_mask, se_mask = dataset_cfg[dataset_name]
                combo_dir = os.path.join(args.save_dir, f"sigma{int(sigma)}", dataset_name, model_label)
                os.makedirs(combo_dir, exist_ok=True)
                summary_name = (
                    f"figure5_dataset-{dataset_name}_denoiser-{model_label}"
                    f"_sigma-{sigma}_G-{args.group_size}_summary.csv"
                )
                summary_path = os.path.join(combo_dir, summary_name)
                print(f"\n[Figure5 grid] sigma={sigma} dataset={dataset_name} model={model_label} G={args.group_size}")
                if not args.no_skip_complete and _combo_complete(summary_path, expected_summary_rows, expected_detail_rows):
                    print("[Figure5 grid] complete; skipping")
                    combo_summary = pd.read_csv(summary_path)
                else:
                    try:
                        out = figure5_angle_psnr.run(
                            dataset_dir=dataset_path(base, dataset_name),
                            denoiser=denoiser,
                            denoiser_name=model_label,
                            rotation_group_name=args.group_name,
                            base_group_size=args.group_size,
                            eval_group_sizes=args.eval_group_sizes,
                            angles_deg=angles,
                            upsample=args.upsample,
                            noise_sigma=sigma,
                            num_noise=args.num_noise,
                            noise_mask=noise_mask,
                            se_mask=se_mask,
                            seed=args.seed,
                            max_images=args.max_images,
                            expand=True,
                            orbit_expand=args.orbit_expand,
                            save_dir=combo_dir,
                            save_csv=True,
                            verbose=True,
                        )
                        combo_summary = out["summary"]
                    except Exception as exc:
                        failures.append({
                            "sigma": sigma,
                            "dataset": dataset_name,
                            "model": model_label,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        })
                        pd.DataFrame(failures).to_csv(failures_path, index=False)
                        print(f"[Figure5 grid] FAILED: {type(exc).__name__}: {exc}")
                        raise
                _rebuild_grid_summary(args.save_dir, grid_summary_path)

    _rebuild_grid_summary(args.save_dir, grid_summary_path)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"\n[Figure5 grid] wrote {grid_summary_path}")


if __name__ == "__main__":
    main()
