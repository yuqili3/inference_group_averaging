#!/usr/bin/env python
"""Run the Q2 orbit-averaging grid requested for the paper experiments."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import q2_orbit_averaging  # noqa: E402


RETRAINED_RESTORMER_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)

DATASETS = [
    ("val_images", "none", "none"),
    ("val_images_circle", "circle", "circle"),
]
GROUP_SIZES = [8, 16, 32]
MODELS = [
    "restormer",
    "restormer-rotated-noise-retrained",
    "wavelet",
    "tv",
    "nlm",
    "bm3d",
]


def _require_cuda(device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this grid, but torch.cuda.is_available() is False.")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA is required for this grid, but device is {device!r}.")


def _make_model(label, base, device):
    if label == "restormer":
        return make_denoiser(
            "restormer",
            weights=restormer_weights(base, sigma=15, color=False),
            color=False,
            device=device,
        )
    if label == "restormer-rotated-noise-retrained":
        return make_denoiser(
            "restormer",
            weights=RETRAINED_RESTORMER_WEIGHTS,
            color=False,
            device=device,
        )
    return make_denoiser(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/q2_orbit_averaging_grid")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=2)
    ap.add_argument("--noise-sigma", type=float, default=15.0)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--upsample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    _require_cuda(device)

    os.makedirs(args.save_dir, exist_ok=True)
    rows = []
    failures = []
    summary_path = os.path.join(args.save_dir, "q2_orbit_averaging_grid_summary.csv")
    failures_path = os.path.join(args.save_dir, "q2_orbit_averaging_grid_failures.csv")

    for model_label in MODELS:
        denoiser = _make_model(model_label, base, device)
        for dataset_name, noise_mask, se_mask in DATASETS:
            ds_path = dataset_path(base, dataset_name)
            for group_size in GROUP_SIZES:
                combo_dir = os.path.join(args.save_dir, dataset_name, model_label)
                os.makedirs(combo_dir, exist_ok=True)
                print(f"\n[Q2 grid] dataset={dataset_name} model={model_label} G={group_size}")
                try:
                    out = q2_orbit_averaging.run(
                        dataset_dir=ds_path,
                        denoiser=denoiser,
                        denoiser_name=model_label,
                        group_name=args.group_name,
                        averaging=group_size,
                        upsample=args.upsample,
                        noise_sigma=args.noise_sigma,
                        num_noise=args.num_noise,
                        noise_mask=noise_mask,
                        se_mask=se_mask,
                        seed=args.seed,
                        max_images=args.max_images,
                        expand=True,
                        save_dir=combo_dir,
                        save_csv=True,
                        verbose=True,
                    )
                    row = dict(out["summary"])
                    row.update({
                        "dataset": dataset_name,
                        "model": model_label,
                        "group": args.group_name,
                        "averaging": group_size,
                        "num_noise": args.num_noise,
                        "noise_sigma": args.noise_sigma,
                        "noise_mask": noise_mask,
                        "se_mask": se_mask,
                        "max_images": args.max_images,
                        "save_dir": combo_dir,
                    })
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(summary_path, index=False)
                except Exception as exc:
                    failures.append({
                        "dataset": dataset_name,
                        "model": model_label,
                        "averaging": group_size,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    pd.DataFrame(failures).to_csv(failures_path, index=False)
                    print(f"[Q2 grid] FAILED: {type(exc).__name__}: {exc}")
                    raise

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"\n[Q2 grid] wrote {summary_path}")


if __name__ == "__main__":
    main()
