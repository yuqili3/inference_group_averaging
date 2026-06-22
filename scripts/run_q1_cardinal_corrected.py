#!/usr/bin/env python
"""Run corrected Q1 equivariance checks for 0/90/180/270 degree rotations."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import q1_cardinal_equivariance  # noqa: E402


RETRAINED_RESTORMER_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)

DATASETS = [
    ("val_images", "none", "none"),
    ("val_images_circle", "circle", "circle"),
]
MODELS = ["restormer", "restormer-rotated-noise-retrained", "wavelet", "tv", "nlm"]
SIGMAS = [15.0, 25.0, 50.0]


def _require_cuda(device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required, but torch.cuda.is_available() is False.")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA is required, but device is {device!r}.")


def _make_model(label, base, device, sigma):
    if label == "restormer":
        return make_denoiser(
            "restormer",
            weights=restormer_weights(base, sigma=sigma, color=False),
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


def _combo_complete(path, expected_rows):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        return len(pd.read_csv(path)) == expected_rows
    except Exception:
        return False


def _rebuild_grid_summary(save_dir, out_path):
    frames = []
    for root, _, files in os.walk(save_dir):
        for name in files:
            if not name.endswith("_summary.csv"):
                continue
            if name == os.path.basename(out_path):
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
        out = out.sort_values(["noise_sigma", "denoiser", "angle_deg"], kind="stable")
    out.to_csv(out_path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/q1_cardinal_corrected")
    ap.add_argument("--device", default=None)
    ap.add_argument("--datasets", nargs="+", default=[d[0] for d in DATASETS])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=4)
    ap.add_argument("--angles", type=int, nargs="+", default=[0, 90, 180, 270])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-retrained-all-sigmas", action="store_true")
    ap.add_argument("--no-skip-complete", action="store_true")
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    _require_cuda(device)

    dataset_cfg = {name: ("circle", "circle") if name.endswith("circle") else ("none", "none") for name in args.datasets}
    os.makedirs(args.save_dir, exist_ok=True)
    grid_summary_path = os.path.join(args.save_dir, "q1_cardinal_corrected_summary.csv")
    failures_path = os.path.join(args.save_dir, "q1_cardinal_corrected_failures.csv")
    failures = []
    if os.path.exists(failures_path):
        os.remove(failures_path)

    expected_summary_rows = len(args.angles)
    for sigma in args.sigmas:
        for model_label in args.models:
            if (
                model_label == "restormer-rotated-noise-retrained"
                and sigma != 15.0
                and not args.include_retrained_all_sigmas
            ):
                print(f"\n[Q1 cardinal] skip model={model_label} sigma={sigma}: no retrained sigma-specific weights configured")
                continue
            denoiser = _make_model(model_label, base, device, sigma)
            for dataset_name in args.datasets:
                noise_mask, se_mask = dataset_cfg[dataset_name]
                combo_dir = os.path.join(args.save_dir, f"sigma{int(sigma)}", dataset_name, model_label)
                os.makedirs(combo_dir, exist_ok=True)
                summary_name = (
                    f"q1_cardinal_folder-{dataset_name}_denoiser-{model_label}"
                    f"_group-cardinal_rot90_sigma-{sigma}_G-{len(args.angles)}_summary.csv"
                )
                summary_path = os.path.join(combo_dir, summary_name)
                print(f"\n[Q1 cardinal] sigma={sigma} dataset={dataset_name} model={model_label}")
                if not args.no_skip_complete and _combo_complete(summary_path, expected_summary_rows):
                    print("[Q1 cardinal] complete; skipping")
                    _rebuild_grid_summary(args.save_dir, grid_summary_path)
                    continue
                try:
                    q1_cardinal_equivariance.run(
                        dataset_dir=dataset_path(base, dataset_name),
                        denoiser=denoiser,
                        denoiser_name=model_label,
                        angles=args.angles,
                        noise_sigma=sigma,
                        num_noise=args.num_noise,
                        noise_mask=noise_mask,
                        se_mask=se_mask,
                        seed=args.seed,
                        max_images=args.max_images,
                        save_dir=combo_dir,
                        save_csv=True,
                        verbose=True,
                    )
                    _rebuild_grid_summary(args.save_dir, grid_summary_path)
                except Exception as exc:
                    failures.append({
                        "sigma": sigma,
                        "dataset": dataset_name,
                        "model": model_label,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    pd.DataFrame(failures).to_csv(failures_path, index=False)
                    print(f"[Q1 cardinal] FAILED: {type(exc).__name__}: {exc}")
                    raise

    _rebuild_grid_summary(args.save_dir, grid_summary_path)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"\n[Q1 cardinal] wrote {grid_summary_path}")


if __name__ == "__main__":
    main()
