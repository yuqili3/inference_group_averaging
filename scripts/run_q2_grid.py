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
SIGMAS = [15.0]
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
            weights=restormer_weights(base, sigma=_make_model.sigma, color=False),
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


_make_model.sigma = 15.0


def _expected_detail_rows(max_images, num_noise, group_size):
    return max_images * num_noise * group_size if max_images is not None else None


def _combo_complete(path, expected_detail_rows):
    summary_files = [f for f in os.listdir(path) if f.endswith("_summary.csv")] if os.path.isdir(path) else []
    per_image_files = [f for f in os.listdir(path) if f.endswith("_per_image.csv")] if os.path.isdir(path) else []
    detail_files = [f for f in os.listdir(path) if f.endswith("_detail.csv")] if os.path.isdir(path) else []
    if len(summary_files) != 1 or len(per_image_files) != 1 or len(detail_files) != 1:
        return False
    if expected_detail_rows is None:
        return True
    try:
        detail = pd.read_csv(os.path.join(path, detail_files[0]))
    except Exception:
        return False
    return len(detail) == expected_detail_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/q2_orbit_averaging_grid")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=2)
    ap.add_argument("--noise-sigma", type=float, default=None)
    ap.add_argument("--sigmas", type=float, nargs="+", default=None)
    ap.add_argument("--include-retrained-all-sigmas", action="store_true")
    ap.add_argument("--no-skip-complete", action="store_true")
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
    sigmas = args.sigmas or ([args.noise_sigma] if args.noise_sigma is not None else SIGMAS)

    for sigma in sigmas:
        for model_label in MODELS:
            if (
                model_label == "restormer-rotated-noise-retrained"
                and sigma != 15.0
                and not args.include_retrained_all_sigmas
            ):
                print(f"\n[Q2 grid] skip model={model_label} sigma={sigma}: no retrained sigma-specific weights configured")
                continue
            _make_model.sigma = sigma
            denoiser = _make_model(model_label, base, device)
            for dataset_name, noise_mask, se_mask in DATASETS:
                ds_path = dataset_path(base, dataset_name)
                for group_size in GROUP_SIZES:
                    combo_dir = os.path.join(args.save_dir, f"sigma{int(sigma)}", dataset_name, model_label)
                    os.makedirs(combo_dir, exist_ok=True)
                    print(f"\n[Q2 grid] sigma={sigma} dataset={dataset_name} model={model_label} G={group_size}")
                    if not args.no_skip_complete and _combo_complete(
                        combo_dir, _expected_detail_rows(args.max_images, args.num_noise, group_size)
                    ):
                        print("[Q2 grid] complete; skipping")
                        continue
                    try:
                        out = q2_orbit_averaging.run(
                            dataset_dir=ds_path,
                            denoiser=denoiser,
                            denoiser_name=model_label,
                            group_name=args.group_name,
                            averaging=group_size,
                            upsample=args.upsample,
                            noise_sigma=sigma,
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
                            "sigma": sigma,
                            "dataset": dataset_name,
                            "model": model_label,
                            "group": args.group_name,
                            "averaging": group_size,
                            "num_noise": args.num_noise,
                            "noise_sigma": sigma,
                            "noise_mask": noise_mask,
                            "se_mask": se_mask,
                            "max_images": args.max_images,
                            "save_dir": combo_dir,
                        })
                        rows.append(row)
                        pd.DataFrame(rows).to_csv(summary_path, index=False)
                    except Exception as exc:
                        failures.append({
                            "sigma": sigma,
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
