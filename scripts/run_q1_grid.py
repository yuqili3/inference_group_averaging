#!/usr/bin/env python
"""Run the Q1 equivariance grid with a single dense orbit size."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import q1_equivariance  # noqa: E402


RETRAINED_RESTORMER_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)

DATASETS = [
    ("val_images", "none", "none"),
    ("val_images_circle", "circle", "circle"),
]
MODELS = [
    "restormer",
    "restormer-rotated-noise-retrained",
    "wavelet",
    "tv",
    "nlm",
    "bm3d",
]
SIGMAS = [15.0, 25.0, 50.0]


def _require_cuda(device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this grid, but torch.cuda.is_available() is False.")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA is required for this grid, but device is {device!r}.")


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


def _combo_complete(path, max_images, num_noise, averaging):
    per_image_files = [f for f in os.listdir(path) if f.endswith("_per_image.csv")] if os.path.isdir(path) else []
    per_element_files = [f for f in os.listdir(path) if f.endswith("_per_element.csv")] if os.path.isdir(path) else []
    if len(per_image_files) != 1 or len(per_element_files) != 1:
        return False
    if max_images is None:
        return True
    try:
        per_image = pd.read_csv(os.path.join(path, per_image_files[0]))
        per_element = pd.read_csv(os.path.join(path, per_element_files[0]))
    except Exception:
        return False
    return len(per_image) == max_images and len(per_element) == max_images * num_noise * averaging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/q1_equivariance_grid")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=4)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--averaging", type=int, default=64)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--upsample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-retrained-all-sigmas", action="store_true")
    ap.add_argument("--include-bm3d-all-sigmas", action="store_true")
    ap.add_argument("--skip-bm3d", action="store_true")
    ap.add_argument("--no-skip-complete", action="store_true")
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    _require_cuda(device)

    os.makedirs(args.save_dir, exist_ok=True)
    rows = []
    failures = []
    summary_path = os.path.join(args.save_dir, "q1_equivariance_grid_summary.csv")
    failures_path = os.path.join(args.save_dir, "q1_equivariance_grid_failures.csv")

    for sigma in args.sigmas:
        for model_label in MODELS:
            if model_label == "bm3d" and args.skip_bm3d:
                print(f"\n[Q1 grid] skip model={model_label} sigma={sigma}: skipped by request")
                continue
            if (
                model_label == "restormer-rotated-noise-retrained"
                and sigma != 15.0
                and not args.include_retrained_all_sigmas
            ):
                print(f"\n[Q1 grid] skip model={model_label} sigma={sigma}: no retrained sigma-specific weights configured")
                continue
            if model_label == "bm3d" and sigma != 15.0 and not args.include_bm3d_all_sigmas:
                print(f"\n[Q1 grid] skip model={model_label} sigma={sigma}: bm3d sigma={sigma} omitted by request")
                continue
            denoiser = _make_model(model_label, base, device, sigma)
            for dataset_name, noise_mask, se_mask in DATASETS:
                combo_dir = os.path.join(args.save_dir, f"sigma{int(sigma)}", dataset_name, model_label)
                os.makedirs(combo_dir, exist_ok=True)
                print(f"\n[Q1 grid] sigma={sigma} dataset={dataset_name} model={model_label} G={args.averaging}")
                if not args.no_skip_complete and _combo_complete(
                    combo_dir, args.max_images, args.num_noise, args.averaging
                ):
                    print("[Q1 grid] complete; skipping")
                    continue
                try:
                    out = q1_equivariance.run(
                        dataset_dir=dataset_path(base, dataset_name),
                        denoiser=denoiser,
                        denoiser_name=model_label,
                        group_name=args.group_name,
                        averaging=args.averaging,
                        upsample=args.upsample,
                        noise_sigma=sigma,
                        num_noise=args.num_noise,
                        noise_mask=noise_mask,
                        se_mask=se_mask,
                        clean_mode=False,
                        seed=args.seed,
                        max_images=args.max_images,
                        expand=True,
                        save_dir=combo_dir,
                        save_csv=True,
                        verbose=True,
                    )
                    row = {
                        "sigma": sigma,
                        "dataset": dataset_name,
                        "model": model_label,
                        "group": args.group_name,
                        "averaging": args.averaging,
                        "num_noise": args.num_noise,
                        "noise_mask": noise_mask,
                        "se_mask": se_mask,
                        "max_images": args.max_images,
                        "mean_rel_err": out["per_image"]["mean_rel_err"].mean(),
                        "mean_max_elem_rel_err": out["per_image"]["max_elem_rel_err"].mean(),
                        "mean_e1": out["per_image"]["e1"].mean(),
                        "save_dir": combo_dir,
                    }
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(summary_path, index=False)
                except Exception as exc:
                    failures.append({
                        "sigma": sigma,
                        "dataset": dataset_name,
                        "model": model_label,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    pd.DataFrame(failures).to_csv(failures_path, index=False)
                    print(f"[Q1 grid] FAILED: {type(exc).__name__}: {exc}")
                    raise

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"\n[Q1 grid] wrote {summary_path}")


if __name__ == "__main__":
    main()
