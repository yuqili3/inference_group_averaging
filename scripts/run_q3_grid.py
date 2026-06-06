#!/usr/bin/env python
"""Run the Q3 degradation grid with resumable sigma sweeps."""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import q3_degradation  # noqa: E402


DATASETS = ["square_val_images", "val_images", "val_images_circle", "val_images_diag_padded"]
GROUPS = ["fourier_rotation", "rotation"]
AVERAGINGS = [2, 4, 8, 16]
NOISE_MASKS = ["circle"]
SIGMAS = [15.0]


def _require_cuda(device):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this grid, but torch.cuda.is_available() is False.")
    if not str(device).startswith("cuda"):
        raise RuntimeError(f"CUDA is required for this grid, but device is {device!r}.")


def _make_restormer(base, device, sigma):
    return make_denoiser(
        "restormer",
        weights=restormer_weights(base, sigma=sigma, color=False),
        color=False,
        device=device,
    )


def _combo_complete(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/q3_degradation_grid")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=4)
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS)
    ap.add_argument("--averagings", type=int, nargs="+", default=AVERAGINGS)
    ap.add_argument("--upsample", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-skip-complete", action="store_true")
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    base["device"] = device
    _require_cuda(device)

    os.makedirs(args.save_dir, exist_ok=True)
    rows = []
    failures = []
    summary_path = os.path.join(args.save_dir, "q3_degradation_grid_summary.csv")
    failures_path = os.path.join(args.save_dir, "q3_degradation_grid_failures.csv")
    if os.path.exists(failures_path):
        os.remove(failures_path)
    datasets = {name: dataset_path(base, name) for name in DATASETS}

    for sigma in args.sigmas:
        out_csv = os.path.join(args.save_dir, f"sigma{int(sigma)}", f"q3_degradation_denoiser-restormer_sigma-{sigma}.csv")
        if not args.no_skip_complete and _combo_complete(out_csv):
            print(f"\n[Q3 grid] sigma={sigma} complete; skipping")
            df = pd.read_csv(out_csv)
        else:
            print(f"\n[Q3 grid] sigma={sigma}")
            combo_dir = os.path.dirname(out_csv)
            os.makedirs(combo_dir, exist_ok=True)
            try:
                df = q3_degradation.run(
                    datasets=datasets,
                    denoiser=_make_restormer(base, device, sigma),
                    denoiser_name="restormer",
                    groups=GROUPS,
                    averagings=args.averagings,
                    noise_sigma=sigma,
                    num_noise=args.num_noise,
                    noise_masks=NOISE_MASKS,
                    se_mask="content",
                    upsample=args.upsample,
                    max_images=args.max_images,
                    seed=args.seed,
                    save_dir=combo_dir,
                    save_csv=True,
                    verbose=True,
                )
            except Exception as exc:
                failures.append({
                    "sigma": sigma,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                pd.DataFrame(failures).to_csv(failures_path, index=False)
                print(f"[Q3 grid] FAILED: {type(exc).__name__}: {exc}")
                raise
        df = df.copy()
        df["sigma"] = sigma
        rows.extend(df.to_dict("records"))
        pd.DataFrame(rows).to_csv(summary_path, index=False)

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(failures_path, index=False)
    print(f"\n[Q3 grid] wrote {summary_path}")


if __name__ == "__main__":
    main()
