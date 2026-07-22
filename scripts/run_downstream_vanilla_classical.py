#!/usr/bin/env python
"""Run vanilla denoiser baselines for downstream stochastic results."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.data import list_images, load_image  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.experiments import downstream_effect as de  # noqa: E402
from groupavg.masks import build_mask, count_pixels  # noqa: E402
from groupavg.metrics import l2sq, se_to_psnr  # noqa: E402
from groupavg.pipeline import denoise_one  # noqa: E402
from groupavg.registry import make_group  # noqa: E402

RESTORMER_AUG_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)


def _tag(denoiser_name, sigma):
    return f"{denoiser_name}_sigma{int(float(sigma))}".replace("-", "_")


def _make_model(label, base, device, sigma):
    if label == "restormer":
        weights = restormer_weights(base, sigma=sigma, color=False)
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        return make_denoiser("restormer", weights=weights, color=False, device=device)
    if label == "restormer-aug":
        if float(sigma) != 15.0:
            raise ValueError("restormer-aug is only configured for sigma=15")
        if not os.path.exists(RESTORMER_AUG_WEIGHTS):
            raise FileNotFoundError(RESTORMER_AUG_WEIGHTS)
        return make_denoiser("restormer", weights=RESTORMER_AUG_WEIGHTS, color=False, device=device)
    return make_denoiser(label)


def _input_variants(clean, group_name):
    clean = np.asarray(clean, dtype=np.float32)
    rot_group = make_group(group_name, K=1, angles=[45.0], expand=True)
    return [
        ("upright", 0.0, clean),
        ("rot45_padded", 45.0, rot_group.forward(clean)[0].astype(np.float32)),
    ]


def _run_one(clean, problem, denoise_fn, algorithm, num_iter, measurement_sigma,
             seed, rho, step, lam):
    clean = np.asarray(clean, dtype=np.float32)
    mask = build_mask(clean, clean_ref=clean, mask_mode="content")
    pixels = count_pixels(mask, clean)

    rng = np.random.default_rng(seed)
    y = problem.A(clean)
    if measurement_sigma and measurement_sigma > 0:
        y = y + rng.normal(0.0, measurement_sigma / 255.0, size=y.shape).astype(np.float32)
    y = np.clip(y, 0.0, 1.0)

    final = de._restore(
        algorithm, y, problem, denoise_fn,
        num_iter=num_iter, rho=rho, step=step, lam=lam,
        data_step=0.4, prior_step=0.25, callback=None,
    )
    degraded_se = l2sq(y, clean, mask=mask) / pixels
    final_se = l2sq(final, clean, mask=mask) / pixels
    return {
        "degraded_se": float(degraded_se),
        "degraded_psnr": float(se_to_psnr(degraded_se)),
        "final_se": float(final_se),
        "final_psnr": float(se_to_psnr(final_se)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--dataset", default="val_images")
    ap.add_argument("--denoisers", nargs="+", default=["wavelet", "tv"],
                    choices=["wavelet", "tv", "restormer", "restormer-aug"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--problems", nargs="+", default=["blur", "inpaint"], choices=["blur", "inpaint"])
    ap.add_argument("--algorithms", nargs="+", default=["pnp_hqs", "red_gd"], choices=["pnp_hqs", "red_gd"])
    ap.add_argument("--input-poses", nargs="+", default=["upright", "rot45_padded"],
                    choices=["upright", "rot45_padded"])
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--measurement-sigma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.15)
    args = ap.parse_args()

    base = load_config(args.base)
    files = list_images(dataset_path(base, args.dataset))[:args.max_images]
    device = args.device or base.get("device", "cuda:0")
    save_dir = Path(args.save_dir or f"results/downstream/vanilla/sigma{int(float(args.sigma))}")
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for denoiser_name in args.denoisers:
        denoiser = _make_model(denoiser_name, base, device, args.sigma)

        def denoise_fn(x):
            return np.clip(denoise_one(x, denoiser, args.sigma), 0.0, 1.0)

        for image_index, path in enumerate(files):
            file_name = os.path.basename(path)
            source_clean = load_image(path)
            for input_pose, input_angle, clean in _input_variants(source_clean, args.group_name):
                if input_pose not in args.input_poses:
                    continue
                for problem_name in args.problems:
                    problem_seed = args.seed + image_index * 1000003
                    if input_pose == "rot45_padded":
                        problem_seed += 45000
                    problem = de._make_problem(problem_name, clean.shape, seed=problem_seed)
                    for algorithm in args.algorithms:
                        print(
                            f"[vanilla-{denoiser_name}] image={file_name} input={input_pose} "
                            f"problem={problem_name} algorithm={algorithm} num_iter={args.num_iter}"
                        )
                        res = _run_one(
                            clean, problem, denoise_fn, algorithm,
                            num_iter=args.num_iter,
                            measurement_sigma=args.measurement_sigma,
                            seed=args.seed + image_index * 1000003 + 17 + int(round(input_angle)) * 997,
                            rho=args.rho, step=args.step, lam=args.lam,
                        )
                        rows.append({
                            "dataset": args.dataset,
                            "file": file_name,
                            "image_index": image_index,
                            "input_pose": input_pose,
                            "input_angle_deg": input_angle,
                            "input_height": clean.shape[0],
                            "input_width": clean.shape[1],
                            "problem": problem_name,
                            "algorithm": algorithm,
                            "denoiser": denoiser_name,
                            "denoiser_sigma": args.sigma,
                            "mode": "vanilla",
                            "sampling": "none",
                            "group_expand": False,
                            "base_group_size": 0,
                            "sample_size": 0,
                            "num_iter": args.num_iter,
                            "final_se": res["final_se"],
                            "final_psnr": res["final_psnr"],
                            "degraded_se": res["degraded_se"],
                            "degraded_psnr": res["degraded_psnr"],
                            "total_denoiser_calls": args.num_iter,
                            "total_group_evals": 0,
                        })

    final = pd.DataFrame(rows)
    summary = (
        final.groupby(
            ["dataset", "input_pose", "input_angle_deg", "problem", "algorithm",
             "denoiser", "denoiser_sigma", "mode", "sampling", "base_group_size",
             "sample_size", "group_expand", "num_iter"],
            as_index=False,
        )
        .agg(
            mean_final_se=("final_se", "mean"),
            mean_final_psnr=("final_psnr", "mean"),
            mean_degraded_se=("degraded_se", "mean"),
            mean_degraded_psnr=("degraded_psnr", "mean"),
            total_group_evals=("total_group_evals", "first"),
            n=("file", "size"),
        )
    )
    summary["final_psnr_from_mean_se"] = summary["mean_final_se"].map(se_to_psnr)
    summary["degraded_psnr_from_mean_se"] = summary["mean_degraded_se"].map(se_to_psnr)
    tag = "_".join(_tag(d, args.sigma) for d in args.denoisers)
    final.to_csv(save_dir / f"downstream_vanilla_{tag}_final.csv", index=False)
    summary.to_csv(save_dir / f"downstream_vanilla_{tag}_summary.csv", index=False)
    print(f"wrote {save_dir / f'downstream_vanilla_{tag}_summary.csv'}")


if __name__ == "__main__":
    main()
