#!/usr/bin/env python
"""Compare fixed vs stochastic group averaging inside PnP/RED iterations.

Initial scope is classical denoising. The stochastic modes sample m angles from
a fixed G=16 angle grid at every denoiser call, then average only those sampled
orbit estimates. Each setting is run on both the original upright image and a
45-degree rotated image padded to the larger rotation canvas. This isolates the
cost/accuracy tradeoff against full G=16 averaging inside downstream solvers.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
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


MODES = [
    ("G16_fixed", 16, "fixed"),
    ("G4_random", 4, "random"),
    ("G2_random", 2, "random"),
    ("G1_random", 1, "random"),
]

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


def _result_paths(save_dir, denoiser_name, sigma):
    stem = f"downstream_stochastic_{_tag(denoiser_name, sigma)}"
    return {
        "detail": save_dir / f"{stem}_detail.csv",
        "final": save_dir / f"{stem}_final.csv",
        "summary": save_dir / f"{stem}_summary.csv",
    }


def _write_tables(save_dir, denoiser_name, sigma, detail_rows, final_rows):
    paths = _result_paths(save_dir, denoiser_name, sigma)
    detail = pd.DataFrame(detail_rows)
    final = pd.DataFrame(final_rows)
    if final.empty:
        summary = pd.DataFrame()
    else:
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

    detail.to_csv(paths["detail"], index=False)
    final.to_csv(paths["final"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def _write_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(img), 0.0, 1.0)
    cv2.imwrite(str(path), (arr * 255.0 + 0.5).astype(np.uint8))


def _angle_string(angles):
    return ";".join(f"{float(a):.6g}" for a in angles)


def _input_variants(clean, group_name):
    """Return downstream input images: upright and 45-degree padded rotation."""
    clean = np.asarray(clean, dtype=np.float32)
    rot_group = make_group(group_name, K=1, angles=[45.0], expand=True)
    rot45 = rot_group.forward(clean)[0].astype(np.float32)
    return [
        ("upright", 0.0, clean),
        ("rot45_padded", 45.0, rot45),
    ]


class OrbitSampler:
    def __init__(self, denoiser, sigma, group_name, base_angles, mode_name, sample_size,
                 sampling, seed, expand=True, clip=True):
        self.denoiser = denoiser
        self.sigma = float(sigma)
        self.group_name = group_name
        self.base_angles = np.asarray(base_angles, dtype=np.float32)
        self.mode_name = mode_name
        self.sample_size = int(sample_size)
        self.sampling = sampling
        self.expand = expand
        self.clip = clip
        self.rng = np.random.default_rng(seed)
        self.call_count = 0
        self.last_angles = []
        self._cache = {}

    def _sample_angles(self):
        if self.sampling == "fixed":
            return self.base_angles
        idx = self.rng.choice(len(self.base_angles), size=self.sample_size, replace=False)
        return self.base_angles[np.sort(idx)]

    def _group_for(self, angles):
        key = tuple(float(a) for a in angles)
        if key not in self._cache:
            self._cache[key] = make_group(
                self.group_name, K=len(key), angles=list(key), expand=self.expand
            )
        return self._cache[key]

    def __call__(self, x):
        angles = self._sample_angles()
        group = self._group_for(angles)
        transformed = group.forward(np.asarray(x, dtype=np.float32))
        estimates = []
        for idx, tg_x in enumerate(transformed):
            d = denoise_one(tg_x, self.denoiser, self.sigma)
            if self.clip:
                d = np.clip(d, 0.0, 1.0)
            z = group.invert(idx, d)
            if self.clip:
                z = np.clip(z, 0.0, 1.0)
            estimates.append(z.astype(np.float32))
        self.call_count += 1
        self.last_angles = [float(a) for a in angles]
        out = np.mean(np.stack(estimates, axis=0), axis=0).astype(np.float32)
        return np.clip(out, 0.0, 1.0) if self.clip else out


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

    trajectory = []

    def _record(k, xk):
        se = l2sq(xk, clean, mask=mask) / pixels
        trajectory.append({
            "iter": int(k),
            "image": np.asarray(xk, dtype=np.float32).copy(),
            "se": float(se),
            "psnr": float(se_to_psnr(se)),
            "sampled_angles": list(denoise_fn.last_angles),
            "denoiser_calls": int(denoise_fn.call_count),
        })

    final = de._restore(
        algorithm, y, problem, denoise_fn,
        num_iter=num_iter, rho=rho, step=step, lam=lam,
        data_step=0.4, prior_step=0.25, callback=_record,
    )
    degraded_se = l2sq(y, clean, mask=mask) / pixels
    final_se = l2sq(final, clean, mask=mask) / pixels
    return {
        "degraded": y,
        "degraded_se": float(degraded_se),
        "degraded_psnr": float(se_to_psnr(degraded_se)),
        "final": final,
        "final_se": float(final_se),
        "final_psnr": float(se_to_psnr(final_se)),
        "trajectory": trajectory,
    }


def _snapshot_iters(num_iter, requested):
    if requested:
        vals = sorted(set(int(v) for v in requested if 0 <= int(v) < num_iter))
    else:
        vals = [0, 1, 2, 4, 8, 12, 16, num_iter - 1]
        vals = sorted(set(v for v in vals if 0 <= v < num_iter))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--dataset", default="val_images")
    ap.add_argument("--denoiser", default="wavelet",
                    choices=["wavelet", "tv", "restormer", "restormer-aug"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--problems", nargs="+", default=["blur", "inpaint"], choices=["blur", "inpaint"])
    ap.add_argument("--algorithms", nargs="+", default=["pnp_hqs", "red_gd"], choices=["pnp_hqs", "red_gd"])
    ap.add_argument("--input-poses", nargs="+", default=["upright", "rot45_padded"],
                    choices=["upright", "rot45_padded"])
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--save-image-count", type=int, default=1)
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--measurement-sigma", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--snapshot-iters", type=int, nargs="*", default=None)
    args = ap.parse_args()

    base = load_config(args.base)
    files = list_images(dataset_path(base, args.dataset))[:args.max_images]
    save_dir = Path(args.save_dir or f"results/downstream/stochastic/{_tag(args.denoiser, args.sigma)}")
    save_dir.mkdir(parents=True, exist_ok=True)
    image_dir = save_dir / "intermediate_images"

    device = args.device or base.get("device", "cuda:0")
    denoiser = _make_model(args.denoiser, base, device, args.sigma)
    base_angles = np.arange(args.group_size, dtype=np.float32) * (360.0 / float(args.group_size))
    snapshots = _snapshot_iters(args.num_iter, args.snapshot_iters)

    detail_rows = []
    final_rows = []
    for image_index, path in enumerate(files):
        file_name = os.path.basename(path)
        stem = Path(file_name).stem
        source_clean = load_image(path)

        for input_pose, input_angle, clean in _input_variants(source_clean, args.group_name):
            if input_pose not in args.input_poses:
                continue
            if image_index < args.save_image_count:
                _write_gray(image_dir / input_pose / stem / "clean.png", clean)

            for problem_name in args.problems:
                problem_seed = args.seed + image_index * 1000003
                if input_pose == "rot45_padded":
                    problem_seed += 45000
                problem = de._make_problem(problem_name, clean.shape, seed=problem_seed)
                for algorithm in args.algorithms:
                    for mode_name, sample_size, sampling in MODES:
                        group_expand = input_pose != "rot45_padded"
                        run_seed = (
                            args.seed + image_index * 1000003 + sample_size * 1009 +
                            len(mode_name) + int(round(input_angle)) * 997
                        )
                        sampler = OrbitSampler(
                            denoiser, args.sigma, args.group_name, base_angles,
                            mode_name, sample_size, sampling, seed=run_seed,
                            expand=group_expand,
                        )
                        print(
                            f"[stochastic-{args.denoiser}] image={file_name} input={input_pose} "
                            f"shape={clean.shape} problem={problem_name} algorithm={algorithm} "
                            f"mode={mode_name} group_expand={group_expand} num_iter={args.num_iter}"
                        )
                        res = _run_one(
                            clean, problem, sampler, algorithm,
                            num_iter=args.num_iter,
                            measurement_sigma=args.measurement_sigma,
                            seed=args.seed + image_index * 1000003 + 17 + int(round(input_angle)) * 997,
                            rho=args.rho, step=args.step, lam=args.lam,
                        )

                        if image_index < args.save_image_count:
                            prefix = image_dir / input_pose / stem / problem_name / algorithm / mode_name
                            _write_gray(prefix / "degraded.png", res["degraded"])
                            _write_gray(prefix / "final.png", res["final"])

                        for item in res["trajectory"]:
                            k = item["iter"]
                            cumulative_group_evals = (k + 1) * sample_size
                            detail_rows.append({
                                "dataset": args.dataset,
                                "file": file_name,
                                "image_index": image_index,
                                "input_pose": input_pose,
                                "input_angle_deg": input_angle,
                                "input_height": clean.shape[0],
                                "input_width": clean.shape[1],
                                "problem": problem_name,
                                "algorithm": algorithm,
                                "denoiser": args.denoiser,
                                "denoiser_sigma": args.sigma,
                                "mode": mode_name,
                                "sampling": sampling,
                                "group_expand": group_expand,
                                "base_group_size": args.group_size,
                                "sample_size": sample_size,
                                "iteration": k + 1,
                                "num_iter": args.num_iter,
                                "se": item["se"],
                                "psnr": item["psnr"],
                                "degraded_se": res["degraded_se"],
                                "degraded_psnr": res["degraded_psnr"],
                                "sampled_angles": _angle_string(item["sampled_angles"]),
                                "cumulative_denoiser_calls": k + 1,
                                "cumulative_group_evals": cumulative_group_evals,
                            })
                            if image_index < args.save_image_count and k in snapshots:
                                out = (
                                    image_dir / input_pose / stem / problem_name / algorithm /
                                    mode_name / f"iter_{k + 1:03d}.png"
                                )
                                _write_gray(out, item["image"])

                        final_rows.append({
                            "dataset": args.dataset,
                            "file": file_name,
                            "image_index": image_index,
                            "input_pose": input_pose,
                            "input_angle_deg": input_angle,
                            "input_height": clean.shape[0],
                            "input_width": clean.shape[1],
                            "problem": problem_name,
                            "algorithm": algorithm,
                            "denoiser": args.denoiser,
                            "denoiser_sigma": args.sigma,
                            "mode": mode_name,
                            "sampling": sampling,
                            "group_expand": group_expand,
                            "base_group_size": args.group_size,
                            "sample_size": sample_size,
                            "num_iter": args.num_iter,
                            "final_se": res["final_se"],
                            "final_psnr": res["final_psnr"],
                            "degraded_se": res["degraded_se"],
                            "degraded_psnr": res["degraded_psnr"],
                            "total_denoiser_calls": args.num_iter,
                            "total_group_evals": args.num_iter * sample_size,
                        })
                        paths = _write_tables(save_dir, args.denoiser, args.sigma, detail_rows, final_rows)
                        print(f"checkpoint wrote {paths['summary']}")

    paths = _write_tables(save_dir, args.denoiser, args.sigma, detail_rows, final_rows)
    print(f"wrote {paths['summary']}")


if __name__ == "__main__":
    main()
