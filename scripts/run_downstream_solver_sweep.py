#!/usr/bin/env python
"""Hyperparameter sweep for downstream restoration solvers.

This targeted experiment checks whether absolute restoration PSNR is driven by
solver hyperparameters rather than group averaging. It compares vanilla
denoisers against fixed G=16 orbit averaging for blur/inpaint and PnP-HQS/RED.
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
from groupavg.metrics import l2sq, masked_ssim, se_to_psnr  # noqa: E402
from groupavg.pipeline import denoise_one  # noqa: E402
from groupavg.registry import make_group  # noqa: E402


RESTORMER_AUG_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)


def _tag(text):
    return str(text).replace("-", "_").replace(".", "p")


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


def _input_variants(clean, group_name, eval_mask_mode, input_rotate_expand=True):
    clean = np.asarray(clean, dtype=np.float32)
    rot_group = make_group(group_name, K=1, angles=[45.0], expand=input_rotate_expand)
    clean_mask = build_mask(clean, clean_ref=clean, mask_mode=eval_mask_mode)
    rot_mask = (rot_group.forward(clean_mask)[0] > 0.5).astype(np.float32)
    return [
        ("upright", 0.0, clean, clean_mask),
        ("rot45_padded", 45.0, rot_group.forward(clean)[0].astype(np.float32), rot_mask),
    ]


def _write_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(img), 0.0, 1.0)
    cv2.imwrite(str(path), (arr * 255.0 + 0.5).astype(np.uint8))


def _schedule_value(schedule, iteration, num_iter):
    kind = schedule["kind"]
    if kind == "fixed":
        return float(schedule["start"])
    if kind == "linear":
        if num_iter <= 1:
            return float(schedule["end"])
        t = float(iteration) / float(num_iter - 1)
        return (1.0 - t) * float(schedule["start"]) + t * float(schedule["end"])
    raise ValueError(f"Unknown schedule kind: {kind}")


def _schedule_label(schedule):
    if schedule["kind"] == "fixed":
        return f"fixed{_tag(schedule['start'])}"
    return f"linear{_tag(schedule['start'])}_to_{_tag(schedule['end'])}"


def _parse_schedule(text):
    if "->" in text:
        start, end = text.split("->", 1)
        return {"kind": "linear", "start": float(start), "end": float(end)}
    return {"kind": "fixed", "start": float(text), "end": float(text)}


class ScheduledDenoiser:
    def __init__(self, denoiser, mode, denoiser_name, train_sigma, schedule,
                 group_name, group_size, group_expand, num_iter, clip=False):
        self.denoiser = denoiser
        self.mode = mode
        self.denoiser_name = denoiser_name
        self.train_sigma = float(train_sigma)
        self.schedule = schedule
        self.group_name = group_name
        self.group_size = int(group_size)
        self.group_expand = bool(group_expand)
        self.num_iter = int(num_iter)
        self.clip = clip
        self.call_count = 0
        self.last_effective_sigma = float(train_sigma)
        self.last_angles = ["none"]
        self.group = None
        if mode == "G16_fixed":
            self.angles = np.arange(self.group_size, dtype=np.float32) * (360.0 / float(self.group_size))
            self.group = make_group(
                self.group_name, K=self.group_size, angles=list(self.angles), expand=self.group_expand
            )
        else:
            self.angles = np.array([], dtype=np.float32)

    def _effective_sigma(self):
        requested = _schedule_value(self.schedule, self.call_count, self.num_iter)
        if self.denoiser_name in {"restormer", "restormer-aug"}:
            return self.train_sigma
        return requested

    def __call__(self, x):
        sigma = self._effective_sigma()
        self.last_effective_sigma = float(sigma)
        x = np.asarray(x, dtype=np.float32)
        if self.mode == "vanilla":
            out = denoise_one(x, self.denoiser, sigma)
            self.last_angles = ["none"]
        elif self.mode == "G16_fixed":
            estimates = []
            transformed = self.group.forward(x)
            for idx, tg_x in enumerate(transformed):
                d = denoise_one(tg_x, self.denoiser, sigma)
                estimates.append(self.group.invert(idx, d).astype(np.float32))
            out = np.mean(np.stack(estimates, axis=0), axis=0).astype(np.float32)
            self.last_angles = [float(a) for a in self.angles]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        self.call_count += 1
        return np.clip(out, 0.0, 1.0).astype(np.float32) if self.clip else out


def _red_step_value(step, lam, red_input_sigma):
    if step is not None:
        return float(step)
    data_scale = 1.0 / (float(red_input_sigma) ** 2)
    return 2.0 / (data_scale + float(lam))


def _run_one(clean, eval_mask, problem, denoise_fn, algorithm, num_iter, measurement_sigma,
             seed, rho, step, lam, red_input_sigma):
    clean = np.asarray(clean, dtype=np.float32)
    mask = np.asarray(eval_mask, dtype=np.float32)
    pixels = count_pixels(mask, clean)

    rng = np.random.default_rng(seed)
    y = problem.A(clean)
    if measurement_sigma and measurement_sigma > 0:
        y = y + rng.normal(0.0, measurement_sigma / 255.0, size=y.shape).astype(np.float32)

    trajectory = []

    def _record(k, xk):
        se = l2sq(xk, clean, mask=mask) / pixels
        trajectory.append({
            "iteration": int(k) + 1,
            "image": np.asarray(xk, dtype=np.float32).copy(),
            "se": float(se),
            "psnr": float(se_to_psnr(se)),
            "ssim": masked_ssim(xk, clean, mask=mask),
            "effective_denoiser_sigma": float(denoise_fn.last_effective_sigma),
            "sampled_angles": ";".join(
                str(a) if isinstance(a, str) else f"{a:.6g}"
                for a in denoise_fn.last_angles
            ),
        })

    final = de._restore(
        algorithm, y, problem, denoise_fn,
        num_iter=num_iter, rho=rho, step=step, lam=lam,
        data_step=0.4, prior_step=0.25, red_input_sigma=red_input_sigma,
        callback=_record,
    )
    degraded_se = l2sq(y, clean, mask=mask) / pixels
    final_se = l2sq(final, clean, mask=mask) / pixels
    return {
        "degraded": y,
        "final": final,
        "degraded_se": float(degraded_se),
        "degraded_psnr": float(se_to_psnr(degraded_se)),
        "degraded_ssim": masked_ssim(y, clean, mask=mask),
        "final_se": float(final_se),
        "final_psnr": float(se_to_psnr(final_se)),
        "final_ssim": masked_ssim(final, clean, mask=mask),
        "trajectory": trajectory,
    }


def _write_tables(save_dir, detail_rows, final_rows):
    detail = pd.DataFrame(detail_rows)
    final = pd.DataFrame(final_rows)
    summary = (
        final.groupby(
            [
                "dataset", "eval_mask", "input_pose", "input_rotate_expand",
                "problem", "algorithm", "denoiser",
                "train_sigma", "schedule", "rho", "step", "red_input_sigma",
                "lambda", "mode", "base_group_size", "num_iter",
            ],
            as_index=False,
        )
        .agg(
            mean_final_psnr=("final_psnr", "mean"),
            mean_degraded_psnr=("degraded_psnr", "mean"),
            mean_final_ssim=("final_ssim", "mean"),
            mean_degraded_ssim=("degraded_ssim", "mean"),
            mean_final_se=("final_se", "mean"),
            mean_degraded_se=("degraded_se", "mean"),
            fail_rate=("beats_degraded", lambda s: 1.0 - float(np.mean(s))),
            mean_gap_psnr=("gap_psnr", "mean"),
            mean_gap_ssim=("gap_ssim", "mean"),
            n=("file", "size"),
        )
    )
    summary["final_psnr_from_mean_se"] = summary["mean_final_se"].map(se_to_psnr)
    summary["degraded_psnr_from_mean_se"] = summary["mean_degraded_se"].map(se_to_psnr)
    detail.to_csv(save_dir / "downstream_solver_sweep_detail.csv", index=False)
    final.to_csv(save_dir / "downstream_solver_sweep_final.csv", index=False)
    summary.to_csv(save_dir / "downstream_solver_sweep_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/downstream/solver_sweep")
    ap.add_argument("--dataset", default="val_images")
    ap.add_argument("--eval-mask", default="none",
                    choices=["none", "content", "rectangle", "circle", "square"],
                    help="Mask used for PSNR/SSIM. Use circle for val_images_circle.")
    ap.add_argument("--denoisers", nargs="+", default=["wavelet", "tv", "restormer"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--train-sigma", type=float, default=15.0)
    ap.add_argument("--problems", nargs="+", default=["blur", "inpaint"], choices=["blur", "inpaint"])
    ap.add_argument("--algorithms", nargs="+", default=["pnp_hqs", "red_gd"], choices=["pnp_hqs", "red_gd"])
    ap.add_argument("--pnp-classical-schedules", nargs="+", default=["3", "5", "7.5", "10", "15", "15->3", "10->2"])
    ap.add_argument("--red-classical-schedules", nargs="+", default=["3.25", "4.1", "5"])
    ap.add_argument("--pnp-restormer-rhos", type=float, nargs="+", default=[0.2, 0.4, 0.8])
    ap.add_argument("--pnp-classical-rhos", type=float, nargs="+", default=[0.4, 0.8])
    ap.add_argument("--red-lambdas", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    ap.add_argument("--red-steps", type=float, nargs="+", default=None)
    ap.add_argument("--red-input-sigma", type=float, default=2 ** 0.5,
                    help="RED input noise sigma on the 0-255 scale, matching Google RED.")
    ap.add_argument("--modes", nargs="+", default=["vanilla", "G16_fixed"], choices=["vanilla", "G16_fixed"])
    ap.add_argument("--input-poses", nargs="+", default=["upright", "rot45_padded"],
                    choices=["upright", "rot45_padded"])
    ap.add_argument("--no-input-rotate-expand", action="store_true",
                    help="Rotate the 45-degree input on the original canvas.")
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-iter", type=int, default=20)
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--measurement-sigma", type=float, default=2 ** 0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-image-count", type=int, default=1)
    ap.add_argument("--snapshot-iters", type=int, nargs="*", default=[1, 2, 4, 8, 12, 16, 20])
    args = ap.parse_args()

    base = load_config(args.base)
    files = list_images(dataset_path(base, args.dataset))[:args.max_images]
    device = args.device or base.get("device", "cuda:0")
    save_dir = Path(args.save_dir)
    image_dir = save_dir / "intermediate_images"
    save_dir.mkdir(parents=True, exist_ok=True)

    detail_rows = []
    final_rows = []
    for denoiser_name in args.denoisers:
        denoiser = _make_model(denoiser_name, base, device, args.train_sigma)
        for algorithm in args.algorithms:
            if algorithm == "pnp_hqs":
                if denoiser_name in {"restormer", "restormer-aug"}:
                    schedules = [{"kind": "fixed", "start": args.train_sigma, "end": args.train_sigma}]
                    rhos = args.pnp_restormer_rhos
                else:
                    schedules = [_parse_schedule(s) for s in args.pnp_classical_schedules]
                    rhos = args.pnp_classical_rhos
                steps = [0.5]
                lams = [0.15]
            else:
                if denoiser_name in {"restormer", "restormer-aug"}:
                    schedules = [{"kind": "fixed", "start": args.train_sigma, "end": args.train_sigma}]
                else:
                    schedules = [_parse_schedule(s) for s in args.red_classical_schedules]
                rhos = [0.8]
                steps = args.red_steps if args.red_steps is not None else [None]
                lams = args.red_lambdas

            for problem_name in args.problems:
                for schedule in schedules:
                    schedule_name = _schedule_label(schedule)
                    for rho in rhos:
                        for step in steps:
                            for lam in lams:
                                for image_index, path in enumerate(files):
                                    file_name = os.path.basename(path)
                                    stem = Path(file_name).stem
                                    source_clean = load_image(path)
                                    for input_pose, input_angle, clean, eval_mask in _input_variants(
                                        source_clean,
                                        args.group_name,
                                        args.eval_mask,
                                        input_rotate_expand=not args.no_input_rotate_expand,
                                    ):
                                        if input_pose not in args.input_poses:
                                            continue
                                        problem_seed = args.seed + image_index * 1000003
                                        if input_pose == "rot45_padded":
                                            problem_seed += 45000
                                        problem = de._make_problem(problem_name, clean.shape, seed=problem_seed)
                                        group_expand = input_pose != "rot45_padded"
                                        for mode in args.modes:
                                            runner = ScheduledDenoiser(
                                                denoiser, mode, denoiser_name, args.train_sigma, schedule,
                                                args.group_name, args.group_size, group_expand, args.num_iter,
                                            )
                                            step_value = _red_step_value(step, lam, args.red_input_sigma) if algorithm == "red_gd" else float(step)
                                            print(
                                                f"[solver-sweep] denoiser={denoiser_name} problem={problem_name} "
                                                f"algorithm={algorithm} schedule={schedule_name} rho={rho} "
                                                f"step={step_value:.6g} lam={lam} image={file_name} input={input_pose} mode={mode}"
                                            )
                                            res = _run_one(
                                                clean, eval_mask, problem, runner, algorithm,
                                                num_iter=args.num_iter,
                                                measurement_sigma=args.measurement_sigma,
                                                seed=args.seed + image_index * 1000003 + 17 + int(round(input_angle)) * 997,
                                                rho=rho, step=step, lam=lam,
                                                red_input_sigma=args.red_input_sigma,
                                            )
                                            if image_index < args.save_image_count:
                                                prefix = (
                                                    image_dir / denoiser_name / problem_name / algorithm /
                                                    schedule_name / f"rho_{_tag(rho)}_step_{_tag(step_value)}_lam_{_tag(lam)}" /
                                                    input_pose / stem / mode
                                                )
                                                _write_gray(prefix / "degraded.png", res["degraded"])
                                                _write_gray(prefix / "final.png", res["final"])

                                            common = {
                                                "dataset": args.dataset,
                                                "eval_mask": args.eval_mask,
                                                "file": file_name,
                                                "image_index": image_index,
                                                "input_pose": input_pose,
                                                "input_angle_deg": input_angle,
                                                "input_rotate_expand": not args.no_input_rotate_expand,
                                                "input_height": clean.shape[0],
                                                "input_width": clean.shape[1],
                                                "eval_mask_pixels": count_pixels(eval_mask, clean),
                                                "eval_mask_fraction": float(np.mean(eval_mask)),
                                                "problem": problem_name,
                                                "algorithm": algorithm,
                                                "denoiser": denoiser_name,
                                                "train_sigma": args.train_sigma,
                                                "schedule": schedule_name,
                                                "rho": float(rho),
                                                "step": float(step_value),
                                                "red_input_sigma": float(args.red_input_sigma) if algorithm == "red_gd" else 0.0,
                                                "lambda": float(lam),
                                                "mode": mode,
                                                "group_expand": group_expand,
                                                "base_group_size": args.group_size if mode == "G16_fixed" else 0,
                                                "num_iter": args.num_iter,
                                                "degraded_se": res["degraded_se"],
                                                "degraded_psnr": res["degraded_psnr"],
                                                "degraded_ssim": res["degraded_ssim"],
                                            }
                                            for item in res["trajectory"]:
                                                row = dict(common)
                                                row.update({
                                                    "iteration": item["iteration"],
                                                    "se": item["se"],
                                                    "psnr": item["psnr"],
                                                    "ssim": item["ssim"],
                                                    "effective_denoiser_sigma": item["effective_denoiser_sigma"],
                                                    "sampled_angles": item["sampled_angles"],
                                                })
                                                detail_rows.append(row)
                                                if image_index < args.save_image_count and item["iteration"] in args.snapshot_iters:
                                                    _write_gray(prefix / f"iter_{item['iteration']:03d}.png", item["image"])

                                            final = dict(common)
                                            final.update({
                                                "final_se": res["final_se"],
                                                "final_psnr": res["final_psnr"],
                                                "final_ssim": res["final_ssim"],
                                                "gap_psnr": res["final_psnr"] - res["degraded_psnr"],
                                                "gap_ssim": res["final_ssim"] - res["degraded_ssim"],
                                                "beats_degraded": res["final_psnr"] > res["degraded_psnr"],
                                                "total_denoiser_calls": args.num_iter,
                                                "total_group_evals": args.num_iter * (args.group_size if mode == "G16_fixed" else 1),
                                            })
                                            final_rows.append(final)
                                            _write_tables(save_dir, detail_rows, final_rows)

    _write_tables(save_dir, detail_rows, final_rows)
    print(f"wrote {save_dir / 'downstream_solver_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
