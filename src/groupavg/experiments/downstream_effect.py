"""Downstream effect of denoiser non-equivariance in restoration pipelines.

This module measures whether replacing a denoiser D by its orbit-averaged
version D_G changes downstream restoration behavior in PnP / RED /
diffusion-style iterations.

For a restoration algorithm R, a degradation A, and rotation T_alpha, we compare

    R(Ax)                         vs.   T_alpha^{-1} R(A T_alpha x)

after using either vanilla D or group-averaged D_G inside R. The resulting
downstream equivariance residual quantifies how denoiser non-equivariance
propagates into iterative inverse-problem solvers.
"""
import os

import cv2
import numpy as np
import pandas as pd

from ..data import list_images, load_image
from ..masks import build_mask, count_pixels
from ..metrics import l2sq, se_to_psnr
from ..pipeline import denoise_one, orbit_average
from ..registry import make_group


def _fft2(x):
    return np.fft.fft2(np.asarray(x, dtype=np.float32))


def _ifft2(x):
    return np.fft.ifft2(x).real.astype(np.float32)


def _gaussian_kernel(size=9, sigma=1.6):
    if size % 2 != 1:
        raise ValueError("Gaussian kernel size must be odd.")
    ax = np.arange(size, dtype=np.float32) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)


def _psf_to_otf(psf, shape):
    padded = np.zeros(shape, dtype=np.float32)
    h, w = psf.shape
    padded[:h, :w] = psf
    padded = np.roll(padded, -h // 2, axis=0)
    padded = np.roll(padded, -w // 2, axis=1)
    return np.fft.fft2(padded)


class BlurProblem:
    name = "blur"

    def __init__(self, shape, blur_size=9, blur_sigma=1.6):
        self.shape = tuple(shape)
        self.kernel = _gaussian_kernel(blur_size, blur_sigma)
        self.H = _psf_to_otf(self.kernel, self.shape)
        self.H_conj = np.conj(self.H)
        self.H_abs2 = np.abs(self.H) ** 2

    def A(self, x):
        return _ifft2(self.H * _fft2(x))

    def AT(self, y):
        return _ifft2(self.H_conj * _fft2(y))

    def prox_data(self, y, z, rho):
        rhs = self.H_conj * _fft2(y) + float(rho) * _fft2(z)
        return _ifft2(rhs / (self.H_abs2 + float(rho)))


class InpaintProblem:
    name = "inpaint"

    def __init__(self, shape, keep_fraction=0.5, seed=0):
        rng = np.random.default_rng(seed)
        self.mask = (rng.random(shape) < keep_fraction).astype(np.float32)

    def A(self, x):
        return np.asarray(x, dtype=np.float32) * self.mask

    def AT(self, y):
        return np.asarray(y, dtype=np.float32) * self.mask

    def prox_data(self, y, z, rho):
        return ((self.mask * y) + float(rho) * z) / (self.mask + float(rho))


def _make_problem(name, shape, seed=0, blur_size=9, blur_sigma=1.6, keep_fraction=0.5):
    if name == "blur":
        return BlurProblem(shape, blur_size=blur_size, blur_sigma=blur_sigma)
    if name == "inpaint":
        return InpaintProblem(shape, keep_fraction=keep_fraction, seed=seed)
    raise ValueError(f"Unknown downstream problem: {name!r}")


def _make_denoise_fn(
    denoiser,
    denoiser_sigma,
    mode,
    group_name="fourier_rotation",
    group_size=8,
    expand=True,
    clip=True,
):
    if mode == "vanilla":
        def _fn(x):
            out = denoise_one(x, denoiser, denoiser_sigma)
            return np.clip(out, 0.0, 1.0) if clip else out
        return _fn

    if mode == "group_avg":
        group = make_group(group_name, K=group_size, expand=expand)

        def _fn(x):
            out = orbit_average(x, group, denoiser, noise_sigma=denoiser_sigma, clip=clip)
            return np.clip(out, 0.0, 1.0) if clip else out
        return _fn

    raise ValueError(f"Unknown denoiser mode: {mode!r}")


def _pnp_hqs(y, problem, denoise_fn, num_iter=8, rho=0.8):
    x = np.asarray(y, dtype=np.float32).copy()
    z = x.copy()
    for _ in range(num_iter):
        x = problem.prox_data(y, z, rho=rho)
        x = np.clip(x, 0.0, 1.0)
        z = denoise_fn(x)
    return np.clip(z, 0.0, 1.0).astype(np.float32)


def _red_gd(y, problem, denoise_fn, num_iter=20, step=0.5, lam=0.15):
    x = np.asarray(y, dtype=np.float32).copy()
    for _ in range(num_iter):
        data_grad = problem.AT(problem.A(x) - y)
        prior_grad = x - denoise_fn(x)
        x = x - float(step) * (data_grad + float(lam) * prior_grad)
        x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def _diffusion_style(y, problem, denoise_fn, num_iter=20, data_step=0.4, prior_step=0.25):
    """A lightweight denoising-score style restoration loop.

    This is not a full DDPM sampler. It isolates the downstream effect of using
    D vs D_G in score-like updates by alternating a denoising drift
    (D(x)-x) with a data-consistency gradient.
    """
    x = np.asarray(y, dtype=np.float32).copy()
    for _ in range(num_iter):
        score_drift = denoise_fn(x) - x
        data_grad = problem.AT(problem.A(x) - y)
        x = x + float(prior_step) * score_drift - float(data_step) * data_grad
        x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def _restore(algorithm, y, problem, denoise_fn, num_iter, rho, step, lam, data_step, prior_step):
    if algorithm == "pnp_hqs":
        return _pnp_hqs(y, problem, denoise_fn, num_iter=num_iter, rho=rho)
    if algorithm == "red_gd":
        return _red_gd(y, problem, denoise_fn, num_iter=num_iter, step=step, lam=lam)
    if algorithm == "diffusion_style":
        return _diffusion_style(
            y,
            problem,
            denoise_fn,
            num_iter=num_iter,
            data_step=data_step,
            prior_step=prior_step,
        )
    raise ValueError(f"Unknown downstream algorithm: {algorithm!r}")


def _rotate_pair(clean, angle_deg, group_name, expand):
    group = make_group(group_name, K=1, angles=[float(angle_deg)], expand=expand)
    clean_rot = group.forward(clean)[0]
    return group, clean_rot


def run(
    dataset_dir,
    denoiser,
    denoiser_name="restormer",
    denoiser_sigma=15.0,
    algorithms=("pnp_hqs", "red_gd", "diffusion_style"),
    problems=("blur",),
    denoiser_modes=("vanilla", "group_avg"),
    group_name="fourier_rotation",
    group_size=8,
    angles_deg=(0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0),
    max_images=None,
    seed=0,
    measurement_sigma=2.0,
    num_iter=12,
    rho=0.8,
    step=0.5,
    lam=0.15,
    data_step=0.4,
    prior_step=0.25,
    blur_size=9,
    blur_sigma=1.6,
    keep_fraction=0.5,
    se_mask="content",
    expand=True,
    save_dir="results",
    save_csv=True,
    verbose=True,
):
    files = list_images(dataset_dir)
    if max_images is not None:
        files = files[:max_images]
    os.makedirs(save_dir, exist_ok=True)
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))

    rows = []
    for fi, path in enumerate(files):
        file_name = os.path.basename(path)
        clean = load_image(path)
        base_mask = build_mask(clean, clean_ref=clean, mask_mode=se_mask)
        base_pixels = count_pixels(base_mask, clean)

        for problem_name in problems:
            problem_seed = seed + fi * 1000003
            problem = _make_problem(
                problem_name,
                clean.shape,
                seed=problem_seed,
                blur_size=blur_size,
                blur_sigma=blur_sigma,
                keep_fraction=keep_fraction,
            )
            rng = np.random.default_rng(seed + fi * 1000003 + 17)
            y = problem.A(clean)
            if measurement_sigma and measurement_sigma > 0:
                y = y + rng.normal(0.0, measurement_sigma / 255.0, size=y.shape).astype(np.float32)
            y = np.clip(y, 0.0, 1.0)

            for mode in denoiser_modes:
                denoise_fn = _make_denoise_fn(
                    denoiser,
                    denoiser_sigma,
                    mode,
                    group_name=group_name,
                    group_size=group_size,
                    expand=expand,
                )
                for algorithm in algorithms:
                    if verbose:
                        print(
                            f"[Downstream] {file_name} problem={problem_name} "
                            f"algorithm={algorithm} denoiser={denoiser_name}/{mode}"
                        )
                    base_rec = _restore(
                        algorithm,
                        y,
                        problem,
                        denoise_fn,
                        num_iter=num_iter,
                        rho=rho,
                        step=step,
                        lam=lam,
                        data_step=data_step,
                        prior_step=prior_step,
                    )
                    base_se = l2sq(base_rec, clean, mask=base_mask) / base_pixels

                    for angle_deg in angles_deg:
                        inv_group, clean_rot = _rotate_pair(clean, angle_deg, group_name, expand)
                        rot_mask = build_mask(clean_rot, clean_ref=clean_rot, mask_mode=se_mask)
                        rot_pixels = count_pixels(rot_mask, clean_rot)
                        rot_problem = _make_problem(
                            problem_name,
                            clean_rot.shape,
                            seed=problem_seed,
                            blur_size=blur_size,
                            blur_sigma=blur_sigma,
                            keep_fraction=keep_fraction,
                        )
                        rot_rng = np.random.default_rng(seed + fi * 1000003 + 17)
                        y_rot = rot_problem.A(clean_rot)
                        if measurement_sigma and measurement_sigma > 0:
                            y_rot = y_rot + rot_rng.normal(
                                0.0, measurement_sigma / 255.0, size=y_rot.shape
                            ).astype(np.float32)
                        y_rot = np.clip(y_rot, 0.0, 1.0)

                        rot_rec = _restore(
                            algorithm,
                            y_rot,
                            rot_problem,
                            denoise_fn,
                            num_iter=num_iter,
                            rho=rho,
                            step=step,
                            lam=lam,
                            data_step=data_step,
                            prior_step=prior_step,
                        )
                        rot_se = l2sq(rot_rec, clean_rot, mask=rot_mask) / rot_pixels
                        aligned = inv_group.invert(0, rot_rec)
                        downstream_residual = l2sq(aligned, base_rec, mask=base_mask) / base_pixels

                        rows.append({
                            "dataset": dataset_name,
                            "file": file_name,
                            "problem": problem_name,
                            "algorithm": algorithm,
                            "denoiser": denoiser_name,
                            "denoiser_mode": mode,
                            "denoiser_sigma": denoiser_sigma,
                            "measurement_sigma": measurement_sigma,
                            "group": group_name,
                            "group_size": group_size if mode == "group_avg" else 0,
                            "angle_deg": float(angle_deg),
                            "base_se": base_se,
                            "base_psnr": se_to_psnr(base_se),
                            "rotated_se": rot_se,
                            "rotated_psnr": se_to_psnr(rot_se),
                            "downstream_equivariance_mse": downstream_residual,
                            "downstream_equivariance_psnr": se_to_psnr(downstream_residual),
                            "num_iter": num_iter,
                            "rho": rho,
                            "step": step,
                            "lambda": lam,
                            "data_step": data_step,
                            "prior_step": prior_step,
                        })

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            [
                "dataset",
                "problem",
                "algorithm",
                "denoiser",
                "denoiser_mode",
                "denoiser_sigma",
                "measurement_sigma",
                "group",
                "group_size",
                "angle_deg",
            ],
            as_index=False,
        )
        .agg(
            mean_base_se=("base_se", "mean"),
            mean_base_psnr=("base_psnr", "mean"),
            mean_rotated_se=("rotated_se", "mean"),
            mean_rotated_psnr=("rotated_psnr", "mean"),
            mean_downstream_equivariance_mse=("downstream_equivariance_mse", "mean"),
            mean_downstream_equivariance_psnr=("downstream_equivariance_psnr", "mean"),
            n=("file", "size"),
        )
    )
    summary["base_psnr_from_mean_se"] = summary["mean_base_se"].map(se_to_psnr)
    summary["rotated_psnr_from_mean_se"] = summary["mean_rotated_se"].map(se_to_psnr)
    summary["downstream_equivariance_psnr_from_mean_mse"] = summary[
        "mean_downstream_equivariance_mse"
    ].map(se_to_psnr)

    if save_csv:
        tag = f"downstream_dataset-{dataset_name}_denoiser-{denoiser_name}_sigma-{denoiser_sigma}"
        detail.to_csv(os.path.join(save_dir, f"{tag}_detail.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, f"{tag}_summary.csv"), index=False)

    return {"detail": detail, "summary": summary}
