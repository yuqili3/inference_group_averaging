"""Figure 5 style PSNR-vs-rotation-angle experiment.

For each external rotation angle alpha, compare:
  solid  : f(T_alpha(x+n)) against T_alpha x
  dashed : orbit-averaged f_G(T_alpha(x+n)) against T_alpha x

The orbit-averaged estimate is computed once with |G|=32 and then subsampled to
report G=4,8,16,32 without rerunning the denoiser.
"""
import os

import numpy as np
import pandas as pd

from ..data import list_images, load_image
from ..group_operators import UpsampleGroup
from ..masks import build_mask, count_pixels
from ..metrics import l2sq, se_to_psnr
from ..pipeline import denoise_one
from ..registry import make_group


def _subset_for_group(z_list, group_size):
    if group_size > len(z_list) or len(z_list) % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide base group size {len(z_list)}")
    stride = len(z_list) // group_size
    return z_list[::stride]


def run(
    dataset_dir,
    denoiser,
    denoiser_name="restormer",
    rotation_group_name="fourier_rotation",
    base_group_size=32,
    eval_group_sizes=(4, 8, 16, 32),
    angles_deg=None,
    upsample=1.0,
    noise_sigma=15.0,
    num_noise=2,
    noise_mask="none",
    se_mask="content",
    seed=0,
    max_images=None,
    clip_noisy=False,
    clip_denoised=True,
    expand=True,
    orbit_expand=False,
    save_dir="results",
    save_csv=True,
    verbose=True,
):
    files = list_images(dataset_dir)
    if max_images is not None:
        files = files[:max_images]
    if angles_deg is None:
        angles_deg = np.arange(0.0, 180.0, 5.0, dtype=np.float32)
    angles_deg = [float(a) for a in angles_deg]
    eval_group_sizes = [int(g) for g in eval_group_sizes]

    os.makedirs(save_dir, exist_ok=True)
    dataset_name = os.path.basename(dataset_dir.rstrip("/"))
    scale_group = UpsampleGroup(scales=[upsample])
    orbit_group = make_group(rotation_group_name, K=base_group_size, expand=orbit_expand)

    detail_rows = []
    for fi, path in enumerate(files):
        file_name = os.path.basename(path)
        clean = load_image(path)
        for noise_id in range(num_noise):
            rng = np.random.default_rng(seed + fi * 1000003 + noise_id * 9176)
            noise = rng.normal(0.0, noise_sigma / 255.0, size=clean.shape).astype(np.float32)
            if noise_mask not in (None, "none"):
                noise = noise * build_mask(clean, clean_ref=clean, mask_mode=noise_mask)
            noisy = clean + noise
            if clip_noisy:
                noisy = np.clip(noisy, 0.0, 1.0)

            clean_up = scale_group.forward(clean)[0]
            noisy_up = scale_group.forward(noisy)[0]
            for angle_id, angle_deg in enumerate(angles_deg):
                if verbose:
                    print(
                        f"[Figure5] {denoiser_name} sigma={noise_sigma} "
                        f"{fi + 1}/{len(files)} noise={noise_id + 1}/{num_noise} "
                        f"angle={angle_deg:g}"
                    )
                angle_group = make_group(rotation_group_name, K=1, angles=[angle_deg], expand=expand)
                clean_rot = angle_group.forward(clean_up)[0]
                noisy_rot = angle_group.forward(noisy_up)[0]
                rot_mask = build_mask(clean_rot, clean_ref=clean_rot, mask_mode=se_mask)
                rot_pixels = count_pixels(rot_mask, clean_rot)

                vanilla = denoise_one(noisy_rot, denoiser, noise_sigma)
                if clip_denoised:
                    vanilla = np.clip(vanilla, 0.0, 1.0)
                vanilla_se = l2sq(vanilla, clean_rot, mask=rot_mask) / rot_pixels
                detail_rows.append({
                    "dataset": dataset_name,
                    "file": file_name,
                    "noise_id": noise_id,
                    "denoiser": denoiser_name,
                    "noise_sigma": noise_sigma,
                    "angle_index": angle_id,
                    "angle_deg": angle_deg,
                    "estimator": "vanilla",
                    "group_size": 0,
                    "se": vanilla_se,
                    "psnr": se_to_psnr(vanilla_se),
                    "num_pixels": rot_pixels,
                })

                orbit_inputs = orbit_group.forward(noisy_rot)
                z_list = []
                for gi, orbit_input in enumerate(orbit_inputs):
                    denoised = denoise_one(orbit_input, denoiser, noise_sigma)
                    if clip_denoised:
                        denoised = np.clip(denoised, 0.0, 1.0)
                    z = orbit_group.invert(gi, denoised).astype(np.float32)
                    if clip_denoised:
                        z = np.clip(z, 0.0, 1.0)
                    z_list.append(z)

                for group_size in eval_group_sizes:
                    avg = np.mean(np.stack(_subset_for_group(z_list, group_size), axis=0), axis=0)
                    if clip_denoised:
                        avg = np.clip(avg, 0.0, 1.0)
                    avg_se = l2sq(avg, clean_rot, mask=rot_mask) / rot_pixels
                    detail_rows.append({
                        "dataset": dataset_name,
                        "file": file_name,
                        "noise_id": noise_id,
                        "denoiser": denoiser_name,
                        "noise_sigma": noise_sigma,
                        "angle_index": angle_id,
                        "angle_deg": angle_deg,
                        "estimator": "group_avg",
                        "group_size": group_size,
                        "se": avg_se,
                        "psnr": se_to_psnr(avg_se),
                        "num_pixels": rot_pixels,
                    })

    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(
            ["dataset", "denoiser", "noise_sigma", "angle_index", "angle_deg", "estimator", "group_size"],
            as_index=False,
        )
        .agg(mean_se=("se", "mean"), mean_psnr=("psnr", "mean"), n=("psnr", "size"))
    )
    summary["psnr_from_mean_se"] = summary["mean_se"].map(se_to_psnr)

    if save_csv:
        tag = (
            f"figure5_dataset-{dataset_name}_denoiser-{denoiser_name}"
            f"_sigma-{noise_sigma}_G-{base_group_size}"
        )
        detail.to_csv(os.path.join(save_dir, f"{tag}_detail.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, f"{tag}_summary.csv"), index=False)

    return {"detail": detail, "summary": summary}
