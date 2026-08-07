#!/usr/bin/env python
"""Pure denoising sanity check for circle-image downstream settings.

This isolates whether Restormer benefits from orbit averaging before it is
placed inside PnP/RED. The default protocol mirrors the current downstream
circle experiment: val_images_circle, 100px zero padding, sigma=15 measurement
noise, no canvas expansion for the input rotations, and circle-mask metrics.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import dataset_path, load_config, restormer_weights  # noqa: E402
from groupavg.data import list_images, load_image  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.masks import build_mask, count_pixels  # noqa: E402
from groupavg.metrics import l2sq, masked_ssim, se_to_psnr  # noqa: E402
from groupavg.pipeline import denoise_one, orbit_average  # noqa: E402
from groupavg.registry import make_group  # noqa: E402


RESTORMER_AUG_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)


def _tag(text):
    return str(text).replace("-", "_").replace(".", "p")


def _pad(x, pad):
    if int(pad) <= 0:
        return np.asarray(x, dtype=np.float32)
    return np.pad(np.asarray(x, dtype=np.float32), int(pad), mode="constant")


def _write_gray(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(img), 0.0, 1.0)
    cv2.imwrite(str(path), (arr * 255.0 + 0.5).astype(np.uint8))


def _make_model(name, base, device, sigma):
    if name == "restormer":
        weights = restormer_weights(base, sigma=sigma, color=False)
    elif name == "restormer-aug":
        if float(sigma) != 15.0:
            raise ValueError("restormer-aug is only configured for sigma=15")
        weights = RESTORMER_AUG_WEIGHTS
    else:
        raise ValueError(f"Unsupported model for this sanity check: {name!r}")
    if not os.path.exists(weights):
        raise FileNotFoundError(weights)
    return make_denoiser("restormer", weights=weights, color=False, device=device)


def _input_variants(clean, mask, angles, expand):
    variants = [("upright", 0.0, clean, mask)]
    for angle in angles:
        group = make_group("fourier_rotation", K=1, angles=[float(angle)], expand=expand)
        rot_clean = group.forward(clean)[0].astype(np.float32)
        rot_mask = (group.forward(mask)[0] > 0.5).astype(np.float32)
        variants.append((f"rot{_tag(f'{float(angle):g}')}_padded", float(angle), rot_clean, rot_mask))
    return variants


def _group_expand_for_pose(input_pose, policy):
    if policy == "downstream":
        return input_pose == "upright"
    if policy == "never":
        return False
    if policy == "always":
        return True
    raise ValueError(f"Unknown group expand policy: {policy!r}")


def _metric_rows(clean, noisy, vanilla, group_avg, mask):
    pixels = count_pixels(mask, clean)
    noisy_se = l2sq(noisy, clean, mask=mask) / pixels
    vanilla_se = l2sq(vanilla, clean, mask=mask) / pixels
    group_se = l2sq(group_avg, clean, mask=mask) / pixels
    diff_se = l2sq(group_avg, vanilla, mask=mask) / pixels
    return {
        "noisy_psnr": se_to_psnr(noisy_se),
        "noisy_ssim": masked_ssim(noisy, clean, mask=mask),
        "vanilla_psnr": se_to_psnr(vanilla_se),
        "vanilla_ssim": masked_ssim(vanilla, clean, mask=mask),
        "group_avg_psnr": se_to_psnr(group_se),
        "group_avg_ssim": masked_ssim(group_avg, clean, mask=mask),
        "group_avg_minus_vanilla_psnr": se_to_psnr(group_se) - se_to_psnr(vanilla_se),
        "group_avg_minus_vanilla_ssim": masked_ssim(group_avg, clean, mask=mask)
        - masked_ssim(vanilla, clean, mask=mask),
        "group_avg_vs_vanilla_mse": diff_se,
        "group_avg_vs_vanilla_psnr": se_to_psnr(diff_se),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--save-dir", default="results/denoising_circle_sanity")
    ap.add_argument("--dataset", default="val_images_circle")
    ap.add_argument("--models", nargs="+", default=["restormer", "restormer-aug"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--input-pad", type=int, default=100)
    ap.add_argument("--angles", type=float, nargs="+", default=[30.0, 45.0, 60.0])
    ap.add_argument("--input-rotate-expand", action="store_true")
    ap.add_argument("--group-expand-policy", default="downstream",
                    choices=["downstream", "never", "always"])
    ap.add_argument("--group-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-image-count", type=int, default=1)
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")
    files = list_images(dataset_path(base, args.dataset))[: args.max_images]
    save_dir = Path(args.save_dir)
    image_dir = save_dir / "images"
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in args.models:
        denoiser = _make_model(model_name, base, device, args.sigma)
        for image_index, path in enumerate(files):
            clean0 = load_image(path)
            mask0 = build_mask(clean0, clean_ref=clean0, mask_mode="circle")
            clean_base = _pad(clean0, args.input_pad)
            mask_base = _pad(mask0, args.input_pad)
            variants = _input_variants(
                clean_base,
                mask_base,
                angles=args.angles,
                expand=args.input_rotate_expand,
            )
            for input_pose, angle, clean, mask in variants:
                rng = np.random.default_rng(args.seed + image_index * 1000003 + int(round(angle)) * 997)
                noisy = clean + rng.normal(0.0, args.sigma / 255.0, size=clean.shape).astype(np.float32)
                group_expand = _group_expand_for_pose(input_pose, args.group_expand_policy)
                group = make_group(
                    "fourier_rotation",
                    K=args.group_size,
                    expand=group_expand,
                )
                print(
                    f"[denoising-sanity] model={model_name} image={os.path.basename(path)} "
                    f"pose={input_pose} group_expand={group_expand}"
                )
                vanilla = denoise_one(noisy, denoiser, args.sigma).astype(np.float32)
                group_avg = orbit_average(
                    noisy,
                    group,
                    denoiser,
                    noise_sigma=args.sigma,
                    clip=False,
                ).astype(np.float32)
                row = {
                    "dataset": args.dataset,
                    "file": os.path.basename(path),
                    "image_index": image_index,
                    "model": model_name,
                    "sigma": args.sigma,
                    "input_pose": input_pose,
                    "input_angle_deg": angle,
                    "input_pad": args.input_pad,
                    "input_rotate_expand": args.input_rotate_expand,
                    "input_height": clean.shape[0],
                    "input_width": clean.shape[1],
                    "eval_mask": "circle",
                    "eval_mask_pixels": count_pixels(mask, clean),
                    "eval_mask_fraction": float(np.mean(mask)),
                    "group": "fourier_rotation",
                    "group_size": args.group_size,
                    "group_expand": group_expand,
                    "group_expand_policy": args.group_expand_policy,
                }
                row.update(_metric_rows(clean, noisy, vanilla, group_avg, mask))
                rows.append(row)

                if image_index < args.save_image_count:
                    prefix = image_dir / model_name / input_pose / Path(path).stem
                    _write_gray(prefix / "clean.png", clean)
                    _write_gray(prefix / "noisy.png", noisy)
                    _write_gray(prefix / "vanilla.png", vanilla)
                    _write_gray(prefix / "group_avg.png", group_avg)

                detail = pd.DataFrame(rows)
                detail.to_csv(save_dir / "denoising_circle_sanity_detail.csv", index=False)

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(
            [
                "dataset",
                "model",
                "sigma",
                "input_pose",
                "input_angle_deg",
                "input_pad",
                "input_rotate_expand",
                "eval_mask",
                "group",
                "group_size",
                "group_expand",
                "group_expand_policy",
            ],
            as_index=False,
        )
        .agg(
            mean_noisy_psnr=("noisy_psnr", "mean"),
            mean_noisy_ssim=("noisy_ssim", "mean"),
            mean_vanilla_psnr=("vanilla_psnr", "mean"),
            mean_vanilla_ssim=("vanilla_ssim", "mean"),
            mean_group_avg_psnr=("group_avg_psnr", "mean"),
            mean_group_avg_ssim=("group_avg_ssim", "mean"),
            mean_group_avg_minus_vanilla_psnr=("group_avg_minus_vanilla_psnr", "mean"),
            mean_group_avg_minus_vanilla_ssim=("group_avg_minus_vanilla_ssim", "mean"),
            mean_group_avg_vs_vanilla_mse=("group_avg_vs_vanilla_mse", "mean"),
            mean_group_avg_vs_vanilla_psnr=("group_avg_vs_vanilla_psnr", "mean"),
            n=("file", "size"),
        )
    )
    detail.to_csv(save_dir / "denoising_circle_sanity_detail.csv", index=False)
    summary.to_csv(save_dir / "denoising_circle_sanity_summary.csv", index=False)
    print(f"wrote {save_dir / 'denoising_circle_sanity_summary.csv'}")


if __name__ == "__main__":
    main()
