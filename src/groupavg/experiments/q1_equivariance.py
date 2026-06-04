"""Q1 — Are common denoisers non-equivariant, and by how much?

For a denoiser D and group G we measure the equivariance residual of the orbit
estimate against the direct estimate:

    r_g(y) = T_g^{-1} D(T_g y) - D(y) ,     y = x + n   (or y = x in clean mode)

and report, on the masked content:
    rel_err_g = ||r_g||^2 / ||D(y)||^2          (per group element)
    mean_rel_err = E_g rel_err_g               (scalar non-equivariance score)
    e1 = E_g||r_g||^2 - ||E_g r_g||^2          (orbit variance — the averageable part)

A perfectly equivariant denoiser gives rel_err_g == 0 for all g. Sweeping sigma
(including ~0) separates genuine non-equivariance from interpolation error.

Writes per_image and per_element CSVs; returns both as DataFrames.
"""
import os

import numpy as np
import pandas as pd

from ..registry import make_group
from ..group_operators import UpsampleGroup
from ..pipeline import denoise_one
from ..masks import build_mask, count_pixels
from ..metrics import l2sq


def run(
    dataset_dir,
    denoiser,
    denoiser_name="restormer",
    group_name="fourier_rotation",
    averaging=8,
    upsample=1.0,
    noise_sigma=15.0,
    num_noise=4,
    noise_mask="none",
    se_mask="content",
    clean_mode=False,
    seed=0,
    max_images=None,
    expand=True,
    save_dir="results",
    save_csv=True,
    verbose=True,
):
    from ..data import list_images, load_image

    files = list_images(dataset_dir)
    if max_images is not None:
        files = files[:max_images]
    os.makedirs(save_dir, exist_ok=True)

    group = make_group(group_name, K=averaging, expand=expand)
    scale_group = UpsampleGroup(scales=[upsample])
    ops = list(group.ops) if hasattr(group, "ops") else list(range(averaging))

    rows, elem_rows = [], []
    for fi, f in enumerate(files):
        file_name = os.path.basename(f)
        if verbose:
            print(f"[Q1] {fi + 1}/{len(files)}: {file_name}")
        clean = load_image(f)
        x_mask = build_mask(clean, clean_ref=clean, mask_mode=se_mask)

        n_draws = 1 if clean_mode else num_noise
        per_elem_rel = np.zeros(len(ops), dtype=np.float64)
        mean_rel_acc, e1_acc = [], []

        for r in range(n_draws):
            rng = np.random.default_rng(seed + fi * 1000003 + r * 9176)
            if clean_mode:
                y = clean.copy()
            else:
                noise = rng.normal(0.0, noise_sigma / 255.0, size=clean.shape).astype(np.float32)
                if noise_mask not in (None, "none"):
                    noise = noise * build_mask(clean, clean_ref=clean, mask_mode=noise_mask)
                y = clean + noise

            D_y = denoise_one(y, denoiser, noise_sigma)
            denom = max(l2sq(D_y, mask=x_mask), 1e-12)

            ys = scale_group.forward(y)[0]
            ty = group.forward(ys)
            r_list = []
            for gi, tg_y in enumerate(ty):
                z = scale_group.invert(0, group.invert(gi, denoise_one(tg_y, denoiser, noise_sigma)))
                z = z.astype(np.float32)
                rg = z - D_y
                r_list.append(rg)
                rel = l2sq(rg, mask=x_mask) / denom
                per_elem_rel[gi] += rel / n_draws
                elem_rows.append({
                    "file": file_name, "denoiser": denoiser_name, "group": group_name,
                    "noise_sigma": (0.0 if clean_mode else noise_sigma),
                    "g_index": gi, "op": ops[gi], "rel_err": rel,
                })

            rel_each = [l2sq(rg, mask=x_mask) / denom for rg in r_list]
            mean_rel_acc.append(float(np.mean(rel_each)))
            r_mean = np.mean(np.stack(r_list, axis=0), axis=0)
            e1_raw = float(np.mean([l2sq(rg, mask=x_mask) for rg in r_list])) - l2sq(r_mean, mask=x_mask)
            e1_acc.append(e1_raw / count_pixels(x_mask))

        rows.append({
            "file": file_name, "denoiser": denoiser_name, "group": group_name,
            "noise_sigma": (0.0 if clean_mode else noise_sigma),
            "averaging": averaging, "upsample": upsample,
            "mean_rel_err": float(np.mean(mean_rel_acc)),
            "max_elem_rel_err": float(np.max(per_elem_rel)),
            "e1": float(np.mean(e1_acc)),
        })

    per_image = pd.DataFrame(rows)
    per_element = pd.DataFrame(elem_rows)
    if save_csv:
        sig = 0.0 if clean_mode else noise_sigma
        tag = f"folder-{os.path.basename(dataset_dir.rstrip('/'))}_denoiser-{denoiser_name}_group-{group_name}_sigma-{sig}_G-{averaging}"
        per_image.to_csv(os.path.join(save_dir, f"q1_{tag}_per_image.csv"), index=False)
        per_element.to_csv(os.path.join(save_dir, f"q1_{tag}_per_element.csv"), index=False)

    return {"per_image": per_image, "per_element": per_element}
