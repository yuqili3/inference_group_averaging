"""Q1 cardinal-angle equivariance check with exact rot90 operators.

The dense Q1 grid uses FourierRotationGroup with a shared expanded canvas. That
is appropriate for arbitrary angles, but it contaminates cardinal angles: at
g=0 the transform is not the identity on the original canvas, so rho(g) need not
be exactly zero.

This experiment reruns only 0, 90, 180, and 270 degrees using numpy rot90
directly. For g=0, rho(g) is therefore exactly zero up to deterministic denoiser
roundoff.
"""
import os

import numpy as np
import pandas as pd

from ..data import list_images, load_image
from ..masks import build_mask, count_pixels
from ..metrics import l2sq
from ..pipeline import denoise_one


ANGLES = (0, 90, 180, 270)


def _forward_rot90(img, angle_deg):
    k = (int(angle_deg) // 90) % 4
    if k == 0:
        return np.asarray(img, dtype=np.float32).copy()
    return np.rot90(img, k=k).copy()


def _inverse_rot90(img, angle_deg):
    k = (int(angle_deg) // 90) % 4
    if k == 0:
        return np.asarray(img, dtype=np.float32).copy()
    return np.rot90(img, k=(-k) % 4).copy()


def run(
    dataset_dir,
    denoiser,
    denoiser_name="restormer",
    angles=ANGLES,
    noise_sigma=15.0,
    num_noise=4,
    noise_mask="none",
    se_mask="content",
    clean_mode=False,
    seed=0,
    max_images=None,
    save_dir="results",
    save_csv=True,
    verbose=True,
):
    files = list_images(dataset_dir)
    if max_images is not None:
        files = files[:max_images]
    os.makedirs(save_dir, exist_ok=True)

    angles = [int(a) for a in angles]
    rows, elem_rows = [], []
    for fi, f in enumerate(files):
        file_name = os.path.basename(f)
        if verbose:
            print(f"[Q1 cardinal] {fi + 1}/{len(files)}: {file_name}")
        clean = load_image(f)
        x_mask = build_mask(clean, clean_ref=clean, mask_mode=se_mask)
        x_num_pixels = count_pixels(x_mask, clean)

        n_draws = 1 if clean_mode else num_noise
        per_elem_rel = np.zeros(len(angles), dtype=np.float64)
        mean_rel_acc, e1_acc = [], []

        for noise_id in range(n_draws):
            rng = np.random.default_rng(seed + fi * 1000003 + noise_id * 9176)
            if clean_mode:
                y = clean.copy()
            else:
                noise = rng.normal(0.0, noise_sigma / 255.0, size=clean.shape).astype(np.float32)
                if noise_mask not in (None, "none"):
                    noise = noise * build_mask(clean, clean_ref=clean, mask_mode=noise_mask)
                y = clean + noise

            direct = denoise_one(y, denoiser, noise_sigma)
            denom = max(l2sq(direct, mask=x_mask), 1e-12)
            r_list = []

            for angle_index, angle_deg in enumerate(angles):
                tg_y = _forward_rot90(y, angle_deg)
                tg_denoised = denoise_one(tg_y, denoiser, noise_sigma)
                aligned = _inverse_rot90(tg_denoised, angle_deg).astype(np.float32)
                residual = aligned - direct
                rho_raw = l2sq(residual, mask=x_mask)
                rho = rho_raw / x_num_pixels
                rel = rho_raw / denom
                r_list.append(residual)
                per_elem_rel[angle_index] += rel / n_draws
                elem_rows.append({
                    "file": file_name,
                    "noise_id": noise_id,
                    "denoiser": denoiser_name,
                    "group": "cardinal_rot90",
                    "noise_sigma": (0.0 if clean_mode else noise_sigma),
                    "angle_index": angle_index,
                    "g_index": angle_index,
                    "op": angle_deg,
                    "angle_deg": angle_deg,
                    "rho_raw": rho_raw,
                    "rho": rho,
                    "rel_err": rel,
                    "x_num_pixels": x_num_pixels,
                })

            rel_each = [l2sq(r, mask=x_mask) / denom for r in r_list]
            mean_rel_acc.append(float(np.mean(rel_each)))
            r_mean = np.mean(np.stack(r_list, axis=0), axis=0)
            e1_raw = float(np.mean([l2sq(r, mask=x_mask) for r in r_list])) - l2sq(r_mean, mask=x_mask)
            e1_acc.append(e1_raw / x_num_pixels)

        rows.append({
            "file": file_name,
            "denoiser": denoiser_name,
            "group": "cardinal_rot90",
            "noise_sigma": (0.0 if clean_mode else noise_sigma),
            "averaging": len(angles),
            "angles": ",".join(str(a) for a in angles),
            "mean_rel_err": float(np.mean(mean_rel_acc)),
            "max_elem_rel_err": float(np.max(per_elem_rel)),
            "e1": float(np.mean(e1_acc)),
        })

    per_image = pd.DataFrame(rows)
    per_element = pd.DataFrame(elem_rows)
    summary = (
        per_element.groupby(["denoiser", "noise_sigma", "angle_deg"], as_index=False)
        .agg(
            mean_rho=("rho", "mean"),
            mean_rho_raw=("rho_raw", "mean"),
            mean_rel_err=("rel_err", "mean"),
            max_rel_err=("rel_err", "max"),
            n=("rel_err", "size"),
        )
    )

    if save_csv:
        folder = os.path.basename(dataset_dir.rstrip("/"))
        sig = 0.0 if clean_mode else noise_sigma
        tag = f"folder-{folder}_denoiser-{denoiser_name}_group-cardinal_rot90_sigma-{sig}_G-{len(angles)}"
        per_image.to_csv(os.path.join(save_dir, f"q1_cardinal_{tag}_per_image.csv"), index=False)
        per_element.to_csv(os.path.join(save_dir, f"q1_cardinal_{tag}_per_element.csv"), index=False)
        summary.to_csv(os.path.join(save_dir, f"q1_cardinal_{tag}_summary.csv"), index=False)

    return {"summary": summary, "per_image": per_image, "per_element": per_element}
