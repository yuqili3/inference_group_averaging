"""Q2 — Does orbit averaging consistently improve MSE/PSNR, and is the gain
predicted by the identity  SE_avg(x) = E_g SE(T_g x) - e1(x) ?

For each image and several noise draws we compute, on the masked content:
  EhSE_hx   = E_g SE(T_g x)              mean per-orbit denoising error
  SEavg_x   = SE of  w = mean_g T_g^{-1} D(T_g x)   (orbit-averaged estimate)
  e1        = E_g||e_g||^2 - ||E_g e_g||^2 ,  e_g = T_g^{-1}D(T_g x) - D(x)
  EhSE - SEavg  (should match e1 — the empirical check of the corollary)

Writes three CSVs (summary / per_image / detail) matching the schema used by the
original study, and returns the summary dict.
"""
import os

import numpy as np
import pandas as pd

from ..registry import make_group
from ..group_operators import UpsampleGroup
from ..pipeline import denoise_one
from ..masks import build_mask, count_pixels
from ..metrics import l2sq, se_to_psnr


def run(
    dataset_dir,
    denoiser,
    denoiser_name="restormer",
    group_name="fourier_rotation",
    averaging=16,
    upsample=1.0,
    noise_sigma=15.0,
    num_noise=2,
    noise_mask="circle",
    se_mask="circle",
    seed=0,
    max_images=None,
    clip_noisy=False,
    clip_denoised=True,
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

    rotation_group = make_group(group_name, K=averaging, expand=expand)
    scale_group = UpsampleGroup(scales=[upsample])

    rows, detail_rows = [], []

    for fi, f in enumerate(files):
        file_name = os.path.basename(f)
        if verbose:
            print(f"[Q2] {fi + 1}/{len(files)}: {file_name}")
        clean = load_image(f)

        x_mask = build_mask(clean, clean_ref=clean, mask_mode=se_mask)
        x_num_pixels = count_pixels(x_mask, clean)

        EhSE_noise_raw, SEavg_noise_raw = [], []
        eh_all, eh_l2sq_all_raw = [], []

        for r in range(num_noise):
            rng = np.random.default_rng(seed + fi * 1000003 + r * 9176)
            noise = rng.normal(0.0, noise_sigma / 255.0, size=clean.shape).astype(np.float32)
            if noise_mask not in (None, "none"):
                noise = noise * build_mask(clean, clean_ref=clean, mask_mode=noise_mask)
            x_noisy = clean + noise
            if clip_noisy:
                x_noisy = np.clip(x_noisy, 0, 1)

            D_x_noisy = denoise_one(x_noisy, denoiser, noise_sigma)
            if clip_denoised:
                D_x_noisy = np.clip(D_x_noisy, 0, 1)

            clean_up = scale_group.forward(clean)[0]
            noisy_up = scale_group.forward(x_noisy)[0]
            hx_list = rotation_group.forward(clean_up)
            hxn_list = rotation_group.forward(noisy_up)

            z_list, SE_h_raw_list, SE_h_pp_list, angle_rows = [], [], [], []
            for hi, (hx, hxn) in enumerate(zip(hx_list, hxn_list)):
                D_hxn = denoise_one(hxn, denoiser, noise_sigma)
                if clip_denoised:
                    D_hxn = np.clip(D_hxn, 0, 1)

                hx_mask = build_mask(hx, clean_ref=hx, mask_mode=se_mask)
                hx_num_pixels = count_pixels(hx_mask, hx)
                SE_h_raw = l2sq(D_hxn, hx, mask=hx_mask)
                SE_h_raw_list.append(SE_h_raw)
                SE_h_pp_list.append(SE_h_raw / hx_num_pixels)
                angle_rows.append({
                    "file": file_name,
                    "noise_id": r,
                    "rotation_index": hi,
                    "rotation_angle_deg": rotation_group.ops[hi],
                    "SE_hx_raw": SE_h_raw,
                    "SE_hx": SE_h_raw / hx_num_pixels,
                    "SE_hx_psnr": se_to_psnr(SE_h_raw / hx_num_pixels),
                    "hx_num_pixels": hx_num_pixels,
                })

                z_h = scale_group.invert(0, rotation_group.invert(hi, D_hxn))
                if clip_denoised:
                    z_h = np.clip(z_h, 0, 1)
                z_h = z_h.astype(np.float32)
                z_list.append(z_h)

                e_h = z_h - D_x_noisy
                eh_all.append(e_h)
                eh_l2sq_all_raw.append(l2sq(e_h, mask=x_mask))

            w = np.mean(np.stack(z_list, axis=0), axis=0)
            if clip_denoised:
                w = np.clip(w, 0, 1)
            SEavg_raw_r = l2sq(w, clean, mask=x_mask)
            EhSE_raw_r = float(np.mean(SE_h_raw_list))

            EhSE_noise_raw.append(EhSE_raw_r)
            SEavg_noise_raw.append(SEavg_raw_r)
            for angle_row in angle_rows:
                angle_row.update({
                    "EhSE_hx_raw_this_noise": EhSE_raw_r,
                    "SEavg_raw_this_noise": SEavg_raw_r,
                    "EhSE_hx_this_noise": EhSE_raw_r / x_num_pixels,
                    "SEavg_this_noise": SEavg_raw_r / x_num_pixels,
                    "EhSE_hx_psnr_this_noise": se_to_psnr(EhSE_raw_r / x_num_pixels),
                    "SEavg_psnr_this_noise": se_to_psnr(SEavg_raw_r / x_num_pixels),
                    "x_num_pixels": x_num_pixels,
                })
            detail_rows.extend(angle_rows)

        # ---- per-image aggregation (raw, then per-pixel) ----
        EhSE_hx_raw = float(np.mean(EhSE_noise_raw))
        SEavg_x_raw = float(np.mean(SEavg_noise_raw))
        Ehn_l2sq_eh_raw = float(np.mean(eh_l2sq_all_raw))
        l2sq_Ehn_eh_raw = l2sq(np.mean(np.stack(eh_all, axis=0), axis=0), mask=x_mask)
        e1_raw = Ehn_l2sq_eh_raw - l2sq_Ehn_eh_raw

        EhSE_hx = EhSE_hx_raw / x_num_pixels
        SEavg_x = SEavg_x_raw / x_num_pixels
        e1 = e1_raw / x_num_pixels
        rows.append({
            "file": file_name,
            "EhSE_hx": EhSE_hx, "SEavg_x": SEavg_x, "e1": e1,
            "EhSE_hx_raw": EhSE_hx_raw, "SEavg_x_raw": SEavg_x_raw, "e1_raw": e1_raw,
            "Ehn_l2sq_eh_raw": Ehn_l2sq_eh_raw, "l2sq_Ehn_eh_raw": l2sq_Ehn_eh_raw,
            "Ehn_l2sq_eh": Ehn_l2sq_eh_raw / x_num_pixels,
            "l2sq_Ehn_eh": l2sq_Ehn_eh_raw / x_num_pixels,
            "EhSE_minus_SEavg": EhSE_hx - SEavg_x,
            "EhSE_minus_SEavg_raw": EhSE_hx_raw - SEavg_x_raw,
            "EhSE_hx_psnr": se_to_psnr(EhSE_hx), "SEavg_x_psnr": se_to_psnr(SEavg_x),
            "x_num_pixels": x_num_pixels, "num_noise": num_noise,
            "averaging": averaging, "upsample": upsample, "noise_sigma": noise_sigma,
            "denoiser": denoiser_name, "dataset_folder": os.path.basename(dataset_dir.rstrip("/")),
        })

    per_image = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    summary = {
        "E_x_EhSE_hx": per_image["EhSE_hx"].mean(),
        "E_x_SEavg_x": per_image["SEavg_x"].mean(),
        "E_x_e1": per_image["e1"].mean(),
        "E_x_EhSE_minus_SEavg": per_image["EhSE_minus_SEavg"].mean(),
        "E_x_EhSE_hx_psnr": se_to_psnr(per_image["EhSE_hx"].mean()),
        "E_x_SEavg_x_psnr": se_to_psnr(per_image["SEavg_x"].mean()),
    }

    if save_csv:
        folder = os.path.basename(dataset_dir.rstrip("/"))
        tag = f"folder-{folder}_sigma-{noise_sigma}_denoiser-{denoiser_name}_G-{averaging}"
        pd.DataFrame([summary]).to_csv(os.path.join(save_dir, f"q2_{tag}_summary.csv"), index=False)
        per_image.to_csv(os.path.join(save_dir, f"q2_{tag}_per_image.csv"), index=False)
        detail.to_csv(os.path.join(save_dir, f"q2_{tag}_detail.csv"), index=False)

    return {"summary": summary, "per_image": per_image, "detail": detail}
