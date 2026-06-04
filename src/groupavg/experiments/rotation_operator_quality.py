"""Rotation-operator round-trip quality diagnostic.

For each test signal x and rotation angle theta, this experiment measures

    PSNR(T_g^{-1} T_g x, x)

with expanded forward rotations. It isolates interpolation/operator error from
denoiser behavior by using identity input/output rather than a restoration model.
"""
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import color, data

from ..group_operators import FourierRotationGroup


def _psnr_unit_peak(a, b, eps=1e-12):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= eps else float(10.0 * np.log10(1.0 / mse))


def _center_crop_or_pad(img, shape):
    target_h, target_w = shape
    h, w = img.shape[:2]

    pad_top = max((target_h - h) // 2, 0)
    pad_bottom = max(target_h - h - pad_top, 0)
    pad_left = max((target_w - w) // 2, 0)
    pad_right = max(target_w - w - pad_left, 0)
    if pad_top or pad_bottom or pad_left or pad_right:
        img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant")
        h, w = img.shape[:2]

    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return img[top:top + target_h, left:left + target_w]


def _cv2_rotation_matrix_expand(angle_deg, h, w):
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_v, sin_v = abs(mat[0, 0]), abs(mat[0, 1])
    bound_w = int(np.round(h * sin_v + w * cos_v))
    bound_h = int(np.round(h * cos_v + w * sin_v))
    mat[0, 2] += ((h * sin_v + w * cos_v) / 2.0) - center[0]
    mat[1, 2] += ((h * cos_v + w * sin_v) / 2.0) - center[1]
    return mat, bound_w, bound_h


def _roundtrip_cv2(img, angle_deg):
    h, w = img.shape
    mat, bound_w, bound_h = _cv2_rotation_matrix_expand(angle_deg, h, w)
    rotated = cv2.warpAffine(
        img, mat, (bound_w, bound_h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    inv = cv2.invertAffineTransform(mat)
    return cv2.warpAffine(
        rotated, inv, (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )


def _roundtrip_fourier(img, angle_deg):
    group = FourierRotationGroup(K=1, angles=[float(angle_deg)], expand=True)
    rotated = group.forward(img)[0]
    return group.invert(0, rotated)


def _roundtrip_scipy(img, angle_deg):
    rotated = ndimage.rotate(
        img, angle_deg, reshape=True, order=3, mode="constant", cval=0.0, prefilter=True
    )
    restored_canvas = ndimage.rotate(
        rotated, -angle_deg, reshape=False, order=3, mode="constant", cval=0.0, prefilter=True
    )
    return _center_crop_or_pad(restored_canvas, img.shape)


def make_signals(noise_sigma=15.0, seed=0):
    rng = np.random.default_rng(seed)
    clean = color.rgb2gray(data.astronaut()).astype(np.float32)
    noisy = np.clip(clean + rng.normal(0.0, noise_sigma / 255.0, clean.shape), 0.0, 1.0).astype(np.float32)
    pure_noise = np.clip(0.5 + rng.normal(0.0, 0.25, clean.shape), 0.0, 1.0).astype(np.float32)
    return {
        "astronaut": clean,
        "astronaut_sigma15": noisy,
        "pure_gaussian_noise": pure_noise,
    }


def run(
    angles=None,
    noise_sigma=15.0,
    seed=0,
    save_dir="results/rotation_operator_quality",
    save_csv=True,
    save_plot=True,
    verbose=True,
):
    if angles is None:
        angles = list(np.arange(0.0, 360.0 + 5.0, 5.0))

    operators = {
        "cv2_affine_rotation": _roundtrip_cv2,
        "fourier_rotation": _roundtrip_fourier,
        "scipy_ndimage_rotate": _roundtrip_scipy,
    }
    signals = make_signals(noise_sigma=noise_sigma, seed=seed)

    os.makedirs(save_dir, exist_ok=True)
    rows = []
    for signal_name, img in signals.items():
        if verbose:
            print(f"[rotation-quality] signal={signal_name}")
        for op_name, fn in operators.items():
            if verbose:
                print(f"  operator={op_name}")
            for angle in angles:
                restored = fn(img, float(angle))
                rows.append({
                    "signal": signal_name,
                    "operator": op_name,
                    "angle_deg": float(angle),
                    "psnr": _psnr_unit_peak(img, restored),
                })

    df = pd.DataFrame(rows)
    if save_csv:
        df.to_csv(os.path.join(save_dir, "rotation_operator_quality_all.csv"), index=False)
        for signal_name, sub in df.groupby("signal"):
            sub.to_csv(os.path.join(save_dir, f"rotation_operator_quality_{signal_name}.csv"), index=False)

    fig_paths = []
    if save_plot:
        fig_paths = _plot(df, save_dir)

    return {"results": df, "figures": fig_paths, "save_dir": save_dir}


def _safe_name(name):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


def _plot(df, save_dir):
    signals = list(df["signal"].drop_duplicates())
    paths = []
    for signal_name in signals:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sub = df[df["signal"] == signal_name]
        for op_name, op_df in sub.groupby("operator"):
            ax.plot(op_df["angle_deg"], op_df["psnr"], marker=".", linewidth=1.5, label=op_name)
        ax.set_title(signal_name)
        ax.set_ylabel(r"PSNR of $T_g^{-1} T_g x$ vs $x$ [dB]")
        ax.set_xlabel("Rotation angle [degrees]")
        ax.set_xticks(np.arange(0, 361, 45))
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = os.path.join(save_dir, f"rotation_operator_quality_{_safe_name(signal_name)}_psnr.png")
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths
