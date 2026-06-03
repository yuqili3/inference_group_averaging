"""Squared-error / PSNR primitives and the orbit-averaging decomposition.

Notation (matching the paper):
    SE(x)            squared error of a denoiser estimate vs the clean target
    EhSE = E_g SE(T_g x)        mean per-orbit denoising error
    SE_avg(x)        squared error of the orbit-averaged estimate
                     w = (1/|G|) sum_g T_g^{-1} D(T_g x)
    e1(x)            orbit variance of the per-element estimates, i.e.
                     E_g ||e_g||^2 - ||E_g e_g||^2 ,  e_g = T_g^{-1}D(T_g x) - D(x)

Corollary (orbit-MSE identity, Q2):
    SE_avg(x) = E_g SE(T_g x) - e1(x)

All images are float32 in [0, 1]; masks are {0,1} weights.
"""
import numpy as np


def l2sq(a, b=None, mask=None):
    """Sum of squared (masked) differences. Returns the RAW (unnormalized) sum;
    divide by ``count_pixels`` for a per-pixel MSE."""
    a = np.asarray(a, dtype=np.float32)
    d = a if b is None else (a - np.asarray(b, dtype=np.float32))
    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        return float(np.sum((d * d) * m))
    return float(np.sum(d * d))


def psnr(a, b, mask=None, eps=1e-12):
    """PSNR with peak = max(a). With a mask, MSE is averaged over the mask area."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if isinstance(mask, np.ndarray):
        mse = np.sum((a * mask - b * mask) ** 2) / max(np.sum(mask), eps)
    else:
        mse = np.mean((a - b) ** 2)
    peak = float(np.max(a)) ** 2
    return 99.0 if mse <= eps else float(10.0 * np.log10(peak / mse))


def se_to_psnr(mse_per_pixel, peak=1.0, eps=1e-12):
    """Convert a per-pixel MSE (in [0,1] image scale) to PSNR (dB)."""
    mse = max(float(mse_per_pixel), eps)
    return float(10.0 * np.log10((peak ** 2) / mse))


def orbit_variance_e1(e_list, mask=None):
    """e1 = E_g ||e_g||^2 - ||E_g e_g||^2  (raw, masked).

    e_list: list of per-orbit residual maps e_g = z_g - D(x).
    Returns (e1_raw, mean_l2sq_raw, l2sq_mean_raw).
    """
    eh_l2sq = np.array([l2sq(e, mask=mask) for e in e_list], dtype=np.float64)
    mean_l2sq = float(np.mean(eh_l2sq))            # E_g ||e_g||^2
    e_mean = np.mean(np.stack(e_list, axis=0), axis=0)
    l2sq_mean = l2sq(e_mean, mask=mask)            # ||E_g e_g||^2
    return float(mean_l2sq - l2sq_mean), mean_l2sq, l2sq_mean
