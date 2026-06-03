"""The orbit-averaging estimator.

Given a denoiser D, a group G (with forward/invert), and an input x, the
group-averaged ("orbit averaged") estimate is

    w(x) = (1/|G|) sum_{g in G} T_g^{-1} D(T_g x).

An optional scale group S (e.g. upsample) is applied *around* the orbit, matching
the up-rotate-denoise-unrotate-down pipeline used in the experiments:

    w(x) = (1/|G|) sum_g S^{-1} T_g^{-1} D(T_g S x).
"""
import numpy as np


def denoise_one(img2d01, denoiser, noise_sigma=15.0):
    """Apply a denoiser to a single (H,W) float01 image, returning (H,W) float01."""
    y = denoiser(np.asarray(img2d01, dtype=np.float32), sigma=noise_sigma)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 3:
        y = y[..., 0]
    return y


def orbit_estimates(x, group, denoiser, noise_sigma=15.0, scale_group=None,
                    clip=True):
    """Return the list of per-element un-transformed estimates

        z_g = S^{-1} T_g^{-1} D(T_g S x)

    one per group element. Averaging this list gives the orbit-averaged estimate.
    """
    x = np.asarray(x, dtype=np.float32)
    xs = scale_group.forward(x)[0] if scale_group is not None else x
    transformed = group.forward(xs)
    z_list = []
    for idx, tg_x in enumerate(transformed):
        d = denoise_one(tg_x, denoiser, noise_sigma)
        if clip:
            d = np.clip(d, 0.0, 1.0)
        inv = group.invert(idx, d)
        z = scale_group.invert(0, inv) if scale_group is not None else inv
        if clip:
            z = np.clip(z, 0.0, 1.0)
        z_list.append(z.astype(np.float32))
    return z_list


def orbit_average(x, group, denoiser, noise_sigma=15.0, scale_group=None,
                  clip=True):
    """Group-averaged estimate w(x) = mean_g S^{-1} T_g^{-1} D(T_g S x)."""
    z_list = orbit_estimates(x, group, denoiser, noise_sigma, scale_group, clip)
    w = np.mean(np.stack(z_list, axis=0), axis=0)
    return np.clip(w, 0.0, 1.0) if clip else w
