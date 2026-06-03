"""Classical (non-learned) denoisers used as PnP/RED baselines.

All accept ``sigma`` on the 0-255 scale and convert to image units internally.
"""
import numpy as np
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_wavelet,
    denoise_nl_means,
    estimate_sigma,
)

from .base import Denoiser


class TVChambolle(Denoiser):
    name = "tv"

    def __init__(self, weight=0.1, scale_weight_with_sigma=False):
        self.weight = weight
        self.scale_weight_with_sigma = scale_weight_with_sigma

    def __call__(self, img01, sigma=15.0):
        w = self.weight
        if self.scale_weight_with_sigma:
            w = float(sigma) / 255.0
        return denoise_tv_chambolle(img01.astype(np.float32), weight=w).astype(np.float32)


class Wavelet(Denoiser):
    name = "wavelet"

    def __init__(self, method="BayesShrink", mode="soft", rescale_sigma=True,
                 use_sigma=True):
        self.method = method
        self.mode = mode
        self.rescale_sigma = rescale_sigma
        self.use_sigma = use_sigma

    def __call__(self, img01, sigma=15.0):
        s = (float(sigma) / 255.0) if self.use_sigma else None
        return denoise_wavelet(
            img01.astype(np.float32),
            method=self.method,
            mode=self.mode,
            sigma=s,
            rescale_sigma=self.rescale_sigma,
        ).astype(np.float32)


class NLMeans(Denoiser):
    name = "nlm"

    def __init__(self, patch_size=5, patch_distance=6, h_factor=1.0,
                 fast_mode=True):
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h_factor = h_factor
        self.fast_mode = fast_mode

    def __call__(self, img01, sigma=15.0):
        img = img01.astype(np.float32)
        s = float(sigma) / 255.0
        if s <= 0:
            s = float(estimate_sigma(img))
        return denoise_nl_means(
            img,
            h=self.h_factor * s,
            sigma=s,
            patch_size=self.patch_size,
            patch_distance=self.patch_distance,
            fast_mode=self.fast_mode,
        ).astype(np.float32)


class BM3D(Denoiser):
    """Optional — requires the ``bm3d`` package (`pip install bm3d`)."""
    name = "bm3d"

    def __init__(self):
        try:
            import bm3d  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BM3D denoiser requires the 'bm3d' package: pip install bm3d"
            ) from e

    def __call__(self, img01, sigma=15.0):
        import bm3d
        s = float(sigma) / 255.0
        return bm3d.bm3d(img01.astype(np.float32), sigma_psd=s).astype(np.float32)
