"""Denoiser registry.

    make_denoiser("tv")
    make_denoiser("wavelet")
    make_denoiser("restormer", weights=".../gaussian_gray_denoising_sigma15.pth",
                  device="cuda:1")

Torch-backed denoisers (Restormer) import torch lazily, so the classical
denoisers work in a torch-free environment.
"""
from .classical import TVChambolle, Wavelet, NLMeans, BM3D
from . import stubs

# name -> factory (kept lazy for the torch-backed ones)
DENOISERS = {
    "tv": TVChambolle,
    "wavelet": Wavelet,
    "nlm": NLMeans,
    "bm3d": BM3D,
    "dummy": None,  # handled specially below
    # stubs (raise NotImplementedError on construction)
    "dncnn": stubs.DnCNN,
    "drunet": stubs.DRUNet,
    "diffusion": stubs.Diffusion,
}


def _make_dummy():
    import cv2
    import numpy as np
    from .base import Denoiser

    class Dummy(Denoiser):
        name = "dummy"

        def __call__(self, img01, sigma=15.0):
            return cv2.GaussianBlur(np.asarray(img01, np.float32), (3, 3), 0)

    return Dummy()


def make_denoiser(name, **kwargs):
    if name == "restormer":
        from .restormer import Restormer
        return Restormer(**kwargs)
    if name == "dummy":
        return _make_dummy()
    if name not in DENOISERS:
        raise KeyError(f"Unknown denoiser '{name}'. Available: {list_denoisers()}")
    return DENOISERS[name](**kwargs)


def list_denoisers():
    return sorted(list(DENOISERS) + ["restormer"])
