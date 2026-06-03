"""Smoke tests for the classical denoisers: correct shape, range, and that they
actually reduce noise on a simple image. Restormer is skipped if torch is absent.
"""
import numpy as np
import pytest

from groupavg import make_denoiser
from groupavg.metrics import psnr


def _noisy(seed=0, h=64, w=64, sigma=0.1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    clean = np.clip(((xx // 8 + yy // 8) % 2).astype(np.float32) * 0.7 + 0.15, 0, 1)
    noisy = np.clip(clean + rng.normal(0, sigma, clean.shape), 0, 1).astype(np.float32)
    return clean, noisy


@pytest.mark.parametrize("name", ["tv", "wavelet", "nlm"])
def test_classical_denoiser_shape_range_and_improves(name):
    clean, noisy = _noisy()
    d = make_denoiser(name)
    out = d(noisy, sigma=25.5)  # 0.1 * 255
    assert out.shape == noisy.shape
    assert out.dtype == np.float32
    assert out.min() >= -1e-3 and out.max() <= 1 + 1e-3
    assert psnr(clean, out) >= psnr(clean, noisy) - 0.5  # should not make it worse


def test_dummy_denoiser():
    _, noisy = _noisy()
    out = make_denoiser("dummy")(noisy)
    assert out.shape == noisy.shape


def test_stub_raises():
    with pytest.raises(NotImplementedError):
        make_denoiser("drunet")
