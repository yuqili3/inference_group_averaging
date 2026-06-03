"""Round-trip fidelity of the group operators: invert(forward) ~ identity.

Thresholds mirror the validated unit-test notebook: exact operators (Fourier
rotation, all-pass FFT) are high-PSNR; cv2 affine rotation and resampling are
lossy but bounded.
"""
import numpy as np
import pytest

from groupavg import make_group


def _synth(h=128, w=128):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    grad = (xx / (w - 1) + yy / (h - 1)) * 0.5
    cy, cx, r = h // 2, w // 2, min(h, w) // 4
    circle = (((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2).astype(np.float32)
    return np.clip(grad * 0.6 + 0.4 * circle, 0, 1)


def _psnr(a, b, eps=1e-12):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= eps else 10 * np.log10(1.0 / mse)


def _roundtrip_psnr(group, img):
    outs = group.forward(img)
    ps = [_psnr(img, group.invert(i, o)) for i, o in enumerate(outs)]
    return float(np.mean(ps))


@pytest.mark.parametrize("name,kwargs,thresh", [
    ("fourier_rotation", dict(K=8, expand=True), 55.0),
    ("allpass_fft", dict(n_filters=4, phase_mode="random", clip01=False), 90.0),
    ("shift", dict(nx_steps=4, ny_steps=4), 90.0),
    ("upsample", dict(scales=[1.5, 2.0]), 30.0),
    ("rotation", dict(K=8, expand=True), 30.0),  # lossy but bounded
])
def test_roundtrip(name, kwargs, thresh):
    img = _synth()
    group = make_group(name, **kwargs)
    assert _roundtrip_psnr(group, img) >= thresh


def test_forward_count_matches_group_size():
    img = _synth(64, 64)
    g = make_group("fourier_rotation", K=6, expand=True)
    assert len(g.forward(img)) == 6
