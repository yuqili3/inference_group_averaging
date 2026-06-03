"""The orbit-MSE identity  SE_avg(x) = E_g SE(T_g x) - e1(x)  must hold
*exactly* (up to float error) when the group action is exactly invertible.

We use the AllPassFFT group (exact, energy-preserving) with a known denoiser, so
the only thing being checked is the algebra of the decomposition, independent of
any operator/interpolation loss.
"""
import numpy as np

from groupavg import make_group
from groupavg.metrics import l2sq, orbit_variance_e1


def _img(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.random((h, w)).astype(np.float32), 0, 1)


def test_se_avg_identity_exact_group():
    rng = np.random.default_rng(1)
    clean = _img()
    noise = rng.normal(0, 0.05, clean.shape).astype(np.float32)
    noisy = clean + noise

    group = make_group("allpass_fft", n_filters=6, phase_mode="random")

    # a simple denoiser: 3x3 box blur (linear, deterministic)
    import cv2
    D = lambda im: cv2.blur(im.astype(np.float32), (3, 3))

    Tg = group.forward(noisy)
    z_list, se_list = [], []
    D_direct = D(noisy)
    for i, t in enumerate(Tg):
        d = D(t)
        z = group.invert(i, d)               # T_g^{-1} D(T_g y)
        z_list.append(z.astype(np.float32))
        se_list.append(l2sq(z, clean))       # SE(T_g x) measured in canonical frame

    w = np.mean(np.stack(z_list, 0), 0)
    SE_avg = l2sq(w, clean)
    EhSE = float(np.mean(se_list))

    # Identity in the form used by the corollary:
    #   E_g||z_g - x||^2 = ||mean_g z_g - x||^2 + E_g||z_g - mean_g z_g||^2
    var_term, _, _ = orbit_variance_e1(z_list)  # E||z||^2 - ||Ez||^2 == E||z-Ez||^2
    assert abs(EhSE - (SE_avg + var_term)) < 1e-3 * max(EhSE, 1.0)


def test_orbit_variance_nonnegative():
    maps = [_img(seed=k) for k in range(5)]
    e1, mean_l2, l2_mean = orbit_variance_e1(maps)
    assert e1 >= -1e-6
    assert mean_l2 >= l2_mean - 1e-6
