"""Group-operator registry.

The concrete operators live in ``group_operators.py`` (ported verbatim from the
validated research code). This module exposes a small factory so experiments and
configs can refer to groups by name.

Every group is a :class:`GroupOperator` with the contract:
    forward(img)      -> list of |G| transformed images  (apply every g)
    invert(idx, img)  -> undo the idx-th group element
"""
from . import group_operators as _go

GROUPS = {
    "rotation": _go.RotationGroup,                  # cv2 affine, lossy on high-freq
    "fourier_rotation": _go.FourierRotationGroup,   # exact FFT 3-shear rotation
    "fourier_rotation_v2": _go.FourierRotationGroup_v2,
    "shift": _go.CircularShiftGroup,                # exact circular translation
    "upsample": _go.UpsampleGroup,
    "downsample": _go.DownsampleGroup,
    "allpass_fft": _go.AllPassFFTGroup,             # exact, energy-preserving
}


def make_group(name, **kwargs):
    """Construct a group operator by registry name.

    Examples
    --------
    >>> make_group("fourier_rotation", K=8, expand=True)
    >>> make_group("shift", nx_steps=4, ny_steps=4)
    """
    if name not in GROUPS:
        raise KeyError(
            f"Unknown group '{name}'. Available: {sorted(GROUPS)}"
        )
    return GROUPS[name](**kwargs)


def list_groups():
    return sorted(GROUPS)
