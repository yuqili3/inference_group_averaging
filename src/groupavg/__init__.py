"""groupavg — orbit averaging for image denoisers.

Public API:
    make_group(name, **kwargs)      -> GroupOperator
    make_denoiser(name, **kwargs)   -> Denoiser
    orbit_average(...)              -> group-averaged estimate
"""
from .registry import make_group, list_groups, GROUPS
from .denoisers import make_denoiser, list_denoisers
from .pipeline import orbit_average, denoise_one

__all__ = [
    "make_group",
    "list_groups",
    "GROUPS",
    "make_denoiser",
    "list_denoisers",
    "orbit_average",
    "denoise_one",
]
