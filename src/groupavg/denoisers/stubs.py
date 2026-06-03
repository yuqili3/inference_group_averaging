"""Placeholder denoisers reserved for future wiring.

These raise a clear NotImplementedError so the registry surface is complete and
configs can reference them, but they are not yet implemented.

  * dncnn     — residual CNN denoiser. (Deferred per current scope; the local
                pretrained .pth files in the reference repo are corrupt HTML and
                need re-downloading before this can be wired.)
  * drunet    — PnP-standard bias-free UNet denoiser (Zhang et al.).
  * diffusion — score/DDPM-based denoiser used in diffusion restoration pipelines.
"""
from .base import Denoiser


class _Stub(Denoiser):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"Denoiser '{self.name}' is not wired yet. "
            f"Register an implementation in groupavg.denoisers."
        )

    def __call__(self, img01, sigma=15.0):  # pragma: no cover
        raise NotImplementedError(self.name)


class DnCNN(_Stub):
    name = "dncnn"


class DRUNet(_Stub):
    name = "drunet"


class Diffusion(_Stub):
    name = "diffusion"
