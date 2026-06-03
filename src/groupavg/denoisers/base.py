"""Denoiser interface.

A Denoiser maps a single grayscale image in [0, 1] to a denoised image in [0, 1].

Sigma convention: ``__call__`` receives ``sigma`` on the 0-255 scale (the
``noise_sigma`` used throughout the pipeline). Learned denoisers whose weights
are tied to a fixed noise level ignore it; classical denoisers convert to image
units internally (sigma / 255).
"""
from abc import ABC, abstractmethod

import numpy as np


class Denoiser(ABC):
    name = "denoiser"

    @abstractmethod
    def __call__(self, img01: np.ndarray, sigma: float = 15.0) -> np.ndarray:
        ...

    def __repr__(self):
        return f"<Denoiser {self.name}>"
