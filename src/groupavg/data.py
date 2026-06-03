"""Image loading and noise injection.

Images are read as grayscale float32 in [0, 1]. Additive Gaussian noise uses the
``noise_sigma`` convention shared across this codebase: sigma is on the 0-255
scale, applied as ``sigma/255`` in image units.
"""
import glob
import os

import cv2
import numpy as np

from .masks import build_mask

_EXTS = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.bmp")


def to_gray_float01(img):
    if img.ndim == 3 and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def list_images(dataset_dir):
    files = []
    for e in _EXTS:
        files += glob.glob(os.path.join(dataset_dir, e))
    if not files:
        raise FileNotFoundError(f"No images found in {dataset_dir}")
    return sorted(files)


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image {path}")
    return to_gray_float01(img)


def iter_dataset(dataset_dir, max_images=None):
    """Yield (filename, clean_float01) pairs."""
    files = list_images(dataset_dir)
    if max_images is not None:
        files = files[:max_images]
    for f in files:
        yield os.path.basename(f), load_image(f)


def add_noise(clean, noise_sigma, rng, mask_mode="none"):
    """Add Gaussian noise (sigma on 0-255 scale), optionally restricted to a mask.

    Returns (noisy, noise). ``mask_mode`` of "circle"/"rectangle"/"content"
    confines the noise to the image support — the "supported / fixed-orientation
    noise" condition probed in Q3.
    """
    noise = rng.normal(0.0, noise_sigma / 255.0, size=clean.shape).astype(np.float32)
    if mask_mode not in (None, "none"):
        n_mask = build_mask(clean, clean_ref=clean, mask_mode=mask_mode)
        noise = noise * n_mask
    return (clean + noise).astype(np.float32), noise
