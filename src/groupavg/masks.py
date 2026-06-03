"""Region masks used for noise injection and squared-error accounting.

Two places need a mask:
  * noise mask  — restrict the additive noise to a region (e.g. the disc of a
    circle image), matching the "fixed-orientation / supported noise" setting.
  * SE mask     — restrict the squared-error / PSNR accounting to the image
    content, so that zero-padding introduced by rotation does not dominate.

Masks are float32 in {0, 1} (hard) unless ``blur_width`` is given for the disc.
"""
import numpy as np


def ones_mask(arr2d):
    return np.ones(arr2d.shape[:2], dtype=np.float32)


def content_mask(clean_ref, thresh=0.0):
    """Nonzero support of the (clean) reference — the natural rectangular/foreground
    mask. For full-frame natural images this is all ones; for circle/padded images
    it is the actual support."""
    return (np.asarray(clean_ref) > thresh).astype(np.float32)


def circular_mask(h, w, blur_width=0.0):
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    r = min(h, w) / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if blur_width and blur_width > 0:
        return np.clip((r - dist) / blur_width, 0, 1).astype(np.float32)
    return (dist <= r).astype(np.float32)


def build_mask(arr2d, clean_ref=None, mask_mode="content"):
    """Return a float32 {0,1} mask shaped like ``arr2d``.

    mask_mode:
      "none"               -> all ones
      "content"/"rectangle"-> nonzero support of clean_ref (defaults to arr2d)
      "circle"             -> centered inscribed disc
      "square"             -> all ones (square images are full-support)
    """
    h, w = arr2d.shape[:2]
    if mask_mode in (None, "none"):
        return ones_mask(arr2d)
    if mask_mode in ("content", "rectangle"):
        ref = arr2d if clean_ref is None else clean_ref
        return content_mask(ref)
    if mask_mode == "circle":
        return circular_mask(h, w)
    if mask_mode == "square":
        return ones_mask(arr2d)
    raise ValueError(f"Unknown mask_mode: {mask_mode!r}")


def count_pixels(mask, arr2d=None):
    """Effective pixel count for per-pixel normalization (mask area, >= 1)."""
    s = float(np.sum(mask))
    return max(s, 1.0)
