"""Illustrative figures for the two image-domain protocols (Sec. exp:protocol).

Disk (theory-aligned): the image is cropped to the largest centred disk, the
exterior set to a constant, and i.i.d. Gaussian noise is added only inside the
disk. The disk is rotation-invariant, so a rotation maps the support (and the
noise law restricted to it) into itself.

Rectangle (deployment-aligned): the full rectangular image plus full-grid i.i.d.
Gaussian noise. Rotating with an expanding canvas pulls constant padding into the
corners, so PSNR is measured on the central inscribed rectangle --- the largest
centred axis-aligned rectangle (same aspect as the image) that stays inside valid
content after rotation.

These panels are illustrative of the protocol geometry, not quantitative
results, so a representative natural image and a bilinear rotation suffice. By
default the image is matplotlib's bundled ``grace_hopper.jpg``; pass --image to
use a file from the real val_images dataset instead.

Writes figures/protocol_disk.png and figures/protocol_rectangle.png at dpi=400
(PNG only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT_DIR = Path("/Users/yuqi.li@newsbreak.com/Downloads/draft/IEEEtran/figures")
SIGMA = 25.0 / 255.0   # display sigma (slightly above the sigma=15 experiment so noise is visible)
ANGLE = 40.0           # illustrative rotation angle (deg), not a cardinal angle
RECT_EDGE = "#ffcc00"  # inscribed-rectangle outline colour


def _load_gray(image: str | None) -> np.ndarray:
    if image is None:
        with cbook.get_sample_data("grace_hopper.jpg", asfileobj=True) as f:
            im = Image.open(f).convert("L")
    else:
        im = Image.open(image).convert("L")
    return np.asarray(im, dtype=np.float32) / 255.0


def load_gray(image: str | None, out_w: int, out_h: int) -> np.ndarray:
    """Grayscale float image in [0,1], centre-cropped to aspect out_w/out_h, resized."""
    arr = _load_gray(image)
    h, w = arr.shape
    target = out_w / out_h
    if w / h > target:                      # too wide -> crop width
        cw = int(round(h * target)); ch = h
    else:                                   # too tall -> crop height
        cw = w; ch = int(round(w / target))
    top, left = (h - ch) // 2, (w - cw) // 2
    arr = arr[top:top + ch, left:left + cw]
    im = Image.fromarray((arr * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def disk_mask(n: int) -> np.ndarray:
    yy, xx = np.ogrid[:n, :n]
    c = (n - 1) / 2.0
    return ((xx - c) ** 2 + (yy - c) ** 2 <= (n / 2.0) ** 2).astype(np.float32)


def rotate(arr: np.ndarray, angle: float, fill: float = 0.0, expand: bool = False) -> np.ndarray:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    out = im.rotate(angle, resample=Image.BILINEAR, expand=expand,
                    fillcolor=int(round(fill * 255)))
    return np.asarray(out, dtype=np.float32) / 255.0


def add_noise(arr: np.ndarray, rng, mask: np.ndarray | None = None) -> np.ndarray:
    noise = rng.normal(0.0, SIGMA, size=arr.shape).astype(np.float32)
    if mask is not None:
        noise = noise * mask
    return np.clip(arr + noise, 0.0, 1.0)


def _imshow(ax, arr, title, fontsize=10):
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=fontsize)
    ax.set_xticks([]); ax.set_yticks([])


def make_disk_figure(img: np.ndarray, rng, out_png: Path):
    n = img.shape[0]
    mask = disk_mask(n)
    const = 0.0  # exterior constant
    base = img * mask + const * (1 - mask)
    noisy = add_noise(base, rng, mask=mask)
    rotated = rotate(noisy, ANGLE, fill=const)
    # re-impose the disk support after rotation (corners stay at the constant)
    rotated = rotated * mask + const * (1 - mask)

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.95))
    _imshow(axes[0], noisy, "Disk input")
    _imshow(axes[1], rotated, rf"Rotated by $g={ANGLE:.0f}^\circ$")
    fig.tight_layout(pad=0.3, w_pad=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def make_rectangle_figure(img: np.ndarray, rng, out_png: Path):
    H, W = img.shape
    noisy = add_noise(img, rng)
    rotated = rotate(noisy, ANGLE, fill=0.0, expand=True)  # canvas grows, corners padded

    # tilted outline tracing the rotated image boundary on the expanded canvas
    Hc, Wc = rotated.shape
    cx, cy = (Wc - 1) / 2.0, (Hc - 1) / 2.0
    th = np.deg2rad(ANGLE)
    c, s = np.cos(th), np.sin(th)
    corners = []
    for dx, dy in [(-W / 2, -H / 2), (W / 2, -H / 2), (W / 2, H / 2), (-W / 2, H / 2)]:
        rx, ry = c * dx + s * dy, -s * dx + c * dy   # PIL rotates CCW; y-axis points down
        corners.append((cx + rx, cy + ry))

    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    _imshow(axes[0], noisy, "Rectangle input")
    _imshow(axes[1], rotated, rf"Rotated by $g={ANGLE:.0f}^\circ$")
    axes[1].add_patch(plt.Polygon(corners, closed=True, fill=False,
                                  edgecolor=RECT_EDGE, linewidth=1.2))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None,
                    help="path to a source image (default: matplotlib grace_hopper sample)")
    args = ap.parse_args()

    disk_img = load_gray(args.image, 320, 320)          # square for the disk crop
    rect_img = load_gray(args.image, 360, 240)          # landscape rectangle
    make_disk_figure(disk_img, np.random.default_rng(0), OUT_DIR / "protocol_disk.png")
    make_rectangle_figure(rect_img, np.random.default_rng(0), OUT_DIR / "protocol_rectangle.png")
