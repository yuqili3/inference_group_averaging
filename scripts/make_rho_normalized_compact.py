"""Compact normalized rho(g) vs rotation angle for Section D (Verifying the orbit-MSE identity).

rho(g) := || T_g^{-1} f(T_g(x+n)) - f(x+n) ||^2 / || f(x+n) ||^2

is the per-rotation equivariance residual, normalised by ||f(y)||^2. The orbit-MSE
identity SE_avg(x) = E_g SE(T_g x) - e_1(x) ties the per-input MSE gain to e_1,
which is the orbit-variance of the residuals T_g^{-1} f(T_g y) - f(y); the angular
profile of rho(g) shows which rotations dominate that variance.

Reads Q1 per-element CSVs at results/q1_equivariance_grid/sigma15/{val_images,
val_images_circle}/<denoiser>/*per_element*.csv, averages rel_err over images and
noise realisations, and produces figures/rho_normalized_compact.png at dpi=400.

Compact one-row layout (disk and rectangle protocols side by side) sized for a
single-column IEEE figure (~3.4 in wide).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results" / "q1_equivariance_grid" / "sigma15"
CARD = REPO / "results" / "q1_cardinal_corrected" / "sigma15"  # 0/90/180/270 recomputed with exact pixel permutation

DENOISER_ORDER = ["tv", "wavelet", "restormer", "restormer-rotated-noise-retrained"]
DENOISER_LABEL = {
    "tv": "TV",
    "wavelet": "Wavelet",
    "restormer": "Restormer",
    "restormer-rotated-noise-retrained": "Restormer-aug",
}
DENOISER_COLOR = {
    "tv": "#1f77b4",
    "wavelet": "#d62728",
    "restormer": "#9467bd",
    "restormer-rotated-noise-retrained": "#8c564b",
}


def _cardinal_corrected(protocol: str, denoiser: str) -> dict:
    """Mean rel_err at the cardinal angles {0,90,180,270}, recomputed with the
    exact pixel-permutation operator so the pad-before-denoise artifact cancels."""
    pat = f"q1_cardinal_folder-*_denoiser-{denoiser}_group-cardinal_rot90_sigma-15.0_G-4_per_element.csv"
    matches = list((CARD / protocol / denoiser).glob(pat))
    if not matches:
        return {}
    df = pd.read_csv(matches[0])
    return df.groupby("op")["rel_err"].mean().to_dict()


def load_curve(protocol: str, denoiser: str, G: int = 64) -> tuple[np.ndarray, np.ndarray]:
    pat = f"q1_folder-*_denoiser-{denoiser}_group-fourier_rotation_sigma-15.0_G-{G}_per_element.csv"
    matches = list((GRID / protocol / denoiser).glob(pat))
    if not matches:
        raise FileNotFoundError(f"missing per_element CSV for {protocol}/{denoiser}")
    df = pd.read_csv(matches[0])
    grp = df.groupby("op")["rel_err"].mean().sort_index()
    angles, rho = grp.index.to_numpy(), grp.to_numpy().astype(float).copy()
    # Replace the cardinal-angle values with the artifact-free recomputation.
    for op, val in _cardinal_corrected(protocol, denoiser).items():
        idx = np.where(np.isclose(angles, float(op)))[0]
        if len(idx):
            rho[idx[0]] = val
    return angles, rho


def make_figure(out_png: Path, G: int = 64, sigma: float = 15.0):
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.0), sharey=True)
    protocols = [("val_images_circle", "Disk"),
                 ("val_images",        "Rectangle")]

    for ax, (protocol, title) in zip(axes, protocols):
        for denoiser in DENOISER_ORDER:
            try:
                angles, rho = load_curve(protocol, denoiser, G=G)
            except FileNotFoundError:
                continue
            ax.plot(angles, rho, "-",
                    color=DENOISER_COLOR[denoiser],
                    linewidth=1.1,
                    marker="o", markersize=2.0,
                    label=DENOISER_LABEL[denoiser])
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1e-1)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel(r"rotation angle $g$  (deg)", fontsize=10)
        ax.set_title(title, fontsize=11, pad=2)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3, linewidth=0.4, which="both")

    axes[0].set_ylabel(r"$\rho(g)$", fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="upper center",
               ncol=len(labels), frameon=False, handlelength=0.9,
               handletextpad=0.25, columnspacing=0.45, labelspacing=0.1,
               bbox_to_anchor=(0.5, 0.135), borderaxespad=0.0,
               borderpad=0.0)
    fig.tight_layout(pad=0.2, rect=(0, 0.14, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    out_dir = Path("/Users/yuqi.li@newsbreak.com/Downloads/Final_Exam_Slides/ga_figures")
    make_figure(out_dir / "rho_normalized_compact.png")
