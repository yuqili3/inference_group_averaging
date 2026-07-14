"""Section D / Q1 figure: rho(g) vs rotation angle for each denoiser.

rho(g) := || T_g^{-1} f(T_g(x+n)) - f(x+n) ||^2  (relative, normalised by ||f(y)||^2)

Reads per_element CSVs under results/q1_equivariance_grid/sigma15 and produces
figures/noneq_rho_vs_angle.pdf (also .png), at dpi=400.

NB: only the orbit-noise model (T_g(x+n)) is in the data — the upright-noise model
(T_g x + n) is not currently produced by q1_equivariance.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results" / "q1_equivariance_grid" / "sigma15"

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


def load_curve(protocol: str, denoiser: str, G: int = 64) -> tuple[np.ndarray, np.ndarray]:
    pat = f"q1_folder-*_denoiser-{denoiser}_group-fourier_rotation_sigma-15.0_G-{G}_per_element.csv"
    matches = list((GRID / protocol / denoiser).glob(pat))
    if not matches:
        raise FileNotFoundError(f"missing per_element CSV for {protocol}/{denoiser}")
    df = pd.read_csv(matches[0])
    # Average over images and noise realisations, group by rotation angle.
    grp = df.groupby("op")["rel_err"].mean().sort_index()
    return grp.index.to_numpy(), grp.to_numpy()


def make_figure(out_pdf: Path, out_png: Path, G: int = 64, sigma: float = 15.0):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    protocols = [("val_images_circle", "Disk protocol"),
                 ("val_images",        "Rectangle protocol")]

    for ax, (protocol, title) in zip(axes, protocols):
        for denoiser in DENOISER_ORDER:
            try:
                angles, rho = load_curve(protocol, denoiser, G=G)
            except FileNotFoundError:
                continue
            ax.plot(angles, rho, "-",
                    color=DENOISER_COLOR[denoiser],
                    linewidth=1.4,
                    marker="o", markersize=2.8,
                    label=DENOISER_LABEL[denoiser])
        # Mark the four 90-degree rotations (axes of symmetry for axis-aligned filters).
        for a in (90, 180, 270):
            ax.axvline(a, color="gray", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xlabel(r"rotation angle $g$  (degrees)", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linewidth=0.5, which="both")

    axes[0].set_ylabel(
        r"$\rho(g) = \|T_g^{-1}f(T_g y) - f(y)\|^2 \,/\, \|f(y)\|^2$",
        fontsize=9,
    )
    axes[1].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(
        rf"Empirical non-equivariance vs rotation angle  ($|\mathcal{{G}}|={G}$, $\sigma={sigma:g}/255$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    out_dir = Path("/Users/yuqi.li@newsbreak.com/Downloads/draft/IEEEtran/figures")
    make_figure(out_dir / "noneq_rho_vs_angle.pdf",
                out_dir / "noneq_rho_vs_angle.png")
