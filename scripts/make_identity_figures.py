"""Generate Section D (Q2) figures: identity SE_avg = E_g SE(T_g x) - e_1.

Produces two PDFs in figures/:
  - identity_disk.pdf   (val_images_circle)
  - identity_rect.pdf   (val_images)

Each PDF has one subplot per denoiser. Per image, a stacked bar shows
SE_avg(x) + (E_g SE - SE_avg) = E_g SE(T_g x); an overlaid marker at
SE_avg + e_1 shows the corollary's prediction. Bar tops and markers
should coincide on the disk protocol, with a controlled residual on
the rectangle protocol.
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results" / "q2_orbit_averaging_grid"

DENOISER_ORDER = ["tv", "wavelet", "nlm", "bm3d", "restormer", "restormer-rotated-noise-retrained"]
DENOISER_LABEL = {
    "tv": "TV",
    "wavelet": "Wavelet",
    "nlm": "NLM",
    "bm3d": "BM3D",
    "restormer": "Restormer",
    "restormer-rotated-noise-retrained": "Restormer-aug",
}


def sigma_root(sigma: float) -> Path:
    """Locate the protocol root for a given noise level.

    sigma=15 data lives at GRID/<protocol>/<denoiser>/ (top-level default),
    sigma=25 and sigma=50 live at GRID/sigma{25,50}/<protocol>/<denoiser>/.
    """
    if sigma == 15.0:
        return GRID
    return GRID / f"sigma{int(sigma)}"


def load_per_image(protocol: str, denoiser: str, G: int, sigma: float = 15.0) -> pd.DataFrame:
    pat = f"q2_folder-*_sigma-{sigma}_denoiser-{denoiser}_G-{G}_per_image.csv"
    protocol_dir = sigma_root(sigma) / protocol
    matches = list((protocol_dir / denoiser).glob(pat))
    if not matches:
        raise FileNotFoundError(f"no per_image CSV under {protocol_dir / denoiser} matching {pat}")
    return pd.read_csv(matches[0])


def short_label(name: str) -> str:
    # "test001.png" -> "01"
    stem = Path(name).stem
    return stem.replace("test", "")


def plot_protocol(protocol: str, G: int, out: Path, title: str, sigma: float = 15.0):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharex=False)
    axes = axes.flatten()

    for ax, denoiser in zip(axes, DENOISER_ORDER):
        df = load_per_image(protocol, denoiser, G, sigma=sigma)
        df = df.sort_values("file").reset_index(drop=True)

        idx = np.arange(len(df))
        se_avg = df["SEavg_x"].to_numpy()
        ehse = df["EhSE_hx"].to_numpy()
        gap = ehse - se_avg            # empirical gap (E_g SE - SE_avg)
        e1 = df["e1"].to_numpy()       # corollary prediction of the gap

        width = 0.78
        ax.bar(idx, se_avg, width=width, color="#3b82f6", label=r"$\mathrm{SE}_{\mathrm{avg}}(x)$")
        ax.bar(idx, gap, width=width, bottom=se_avg, color="#f59e0b",
               label=r"$\mathbb{E}_g\,\mathrm{SE}(T_gx)-\mathrm{SE}_{\mathrm{avg}}$")
        # Marker for SE_avg + e_1 — should coincide with the bar top under the identity.
        ax.scatter(idx, se_avg + e1, marker="_", s=110, linewidths=2.0,
                   color="#111827", zorder=5, label=r"$\mathrm{SE}_{\mathrm{avg}}+e_1(x)$")

        ax.set_title(DENOISER_LABEL[denoiser], fontsize=10, pad=2)
        ax.set_xticks(idx)
        ax.set_xticklabels([short_label(f) for f in df["file"]], fontsize=7)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("image", fontsize=8)
        ax.set_ylabel("squared error", fontsize=8)
        ax.margins(x=0.02)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 0.055), frameon=False, fontsize=11,
               handlelength=0.9, handletextpad=0.25, columnspacing=0.45,
               labelspacing=0.1, borderaxespad=0.0, borderpad=0.0)
    fig.suptitle(f"{title}  ($|\\mathcal{{G}}|={G}$, $\\sigma={sigma:g}/255$)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96), pad=0.2)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")


def plot_identity_scatter_multisigma(G: int, sigmas: list[float], out_png: Path):
    """Per-image identity check for the orbit-MSE identity SE_avg = E_g SE(T_g x) - e_1,
    across three noise levels.

    Layout: rows = noise levels (sigma in sigmas), columns = (disk protocol, rectangle
    protocol). Each panel scatters the empirical per-input gap
    E_g SE(T_g x) - SE_avg(x) against the denoiser-only diagnostic e_1(x), one point
    per image per denoiser, log-log axes. The diagonal y = x marks the prediction of
    Theorem main(b): points on the line validate the identity within Monte-Carlo
    error; points above the line indicate that e_1 underpredicts the gap, a
    signature of the approximate-action regime or of broken noise invariance.

    Disk protocol (left column): theory-aligned, G-invariant disk support and
    rotation-invariant noise restricted to the disk.
    Rectangle protocol (right column): G-invariance broken by rectangular support
    and full-image i.i.d. Gaussian noise.

    Denoisers without data at a given sigma (e.g. Restormer-aug at sigma in {25, 50})
    are silently skipped on the corresponding panel.

    Saves PNG at dpi=400.
    """
    markers = ["o", "s", "^", "D", "v", "P"]
    n = len(sigmas)
    fig, axes = plt.subplots(n, 2, figsize=(4.2, 1.55 * n), sharex=True, sharey=True)
    if n == 1:
        axes = axes[None, :]

    for row, sigma in enumerate(sigmas):
        for col, (protocol, label) in enumerate([("val_images_circle", "Disk"),
                                                  ("val_images",        "Rectangle")]):
            ax = axes[row, col]
            for denoiser, marker in zip(DENOISER_ORDER, markers):
                try:
                    df = load_per_image(protocol, denoiser, G, sigma=sigma)
                except FileNotFoundError:
                    continue
                ax.scatter(df["e1"], df["EhSE_hx"] - df["SEavg_x"],
                           s=18, alpha=0.8, marker=marker, label=DENOISER_LABEL[denoiser])
            ax.set_xscale("log"); ax.set_yscale("log")
            lo, hi = 1e-6, 1.0
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.7, label=r"diagonal")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.tick_params(labelsize=8)
            ax.grid(alpha=0.3, linewidth=0.4, which="both")
            if row == n - 1:
                ax.set_xlabel(r"$e_1(x)$", fontsize=9)
            if col == 0:
                ax.set_ylabel(rf"$\sigma={sigma:g}/255$" + "\n"
                              + r"$\mathbb{E}_g\,E(T_gx)-E_{\mathcal{G}}(x)$",
                              fontsize=9)
            if row == 0:
                ax.set_title(f"{label} protocol", fontsize=10, pad=2)

    # Single shared legend below the panel grid, collecting unique
    # entries across all panels so Restormer-aug (only present at sigma=15) is kept.
    handles, labels, seen = [], [], set()
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l); handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.095),
               ncol=4, fontsize=10, frameon=False, handlelength=0.8,
               handletextpad=0.25, columnspacing=0.45, labelspacing=0.1,
               borderaxespad=0.0, borderpad=0.0)

    fig.suptitle(rf"Orbit-MSE identity check  ($|\mathcal{{G}}|={G}$)", fontsize=10, y=0.965)
    fig.tight_layout(rect=(0, 0.10, 1, 0.935), pad=0.2, h_pad=0.55)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def plot_psnr_gain(out: Path, sigma: float = 15.0):
    """PSNR of vanilla single-view vs orbit-averaged denoiser, by |G|."""
    summary = pd.read_csv(GRID / "q2_orbit_averaging_grid_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for ax, dataset, label in [(axes[0], "val_images_circle", "Disk protocol"),
                                (axes[1], "val_images", "Rectangle protocol")]:
        sub = summary[(summary["dataset"] == dataset) & (summary["noise_sigma"] == sigma)]
        Gs = sorted(sub["averaging"].unique())
        x = np.arange(len(DENOISER_ORDER))
        width = 0.22
        cmap = plt.get_cmap("viridis")
        for i, G in enumerate(Gs):
            gains = []
            for d in DENOISER_ORDER:
                row = sub[(sub["model"] == d) & (sub["averaging"] == G)]
                if row.empty:
                    gains.append(np.nan)
                else:
                    gains.append(float(row["E_x_SEavg_x_psnr"].iloc[0] - row["E_x_EhSE_hx_psnr"].iloc[0]))
            ax.bar(x + (i - len(Gs) / 2 + 0.5) * width, gains, width,
                   label=f"|G|={G}", color=cmap(0.15 + 0.7 * i / max(len(Gs) - 1, 1)))
        ax.set_xticks(x)
        ax.set_xticklabels([DENOISER_LABEL[d] for d in DENOISER_ORDER], rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("PSNR gain (dB)", fontsize=9)
        ax.set_title(label, fontsize=10, pad=2)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.08), ncol=len(labels),
               handlelength=0.8, handletextpad=0.25, columnspacing=0.45,
               labelspacing=0.1, borderaxespad=0.0, borderpad=0.0)
    fig.suptitle(f"Orbit-averaging PSNR gain vs single-view  ($\\sigma={sigma:g}/255$)", fontsize=11)
    fig.tight_layout(rect=(0, 0.09, 1, 0.94), pad=0.2)
    fig.savefig(out, dpi=400, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/Users/yuqi.li@newsbreak.com/Downloads/Final_Exam_Slides/ga_figures"))
    ap.add_argument("--G", type=int, default=16)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    plot_identity_scatter_multisigma(
        G=args.G,
        sigmas=[15.0, 25.0, 50.0],
        out_png=args.out / "identity_scatter_multisigma.png",
    )


if __name__ == "__main__":
    main()
