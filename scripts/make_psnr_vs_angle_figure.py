"""PSNR vs. rotation angle for vanilla vs. group-averaged denoising (Sec. PSNR-vs-angle).

For a continuous sweep of the input rotation angle g over [0, 180) deg (the
per-angle PSNR is 180-periodic), we plot two curves per denoiser:
  - vanilla:   E(g x)        -- PSNR of the plain denoiser on the rotated scene;
  - group-avg: E_G(g x)      -- PSNR of the group-averaged denoiser, |G|=16.
The group-averaged curve is (approximately) flat across the discrete subgroup
C_16 (angles that are multiples of 360/16 = 22.5 deg, marked by ticks), while
the vanilla curve dips between those angles; the vertical gap between the two
curves is the deployment-time improvement from group averaging.

Reads results/figure5_angle_psnr_grid/figure5_angle_psnr_grid_summary.csv and
writes <draft>/figures/psnr_vs_angle.png at dpi=400 (PNG only).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "figure5_angle_psnr_grid" / "figure5_angle_psnr_grid_summary.csv"
OUT = Path("/Users/yuqi.li@newsbreak.com/Downloads/Final_Exam_Slides/ga_figures/psnr_vs_angle.png")

SIGMA = 15.0
GROUP = 16                      # |G| for the group-averaged curve
DATASET = "val_images"
PANELS = [("tv", "TV"), ("wavelet", "Wavelet"),
          ("restormer", "Restormer"), ("restormer-aug", "Restormer-aug")]

VANILLA_COLOR = "#1f77b4"
GROUPAVG_COLOR = "#d62728"


def curve(df, denoiser, estimator, group_size):
    sub = df[(df["denoiser"] == denoiser)
             & (df["noise_sigma"] == SIGMA)
             & (df["estimator"] == estimator)
             & (df["group_size"] == group_size)]
    if DATASET in set(df["dataset"]):
        sub = sub[sub["dataset"] == DATASET]
    sub = sub.sort_values("angle_deg")
    return sub["angle_deg"].to_numpy(), sub["mean_psnr"].to_numpy()


def make_figure(out_png: Path):
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.7), sharex=True)
    axes = axes.ravel()
    subgroup = np.arange(0.0, 180.0, 360.0 / GROUP)   # C_16 angles in [0,180)

    for i, (ax, (denoiser, title)) in enumerate(zip(axes, PANELS)):
        a_v, p_v = curve(df, denoiser, "vanilla", 0)
        a_g, p_g = curve(df, denoiser, "group_avg", GROUP)
        if len(a_v):
            ax.plot(a_v, p_v, "-", color=VANILLA_COLOR, linewidth=1.2,
                    marker="o", markersize=2.2, label="vanilla")
        if len(a_g):
            ax.plot(a_g, p_g, "--", color=GROUPAVG_COLOR, linewidth=1.2,
                    marker="s", markersize=2.2,
                    label=r"group-averaged ($|\mathcal{G}|=16$)")
        for s in subgroup:
            ax.axvline(s, color="gray", linewidth=0.4, linestyle=":", alpha=0.4)
        ax.set_xlim(0, 175)
        ax.set_xticks([0, 45, 90, 135, 180])
        if i // 2 == 1:                      # bottom row
            ax.set_xlabel(r"rotation angle $g$  (deg)", fontsize=10)
        if i % 2 == 0:                       # left column
            ax.set_ylabel("PSNR  (dB)", fontsize=10)
        ax.set_title(title, fontsize=11, pad=2)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3, linewidth=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="upper center",
               ncol=len(labels), frameon=False, handlelength=0.9,
               handletextpad=0.25, columnspacing=0.45, labelspacing=0.1,
               bbox_to_anchor=(0.5, 0.13), borderaxespad=0.0,
               borderpad=0.0)
    fig.tight_layout(pad=0.2, rect=(0, 0.14, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    make_figure(OUT)
