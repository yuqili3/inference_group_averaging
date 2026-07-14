"""Rotation operator quality: PSNR of T_g^{-1} T_g x as a function of the rotation angle.

For each rotation operator implementation (FFT 3-shear, OpenCV bilinear affine, SciPy
ndimage rotate), we apply T_g then T_g^{-1} to a reference signal x and report the
round-trip reconstruction PSNR vs. the rotation angle. A perfectly isometric and
invertible operator would yield 99 dB (capped) at every angle. The shortfall is the
implementation residual epsilon_T analysed in Sec. approx.

Three reference signals isolate different failure modes:
  - astronaut          : natural image (smooth + high-frequency edges);
  - astronaut_sigma15  : same with additive Gaussian noise sigma=15/255;
  - pure_gaussian_noise: white noise (worst case for any interpolation-based operator).

Reads results/rotation_operator_quality/rotation_operator_quality_all.csv and writes
figures/rotation_operator_quality.png at dpi=400.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "rotation_operator_quality" / "rotation_operator_quality_all.csv"

SIGNAL_ORDER = ["astronaut", "astronaut_sigma15", "pure_gaussian_noise"]
SIGNAL_LABEL = {
    "astronaut": "Clean image",
    "astronaut_sigma15": r"Noisy image ($\sigma=15/255$)",
    "pure_gaussian_noise": "White Gaussian noise",
}
OPERATOR_ORDER = ["fourier_rotation", "cv2_affine_rotation", "scipy_ndimage_rotate"]
OPERATOR_LABEL = {
    "fourier_rotation": "FFT 3-shear",
    "cv2_affine_rotation": "OpenCV bilinear",
    "scipy_ndimage_rotate": "SciPy ndimage spline",
}
OPERATOR_COLOR = {
    "fourier_rotation": "#1f77b4",
    "cv2_affine_rotation": "#ff7f0e",
    "scipy_ndimage_rotate": "#2ca02c",
}


def make_figure(out_png: Path):
    df = pd.read_csv(CSV)
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.0), sharey=True)
    for ax, signal in zip(axes, SIGNAL_ORDER):
        sub = df[df["signal"] == signal]
        for op in OPERATOR_ORDER:
            ops = sub[sub["operator"] == op].sort_values("angle_deg")
            if ops.empty:
                continue
            ax.plot(ops["angle_deg"], ops["psnr"],
                    color=OPERATOR_COLOR[op],
                    linewidth=1.0,
                    marker="o", markersize=2.2,
                    label=OPERATOR_LABEL[op])
        for a in (90, 180, 270):
            ax.axvline(a, color="gray", linewidth=0.5, linestyle=":", alpha=0.5)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("rotation angle (deg)", fontsize=10)
        ax.set_title(SIGNAL_LABEL[signal], fontsize=11, pad=2)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.3, linewidth=0.4)
    axes[0].set_ylabel(r"PSNR$(x,\, T_g^{-1} T_g x)$  (dB)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12, loc="upper center",
               ncol=len(labels), frameon=False, handlelength=0.9,
               handletextpad=0.25, columnspacing=0.45, labelspacing=0.1,
               bbox_to_anchor=(0.5, 0.15), borderaxespad=0.0,
               borderpad=0.0)
    fig.tight_layout(pad=0.2, rect=(0, 0.15, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    out_dir = Path("/Users/yuqi.li@newsbreak.com/Downloads/Final_Exam_Slides/ga_figures")
    make_figure(out_dir / "rotation_operator_quality.png")
