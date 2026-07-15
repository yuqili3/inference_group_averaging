"""Downstream effect of orbit averaging inside PnP / RED solvers.

Two-panel headline figure, read from the grid summary produced by
`scripts/run_downstream_effect_grid.py`
(results/downstream_effect_grid/downstream_effect_grid_summary.csv):

  Left  — Reconstruction quality. Paired bars of reconstruction PSNR
          (vanilla denoiser D vs. orbit-averaged D_G) for each
          problem x solver, annotated with the PSNR gain (dB).
  Right — Downstream equivariance. Downstream equivariance PSNR
          (higher = the reconstruction is more equivariant) as a
          function of the rotation angle, for D vs. D_G, averaged over
          problems and solvers.

The figure is drawn for one denoiser at one sigma (default restormer,
sigma=15); pass --denoiser / --sigma to select another slice.

Writes figures/downstream_effect.png at dpi=400 (PNG only).
"""

from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "results" / "downstream_effect_grid" / "downstream_effect_grid_summary.csv"

MODE_ORDER = ["vanilla", "group_avg"]
MODE_LABEL = {"vanilla": r"Vanilla $D$", "group_avg": r"Group-averaged $D_G$"}
MODE_COLOR = {"vanilla": "#7f7f7f", "group_avg": "#2ca02c"}

ALGO_LABEL = {"pnp_hqs": "PnP-HQS", "red_gd": "RED", "diffusion_style": "Diffusion"}
PROBLEM_LABEL = {"blur": "Deblur", "inpaint": "Inpaint"}


def _pretty_algo(a):
    return ALGO_LABEL.get(a, a)


def _pretty_problem(p):
    return PROBLEM_LABEL.get(p, p)


def _select_slice(df, denoiser, sigma):
    """Filter to one (denoiser, sigma); fall back to the first available."""
    if denoiser is not None and denoiser in set(df["denoiser"]):
        df = df[df["denoiser"] == denoiser]
    else:
        pick = sorted(df["denoiser"].unique())[0]
        if denoiser is not None:
            print(f"[warn] denoiser {denoiser!r} not in CSV; using {pick!r}")
        df = df[df["denoiser"] == pick]
    sigmas = sorted(df["denoiser_sigma"].unique())
    if sigma is not None and float(sigma) in set(float(s) for s in sigmas):
        df = df[df["denoiser_sigma"].astype(float) == float(sigma)]
    else:
        pick = sigmas[0]
        if sigma is not None:
            print(f"[warn] sigma {sigma!r} not in CSV; using {pick}")
        df = df[df["denoiser_sigma"].astype(float) == float(pick)]
    return df.copy()


def make_figure(out_png: Path, csv_path: Path = CSV, denoiser="restormer", sigma=15.0):
    df = pd.read_csv(csv_path)
    df = _select_slice(df, denoiser, sigma)
    if df.empty:
        raise ValueError(f"No rows to plot after filtering {csv_path}")
    den = df["denoiser"].iloc[0]
    sig = float(df["denoiser_sigma"].iloc[0])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.0))

    # ---- Left: reconstruction PSNR, vanilla vs group_avg -------------------
    # base_psnr is constant across angle; average to collapse the angle axis.
    recon = (df.groupby(["problem", "algorithm", "denoiser_mode"], as_index=False)
               ["mean_base_psnr"].mean())
    combos = (recon[["problem", "algorithm"]].drop_duplicates()
                    .sort_values(["problem", "algorithm"]).values.tolist())
    xlabels = [f"{_pretty_problem(p)}\n{_pretty_algo(a)}" for p, a in combos]
    xs = range(len(combos))
    width = 0.38
    for k, mode in enumerate(MODE_ORDER):
        vals = []
        for p, a in combos:
            r = recon[(recon["problem"] == p) & (recon["algorithm"] == a)
                      & (recon["denoiser_mode"] == mode)]
            vals.append(float(r["mean_base_psnr"].iloc[0]) if len(r) else float("nan"))
        offset = (k - 0.5) * width
        ax_l.bar([x + offset for x in xs], vals, width,
                 color=MODE_COLOR[mode], label=MODE_LABEL[mode])
    # annotate the per-combo gain (group_avg - vanilla)
    for i, (p, a) in enumerate(combos):
        def _v(mode):
            r = recon[(recon["problem"] == p) & (recon["algorithm"] == a)
                      & (recon["denoiser_mode"] == mode)]
            return float(r["mean_base_psnr"].iloc[0]) if len(r) else float("nan")
        v0, v1 = _v("vanilla"), _v("group_avg")
        if v0 == v0 and v1 == v1:
            ax_l.annotate(f"{v1 - v0:+.2f}", (i, max(v0, v1)),
                          textcoords="offset points", xytext=(0, 3),
                          ha="center", fontsize=8, color=MODE_COLOR["group_avg"])
    ax_l.set_xticks(list(xs))
    ax_l.set_xticklabels(xlabels, fontsize=9)
    ax_l.set_ylabel("Reconstruction PSNR (dB)", fontsize=10)
    ax_l.set_title("Reconstruction quality (gain in dB)", fontsize=10)
    ax_l.grid(axis="y", alpha=0.3, linewidth=0.4)

    # ---- Right: downstream equivariance PSNR vs angle ----------------------
    eqv = (df.groupby(["denoiser_mode", "angle_deg"], as_index=False)
             ["mean_downstream_equivariance_psnr"].mean())
    for mode in MODE_ORDER:
        sub = eqv[eqv["denoiser_mode"] == mode].sort_values("angle_deg")
        if sub.empty:
            continue
        ax_r.plot(sub["angle_deg"], sub["mean_downstream_equivariance_psnr"],
                  color=MODE_COLOR[mode], marker="o", markersize=3, linewidth=1.2,
                  label=MODE_LABEL[mode])
    ax_r.set_xlabel("rotation angle (deg)", fontsize=10)
    ax_r.set_ylabel("Downstream equivariance PSNR (dB)", fontsize=10)
    ax_r.set_title("Reconstruction equivariance (higher = better)", fontsize=10)
    ax_r.grid(alpha=0.3, linewidth=0.4)

    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=10, loc="upper center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Downstream effect of orbit averaging  ({den}, $\\sigma={sig:g}$)",
                 fontsize=11, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--denoiser", default="restormer")
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--out", default=str(REPO / "figures" / "downstream_effect.png"))
    args = ap.parse_args()
    make_figure(Path(args.out), csv_path=Path(args.csv),
                denoiser=args.denoiser, sigma=args.sigma)
