"""Visualize the downstream effect of orbit averaging on a single image.

Runs a PnP/RED solver on one clean image and one degradation (blur/inpaint) and
lays out:  Clean | Degraded | <intermediate iterates> | Final,  with PSNR/MSE
under each panel. Two rows compare the vanilla denoiser D against its
orbit-averaged version D_G, so you can watch how making the inner denoiser
equivariant changes the reconstruction trajectory.

The data side (denoiser + solver) needs the groupavg deps (cv2/skimage/pywt;
torch + CUDA for Restormer) and runs on the GPU box. The plotting function
`plot_panels` is dependency-light (numpy + matplotlib) and unit-testable alone.

Example (GPU box):
  ~/anaconda3/envs/restormer37/bin/python scripts/make_downstream_visualization.py \
      --denoiser wavelet --problem inpaint --algorithm pnp_hqs \
      --dataset val_images --image-index 0 --num-iter 12 --snapshots 3 \
      --out figures/downstream_vis.png
"""
from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]

MODE_LABEL = {"vanilla": r"Vanilla $D$", "group_avg": r"Group-avg $D_G$"}


def snapshot_indices(n_iter, k):
    """k evenly spaced iterate indices in [0, n_iter-2] (final shown separately)."""
    hi = n_iter - 2  # exclude the last iterate; it is shown as 'Final'
    if hi < 0:
        return []
    k = max(1, min(k, hi + 1))
    idx = np.linspace(0, hi, k)
    return sorted(set(int(round(v)) for v in idx))


def _fmt(psnr, mse):
    return f"{psnr:.2f} dB\nMSE {mse:.1e}"


def plot_panels(rows, out_png, suptitle=None):
    """Render one row of panels per denoiser mode.

    rows: list of dicts, each with keys:
        label, clean, degraded, degraded_psnr, degraded_mse,
        snapshots=[{iter, image, psnr, mse}, ...], final, final_psnr, final_mse
    Columns: Clean | Degraded | snapshots... | Final.
    """
    n_snap = max((len(r["snapshots"]) for r in rows), default=0)
    ncols = 2 + n_snap + 1
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * ncols, 2.55 * nrows),
                             squeeze=False, layout="constrained")

    def show(ax, img, title="", sub=None):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)
        if sub:
            ax.set_xlabel(sub, fontsize=7.5)

    for r, row in enumerate(rows):
        top = (r == 0)
        show(axes[r][0], row["clean"], "Clean" if top else "")
        axes[r][0].set_ylabel(row["label"], fontsize=10)
        show(axes[r][1], row["degraded"], "Degraded" if top else "",
             _fmt(row["degraded_psnr"], row["degraded_mse"]))
        for j in range(n_snap):
            ax = axes[r][2 + j]
            if j < len(row["snapshots"]):
                s = row["snapshots"][j]
                show(ax, s["image"], (f"iter {s['iter'] + 1}" if top else ""),
                     _fmt(s["psnr"], s["mse"]))
            else:
                ax.axis("off")
        show(axes[r][-1], row["final"], "Final" if top else "",
             _fmt(row["final_psnr"], row["final_mse"]))

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


def build_row(res, label, n_snap):
    """Turn a run_single_with_trajectory result into a plot_panels row."""
    idx = snapshot_indices(len(res["trajectory"]), n_snap)
    snaps = [{
        "iter": res["trajectory"][i]["iter"],
        "image": res["trajectory"][i]["image"],
        "psnr": res["trajectory"][i]["psnr"],
        "mse": res["trajectory"][i]["se"],
    } for i in idx]
    return {
        "label": label,
        "clean": res["clean"],
        "degraded": res["degraded"],
        "degraded_psnr": res["degraded_psnr"],
        "degraded_mse": res["degraded_se"],
        "snapshots": snaps,
        "final": res["final"],
        "final_psnr": res["final_psnr"],
        "final_mse": res["final_se"],
    }


def main():
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from groupavg.config import load_config, dataset_path, restormer_weights
    from groupavg.denoisers import make_denoiser
    from groupavg.data import list_images, load_image
    from groupavg.experiments import downstream_effect as de

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(REPO / "configs" / "base.yaml"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--denoiser", default="wavelet")
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--dataset", default="val_images")
    ap.add_argument("--image", default=None, help="path to a single image (overrides --dataset)")
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--problem", default="inpaint", choices=["blur", "inpaint"])
    ap.add_argument("--algorithm", default="pnp_hqs",
                    choices=["pnp_hqs", "red_gd", "diffusion_style"])
    ap.add_argument("--modes", nargs="+", default=["vanilla", "group_avg"])
    ap.add_argument("--group-name", default="fourier_rotation")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--num-iter", type=int, default=12)
    ap.add_argument("--snapshots", type=int, default=3)
    ap.add_argument("--measurement-sigma", type=float, default=2.0)
    ap.add_argument("--keep-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rho", type=float, default=0.8)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--data-step", type=float, default=0.4)
    ap.add_argument("--prior-step", type=float, default=0.25)
    ap.add_argument("--out", default=str(REPO / "figures" / "downstream_vis.png"))
    args = ap.parse_args()

    base = load_config(args.base)
    device = args.device or base.get("device", "cuda:0")

    if args.denoiser == "restormer":
        weights = restormer_weights(base, sigma=args.sigma, color=False)
        denoiser = make_denoiser("restormer", weights=weights, color=False, device=device)
    else:
        denoiser = make_denoiser(args.denoiser)

    if args.image:
        clean = load_image(args.image)
    else:
        files = list_images(dataset_path(base, args.dataset))
        clean = load_image(files[args.image_index])

    problem = de._make_problem(args.problem, clean.shape, seed=args.seed,
                               keep_fraction=args.keep_fraction)

    rows = []
    for mode in args.modes:
        denoise_fn = de._make_denoise_fn(denoiser, args.sigma, mode,
                                         group_name=args.group_name,
                                         group_size=args.group_size)
        res = de.run_single_with_trajectory(
            clean, problem, denoise_fn, algorithm=args.algorithm,
            num_iter=args.num_iter, rho=args.rho, step=args.step, lam=args.lam,
            data_step=args.data_step, prior_step=args.prior_step,
            measurement_sigma=args.measurement_sigma, seed=args.seed,
        )
        rows.append(build_row(res, MODE_LABEL.get(mode, mode), args.snapshots))
        print(f"[{mode}] degraded {res['degraded_psnr']:.2f} dB "
              f"-> final {res['final_psnr']:.2f} dB")

    suptitle = (f"Downstream {args.algorithm} on {args.problem}  "
                f"({args.denoiser}, $\\sigma={args.sigma:g}$)")
    plot_panels(rows, Path(args.out), suptitle=suptitle)


if __name__ == "__main__":
    main()
