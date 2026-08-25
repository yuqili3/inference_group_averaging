#!/usr/bin/env python
"""Pre-submission diagnostics for the orbit-averaging paper.

The heavy per-angle/detail outputs go under --data2-dir.  The repo output
contains only compact CSVs, figures, and notes suitable for git.
"""
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from groupavg.config import load_config, dataset_path, restormer_weights  # noqa: E402
from groupavg.data import list_images, load_image  # noqa: E402
from groupavg.denoisers import make_denoiser  # noqa: E402
from groupavg.group_operators import FourierRotationGroup  # noqa: E402
from groupavg.metrics import l2sq, se_to_psnr  # noqa: E402
from groupavg.pipeline import denoise_one  # noqa: E402


RESTORMER_AUG_WEIGHTS = (
    "/data2/yuqi/Restormer/experiments/"
    "GaussianGrayDenoising_x_Rn_RestormerSigma15/models/net_g_148000.pth"
)

MODELS = ["wavelet", "restormer", "restormer-aug"]


class Rot90Group:
    """Exact C4 rotations by array permutation, with no interpolation/padding."""

    def __init__(self):
        self._ops = [0.0, 90.0, 180.0, 270.0]

    @property
    def ops(self):
        return self._ops

    def forward(self, img):
        return [np.rot90(np.asarray(img), k=k).copy() for k in range(4)]

    def invert(self, idx, img):
        return np.rot90(np.asarray(img), k=(-idx) % 4).copy()


def _make_model(label, base, device, sigma):
    if label == "restormer":
        return make_denoiser(
            "restormer",
            weights=restormer_weights(base, sigma=sigma, color=False),
            color=False,
            device=device,
        )
    if label == "restormer-aug":
        if float(sigma) != 15.0:
            raise ValueError("restormer-aug is only configured for sigma=15")
        return make_denoiser(
            "restormer",
            weights=RESTORMER_AUG_WEIGHTS,
            color=False,
            device=device,
        )
    return make_denoiser(label)


def _center_square(x):
    h, w = x.shape[:2]
    n = min(h, w)
    y0 = (h - n) // 2
    x0 = (w - n) // 2
    return np.asarray(x[y0:y0 + n, x0:x0 + n], dtype=np.float32)


def _noise(shape, sigma, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma / 255.0, size=shape).astype(np.float32)


def _stable_int(*parts):
    text = "::".join(str(p) for p in parts)
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:12], 16) % 1000003


def _rot_pose(clean, angle, method):
    if method == "rot90":
        k = int(round(angle / 90.0)) % 4
        return np.rot90(clean, k=k).copy()
    group = FourierRotationGroup(K=1, angles=[float(angle)], expand=False)
    return group.forward(clean)[0].astype(np.float32)


def _q2_from_noisy(clean, noisy, denoiser, denoiser_name, group, sigma,
                   clip_denoised=True):
    clean = np.asarray(clean, dtype=np.float32)
    noisy = np.asarray(noisy, dtype=np.float32)
    mask = np.ones(clean.shape, dtype=np.float32)
    num_pixels = float(clean.size)

    d_y = denoise_one(noisy, denoiser, sigma)
    if clip_denoised:
        d_y = np.clip(d_y, 0.0, 1.0)

    z_list = []
    se_h_raw = []
    detail = []
    for hi, (hx, hy) in enumerate(zip(group.forward(clean), group.forward(noisy))):
        d_hy = denoise_one(hy, denoiser, sigma)
        if clip_denoised:
            d_hy = np.clip(d_hy, 0.0, 1.0)
        se_raw = l2sq(d_hy, hx)
        se_h_raw.append(se_raw)
        z = group.invert(hi, d_hy).astype(np.float32)
        if clip_denoised:
            z = np.clip(z, 0.0, 1.0)
        z_list.append(z)
        detail.append({
            "denoiser": denoiser_name,
            "rotation_index": hi,
            "rotation_angle_deg": float(group.ops[hi]),
            "SE_hx": se_raw / num_pixels,
            "SE_hx_psnr": se_to_psnr(se_raw / num_pixels),
        })

    w = np.mean(np.stack(z_list, axis=0), axis=0).astype(np.float32)
    if clip_denoised:
        w = np.clip(w, 0.0, 1.0)

    e_maps = [z - d_y for z in z_list]
    e_l2 = [l2sq(e, mask=mask) for e in e_maps]
    e_mean = np.mean(np.stack(e_maps, axis=0), axis=0)
    e1_raw = float(np.mean(e_l2) - l2sq(e_mean, mask=mask))
    ehse_raw = float(np.mean(se_h_raw))
    seavg_raw = l2sq(w, clean)
    vanilla_raw = l2sq(d_y, clean)
    return {
        "EhSE_hx": ehse_raw / num_pixels,
        "SEavg_x": seavg_raw / num_pixels,
        "e1": e1_raw / num_pixels,
        "vanilla_SE": vanilla_raw / num_pixels,
        "EhSE_minus_SEavg": (ehse_raw - seavg_raw) / num_pixels,
        "vanilla_minus_SEavg": (vanilla_raw - seavg_raw) / num_pixels,
        "EhSE_hx_psnr": se_to_psnr(ehse_raw / num_pixels),
        "SEavg_x_psnr": se_to_psnr(seavg_raw / num_pixels),
        "vanilla_psnr": se_to_psnr(vanilla_raw / num_pixels),
        "detail": detail,
    }


def _load_square_images(base, max_images):
    files = list_images(dataset_path(base, "val_images"))[:max_images]
    return [(os.path.basename(f), _center_square(load_image(f))) for f in files]


def _write(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_exp1(base, args, repo_dir, data2_dir):
    rows = []
    details = []
    images = _load_square_images(base, args.max_images)
    group = Rot90Group()
    for model in MODELS:
        denoiser = _make_model(model, base, args.device, args.sigma)
        for clip_denoised in ([True, False] if args.exp1_both_clip else [True]):
            for file_name, clean in images:
                for noise_id in range(args.num_noise):
                    noisy = clean + _noise(clean.shape, args.sigma, args.seed + noise_id * 9176 + _stable_int(file_name))
                    out = _q2_from_noisy(clean, noisy, denoiser, model, group, args.sigma, clip_denoised)
                    row = {k: v for k, v in out.items() if k != "detail"}
                    row.update({
                        "experiment": "exp1_c4_identity",
                        "denoiser": model,
                        "file": file_name,
                        "noise_id": noise_id,
                        "sigma": args.sigma,
                        "clip_denoised": clip_denoised,
                        "group": "exact_rot90_C4",
                    })
                    rows.append(row)
                    for d in out["detail"]:
                        d.update(row)
                        details.append(d)
    per = pd.DataFrame(rows)
    summary = _summarize_identity(per, ["denoiser", "clip_denoised"])
    _write(per, repo_dir / "exp1_c4_identity_per_noise.csv")
    _write(summary, repo_dir / "exp1_c4_identity_summary.csv")
    _write(pd.DataFrame(details), data2_dir / "exp1_c4_identity_detail.csv")
    _plot_identity(per, repo_dir / "exp1_c4_identity_scatter.png", "Exp 1: exact C4 identity")
    return per, summary


def run_exp2(base, args, repo_dir, data2_dir):
    rows = []
    details = []
    images = _load_square_images(base, args.max_images)
    c16 = FourierRotationGroup(K=16, expand=False)
    poses = [float(a) for a in args.pose_angles]
    for model in MODELS:
        denoiser = _make_model(model, base, args.device, args.sigma)
        for file_name, clean0 in images:
            for pose in poses:
                pose_methods = ["fft3shear"]
                if pose in {0.0, 90.0}:
                    pose_methods.append("rot90")
                for method in pose_methods:
                    clean = _rot_pose(clean0, pose, "rot90" if method == "rot90" else "fft3shear")
                    for noise_id in range(args.num_noise):
                        noisy = clean + _noise(
                            clean.shape,
                            args.sigma,
                            args.seed + noise_id * 9176 + _stable_int(file_name, pose, method),
                        )
                        out = _q2_from_noisy(clean, noisy, denoiser, model, c16, args.sigma, True)
                        row = {k: v for k, v in out.items() if k != "detail"}
                        row.update({
                            "experiment": "exp2_pose_constancy",
                            "denoiser": model,
                            "file": file_name,
                            "noise_id": noise_id,
                            "sigma": args.sigma,
                            "input_pose_deg": pose,
                            "pose_method": method,
                            "group": "fourier_C16_noexpand",
                            "clip_denoised": True,
                        })
                        rows.append(row)
                        for d in out["detail"]:
                            d.update(row)
                            details.append(d)
    per_noise = pd.DataFrame(rows)
    per_pose = (
        per_noise.groupby(["denoiser", "input_pose_deg", "pose_method"], as_index=False)
        .agg(
            mean_SEavg_x=("SEavg_x", "mean"),
            mean_e1=("e1", "mean"),
            mean_EhSE_minus_SEavg=("EhSE_minus_SEavg", "mean"),
            mean_SEavg_psnr=("SEavg_x_psnr", "mean"),
            n=("file", "size"),
        )
    )
    summary = (
        per_pose.groupby(["denoiser", "pose_method"], as_index=False)
        .agg(
            SEavg_range=("mean_SEavg_x", lambda s: float(np.max(s) - np.min(s))),
            e1_range=("mean_e1", lambda s: float(np.max(s) - np.min(s))),
            mean_e1=("mean_e1", "mean"),
            mean_SEavg_x=("mean_SEavg_x", "mean"),
            n=("n", "sum"),
        )
    )
    _write(per_pose, repo_dir / "exp2_pose_constancy_per_pose.csv")
    _write(summary, repo_dir / "exp2_pose_constancy_summary.csv")
    _write(per_noise, data2_dir / "exp2_pose_constancy_per_noise.csv")
    _write(pd.DataFrame(details), data2_dir / "exp2_pose_constancy_detail.csv")
    _plot_pose(per_pose, repo_dir / "exp2_pose_constancy.png")
    return per_pose, summary


def run_exp3(base, args, repo_dir, data2_dir):
    rows = []
    images = _load_square_images(base, 2)
    c16 = FourierRotationGroup(K=16, expand=False)
    fig1_e = _load_fig1_rectangle_e()
    for model in MODELS:
        denoiser = _make_model(model, base, args.device, args.sigma)
        for file_name, clean0 in images:
            for pose in args.exp3_angles:
                clean = _rot_pose(clean0, pose, "fft3shear")
                for noise_id in range(args.num_noise):
                    noisy = clean + _noise(
                        clean.shape,
                        args.sigma,
                        args.seed + noise_id * 9176 + _stable_int(file_name, pose),
                    )
                    out = _q2_from_noisy(clean, noisy, denoiser, model, c16, args.sigma, True)
                    row = {k: v for k, v in out.items() if k != "detail"}
                    row.update({
                        "experiment": "exp3_fixed_canvas_gain_vs_e",
                        "denoiser": model,
                        "file": file_name,
                        "noise_id": noise_id,
                        "sigma": args.sigma,
                        "input_pose_deg": pose,
                        "group": "fourier_C16_noexpand",
                        "clip_denoised": True,
                        "fig1_rectangle_e1_mean": fig1_e.get(model, np.nan),
                    })
                    rows.append(row)
    per_noise = pd.DataFrame(rows)
    per_pose = (
        per_noise.groupby(["denoiser", "input_pose_deg"], as_index=False)
        .agg(
            fixed_canvas_gain=("vanilla_minus_SEavg", "mean"),
            fixed_canvas_e1=("e1", "mean"),
            identity_gain=("EhSE_minus_SEavg", "mean"),
            fig1_rectangle_e1_mean=("fig1_rectangle_e1_mean", "mean"),
            vanilla_psnr=("vanilla_psnr", "mean"),
            avg_psnr=("SEavg_x_psnr", "mean"),
            n=("file", "size"),
        )
    )
    summary = (
        per_pose.groupby("denoiser", as_index=False)
        .agg(
            mean_fixed_canvas_gain=("fixed_canvas_gain", "mean"),
            mean_fixed_canvas_e1=("fixed_canvas_e1", "mean"),
            mean_identity_gain=("identity_gain", "mean"),
            mean_fig1_rectangle_e1=("fig1_rectangle_e1_mean", "mean"),
            mean_vanilla_psnr=("vanilla_psnr", "mean"),
            mean_avg_psnr=("avg_psnr", "mean"),
        )
    )
    _write(per_pose, repo_dir / "exp3_fixed_canvas_gain_vs_e_per_pose.csv")
    _write(summary, repo_dir / "exp3_fixed_canvas_gain_vs_e_summary.csv")
    _write(per_noise, data2_dir / "exp3_fixed_canvas_gain_vs_e_per_noise.csv")
    _plot_exp3(per_pose, repo_dir / "exp3_fixed_canvas_gain_vs_e.png")
    return per_pose, summary


def _summarize_identity(per, keys):
    return (
        per.groupby(keys, as_index=False)
        .agg(
            mean_e1=("e1", "mean"),
            mean_gain=("EhSE_minus_SEavg", "mean"),
            mean_abs_identity_error=("identity_abs_error", "mean") if "identity_abs_error" in per else ("e1", "mean"),
            mean_EhSE_psnr=("EhSE_hx_psnr", "mean"),
            mean_SEavg_psnr=("SEavg_x_psnr", "mean"),
            n=("file", "size"),
        )
    )


def _add_identity_error(df):
    df = df.copy()
    df["identity_error"] = df["EhSE_minus_SEavg"] - df["e1"]
    df["identity_abs_error"] = df["identity_error"].abs()
    df["gain_over_e"] = df["EhSE_minus_SEavg"] / df["e1"].replace(0.0, np.nan)
    return df


def _load_fig1_rectangle_e():
    out = {}
    roots = [
        Path("results/q2_orbit_averaging_grid/sigma15/val_images"),
        Path("results/q2_orbit_averaging_grid/val_images"),
    ]
    aliases = {
        "restormer": "restormer",
        "restormer-aug": "restormer-rotated-noise-retrained",
        "wavelet": "wavelet",
    }
    for model, folder in aliases.items():
        files = []
        for root in roots:
            files.extend(sorted((root / folder).glob("*_per_image.csv")))
        vals = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            if "e1" in df:
                vals.extend(df["e1"].dropna().tolist())
        if vals:
            out[model] = float(np.mean(vals))
    return out


def _plot_identity(per, path, title):
    df = _add_identity_error(per)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for model, sub in df.groupby("denoiser"):
        sub = sub[sub["clip_denoised"] == True]  # noqa: E712
        ax.scatter(sub["e1"], sub["EhSE_minus_SEavg"], s=22, alpha=0.8, label=model)
    vals = np.r_[df["e1"].to_numpy(), df["EhSE_minus_SEavg"].to_numpy()]
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("e")
    ax.set_ylabel("E_g SE(T_g x) - SE_G(x)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_pose(per_pose, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), sharex=True)
    for model, sub in per_pose[per_pose["pose_method"] == "fft3shear"].groupby("denoiser"):
        axes[0].plot(sub["input_pose_deg"], sub["mean_SEavg_x"], marker="o", ms=3, label=model)
        axes[1].plot(sub["input_pose_deg"], sub["mean_e1"], marker="o", ms=3, label=model)
    for ax, ylabel in zip(axes, ["SE_G(x')", "e(x')"]):
        ax.set_xlabel("input pose (deg)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_exp3(per_pose, path):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for model, sub in per_pose.groupby("denoiser"):
        ax.plot(sub["input_pose_deg"], sub["fixed_canvas_gain"], marker="o", ms=4, label=f"{model}: gain")
        ax.plot(sub["input_pose_deg"], sub["fixed_canvas_e1"], linestyle="--", marker="x", ms=4, label=f"{model}: e")
    ax.set_xlabel("input pose (deg)")
    ax.set_ylabel("MSE reduction")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _markdown_table(df):
    if df.empty:
        return "(empty)"
    view = df.copy()
    for col in view.columns:
        if np.issubdtype(view[col].dtype, np.floating):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_notes(repo_dir, summaries):
    lines = [
        "# Pre-submission Diagnostics",
        "",
        "These diagnostics address the reviewer/YB questions about interpolation,",
        "noise coloring, and whether the denoising diagnostic measured in one",
        "geometry transfers to the downstream fixed-canvas geometry.",
        "",
        "Heavy per-noise/detail CSVs were written outside the repository under",
        "`/data2/yuqi/inference_group_averaging/presubmission_diagnostics/`.",
        "",
    ]
    if "exp1" in summaries:
        s = _add_identity_error(summaries["exp1"])
        lines += [
            "## Exp 1: exact C4 identity",
            "",
            "Exact `np.rot90` removes interpolation and padding. The main diagnostic",
            "is whether `EhSE_minus_SEavg` returns to the diagonal with `e1`.",
            "",
            _markdown_table(s.groupby(["denoiser", "clip_denoised"], as_index=False)
            .agg(mean_e1=("e1", "mean"), mean_gain=("EhSE_minus_SEavg", "mean"),
                 mean_abs_error=("identity_abs_error", "mean"), n=("file", "size"))),
            "",
        ]
    if "exp2" in summaries:
        lines += [
            "## Exp 2: orbit constancy across input pose",
            "",
            "The pose sweep rotates the clean image first and then adds upright",
            "sensor noise. Departures from flat `SE_G` and `e` curves measure",
            "geometry/noise-statistic mismatch in the deployment-like setting.",
            "",
            _markdown_table(summaries["exp2"]),
            "",
        ]
    if "exp3" in summaries:
        lines += [
            "## Exp 3: fixed-canvas gain vs e",
            "",
            "This compares the pure-denoising gain in the downstream fixed-canvas",
            "geometry to the corresponding same-geometry `e`, and contrasts both",
            "with the Fig. 1 rectangle-protocol `e` when available.",
            "",
            _markdown_table(summaries["exp3"]),
            "",
        ]
    (repo_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml"))
    ap.add_argument("--repo-dir", default="results/presubmission_diagnostics")
    ap.add_argument("--data2-dir", default="/data2/yuqi/inference_group_averaging/presubmission_diagnostics")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sigma", type=float, default=15.0)
    ap.add_argument("--max-images", type=int, default=10)
    ap.add_argument("--num-noise", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pose-angles", type=float, nargs="+", default=list(np.arange(0, 91, 5)))
    ap.add_argument("--exp3-angles", type=float, nargs="+", default=[0, 15, 30, 45, 60, 75, 90])
    ap.add_argument("--only", nargs="+", choices=["exp1", "exp2", "exp3"], default=["exp1", "exp2", "exp3"])
    ap.add_argument("--exp1-both-clip", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    base = load_config(args.base)
    args.device = args.device or base.get("device", "cuda:0")
    repo_dir = Path(args.repo_dir)
    data2_dir = Path(args.data2_dir)
    repo_dir.mkdir(parents=True, exist_ok=True)
    data2_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    if "exp1" in args.only:
        per, summary = run_exp1(base, args, repo_dir, data2_dir)
        per = _add_identity_error(per)
        _write(per, repo_dir / "exp1_c4_identity_per_noise.csv")
        _write(_summarize_identity(per, ["denoiser", "clip_denoised"]), repo_dir / "exp1_c4_identity_summary.csv")
        summaries["exp1"] = per
    if "exp2" in args.only:
        _, summary = run_exp2(base, args, repo_dir, data2_dir)
        summaries["exp2"] = summary
    if "exp3" in args.only:
        _, summary = run_exp3(base, args, repo_dir, data2_dir)
        summaries["exp3"] = summary
    write_notes(repo_dir, summaries)

    if args.push:
        subprocess.run(["git", "add", "scripts/run_presubmission_diagnostics.py"], check=True)
        subprocess.run(["git", "add", "-f", str(repo_dir)], check=True)
        subprocess.run(["git", "commit", "-m", "Add presubmission diagnostic results"], check=False)
        env = os.environ.copy()
        env["GIT_SSH_COMMAND"] = "ssh -i ~/.ssh/github_codex -o IdentitiesOnly=yes"
        subprocess.run(["git", "push", "origin", "main"], check=True, env=env)


if __name__ == "__main__":
    main()
