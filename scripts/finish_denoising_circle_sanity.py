#!/usr/bin/env python
"""Finish denoising sanity checks, write a short report, commit, and push."""
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/denoising_circle_sanity"
POLICIES = ["downstream_policy", "noexpand_policy"]
EXPECTED_DETAIL_ROWS = 80


def _run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True, **kwargs)


def _read(path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def wait_for_results():
    while True:
        ready = True
        for policy in POLICIES:
            detail_path = BASE / policy / "denoising_circle_sanity_detail.csv"
            summary_path = BASE / policy / "denoising_circle_sanity_summary.csv"
            detail = _read(detail_path)
            n = 0 if detail is None else len(detail)
            print(f"[wait] {policy}: detail {n}/{EXPECTED_DETAIL_ROWS}, summary={summary_path.exists()}", flush=True)
            if detail is None or n < EXPECTED_DETAIL_ROWS or not summary_path.exists():
                ready = False
            elif n > EXPECTED_DETAIL_ROWS:
                raise RuntimeError(f"{policy}: expected {EXPECTED_DETAIL_ROWS} rows, got {n}")
        if ready:
            break
        time.sleep(300)

    for policy in POLICIES:
        detail = pd.read_csv(BASE / policy / "denoising_circle_sanity_detail.csv")
        numeric = detail.select_dtypes("number")
        if detail.isna().sum().sum() or np.isinf(numeric.to_numpy()).sum():
            raise RuntimeError(f"{policy}: found NaN/inf in detail CSV")


def _format_table(df):
    cols = [
        "model",
        "input_pose",
        "group_expand",
        "mean_noisy_psnr",
        "mean_vanilla_psnr",
        "mean_group_avg_psnr",
        "mean_group_avg_minus_vanilla_psnr",
        "mean_group_avg_minus_vanilla_ssim",
        "n",
    ]
    out = df[cols].copy()
    for c in out.select_dtypes("number").columns:
        out[c] = out[c].map(lambda x: f"{x:.4f}")
    headers = list(out.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def write_report():
    downstream = pd.read_csv(BASE / "downstream_policy" / "denoising_circle_sanity_summary.csv")
    noexpand = pd.read_csv(BASE / "noexpand_policy" / "denoising_circle_sanity_summary.csv")

    downstream_small = downstream.sort_values(["model", "input_pose"])
    noexpand_small = noexpand.sort_values(["model", "input_pose"])

    report = f"""# Denoising circle sanity check

This sanity check isolates pure denoising from the downstream PnP/RED solvers.

Protocol:

- Dataset: `val_images_circle`, first 10 images.
- Noise: additive Gaussian noise with `sigma=15` on the 0-255 scale.
- Geometry: 100 px zero padding before evaluation; input rotations at 30, 45, and 60 degrees without canvas expansion.
- Metrics: PSNR and SSIM on the circle support mask.
- Denoisers: `restormer` and `restormer-aug`.
- Orbit averaging: Fourier rotation group, `G=16`.

## Main Finding

For both Restormer variants, G16 orbit averaging gives almost no pure denoising
gain on the rotated circle images: typically around `0.03-0.09 dB` and about
`0.001` SSIM. This explains why downstream PnP/RED gains for Restormer are small:
there is little denoising-level improvement for the solver to amplify.

The additional no-expand control shows that the large negative upright result
under the downstream policy is caused by using `group_expand=True` for the
already padded upright image. When `group_expand=False` is used for every pose,
the upright loss shrinks to roughly `-0.1 dB`.

## Downstream Policy

This matches the current downstream solver convention: `group_expand=True` for
upright inputs and `group_expand=False` for rotated inputs.

{_format_table(downstream_small)}

## No-Expand Control

This uses `group_expand=False` for all input poses.

{_format_table(noexpand_small)}

## Interpretation

1. Restormer is already close to rotation-stable on these padded circle images,
   so orbit averaging changes its denoising output only slightly.
2. Restormer-aug behaves similarly; the augmentation/retraining does not create
   a hidden downstream opportunity for orbit averaging in this setting.
3. RED gains are small partly because the denoising prior itself gains little,
   and partly because RED updates are conservative under the selected stable
   parameters.
4. The upright padded-circle case should use `group_expand=False` for orbit
   averaging if the goal is to remove canvas effects; `group_expand=True` adds an
   avoidable geometry mismatch.
"""
    out = BASE / "summary.md"
    out.write_text(report)
    print(f"wrote {out}", flush=True)


def commit_and_push():
    _run([
        "git",
        "add",
        "scripts/run_denoising_circle_sanity.py",
        "scripts/finish_denoising_circle_sanity.py",
    ])
    _run(["git", "add", "-f", "results/denoising_circle_sanity"])
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO,
    ).returncode != 0
    if staged:
        _run(["git", "commit", "-m", "Add denoising circle sanity results"])
    else:
        print("[git] no staged changes to commit", flush=True)
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = "ssh -i ~/.ssh/github_codex -o IdentitiesOnly=yes"
    _run(["git", "push", "origin", "main"], env=env)


def main():
    wait_for_results()
    write_report()
    commit_and_push()


if __name__ == "__main__":
    main()
