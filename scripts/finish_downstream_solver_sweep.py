#!/usr/bin/env python
"""Finish the aligned downstream solver sweep and push results.

This script is intended to run in tmux after the clean vanilla parameter sweeps
start. It waits for sweep completion, selects the best parameter setting for
each denoiser/problem/algorithm, reruns the selected settings with
vanilla+G16_fixed, aggregates the final results, and pushes everything to main.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path.home() / "anaconda3/envs/restormer37/bin/python"
SWEEP_SCRIPT = REPO / "scripts/run_downstream_solver_sweep.py"
BASE = REPO / "results/downstream/solver_sweep"
SWEEPS = {
    "classical": (BASE / "parameter_sweep_classical", 1840),
    "restormer": (BASE / "parameter_sweep_restormer", 240),
}
FINAL_DIR = BASE / "selected_final"
MEASUREMENT_SIGMA = float(np.sqrt(2.0))


def _run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO, **kwargs)


def _read_final(path):
    csv = path / "downstream_solver_sweep_final.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def _validate(df, name, expected):
    if len(df) != expected:
        raise RuntimeError(f"{name}: expected {expected} final rows, got {len(df)}")
    numeric = df.select_dtypes("number")
    if df.isna().sum().sum():
        raise RuntimeError(f"{name}: found NaN values")
    if np.isinf(numeric.to_numpy()).sum():
        raise RuntimeError(f"{name}: found inf values")


def wait_for_sweeps():
    while True:
        ready = True
        for name, (path, expected) in SWEEPS.items():
            df = _read_final(path)
            n = 0 if df is None else len(df)
            print(f"[wait] {name}: {n}/{expected}", flush=True)
            if df is None or n < expected:
                ready = False
            elif n > expected:
                raise RuntimeError(f"{name}: got more rows than expected: {n}>{expected}")
        if ready:
            break
        time.sleep(300)

    frames = []
    for name, (path, expected) in SWEEPS.items():
        df = _read_final(path)
        _validate(df, name, expected)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def schedule_arg(label):
    if label.startswith("fixed"):
        return label[len("fixed"):].replace("p", ".")
    if label.startswith("linear") and "_to_" in label:
        start, end = label[len("linear"):].split("_to_", 1)
        return f"{start.replace('p', '.')}->{end.replace('p', '.')}"
    raise ValueError(f"Cannot parse schedule label: {label}")


def select_params(df):
    keys = ["denoiser", "problem", "algorithm", "schedule", "rho", "step", "red_input_sigma", "lambda"]
    summary = (
        df.groupby(keys, as_index=False)
        .agg(
            mean_final_psnr=("final_psnr", "mean"),
            mean_gap_psnr=("gap_psnr", "mean"),
            fail_rate=("beats_degraded", lambda s: 1.0 - float(np.mean(s))),
            n=("file", "size"),
        )
    )
    selected_rows = []
    for combo, sub in summary.groupby(["denoiser", "problem", "algorithm"], as_index=False):
        sub = sub.sort_values(
            ["mean_final_psnr", "mean_gap_psnr", "fail_rate"],
            ascending=[False, False, True],
        )
        selected_rows.append(sub.iloc[0])
    selected = pd.DataFrame(selected_rows).reset_index(drop=True)
    BASE.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BASE / "parameter_sweep_ranked_summary.csv", index=False)
    selected.to_csv(BASE / "selected_params.csv", index=False)
    return selected


def run_selected(selected):
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        denoiser = row["denoiser"]
        problem = row["problem"]
        algorithm = row["algorithm"]
        out = FINAL_DIR / denoiser / problem / algorithm
        cmd = [
            PYTHON,
            "-u",
            SWEEP_SCRIPT,
            "--save-dir",
            out,
            "--denoisers",
            denoiser,
            "--problems",
            problem,
            "--algorithms",
            algorithm,
            "--modes",
            "vanilla",
            "G16_fixed",
            "--max-images",
            "10",
            "--num-iter",
            "20",
            "--save-image-count",
            "1",
            "--measurement-sigma",
            MEASUREMENT_SIGMA,
        ]
        if algorithm == "pnp_hqs":
            if denoiser in {"wavelet", "tv"}:
                cmd += ["--pnp-classical-schedules", schedule_arg(row["schedule"])]
                cmd += ["--pnp-classical-rhos", row["rho"]]
            else:
                cmd += ["--pnp-restormer-rhos", row["rho"]]
        elif algorithm == "red_gd":
            if denoiser in {"wavelet", "tv"}:
                cmd += ["--red-classical-schedules", schedule_arg(row["schedule"])]
            cmd += ["--red-lambdas", row["lambda"]]
            cmd += ["--red-input-sigma", row["red_input_sigma"]]
        _run(cmd)


def aggregate_final():
    frames = []
    summaries = []
    for csv in FINAL_DIR.glob("*/*/*/downstream_solver_sweep_final.csv"):
        frames.append(pd.read_csv(csv))
    for csv in FINAL_DIR.glob("*/*/*/downstream_solver_sweep_summary.csv"):
        summaries.append(pd.read_csv(csv))
    if not frames:
        raise RuntimeError("No selected final CSV files found.")
    final = pd.concat(frames, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    final.to_csv(BASE / "selected_final_all.csv", index=False)
    summary.to_csv(BASE / "selected_final_summary_all.csv", index=False)
    numeric = final.select_dtypes("number")
    if final.isna().sum().sum() or np.isinf(numeric.to_numpy()).sum():
        raise RuntimeError("Aggregated selected final results contain NaN/inf.")


def push_results():
    add_cmd = [
        "git",
        "add",
        "scripts/run_downstream_solver_sweep.py",
        "scripts/finish_downstream_solver_sweep.py",
        "scripts/run_downstream_stochastic_wavelet.py",
        "scripts/run_downstream_vanilla_classical.py",
        "src/groupavg/experiments/downstream_effect.py",
    ]
    _run(add_cmd)
    _run(["git", "add", "-f", "results/README.md", "results/downstream/solver_sweep"])
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if status:
        _run(["git", "commit", "-m", "Add aligned downstream solver sweep results"])
    else:
        print("[git] no changes to commit", flush=True)
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = "ssh -i ~/.ssh/github_codex -o IdentitiesOnly=yes"
    _run(["git", "push", "origin", "main"], env=env)


def main():
    print("[finish] waiting for aligned parameter sweeps", flush=True)
    sweep_df = wait_for_sweeps()
    print("[finish] selecting parameters", flush=True)
    selected = select_params(sweep_df)
    print(selected.to_string(index=False), flush=True)
    print("[finish] running selected vanilla+G16 experiments", flush=True)
    run_selected(selected)
    print("[finish] aggregating selected final results", flush=True)
    aggregate_final()
    print("[finish] committing and pushing", flush=True)
    push_results()
    print("[finish] done", flush=True)


if __name__ == "__main__":
    main()
