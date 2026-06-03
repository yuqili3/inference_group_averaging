#!/usr/bin/env python
"""Render figures from experiment CSVs in results/.

  python scripts/make_figures.py --results results --out results/figures
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def fig_q1(results, out):
    for f in glob.glob(os.path.join(results, "q1_*_per_element.csv")):
        df = pd.read_csv(f)
        g = df.groupby("op")["rel_err"].mean()
        plt.figure(figsize=(5, 3))
        plt.plot(g.index, g.values, "o-")
        plt.xlabel("group element (op)"); plt.ylabel("rel. equivariance error")
        plt.title(os.path.basename(f).replace("q1_", "").replace("_per_element.csv", ""))
        plt.tight_layout()
        p = os.path.join(out, os.path.basename(f).replace(".csv", ".png"))
        plt.savefig(p, dpi=130); plt.close()
        print("wrote", p)


def fig_q3(results, out):
    for f in glob.glob(os.path.join(results, "q3_degradation_*.csv")):
        df = pd.read_csv(f)
        plt.figure(figsize=(6, 4))
        for (ds, grp), sub in df.groupby(["dataset", "group"]):
            sub = sub.sort_values("averaging")
            plt.plot(sub["averaging"], sub["gain_psnr"], "o-", label=f"{ds}/{grp}")
        plt.xlabel("|G| (orbit size)"); plt.ylabel("PSNR gain from averaging (dB)")
        plt.title("Q3: graceful degradation"); plt.legend(fontsize=7)
        plt.tight_layout()
        p = os.path.join(out, os.path.basename(f).replace(".csv", ".png"))
        plt.savefig(p, dpi=130); plt.close()
        print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_q1(args.results, args.out)
    fig_q3(args.results, args.out)


if __name__ == "__main__":
    main()
