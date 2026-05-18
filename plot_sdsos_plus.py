#!/usr/bin/env python3
"""
Plot SDSOS+ refinement results.

Usage:
  python plot_sdsos_plus.py \
    --summary runs/sdsos_plus_hard/summary.csv \
    --outdir figures_sdsos_plus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def savefig(fig, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def cell_label(n: int, d: int, m: int) -> str:
    return f"$n={n}, d={d}, m={m}$"


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["label"] = [cell_label(int(r.n), int(r.d), int(r.basis_size_m)) for r in df.itertuples()]
    return df


def plot_success(df: pd.DataFrame, outdir: Path) -> None:
    sub = df[df["method"] == "sdsos_plus"].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for label, g in sub.groupby("label"):
        g = g.sort_values("round")
        ax.plot(g["round"], g["success_rate"], marker="o", linewidth=1.7, markersize=4, label=label)
    ax.set_xlabel("SDSOS+ refinement round")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("SDSOS+ feasibility over basis-refinement rounds")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig(fig, outdir / "sdsos_plus_success")


def plot_gap(df: pd.DataFrame, outdir: Path) -> None:
    sub = df[(df["method"] == "sdsos_plus") & np.isfinite(df["mean_rel_gap_to_sos"])].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for label, g in sub.groupby("label"):
        g = g.sort_values("round")
        y = g["mean_rel_gap_to_sos"].clip(lower=1e-10, upper=1e3)
        ax.plot(g["round"], y, marker="o", linewidth=1.7, markersize=4, label=label)
    ax.set_xlabel("SDSOS+ refinement round")
    ax.set_ylabel("mean relative gap to SOS")
    ax.set_yscale("log")
    ax.set_title("SDSOS+ closes the gap when initialized")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    savefig(fig, outdir / "sdsos_plus_gap")


def plot_oracle_basis(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for keys, g in df.groupby(["n", "d", "basis_size_m"]):
        n, d, m = keys
        lab = cell_label(int(n), int(d), int(m))
        fixed = g[(g["method"] == "sdsos_plus") & (g["round"] == 0)]
        oracle = g[g["method"] == "sos_basis_sdsos"]
        sos = g[g["method"] == "sos"]
        rows.append({
            "label": lab,
            "fixed_sdsos": fixed["success_rate"].iloc[0] if not fixed.empty else np.nan,
            "sos_basis_sdsos": oracle["success_rate"].iloc[0] if not oracle.empty else np.nan,
            "sos": sos["success_rate"].iloc[0] if not sos.empty else np.nan,
        })
    comp = pd.DataFrame(rows)
    if comp.empty or comp["sos_basis_sdsos"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    x = np.arange(len(comp))
    width = 0.25
    ax.bar(x - width, comp["fixed_sdsos"], width=width, label="fixed SDSOS")
    ax.bar(x, comp["sos_basis_sdsos"], width=width, label="SDSOS in SOS basis")
    ax.bar(x + width, comp["sos"], width=width, label="SOS")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["label"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("Diagnostic: does the SOS basis make SDSOS feasible?")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    savefig(fig, outdir / "sdsos_plus_oracle_basis")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("figures_sdsos_plus"))
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_summary(args.summary)
    plot_success(df, args.outdir)
    plot_gap(df, args.outdir)
    plot_oracle_basis(df, args.outdir)
    print(f"Wrote SDSOS+ figures to {args.outdir}")
    for p in sorted(args.outdir.glob("sdsos_plus_*")):
        print(" ", p)


if __name__ == "__main__":
    main()
