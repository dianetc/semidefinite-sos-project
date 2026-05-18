#!/usr/bin/env python3
"""
Cleaner Figures.

Inputs:
  runs/easy_dsos_10seeds/summary.csv
  runs/hard_sos_10seeds/summary.csv

Outputs:
  figures_clean/benchmark1_main_story.{png,pdf}
  figures_clean/benchmark1_timing_clean.{png,pdf}
  figures_clean/benchmark1_aggregate_success.{png,pdf}
  figures_clean/benchmark1_plot_data.csv

Usage:
  python plot_benchmark1_clean.py \
    --easy runs/easy_dsos_10seeds/summary.csv \
    --hard runs/hard_sos_10seeds/summary.csv \
    --outdir figures_clean
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONE_ORDER = ["dsos", "sdsos", "sos"]
CONE_LABELS = {"dsos": "DSOS", "sdsos": "SDSOS", "sos": "SOS"}
ENSEMBLE_LABELS = {
    "easy_dsos": "DD-generated",
    "hard_sos": "Random PSD Gram",
}


def savefig(fig, outbase: Path) -> None:
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_data(easy: Path, hard: Path) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(easy), pd.read_csv(hard)], ignore_index=True)
    df["cone"] = df["cone"].str.lower()
    df["ensemble"] = df["ensemble"].str.lower()
    df["cone_label"] = df["cone"].map(CONE_LABELS).fillna(df["cone"])
    df["ensemble_label"] = df["ensemble"].map(ENSEMBLE_LABELS).fillna(df["ensemble"])

    df["cell"] = list(zip(df["n"], df["d"], df["basis_size_m"]))
    cells = (
        df[["n", "d", "basis_size_m"]]
        .drop_duplicates()
        .sort_values(["d", "n"])
        .reset_index(drop=True)
    )
    cell_to_x = {
        (int(r.n), int(r.d), int(r.basis_size_m)): i
        for i, r in cells.iterrows()
    }
    cell_to_label = {
        (int(r.n), int(r.d), int(r.basis_size_m)): f"$n={int(r.n)}$\n$d={int(r.d)}$\n$m={int(r.basis_size_m)}$"
        for _, r in cells.iterrows()
    }

    df["x"] = df["cell"].map(cell_to_x)
    df["cell_label"] = df["cell"].map(cell_to_label)
    return df


def plot_main_story(df: pd.DataFrame, outdir: Path) -> None:
    """
    Main figure:
      A. On easy/DD-generated instances: gap to SOS.
      B. On hard/random-PSD instances: success rate.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), constrained_layout=True)

    # Panel A: easy ensemble, gap to SOS
    ax = axes[0]
    easy = df[df["ensemble"] == "easy_dsos"].copy()
    for cone in ["dsos", "sdsos"]:
        sub = easy[easy["cone"] == cone].sort_values(["d", "n"])
        y = sub["mean_rel_gap_to_sos_best"].replace([np.inf, -np.inf], np.nan)
        y = y.clip(lower=1e-8)
        ax.plot(
            sub["x"],
            y,
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=CONE_LABELS[cone],
        )

    labels = (
        easy[["x", "cell_label"]]
        .drop_duplicates()
        .sort_values("x")
    )
    ax.set_xticks(labels["x"])
    ax.set_xticklabels(labels["cell_label"], fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("mean relative gap to SOS")
    ax.set_title("(a) Gap on DSOS-friendly instances")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    # Panel B: hard ensemble, success rate
    ax = axes[1]
    hard = df[df["ensemble"] == "hard_sos"].copy()
    for cone in ["dsos", "sdsos", "sos"]:
        sub = hard[hard["cone"] == cone].sort_values(["d", "n"])
        ax.plot(
            sub["x"],
            sub["success_rate"],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            label=CONE_LABELS[cone],
        )

    labels = (
        hard[["x", "cell_label"]]
        .drop_duplicates()
        .sort_values("x")
    )
    ax.set_xticks(labels["x"])
    ax.set_xticklabels(labels["cell_label"], fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("success rate")
    ax.set_title("(b) Success on generic SOS instances")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Benchmark I: instance ensemble determines DSOS/SDSOS behavior", fontsize=13)
    savefig(fig, outdir / "benchmark1_main_story")


def plot_timing_clean(df: pd.DataFrame, outdir: Path) -> None:
    """
    Cleaner timing figure:
      one panel per ensemble, avoiding the six-line spaghetti plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.6), sharey=True, constrained_layout=True)

    for ax, ensemble in zip(axes, ["easy_dsos", "hard_sos"]):
        subens = df[df["ensemble"] == ensemble].copy()

        for cone in CONE_ORDER:
            sub = subens[subens["cone"] == cone].sort_values(["d", "n"])
            ax.plot(
                sub["basis_size_m"],
                sub["median_wall_time"],
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                label=CONE_LABELS[cone],
            )

        ax.set_title(ENSEMBLE_LABELS[ensemble])
        ax.set_xlabel("Gram basis size $m$")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=9)

    axes[0].set_ylabel("median wall time (seconds)")
    fig.suptitle("Benchmark I: wall-clock scaling by ensemble", fontsize=13)
    savefig(fig, outdir / "benchmark1_timing_clean")


def plot_aggregate_success(df: pd.DataFrame, outdir: Path) -> None:
    """
    Very simple summary: average success rate across all (n,d) cells.
    Useful as a compact figure in slides or a paper margin.
    """
    agg = (
        df.groupby(["ensemble", "cone"], as_index=False)
        .agg(
            avg_success=("success_rate", "mean"),
            avg_gap=("mean_rel_gap_to_sos_best", "mean"),
            avg_time=("median_wall_time", "mean"),
        )
    )

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)

    width = 0.35
    x = np.arange(len(CONE_ORDER))

    for offset, ensemble in [(-width / 2, "easy_dsos"), (width / 2, "hard_sos")]:
        sub = agg[agg["ensemble"] == ensemble].set_index("cone").reindex(CONE_ORDER)
        ax.bar(
            x + offset,
            sub["avg_success"],
            width=width,
            label=ENSEMBLE_LABELS[ensemble],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([CONE_LABELS[c] for c in CONE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("average success rate")
    ax.set_title("Average feasibility across all benchmark cells")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    savefig(fig, outdir / "benchmark1_aggregate_success")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("figures_clean"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.easy, args.hard)
    df.to_csv(args.outdir / "benchmark1_plot_data.csv", index=False)

    plot_main_story(df, args.outdir)
    plot_timing_clean(df, args.outdir)
    plot_aggregate_success(df, args.outdir)

    print(f"Wrote cleaner figures to {args.outdir}")
    for path in sorted(args.outdir.glob("benchmark1_*")):
        print(" ", path)


if __name__ == "__main__":
    main()
