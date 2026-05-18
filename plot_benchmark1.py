#!/usr/bin/env python3
"""
Plot Benchmark I results.

Inputs:
  runs/easy_dsos_10seeds/summary.csv
  runs/hard_sos_10seeds/summary.csv

Outputs:
  figures/benchmark1_success_rate_heatmaps.{png,pdf}
  figures/benchmark1_gap_heatmaps.{png,pdf}
  figures/benchmark1_timing.{png,pdf}
  figures/benchmark1_gap_vs_m.{png,pdf}
  figures/benchmark1_combined_summary.csv

Usage:
  python plot_benchmark1.py \
    --easy runs/easy_dsos_10seeds/summary.csv \
    --hard runs/hard_sos_10seeds/summary.csv \
    --outdir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(easy_path: Path, hard_path: Path) -> pd.DataFrame:
    easy = pd.read_csv(easy_path)
    hard = pd.read_csv(hard_path)
    df = pd.concat([easy, hard], ignore_index=True)

    # Standardize labels for prettier plots.
    df["ensemble_label"] = df["ensemble"].map(
        {
            "easy_dsos": "DD-generated / DSOS-friendly",
            "hard_sos": "Random PSD Gram / SOS-only stress",
        }
    ).fillna(df["ensemble"])

    df["cone_label"] = df["cone"].str.upper()
    df["cell_label"] = "n=" + df["n"].astype(str) + ", d=" + df["d"].astype(str)

    # Infeasible entries have NaN gap. For heatmaps of failure/gap, keep NaN.
    # For display, clip very large gaps only when plotting so one outlier
    # does not wash out the entire figure.
    return df


def savefig(fig, outpath_base: Path):
    outpath_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(outpath_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def heatmap_panel(ax, mat, row_labels, col_labels, title, vmin=None, vmax=None, fmt=".2f"):
    im = ax.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11, pad=8)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isfinite(val):
                text = format(val, fmt)
            else:
                text = "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    return im


def make_success_heatmaps(df: pd.DataFrame, outdir: Path):
    cones = ["DSOS", "SDSOS", "SOS"]
    cells = (
        df[["n", "d", "basis_size_m"]]
        .drop_duplicates()
        .sort_values(["d", "n"])
        .assign(label=lambda x: "n=" + x["n"].astype(str) + "\nd=" + x["d"].astype(str) + "\nm=" + x["basis_size_m"].astype(str))
    )
    cell_order = list(zip(cells["n"], cells["d"]))
    cell_labels = cells["label"].tolist()

    ensembles = [
        ("easy_dsos", "DD-generated / DSOS-friendly"),
        ("hard_sos", "Random PSD Gram / SOS-only stress"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.2), constrained_layout=True)

    for ax, (ens, title) in zip(axes, ensembles):
        sub = df[df["ensemble"] == ens]
        mat = np.full((len(cones), len(cell_order)), np.nan)

        for i, cone in enumerate(["dsos", "sdsos", "sos"]):
            for j, (n, d) in enumerate(cell_order):
                rows = sub[(sub["cone"] == cone) & (sub["n"] == n) & (sub["d"] == d)]
                if not rows.empty:
                    mat[i, j] = rows["success_rate"].iloc[0]

        im = heatmap_panel(
            ax,
            mat,
            row_labels=cones,
            col_labels=cell_labels,
            title=title,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
        )
        ax.set_xlabel("problem cell")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label("success rate")

    fig.suptitle("Benchmark I: feasibility/success rate over random seeds", fontsize=13)
    savefig(fig, outdir / "benchmark1_success_rate_heatmaps")


def make_gap_heatmaps(df: pd.DataFrame, outdir: Path, gap_clip: float = 1.0):
    cones = ["DSOS", "SDSOS"]
    cells = (
        df[["n", "d", "basis_size_m"]]
        .drop_duplicates()
        .sort_values(["d", "n"])
        .assign(label=lambda x: "n=" + x["n"].astype(str) + "\nd=" + x["d"].astype(str) + "\nm=" + x["basis_size_m"].astype(str))
    )
    cell_order = list(zip(cells["n"], cells["d"]))
    cell_labels = cells["label"].tolist()

    ensembles = [
        ("easy_dsos", "DD-generated / DSOS-friendly"),
        ("hard_sos", "Random PSD Gram / SOS-only stress"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 3.7), constrained_layout=True)

    for ax, (ens, title) in zip(axes, ensembles):
        sub = df[df["ensemble"] == ens]
        mat = np.full((len(cones), len(cell_order)), np.nan)

        for i, cone in enumerate(["dsos", "sdsos"]):
            for j, (n, d) in enumerate(cell_order):
                rows = sub[(sub["cone"] == cone) & (sub["n"] == n) & (sub["d"] == d)]
                if not rows.empty:
                    val = rows["mean_rel_gap_to_sos_best"].iloc[0]
                    if np.isfinite(val):
                        mat[i, j] = min(val, gap_clip)

        im = heatmap_panel(
            ax,
            mat,
            row_labels=cones,
            col_labels=cell_labels,
            title=title,
            vmin=0.0,
            vmax=gap_clip,
            fmt=".3f",
        )
        ax.set_xlabel("problem cell")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label(f"mean relative gap to SOS, clipped at {gap_clip:g}")

    fig.suptitle("Benchmark I: DSOS/SDSOS optimality gap relative to SOS", fontsize=13)
    savefig(fig, outdir / "benchmark1_gap_heatmaps")


def make_timing_plot(df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for ensemble in ["easy_dsos", "hard_sos"]:
        for cone in ["dsos", "sdsos", "sos"]:
            sub = df[(df["ensemble"] == ensemble) & (df["cone"] == cone)].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("basis_size_m")
            label = f"{ensemble.replace('_', ' ')} / {cone.upper()}"
            marker = {"dsos": "o", "sdsos": "s", "sos": "^"}[cone]
            linestyle = "-" if ensemble == "easy_dsos" else "--"

            ax.plot(
                sub["basis_size_m"],
                sub["median_wall_time"],
                marker=marker,
                linestyle=linestyle,
                label=label,
                linewidth=1.6,
                markersize=5,
            )

    ax.set_xlabel("Gram basis size m")
    ax.set_ylabel("median wall time (seconds)")
    ax.set_title("Benchmark I: wall time versus Gram basis size")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=1)
    savefig(fig, outdir / "benchmark1_timing")


def make_gap_vs_m_plot(df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for ensemble in ["easy_dsos", "hard_sos"]:
        for cone in ["dsos", "sdsos"]:
            sub = df[(df["ensemble"] == ensemble) & (df["cone"] == cone)].copy()
            sub = sub[np.isfinite(sub["mean_rel_gap_to_sos_best"])]
            if sub.empty:
                continue

            sub = sub.sort_values("basis_size_m")
            label = f"{ensemble.replace('_', ' ')} / {cone.upper()}"
            marker = {"dsos": "o", "sdsos": "s"}[cone]
            linestyle = "-" if ensemble == "easy_dsos" else "--"

            y = sub["mean_rel_gap_to_sos_best"].clip(lower=1e-12, upper=1e3)

            ax.plot(
                sub["basis_size_m"],
                y,
                marker=marker,
                linestyle=linestyle,
                label=label,
                linewidth=1.6,
                markersize=5,
            )

    ax.set_xlabel("Gram basis size m")
    ax.set_ylabel("mean relative gap to SOS")
    ax.set_title("Benchmark I: gap to SOS versus Gram basis size")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    savefig(fig, outdir / "benchmark1_gap_vs_m")


def write_combined_summary(df: pd.DataFrame, outdir: Path):
    keep = [
        "ensemble",
        "n",
        "d",
        "basis_size_m",
        "cone",
        "solver",
        "num_attempts",
        "success_rate",
        "reliable_success_rate",
        "mean_rel_gap_to_sos_best",
        "std_rel_gap_to_sos_best",
        "median_wall_time",
    ]
    df[keep].sort_values(["ensemble", "d", "n", "cone"]).to_csv(
        outdir / "benchmark1_combined_summary.csv",
        index=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--easy", required=True, type=Path)
    parser.add_argument("--hard", required=True, type=Path)
    parser.add_argument("--outdir", default="figures", type=Path)
    parser.add_argument("--gap-clip", default=1.0, type=float)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.easy, args.hard)
    write_combined_summary(df, args.outdir)
    make_success_heatmaps(df, args.outdir)
    make_gap_heatmaps(df, args.outdir, gap_clip=args.gap_clip)
    make_timing_plot(df, args.outdir)
    make_gap_vs_m_plot(df, args.outdir)

    print(f"Wrote figures and combined summary to: {args.outdir}")
    for p in sorted(args.outdir.glob("benchmark1_*")):
        print(" ", p)


if __name__ == "__main__":
    main()
