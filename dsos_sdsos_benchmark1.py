#!/usr/bin/env python3
"""
DSOS/SDSOS/SOS Benchmark I starter.

What this runs does:
  - Generates reproducible polynomial instances.
  - Supports two ensembles:
      1. easy_dsos: DD/DSOS-friendly Gram matrices.
      2. hard_sos: generic random PSD Gram matrices.
  - Solves the same instance with DSOS, SDSOS, and SOS relaxations.
  - Repeats over n, degree, seed, and solver.
  - Saves:
      instances.jsonl
      results.csv
      results_with_gaps.csv
      summary.csv

Install:
  pip install numpy pandas cvxpy scipy tqdm

Optional commercial solver:
  MOSEK is strongly recommended for larger SDP/SOCP runs if you have a license.
  Otherwise, CLARABEL is the best open-source default for this script.
  SCS is useful for solver-sensitivity checks, but not as the main quality solver.

Example smoke tests:
  python dsos_sdsos_benchmark1.py --ensemble easy_dsos --n-list 2 3 --d-list 2 --seeds 0 1 --solvers CLARABEL --outdir runs/smoke_easy
  python dsos_sdsos_benchmark1.py --ensemble hard_sos  --n-list 2 3 --d-list 2 --seeds 0 1 --solvers CLARABEL --outdir runs/smoke_hard

Main first-pass:
  python dsos_sdsos_benchmark1.py --ensemble easy_dsos --n-list 2 3 4 5 6 --d-list 2 3 --seeds 0 1 2 3 4 5 6 7 8 9 --solvers CLARABEL --outdir runs/easy_dsos_main
  python dsos_sdsos_benchmark1.py --ensemble hard_sos  --n-list 2 3 4 5 6 --d-list 2 3 --seeds 0 1 2 3 4 5 6 7 8 9 --solvers CLARABEL --outdir runs/hard_sos_main
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm


Exponent = Tuple[int, ...]
Poly = Dict[Exponent, float]


# ---------------------------------------------------------------------
# Polynomial utilities
# ---------------------------------------------------------------------

def monomials_upto_degree(n: int, d: int) -> List[Exponent]:
    """All exponent tuples alpha in N^n with |alpha| <= d, graded lexicographic."""
    out: List[Exponent] = []

    def rec(pos: int, remaining: int, cur: List[int]):
        if pos == n - 1:
            for k in range(remaining + 1):
                out.append(tuple(cur + [k]))
            return
        for k in range(remaining + 1):
            rec(pos + 1, remaining - k, cur + [k])

    rec(0, d, [])
    out.sort(key=lambda a: (sum(a), a))
    return out


def add_exp(a: Exponent, b: Exponent) -> Exponent:
    return tuple(x + y for x, y in zip(a, b))


def poly_from_gram(Q: np.ndarray, mons: Sequence[Exponent]) -> Poly:
    """Return coefficient dictionary for z(x)^T Q z(x)."""
    p: Poly = {}
    m = len(mons)
    for i in range(m):
        for j in range(m):
            alpha = add_exp(mons[i], mons[j])
            p[alpha] = p.get(alpha, 0.0) + float(Q[i, j])
    return {k: v for k, v in p.items() if abs(v) > 1e-12}


def instance_to_poly(instance: dict) -> Poly:
    return {tuple(item["alpha"]): float(item["coef"]) for item in instance["coefficients"]}


# ---------------------------------------------------------------------
# Instance generators
# ---------------------------------------------------------------------

def generate_sos_instance(
    n: int,
    d: int,
    seed: int,
    rank_frac: float = 1.0,
    condition: float = 1e2,
    linear_scale: float = 0.0,
    constant_shift: float = 0.0,
) -> dict:
    """
    Hard ensemble: generate p(x) = z^T Q z from a generic random PSD Gram matrix.

    This creates a polynomial with a known SOS certificate, but the Gram matrix
    is generally not diagonally dominant or scaled diagonally dominant. Therefore,
    DSOS/SDSOS may be infeasible or may need very conservative gamma values.

    This is useful as a stress test and as the empirical counterpart of the
    theoretical message that DD/SDD are strict inner approximations of PSD.
    """
    rng = np.random.default_rng(seed)
    mons = monomials_upto_degree(n, d)
    m = len(mons)

    r = max(1, int(math.ceil(rank_frac * m)))
    A = rng.normal(size=(r, m))

    # Mild condition control. Larger condition makes the Gram spectrum more uneven.
    row_scales = np.geomspace(1.0, condition, r)
    A = A / row_scales[:, None]

    Q = A.T @ A
    Q = 0.5 * (Q + Q.T)

    p = poly_from_gram(Q, mons)

    if linear_scale > 0:
        for i in range(n):
            alpha = tuple(1 if j == i else 0 for j in range(n))
            p[alpha] = p.get(alpha, 0.0) + float(linear_scale * rng.normal())

    zero = tuple([0] * n)
    p[zero] = p.get(zero, 0.0) + float(constant_shift)

    return {
        "n": n,
        "d": d,
        "seed": seed,
        "basis": [list(a) for a in mons],
        "coefficients": [{"alpha": list(k), "coef": v} for k, v in sorted(p.items())],
        "generator": {
            "type": "hard_sos_random_psd_gram",
            "rank_frac": rank_frac,
            "condition": condition,
            "linear_scale": linear_scale,
            "constant_shift": constant_shift,
        },
    }


def generate_dsos_friendly_instance(
    n: int,
    d: int,
    seed: int,
    diag_scale: float = 1.0,
    offdiag_scale: float = 0.02,
    constant_shift: float = 0.0,
) -> dict:
    """
    Easy ensemble: generate p(x) = z^T Q z from a deliberately DD Gram matrix.

    Since Q is diagonally dominant by construction, the polynomial has a DSOS
    certificate. This gives a sanity-check ensemble where DSOS/SDSOS/SOS should
    usually all succeed and be close.
    """
    rng = np.random.default_rng(seed)
    mons = monomials_upto_degree(n, d)
    m = len(mons)

    Q = rng.normal(scale=offdiag_scale, size=(m, m))
    Q = 0.5 * (Q + Q.T)

    for i in range(m):
        offsum = np.sum(np.abs(Q[i, :])) - abs(Q[i, i])
        Q[i, i] = offsum + diag_scale * (1.0 + rng.random())

    Q = 0.5 * (Q + Q.T)

    p = poly_from_gram(Q, mons)

    zero = tuple([0] * n)
    p[zero] = p.get(zero, 0.0) + float(constant_shift)

    return {
        "n": n,
        "d": d,
        "seed": seed,
        "basis": [list(a) for a in mons],
        "coefficients": [{"alpha": list(k), "coef": v} for k, v in sorted(p.items())],
        "generator": {
            "type": "easy_dsos_dd_gram",
            "diag_scale": diag_scale,
            "offdiag_scale": offdiag_scale,
            "constant_shift": constant_shift,
        },
    }


# ---------------------------------------------------------------------
# Cone constraints
# ---------------------------------------------------------------------

def coefficient_constraints(
    Q: cp.Expression,
    gamma: cp.Variable,
    p: Poly,
    mons: Sequence[Exponent],
    n: int,
) -> List[cp.Constraint]:
    """
    Enforce p(x) - gamma == z(x)^T Q z(x) by coefficient matching.
    """
    coeff_expr: Dict[Exponent, cp.Expression] = {}
    m = len(mons)

    for i in range(m):
        for j in range(m):
            alpha = add_exp(mons[i], mons[j])
            coeff_expr[alpha] = coeff_expr.get(alpha, 0) + Q[i, j]

    all_alphas = set(p.keys()) | set(coeff_expr.keys())
    zero_alpha = tuple([0] * n)

    constraints: List[cp.Constraint] = []
    for alpha in all_alphas:
        target = float(p.get(alpha, 0.0))
        if alpha == zero_alpha:
            target_expr = target - gamma
        else:
            target_expr = target
        constraints.append(coeff_expr.get(alpha, 0) == target_expr)

    return constraints


def add_dd_constraints(Q: cp.Variable) -> List[cp.Constraint]:
    """
    Diagonally dominant symmetric Q.

    Q is DD if Q_ii >= sum_{j != i} |Q_ij| for all i.
    This encoding introduces auxiliary variables for absolute values.
    """
    m = Q.shape[0]
    constraints: List[cp.Constraint] = [Q == Q.T]

    # CVXPY uses nonneg=True, not nonnegative=True.
    abs_off = cp.Variable((m, m), nonneg=True)

    for i in range(m):
        for j in range(m):
            if i == j:
                constraints.append(abs_off[i, j] == 0)
            else:
                constraints += [
                    abs_off[i, j] >= Q[i, j],
                    abs_off[i, j] >= -Q[i, j],
                ]
        constraints.append(Q[i, i] >= cp.sum(abs_off[i, :]))

    return constraints


def add_sdd_constraints(Q: cp.Variable) -> List[cp.Constraint]:
    """
    Scaled diagonally dominant Q via decomposition into 2x2 PSD-supported blocks.

    A matrix is SDD iff it can be decomposed as a sum of symmetric matrices each
    supported on a 2x2 principal block that is PSD. Each 2x2 PSD block is encoded
    with an SOC constraint:

        [[a, b], [b, c]] >= 0  <=>  a >= 0, c >= 0,
                                    ||(2b, a-c)||_2 <= a+c.

    This is the standard SOCP representation used for SDSOS-style constraints.
    """
    m = Q.shape[0]
    constraints: List[cp.Constraint] = [Q == Q.T]

    diag_terms = [[None for _ in range(m)] for __ in range(m)]
    off_terms = {}

    for i in range(m):
        for j in range(i + 1, m):
            a = cp.Variable(nonneg=True, name=f"sdd_a_{i}_{j}")
            c = cp.Variable(nonneg=True, name=f"sdd_c_{i}_{j}")
            b = cp.Variable(name=f"sdd_b_{i}_{j}")

            diag_terms[i][j] = a
            diag_terms[j][i] = c
            off_terms[(i, j)] = b

            constraints.append(cp.SOC(a + c, cp.hstack([2 * b, a - c])))
            constraints.append(Q[i, j] == b)
            constraints.append(Q[j, i] == b)

    for i in range(m):
        terms = [diag_terms[i][j] for j in range(m) if j != i]
        if terms:
            constraints.append(Q[i, i] == cp.sum(cp.hstack(terms)))
        else:
            constraints.append(Q[i, i] >= 0)

    return constraints


# ---------------------------------------------------------------------
# Solving
# ---------------------------------------------------------------------

def solve_bound(
    instance: dict,
    cone: str,
    solver: str,
    verbose: bool = False,
) -> dict:
    n = int(instance["n"])
    mons = [tuple(a) for a in instance["basis"]]
    p = instance_to_poly(instance)
    m = len(mons)

    Q = cp.Variable((m, m), symmetric=True)
    gamma = cp.Variable(name="gamma")

    constraints = coefficient_constraints(Q, gamma, p, mons, n=n)

    cone = cone.lower()
    if cone == "sos":
        constraints.append(Q >> 0)
    elif cone == "dsos":
        constraints += add_dd_constraints(Q)
    elif cone == "sdsos":
        constraints += add_sdd_constraints(Q)
    else:
        raise ValueError(f"Unknown cone: {cone}")

    problem = cp.Problem(cp.Maximize(gamma), constraints)

    installed = set(cp.installed_solvers())
    if solver not in installed:
        return {
            "status": "solver_not_installed",
            "value": np.nan,
            "solve_time": np.nan,
            "wall_time": 0.0,
            "error": f"{solver} not in installed solvers: {sorted(installed)}",
        }

    kwargs = {"solver": solver, "verbose": verbose}

    # Conservative solver-specific defaults.
    # If these ever break under a different CVXPY version, the exception is recorded.
    if solver.upper() == "SCS":
        kwargs.update({"eps": 1e-5, "max_iters": 50_000})
    if solver.upper() == "CLARABEL":
        kwargs.update({"tol_gap_abs": 1e-7, "tol_feas": 1e-7})
    if solver.upper() == "MOSEK":
        # Keep options minimal here for portability across MOSEK versions.
        pass

    start = time.perf_counter()
    try:
        val = problem.solve(**kwargs)
        wall = time.perf_counter() - start

        return {
            "status": problem.status,
            "value": float(val) if val is not None else np.nan,
            "solve_time": (
                float(problem.solver_stats.solve_time)
                if problem.solver_stats.solve_time is not None
                else np.nan
            ),
            "wall_time": wall,
            "error": "",
        }
    except Exception as e:
        wall = time.perf_counter() - start
        return {
            "status": "exception",
            "value": np.nan,
            "solve_time": np.nan,
            "wall_time": wall,
            "error": repr(e),
        }


def available_default_solvers(cone: str) -> List[str]:
    installed = set(cp.installed_solvers())

    # Good defaults:
    # - MOSEK if installed.
    # - CLARABEL is the preferred open-source conic solver here.
    # - SCS is included as a robustness/sensitivity baseline.
    # - SCIPY only applies to LP-style DSOS constraints.
    if cone == "sos":
        preferred = ["MOSEK", "CLARABEL", "SCS"]
    elif cone == "sdsos":
        preferred = ["MOSEK", "CLARABEL", "ECOS", "SCS"]
    elif cone == "dsos":
        preferred = ["MOSEK", "CLARABEL", "ECOS", "SCIPY", "SCS"]
    else:
        preferred = ["MOSEK", "CLARABEL", "SCS"]

    return [s for s in preferred if s in installed]


# ---------------------------------------------------------------------
# CLI and output
# ---------------------------------------------------------------------

def parse_int_list(xs: Sequence[str]) -> List[int]:
    return [int(x) for x in xs]


def build_instance(args, n: int, d: int, seed: int) -> dict:
    if args.ensemble == "hard_sos":
        return generate_sos_instance(
            n=n,
            d=d,
            seed=seed,
            rank_frac=args.rank_frac,
            condition=args.condition,
            linear_scale=args.linear_scale,
            constant_shift=args.constant_shift,
        )

    if args.ensemble == "easy_dsos":
        return generate_dsos_friendly_instance(
            n=n,
            d=d,
            seed=seed,
            diag_scale=args.diag_scale,
            offdiag_scale=args.offdiag_scale,
            constant_shift=args.constant_shift,
        )

    raise ValueError(f"Unknown ensemble: {args.ensemble}")


def write_summary(df: pd.DataFrame, outdir: Path, summary_path: Path) -> None:
    """
    Write honest summaries.

    Important:
      We summarize over all attempted solves, not only successful solves.
      Otherwise infeasible DSOS/SDSOS rows disappear and success_rate becomes false.
    """
    df = df.copy()

    df["is_success"] = df["status"].isin(["optimal", "optimal_inaccurate"])
    df["is_reliable_success"] = df["status"].eq("optimal")
    df["finite_value"] = np.where(np.isfinite(df["value"]), df["value"], np.nan)

    sos_best = (
        df[(df["cone"] == "sos") & (df["is_success"]) & (np.isfinite(df["value"]))]
        .groupby(["n", "d", "seed"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "sos_best_value"})
    )

    df2 = df.merge(sos_best, on=["n", "d", "seed"], how="left")
    df2["gap_to_sos_best"] = df2["sos_best_value"] - df2["finite_value"]
    df2["rel_gap_to_sos_best"] = df2["gap_to_sos_best"] / np.maximum(
        1.0, np.abs(df2["sos_best_value"])
    )

    summary = (
        df2.groupby(["ensemble", "n", "d", "basis_size_m", "cone", "solver"], as_index=False)
        .agg(
            num_attempts=("value", "size"),
            success_rate=("is_success", "mean"),
            reliable_success_rate=("is_reliable_success", "mean"),
            num_finite=("finite_value", lambda x: int(np.isfinite(x).sum())),
            mean_value=("finite_value", "mean"),
            std_value=("finite_value", "std"),
            mean_rel_gap_to_sos_best=("rel_gap_to_sos_best", "mean"),
            std_rel_gap_to_sos_best=("rel_gap_to_sos_best", "std"),
            mean_wall_time=("wall_time", "mean"),
            median_wall_time=("wall_time", "median"),
        )
    )

    df2.to_csv(outdir / "results_with_gaps.csv", index=False)
    summary.to_csv(summary_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ensemble",
        choices=["easy_dsos", "hard_sos"],
        default="hard_sos",
        help=(
            "Instance generator. easy_dsos uses diagonally dominant Gram matrices; "
            "hard_sos uses generic random PSD Gram matrices."
        ),
    )

    parser.add_argument("--n-list", nargs="+", default=["2", "3", "4"], help="List of dimensions n.")
    parser.add_argument(
        "--d-list",
        nargs="+",
        default=["2"],
        help="List of half-degrees d; polynomial degree is 2d.",
    )
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"], help="Random seeds.")
    parser.add_argument("--cones", nargs="+", default=["dsos", "sdsos", "sos"])
    parser.add_argument("--solvers", nargs="*", default=None, help="Optional explicit solver list.")
    parser.add_argument("--outdir", default="runs/dsos_sdsos_benchmark1")

    # hard_sos generator controls
    parser.add_argument("--rank-frac", type=float, default=1.0)
    parser.add_argument("--condition", type=float, default=1e2)
    parser.add_argument("--linear-scale", type=float, default=0.0)

    # easy_dsos generator controls
    parser.add_argument("--diag-scale", type=float, default=1.0)
    parser.add_argument("--offdiag-scale", type=float, default=0.02)

    # shared generator control
    parser.add_argument("--constant-shift", type=float, default=0.0)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    n_list = parse_int_list(args.n_list)
    d_list = parse_int_list(args.d_list)
    seeds = parse_int_list(args.seeds)
    cones = [c.lower() for c in args.cones]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inst_path = outdir / "instances.jsonl"
    res_path = outdir / "results.csv"
    summary_path = outdir / "summary.csv"

    # Generate and save instance library.
    instances = []
    with inst_path.open("w") as f:
        for n, d, seed in itertools.product(n_list, d_list, seeds):
            inst = build_instance(args, n=n, d=d, seed=seed)
            f.write(json.dumps(inst) + "\n")
            instances.append(inst)

    # Build solve tasks.
    tasks = []
    for inst in instances:
        for cone in cones:
            solvers = args.solvers if args.solvers else available_default_solvers(cone)
            if not solvers:
                tasks.append((inst, cone, "NO_AVAILABLE_SOLVER"))
            else:
                for solver in solvers:
                    tasks.append((inst, cone, solver))

    # Solve and stream raw results to disk.
    rows = []
    for inst, cone, solver in tqdm(tasks, desc="Solving"):
        if solver == "NO_AVAILABLE_SOLVER":
            result = {
                "status": "solver_not_installed",
                "value": np.nan,
                "solve_time": np.nan,
                "wall_time": 0.0,
                "error": "No compatible installed solver found.",
            }
        else:
            result = solve_bound(inst, cone=cone, solver=solver, verbose=args.verbose)

        row = {
            "ensemble": args.ensemble,
            "n": inst["n"],
            "d": inst["d"],
            "degree": 2 * inst["d"],
            "seed": inst["seed"],
            "basis_size_m": len(inst["basis"]),
            "generator_type": inst["generator"]["type"],
            "cone": cone,
            "solver": solver,
            **result,
        }
        rows.append(row)

        # Write partial progress so interrupted runs still leave useful output.
        pd.DataFrame(rows).to_csv(res_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(res_path, index=False)
    write_summary(df, outdir=outdir, summary_path=summary_path)

    print(f"\nWrote instance library: {inst_path}")
    print(f"Wrote raw results:      {res_path}")
    print(f"Wrote results + gaps:   {outdir / 'results_with_gaps.csv'}")
    print(f"Wrote summary:          {summary_path}")
    print("\nInstalled CVXPY solvers:", cp.installed_solvers())


if __name__ == "__main__":
    main()
