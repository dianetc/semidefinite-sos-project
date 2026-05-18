#!/usr/bin/env python3
"""
Experiment II: SDSOS+ / basis-refinement experiment.

This script reads an existing instance library produced by dsos_sdsos_benchmark1.py,
then runs:

  1. SOS baseline.
  2. Fixed-basis SDSOS.
  3. Iterative SDSOS+ basis refinement.
  4. Optional SOS-basis diagnostic.

The SDSOS+ update implemented here follows the proposal:
after a successful SDSOS solve in the current basis y = U z, factor the returned
Gram matrix Q ≈ R^T R and update the working basis to

    y_new = R y = R U z.

Then the previous iterate remains feasible in the new basis with Q_new = I, so
the next SDSOS solve can try to improve gamma in a better basis.

Important:
  - If fixed-basis SDSOS is infeasible, there is no SDSOS Gram matrix to factor,
    so SDSOS+ cannot start from that instance.
  - The optional --run-sos-basis diagnostic uses the SOS Gram matrix to construct
    a basis in which the SOS certificate itself is approximately represented by
    Q = I. This is not a valid cheap SDSOS+ algorithm, because it uses the SDP
    solution, but it is a useful diagnostic: it separates "bad basis" from
    "structural SDD limitation."

Usage examples:

  python run_sdsos_plus.py \
    --instances runs/hard_sos_10seeds/instances.jsonl \
    --outdir runs/sdsos_plus_hard_smoke \
    --solver CLARABEL \
    --max-rounds 5 \
    --max-m 20 \
    --run-sos-basis

  python run_sdsos_plus.py \
    --instances runs/hard_sos_10seeds/instances.jsonl \
    --outdir runs/sdsos_plus_hard \
    --solver CLARABEL \
    --max-rounds 10 \
    --max-m 35 \
    --run-sos-basis

Outputs:
  results.csv
  results_with_gaps.csv
  summary.csv
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd
from tqdm import tqdm


Exponent = Tuple[int, ...]
Poly = Dict[Exponent, float]


def add_exp(a: Exponent, b: Exponent) -> Exponent:
    return tuple(x + y for x, y in zip(a, b))


def instance_to_poly(instance: dict) -> Poly:
    return {tuple(item["alpha"]): float(item["coef"]) for item in instance["coefficients"]}


def load_instances(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sqrtm_psd(Q: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    """Return R such that R.T @ R is approximately Q + ridge*I."""
    Q = 0.5 * (Q + Q.T)
    w, V = np.linalg.eigh(Q)
    w = np.maximum(w, 0.0)
    if ridge > 0:
        w = w + ridge
    return np.diag(np.sqrt(w)) @ V.T


def coefficient_constraints_with_basis(
    Q: cp.Expression,
    gamma: cp.Variable,
    p: Poly,
    mons: Sequence[Exponent],
    U: np.ndarray,
    n: int,
) -> List[cp.Constraint]:
    """Enforce p(x) - gamma = (U z(x))^T Q (U z(x))."""
    M = U.T @ Q @ U
    coeff_expr: Dict[Exponent, cp.Expression] = {}
    m = len(mons)

    for i in range(m):
        for j in range(m):
            alpha = add_exp(mons[i], mons[j])
            coeff_expr[alpha] = coeff_expr.get(alpha, 0) + M[i, j]

    all_alphas = set(p.keys()) | set(coeff_expr.keys())
    zero_alpha = tuple([0] * n)
    constraints: List[cp.Constraint] = []

    for alpha in all_alphas:
        target = float(p.get(alpha, 0.0))
        target_expr = target - gamma if alpha == zero_alpha else target
        constraints.append(coeff_expr.get(alpha, 0) == target_expr)

    return constraints


def add_dd_constraints(Q: cp.Variable) -> List[cp.Constraint]:
    m = Q.shape[0]
    constraints: List[cp.Constraint] = [Q == Q.T]
    abs_off = cp.Variable((m, m), nonneg=True)

    for i in range(m):
        for j in range(m):
            if i == j:
                constraints.append(abs_off[i, j] == 0)
            else:
                constraints += [abs_off[i, j] >= Q[i, j], abs_off[i, j] >= -Q[i, j]]
        constraints.append(Q[i, i] >= cp.sum(abs_off[i, :]))
    return constraints


def add_sdd_constraints(Q: cp.Variable) -> List[cp.Constraint]:
    """SDD decomposition via 2x2 PSD-supported blocks encoded as SOC constraints."""
    m = Q.shape[0]
    constraints: List[cp.Constraint] = [Q == Q.T]
    diag_terms = [[None for _ in range(m)] for __ in range(m)]

    for i in range(m):
        for j in range(i + 1, m):
            a = cp.Variable(nonneg=True, name=f"sdd_a_{i}_{j}")
            c = cp.Variable(nonneg=True, name=f"sdd_c_{i}_{j}")
            b = cp.Variable(name=f"sdd_b_{i}_{j}")
            diag_terms[i][j] = a
            diag_terms[j][i] = c
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


def solve_gram_bound(instance: dict, cone: str, solver: str, U: np.ndarray | None = None, verbose: bool = False) -> dict:
    n = int(instance["n"])
    mons = [tuple(a) for a in instance["basis"]]
    p = instance_to_poly(instance)
    m = len(mons)
    U = np.eye(m) if U is None else np.asarray(U, dtype=float)

    Q = cp.Variable((m, m), symmetric=True)
    gamma = cp.Variable(name="gamma")
    constraints = coefficient_constraints_with_basis(Q, gamma, p, mons, U=U, n=n)

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
            "error": f"{solver} not installed. Installed solvers: {sorted(installed)}",
            "Q_value": None,
        }

    kwargs = {"solver": solver, "verbose": verbose}
    if solver.upper() == "SCS":
        kwargs.update({"eps": 1e-5, "max_iters": 50_000})
    if solver.upper() == "CLARABEL":
        kwargs.update({"tol_gap_abs": 1e-7, "tol_feas": 1e-7})

    start = time.perf_counter()
    try:
        val = problem.solve(**kwargs)
        wall = time.perf_counter() - start
        return {
            "status": problem.status,
            "value": float(val) if val is not None else np.nan,
            "solve_time": float(problem.solver_stats.solve_time) if problem.solver_stats.solve_time is not None else np.nan,
            "wall_time": wall,
            "error": "",
            "Q_value": Q.value.copy() if Q.value is not None else None,
        }
    except Exception as e:
        wall = time.perf_counter() - start
        return {
            "status": "exception",
            "value": np.nan,
            "solve_time": np.nan,
            "wall_time": wall,
            "error": repr(e),
            "Q_value": None,
        }


def is_success(result: dict) -> bool:
    return result["status"] in {"optimal", "optimal_inaccurate"} and np.isfinite(result["value"])


def run_sdsos_plus_on_instance(instance: dict, solver: str, max_rounds: int, ridge: float, verbose: bool = False) -> List[dict]:
    rows: List[dict] = []
    m = len(instance["basis"])
    U = np.eye(m)

    for k in range(max_rounds + 1):
        res = solve_gram_bound(instance, cone="sdsos", solver=solver, U=U, verbose=verbose)
        rows.append({
            "method": "sdsos_plus",
            "round": k,
            "status": res["status"],
            "value": res["value"],
            "solve_time": res["solve_time"],
            "wall_time": res["wall_time"],
            "error": res["error"],
            "basis_condition": float(np.linalg.cond(U)) if np.all(np.isfinite(U)) else np.nan,
        })
        if not is_success(res) or res["Q_value"] is None:
            break
        R = sqrtm_psd(res["Q_value"], ridge=ridge)
        U = R @ U
        if not np.all(np.isfinite(U)):
            rows.append({
                "method": "sdsos_plus",
                "round": k + 1,
                "status": "basis_nonfinite",
                "value": np.nan,
                "solve_time": np.nan,
                "wall_time": 0.0,
                "error": "Basis update produced non-finite entries.",
                "basis_condition": np.nan,
            })
            break
    return rows


def solve_sos_basis_diagnostic(instance: dict, solver: str, sos_result: dict, ridge: float, verbose: bool = False) -> dict:
    if not is_success(sos_result) or sos_result["Q_value"] is None:
        return {
            "method": "sos_basis_sdsos",
            "round": 0,
            "status": "skipped_no_sos_gram",
            "value": np.nan,
            "solve_time": np.nan,
            "wall_time": 0.0,
            "error": "SOS solve did not produce a usable Gram matrix.",
            "basis_condition": np.nan,
        }
    U = sqrtm_psd(sos_result["Q_value"], ridge=ridge)
    res = solve_gram_bound(instance, cone="sdsos", solver=solver, U=U, verbose=verbose)
    return {
        "method": "sos_basis_sdsos",
        "round": 0,
        "status": res["status"],
        "value": res["value"],
        "solve_time": res["solve_time"],
        "wall_time": res["wall_time"],
        "error": res["error"],
        "basis_condition": float(np.linalg.cond(U)) if np.all(np.isfinite(U)) else np.nan,
    }


def summarize_results(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    df = df.copy()
    df["is_success"] = df["status"].isin(["optimal", "optimal_inaccurate"])
    df["finite_value"] = np.where(np.isfinite(df["value"]), df["value"], np.nan)
    sos = (
        df[(df["method"] == "sos") & df["is_success"] & np.isfinite(df["value"])]
        .groupby(["n", "d", "seed"], as_index=False)["value"]
        .max()
        .rename(columns={"value": "sos_value"})
    )
    df = df.merge(sos, on=["n", "d", "seed"], how="left")
    df["gap_to_sos"] = df["sos_value"] - df["finite_value"]
    df["rel_gap_to_sos"] = df["gap_to_sos"] / np.maximum(1.0, np.abs(df["sos_value"]))
    df.to_csv(outdir / "results_with_gaps.csv", index=False)
    summary = (
        df.groupby(["generator_type", "n", "d", "basis_size_m", "method", "round"], as_index=False)
        .agg(
            num_attempts=("value", "size"),
            success_rate=("is_success", "mean"),
            num_finite=("finite_value", lambda x: int(np.isfinite(x).sum())),
            mean_value=("finite_value", "mean"),
            std_value=("finite_value", "std"),
            mean_rel_gap_to_sos=("rel_gap_to_sos", "mean"),
            std_rel_gap_to_sos=("rel_gap_to_sos", "std"),
            median_wall_time=("wall_time", "median"),
            mean_basis_condition=("basis_condition", "mean"),
        )
    )
    summary.to_csv(outdir / "summary.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--solver", default="CLARABEL")
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--max-m", type=int, default=None)
    parser.add_argument("--n-list", nargs="*", type=int, default=None)
    parser.add_argument("--d-list", nargs="*", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--run-sos-basis", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    instances = load_instances(args.instances)
    if args.max_m is not None:
        instances = [x for x in instances if len(x["basis"]) <= args.max_m]
    if args.n_list:
        instances = [x for x in instances if int(x["n"]) in set(args.n_list)]
    if args.d_list:
        instances = [x for x in instances if int(x["d"]) in set(args.d_list)]
    if args.seeds:
        instances = [x for x in instances if int(x["seed"]) in set(args.seeds)]

    print(f"Loaded {len(instances)} filtered instances from {args.instances}")
    print(f"Solver: {args.solver}")
    print(f"Max rounds: {args.max_rounds}")

    with (args.outdir / "instances_filtered.jsonl").open("w") as f:
        for inst in instances:
            f.write(json.dumps(inst) + "\n")

    rows = []
    for inst in tqdm(instances, desc="SDSOS+ instances"):
        n = int(inst["n"])
        d = int(inst["d"])
        seed = int(inst["seed"])
        m = len(inst["basis"])
        generator_type = inst.get("generator", {}).get("type", "unknown")
        base = {
            "n": n,
            "d": d,
            "degree": 2 * d,
            "seed": seed,
            "basis_size_m": m,
            "generator_type": generator_type,
            "solver": args.solver,
        }
        sos_res = solve_gram_bound(inst, cone="sos", solver=args.solver, U=np.eye(m), verbose=args.verbose)
        rows.append({
            **base,
            "method": "sos",
            "round": 0,
            "status": sos_res["status"],
            "value": sos_res["value"],
            "solve_time": sos_res["solve_time"],
            "wall_time": sos_res["wall_time"],
            "error": sos_res["error"],
            "basis_condition": 1.0,
        })
        for r in run_sdsos_plus_on_instance(inst, solver=args.solver, max_rounds=args.max_rounds, ridge=args.ridge, verbose=args.verbose):
            rows.append({**base, **r})
        if args.run_sos_basis:
            rows.append({**base, **solve_sos_basis_diagnostic(inst, solver=args.solver, sos_result=sos_res, ridge=args.ridge, verbose=args.verbose)})
        pd.DataFrame(rows).to_csv(args.outdir / "results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.outdir / "results.csv", index=False)
    summarize_results(df, args.outdir)
    print(f"Wrote: {args.outdir / 'results.csv'}")
    print(f"Wrote: {args.outdir / 'results_with_gaps.csv'}")
    print(f"Wrote: {args.outdir / 'summary.csv'}")
    print("Installed CVXPY solvers:", cp.installed_solvers())


if __name__ == "__main__":
    main()
