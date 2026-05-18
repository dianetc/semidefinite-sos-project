# DSOS/SDSOS/SOS Benchmark I

Run a smoke test:

```bash
pip install numpy pandas cvxpy scipy tqdm
python dsos_sdsos_benchmark1.py --n-list 2 3 --d-list 2 --seeds 0 1 --outdir runs/smoke
```

Run a main small-grid experiment:

```bash
python dsos_sdsos_benchmark1.py \
  --n-list 2 3 4 5 6 \
  --d-list 2 3 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --outdir runs/main
```

Outputs:
- `instances.jsonl`: reproducible polynomial instance library.
- `results.csv`: every cone/solver/seed result.
- `summary.csv`: grouped mean, standard deviation, time, and gap-to-SOS summaries.

Notes:
- The first-pass generator uses known-SOS random Gram instances, avoiding the problem that arbitrary random even-degree polynomials can be unbounded below.
- Use MOSEK if available; CLARABEL and SCS are useful for solver-sensitivity experiments but may be less stable on hard SDP instances.
