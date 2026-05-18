import pandas as pd
import numpy as np
from pathlib import Path

summary_path = Path("runs/initialization_bottleneck/summary.csv")
df = pd.read_csv(summary_path)

rows = []
for (n, d, m), g in df.groupby(["n", "d", "basis_size_m"]):
    fixed = g[(g["method"] == "sdsos_plus") & (g["round"] == 0)]
    oracle = g[g["method"] == "sos_basis_sdsos"]
    sos = g[g["method"] == "sos"]

    def val(frame, col):
        if frame.empty:
            return np.nan
        return frame[col].iloc[0]

    rows.append({
        "n": int(n),
        "d": int(d),
        "m": int(m),
        "fixed_sdsos_success": val(fixed, "success_rate"),
        "fixed_sdsos_gap": val(fixed, "mean_rel_gap_to_sos"),
        "sos_basis_sdsos_success": val(oracle, "success_rate"),
        "sos_basis_sdsos_gap": val(oracle, "mean_rel_gap_to_sos"),
        "sos_success": val(sos, "success_rate"),
    })

out = pd.DataFrame(rows).sort_values(["d", "n"])
out.to_csv("runs/initialization_bottleneck/initialization_bottleneck_table.csv", index=False)

def fmt_rate(x):
    return "--" if pd.isna(x) else f"{x:.2f}"

def fmt_gap(x):
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 100:
        return f"{x:.1e}"
    return f"{x:.3f}"

tex = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{ccc cc cc c}
\toprule
& & & \multicolumn{2}{c}{Fixed SDSOS} & \multicolumn{2}{c}{SOS-basis SDSOS} & SOS \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}
\(n\) & \(d\) & \(m\) & Succ. & Gap & Succ. & Gap & Succ. \\
\midrule
"""

for _, r in out.iterrows():
    tex += (
        f"{int(r.n)} & {int(r.d)} & {int(r.m)} & "
        f"{fmt_rate(r.fixed_sdsos_success)} & {fmt_gap(r.fixed_sdsos_gap)} & "
        f"{fmt_rate(r.sos_basis_sdsos_success)} & {fmt_gap(r.sos_basis_sdsos_gap)} & "
        f"{fmt_rate(r.sos_success)} \\\\\n"
    )

tex += r"""\bottomrule
\end{tabular}
\caption{
Initialization-bottleneck diagnostic on the generic SOS ensemble. 
Fixed SDSOS solves the problem in the monomial basis. 
SOS-basis SDSOS first solves the SOS relaxation, factors the SOS Gram matrix, and then re-solves SDSOS in the induced basis. 
This is an oracle diagnostic rather than a practical algorithm, since the basis is obtained from the SDP solution. 
The comparison tests whether SDSOS failure is partly caused by a poor initial basis.
}
\label{tab:initialization-bottleneck}
\end{table}
"""

Path("runs/initialization_bottleneck/initialization_bottleneck_table.tex").write_text(tex)
print(out)
print("\nWrote:")
print("  runs/initialization_bottleneck/initialization_bottleneck_table.csv")
print("  runs/initialization_bottleneck/initialization_bottleneck_table.tex")
