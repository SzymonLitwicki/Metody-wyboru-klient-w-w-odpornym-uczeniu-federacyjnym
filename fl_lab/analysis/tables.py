from pathlib import Path

import numpy as np
import pandas as pd

from fl_lab.analysis.statistics import rounds_to_target


def main_results_table(
    grouped: dict[str, list[dict]], target_accuracy: float = 0.5
) -> pd.DataFrame:
    rows = []
    for strategy, runs in grouped.items():
        accs = [r["summary"]["final_accuracy"] for r in runs if "final_accuracy" in r["summary"]]
        rtt_values = [rounds_to_target(r["rounds"], target_accuracy) for r in runs]
        rtt_values = [v for v in rtt_values if v is not None]
        rows.append({
            "strategy": strategy,
            "seeds": len(runs),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "rounds_to_target": int(np.mean(rtt_values)) if rtt_values else float("nan"),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("mean_accuracy", ascending=False).reset_index(drop=True)


def export_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def export_latex(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_latex(index=False, float_format="%.4f"))


def interaction_table(runs: list[dict], row_field: str, col_field: str) -> pd.DataFrame:
    data: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        row_key = str(run["summary"].get(row_field, ""))
        col_key = str(run["summary"].get(col_field, ""))
        data.setdefault((row_key, col_key), []).append(run["summary"]["final_accuracy"])

    rows = sorted({k[0] for k in data})
    cols = sorted({k[1] for k in data})
    matrix = {
        r: {c: float(np.mean(data[(r, c)])) if (r, c) in data else float("nan")
            for c in cols}
        for r in rows
    }
    df = pd.DataFrame(matrix).T
    df.index.name = row_field
    return df.reset_index()


