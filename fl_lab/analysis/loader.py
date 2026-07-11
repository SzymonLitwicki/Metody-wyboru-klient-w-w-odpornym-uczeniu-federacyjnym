import csv
import json
from pathlib import Path


def load_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "final_summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text())

    rounds: list[dict] = []
    csv_path = run_dir / "round_metrics.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            for row in csv.DictReader(f):

                round_num = row.get("round_num") or row.get("round", "0")
                rounds.append({
                    "round": int(round_num),
                    "test_accuracy": float(row["test_accuracy"]),
                    "train_loss": float(row["train_loss"]),
                    "selection_time_ms": float(row.get("selection_time_ms", 0)),
                    "avg_train_time_ms": float(row.get("avg_train_time_ms", 0)),
                    "bytes_per_round": int(float(row.get("bytes_per_round", 0))),
                    "n_byzantine_in_cohort": int(float(row.get("n_byzantine_in_cohort", 0))),
                    "selector_state_bytes": int(float(row.get("selector_state_bytes", 0))),
                    "selection_peak_mem_bytes": int(float(row.get("selection_peak_mem_bytes", 0))),
                    "byzantine_in_cohort_ids": str(row.get("byzantine_in_cohort_ids", "[]")),
                })

    client_csv = run_dir / "client_metrics.csv"
    return {
        "summary": summary,
        "rounds": rounds,
        "run_dir": str(run_dir),
        "client_metrics_path": str(client_csv) if client_csv.exists() else None,
    }


def load_phase(phase_dir: Path) -> list[dict]:
    runs = []
    for summary_path in sorted(phase_dir.glob("**/final_summary.json")):
        run = load_run(summary_path.parent)
        if run is not None:
            runs.append(run)
    return runs


def group_by_strategy(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for run in runs:
        strategy = run["summary"].get("selection_strategy", "unknown")
        groups.setdefault(strategy, []).append(run)
    return groups
