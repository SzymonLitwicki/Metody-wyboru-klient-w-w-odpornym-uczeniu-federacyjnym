import argparse
import csv
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import yaml

from fl_lab.fl.simulation import run_simulation
from fl_lab.utils.config import ExperimentConfig


_SMOKE_RUNS = [
    ("random_fedavg_gaussian_pct10",      "random",       "fedavg",       "gaussian_noise", 0.10),
    ("cyclic_trimmed_label_pct20",         "cyclic",       "trimmed_mean", "label_flip",     0.20),
    ("reputation_krum_sign_pct40",         "reputation",   "krum",         "sign_flip",      0.40),  
    ("adaptive_fltrust_gaussian_pct10",    "adaptive_mab", "fltrust",      "gaussian_noise", 0.10),
    ("random_bulyan_label_pct40",          "random",       "bulyan",       "label_flip",     0.40), 
    ("cyclic_fedavg_sign_pct20",           "cyclic",       "fedavg",       "sign_flip",      0.20),
    ("reputation_trimmed_gaussian_pct40",  "reputation",   "trimmed_mean", "gaussian_noise", 0.40),
    ("adaptive_krum_label_pct20",          "adaptive_mab", "krum",         "label_flip",     0.20),
    ("random_fltrust_sign_pct10",          "random",       "fltrust",      "sign_flip",      0.10),
    ("cyclic_bulyan_gaussian_pct40",       "cyclic",       "bulyan",       "gaussian_noise", 0.40),  
    ("reputation_fedavg_label_pct10",      "reputation",   "fedavg",       "label_flip",     0.10),
    ("adaptive_trimmed_sign_pct40",        "adaptive_mab", "trimmed_mean", "sign_flip",      0.40),
    ("random_krum_gaussian_pct20",         "random",       "krum",         "gaussian_noise", 0.20),
    ("cyclic_fltrust_label_pct10",         "cyclic",       "fltrust",      "label_flip",     0.10),
    ("reputation_bulyan_sign_pct20",       "reputation",   "bulyan",       "sign_flip",      0.20),
]

_NEW_ROUND_COLS = {
    "round_num", "test_accuracy", "test_loss", "train_loss",
    "selected_clients", "n_byzantine_in_cohort", "byzantine_in_cohort_ids",
    "selection_time_ms", "avg_train_time_ms", "bytes_per_round",
    "selector_state_bytes", "selection_peak_mem_bytes",
}

_CLIENT_COLS = {
    "round_num", "client_id", "is_malicious", "was_selected",
    "local_loss", "gradient_norm",
}


def _build_cfg(
    run_name: str,
    strategy: str,
    aggregation: str,
    attack: str,
    byzantine_pct: float,
    output_root: Path,
    seed: int,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate({
        "name": run_name,
        "seed": seed,
        "data": {"dataset": "cifar10", "distribution": "dirichlet", "alpha": 0.5},
        "model": {"name": "simple_cnn"},
        "fl": {
            "num_clients": 10,
            "clients_per_round": 4,
            "rounds": 2,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "batch_size": 32,
        },
        "attack": {"type": attack, "byzantine_pct": byzantine_pct, "scale": 1.0},
        "selection": {"strategy": strategy},
        "aggregation": {"method": aggregation, "krum_m": 1, "trim_ratio": 0.1},
        "logging": {"output_dir": str(output_root), "log_every": 1},
    })


def _check_round_csv(path: Path, expected_rounds: int) -> list[str]:
    errors = []
    if not path.exists():
        return [f"round_metrics.csv missing at {path}"]
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header_set = set(reader.fieldnames or [])
        rows = list(reader)

    missing = _NEW_ROUND_COLS - header_set
    if missing:
        errors.append(f"round_metrics.csv missing columns: {sorted(missing)}")

    if len(rows) != expected_rounds:
        errors.append(f"round_metrics.csv has {len(rows)} rows, expected {expected_rounds}")
        return errors

    for i, row in enumerate(rows):
        if not row.get("selection_time_ms", ""):
            errors.append(f"round {i}: selection_time_ms empty")
        if not row.get("bytes_per_round", ""):
            errors.append(f"round {i}: bytes_per_round empty")
        try:
            n_byz = int(row.get("n_byzantine_in_cohort", -1))
            if n_byz < 0:
                errors.append(f"round {i}: n_byzantine_in_cohort negative or missing")
        except (ValueError, TypeError):
            errors.append(f"round {i}: n_byzantine_in_cohort not an integer")

    return errors


def _check_client_csv(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        return [f"client_metrics.csv missing at {path}"]
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        header_set = set(reader.fieldnames or [])
        rows = list(reader)

    missing = _CLIENT_COLS - header_set
    if missing:
        errors.append(f"client_metrics.csv missing columns: {sorted(missing)}")

    if not rows:
        errors.append("client_metrics.csv has no data rows")

    return errors


def _check_cost_fields_nonzero(
    run_dir: Path, run_name: str, strategy: str, expected_rounds: int
) -> list[str]:
    errors = []
    csv_path = run_dir / "round_metrics.csv"
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []

    for i, row in enumerate(rows):
        try:
            t = float(row.get("selection_time_ms", 0))
            if t <= 0:
                errors.append(f"round {i}: selection_time_ms={t} not > 0")
        except (ValueError, TypeError):
            errors.append(f"round {i}: selection_time_ms not a float")


        if strategy in ("cyclic", "reputation", "adaptive_mab") and i > 0:
            try:
                b = int(row.get("selector_state_bytes", 0))
                if b <= 0:
                    errors.append(f"round {i}: selector_state_bytes={b} not > 0 for {strategy}")
            except (ValueError, TypeError):
                errors.append(f"round {i}: selector_state_bytes not an int")

    return errors


def _check_summary(path: Path, expected_rounds: int) -> list[str]:
    errors = []
    if not path.exists():
        return [f"final_summary.json missing at {path}"]
    try:
        summary = json.loads(path.read_text())
    except Exception as exc:
        return [f"final_summary.json parse error: {exc}"]

    if not summary.get("simulation_completed", False):
        errors.append(f"simulation_completed=False (rounds_completed={summary.get('rounds_completed')})")
    if summary.get("rounds_completed", 0) != expected_rounds:
        errors.append(
            f"rounds_completed={summary.get('rounds_completed')}, expected {expected_rounds}"
        )
    if "total_selection_time_ms" not in summary:
        errors.append("total_selection_time_ms missing from final_summary.json")

    return errors


def build_smoke_grid(
    grid_path: Path,
    output_root: Path,
    rounds: int,
    num_clients: int,
    clients_per_round: int,
    local_epochs: int,
) -> Path:
    grid_path = Path(grid_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(grid_path) as f:
        grid_spec = yaml.safe_load(f)

    base_path = Path(grid_spec["base"])
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)

    base_cfg.setdefault("fl", {})
    base_cfg["fl"]["rounds"] = rounds
    base_cfg["fl"]["num_clients"] = num_clients
    base_cfg["fl"]["clients_per_round"] = clients_per_round
    base_cfg["fl"]["local_epochs"] = local_epochs

    smoke_base_path = output_root / "smoke_base.yaml"
    with open(smoke_base_path, "w") as f:
        yaml.dump(base_cfg, f)

    smoke_grid = dict(grid_spec)
    smoke_grid["base"] = str(smoke_base_path)
    smoke_grid["output_dir"] = str(output_root)

    smoke_grid_path = output_root / "smoke_grid.yaml"
    with open(smoke_grid_path, "w") as f:
        yaml.dump(smoke_grid, f)

    return smoke_grid_path


def validate_run_output(run_dir: Path, expected_rounds: int) -> dict:
    run_dir = Path(run_dir)

    summary_path = run_dir / "final_summary.json"
    if not summary_path.exists():
        return {"ok": False, "error": f"final_summary.json missing at {run_dir}"}

    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        return {"ok": False, "error": f"final_summary.json parse error: {exc}"}

    if not summary.get("simulation_completed", False):
        return {
            "ok": False,
            "error": f"simulation_completed=False (rounds_completed={summary.get('rounds_completed')})",
        }

    csv_path = run_dir / "round_metrics.csv"
    if not csv_path.exists():
        return {"ok": False, "error": f"round_metrics.csv missing at {run_dir}"}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        rows = list(reader)

    required_cost_cols = {"selection_time_ms", "avg_train_time_ms", "bytes_per_round"}
    missing = required_cost_cols - headers
    if missing:
        return {"ok": False, "error": f"missing cost columns: {', '.join(sorted(missing))}"}

    if len(rows) != expected_rounds:
        return {
            "ok": False,
            "error": f"round_metrics.csv has {len(rows)} rows, expected {expected_rounds}",
        }

    return {"ok": True}


def run_smoke(output_root: Path, seed: int, expected_rounds: int = 2) -> bool:
    checks: list[tuple[str, bool, str]] = [] 

    for run_name, strategy, aggregation, attack, byz_pct in _SMOKE_RUNS:
        run_dir = output_root / run_name
        cfg = _build_cfg(run_name, strategy, aggregation, attack, byz_pct, output_root, seed)

        sim_ok = True
        sim_err = ""
        try:
            run_simulation(cfg)
        except Exception:
            sim_ok = False
            sim_err = traceback.format_exc().strip().splitlines()[-1]

        checks.append((f"{run_name} | simulation_no_exception", sim_ok, sim_err))
        if not sim_ok:
            for label in ("round_metrics.csv", "client_metrics.csv", "summary", "cost_fields"):
                checks.append((f"{run_name} | {label}", False, "simulation did not complete"))
            continue

      
        seed_dir = run_dir

        summary_errors = _check_summary(seed_dir / "final_summary.json", expected_rounds)
        checks.append((f"{run_name} | final_summary.json", not summary_errors,
                        "; ".join(summary_errors)))

        round_errors = _check_round_csv(seed_dir / "round_metrics.csv", expected_rounds)
        checks.append((f"{run_name} | round_metrics.csv", not round_errors,
                        "; ".join(round_errors)))

        client_errors = _check_client_csv(seed_dir / "client_metrics.csv")
        checks.append((f"{run_name} | client_metrics.csv", not client_errors,
                        "; ".join(client_errors)))

        cost_errors = _check_cost_fields_nonzero(seed_dir, run_name, strategy, expected_rounds)
        checks.append((f"{run_name} | cost_fields_nonzero", not cost_errors,
                        "; ".join(cost_errors)))

    print()
    print("=" * 72)
    print("  SMOKE TEST REPORT")
    print("=" * 72)
    passed = 0
    failed = 0
    for label, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
            print(f"  [PASS] {label}")
        else:
            failed += 1
            print(f"  [FAIL] {label}")
            if detail:
                print(f"         → {detail}")

    print()
    print(f"  Total checks : {len(checks)}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")
    print("=" * 72)
    print()

    return failed == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the FL cross-grid pipeline")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_root", default="results/smoke")
    parser.add_argument("--keep", action="store_true",
                        help="Keep output directory after test (default: clean up)")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Smoke test: N=10, K=4, R=2, E=1, seed={args.seed}")
    print(f"Output: {output_root}")
    print(f"Runs: {len(_SMOKE_RUNS)}")
    print()

    success = run_smoke(output_root=output_root, seed=args.seed)

    if not args.keep:
        shutil.rmtree(output_root, ignore_errors=True)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
