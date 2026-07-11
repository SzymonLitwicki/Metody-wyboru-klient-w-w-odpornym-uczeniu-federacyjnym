import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from fl_lab.utils.config import deep_merge


def check_done(output_dir: str, run_name: str, seed: int) -> bool:
    summary = Path(output_dir) / run_name / f"seed_{seed}" / "final_summary.json"
    if not summary.exists():
        return False
    try:
        content = summary.read_text().strip()
        return bool(content) and json.loads(content).get("simulation_completed", False)
    except (json.JSONDecodeError, OSError):
        return False


def build_run_config(base_raw: dict, run: dict, seed: int, output_dir: str) -> dict:
    overrides = {k: v for k, v in run.items() if k != "name"}
    overrides["seed"] = seed
    overrides["name"] = f"seed_{seed}"
    overrides["logging"] = {"output_dir": str(Path(output_dir) / run["name"])}
    return deep_merge(base_raw, overrides)


def _execute_one(task: dict) -> tuple[str, int]:
    script_path = task["script_path"]
    executable = task["executable"]
    merged = task["merged"]
    tag = task["tag"]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="fl_run_"
    ) as f:
        yaml.dump(merged, f)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [executable, script_path, "--config", tmp_path],
            check=False,
        )
        return tag, result.returncode
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def run_grid(grid_path: str, seeds: list[int], workers: int = 1) -> tuple[int, int]:
    with open(grid_path) as f:
        grid_spec = yaml.safe_load(f)

    base_path = Path(grid_spec["base"])
    output_dir = grid_spec.get("output_dir", "results")
    runs = grid_spec["runs"]

    with open(base_path) as f:
        base_raw = yaml.safe_load(f) or {}

    script_path = str(Path(__file__).parent / "run_experiment.py")
    executable = sys.executable

    total = len(runs) * len(seeds)
    skipped = 0
    failed = 0

    pending: list[dict] = []
    for run in runs:
        for seed in seeds:
            tag = f"{run['name']} seed={seed}"
            if check_done(output_dir, run["name"], seed):
                print(f"[SKIP] {tag}")
                skipped += 1
                continue
            pending.append({
                "script_path": script_path,
                "executable": executable,
                "merged": build_run_config(base_raw, run, seed, output_dir),
                "tag": tag,
            })

    completed_count = skipped
    n_pending = len(pending)

    if workers == 1:
        for i, task in enumerate(pending):
            print(f"[RUN ] {task['tag']} ({completed_count + 1}/{total})")
            tag, rc = _execute_one(task)
            completed_count += 1
            if rc != 0:
                print(f"[FAIL] {tag} exited with code {rc}")
                failed += 1
    else:
        print(f"Running {n_pending} experiments with {workers} parallel workers.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_tag = {
                executor.submit(_execute_one, task): task["tag"]
                for task in pending
            }
            for future in concurrent.futures.as_completed(future_to_tag):
                tag, rc = future.result()
                completed_count += 1
                status = "DONE" if rc == 0 else "FAIL"
                print(f"[{status}] {tag} ({completed_count}/{total})")
                if rc != 0:
                    failed += 1

    if failed:
        print(f"\nGrid done: {completed_count}/{total} runs ({skipped} skipped, {failed} failed, {completed_count - skipped - failed} succeeded)")
    else:
        print(f"\nGrid done: {completed_count}/{total} runs ({skipped} skipped, {completed_count - skipped} succeeded)")
    return skipped, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grid of FL experiments")
    parser.add_argument("--grid", required=True, help="Path to grid-spec YAML")
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Seeds to run"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of experiments to run in parallel (default: 1). "
             "Use 2-4 on a single GPU to keep it saturated between runs.",
    )
    args = parser.parse_args()
    run_grid(args.grid, args.seeds, workers=args.workers)


if __name__ == "__main__":
    main()
