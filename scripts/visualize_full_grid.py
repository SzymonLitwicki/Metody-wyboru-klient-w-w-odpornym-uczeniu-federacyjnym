import argparse
from pathlib import Path

import pandas as pd

from fl_lab.analysis.constants import ATTACK_ORDER, AGGREGATION_ORDER
from fl_lab.analysis.loader import load_phase
from fl_lab.analysis.full_grid import (
    accuracy_vs_byzantine_pct,
    attack_comparison_bars,
    cohort_composition_vs_robustness,
    convergence_small_multiples,
    heatmap_accuracy_drop_grid,
    heatmap_strategy_aggregation_grid,
    memory_cost_bars,
    selection_time_boxplot,
    summary_table_csv,
    tradeoff_scatter,
)


def _build_dataframes(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows = []
    round_rows = []

    for run in runs:
        s = run["summary"]
        meta = {
            "strategy": s.get("selection_strategy"),
            "aggregation": s.get("aggregation_method"),
            "attack": s.get("attack_type"),
            "byzantine_pct": float(s.get("byzantine_pct", 0)),
        }
        run_rows.append({
            **meta,
            "final_accuracy": float(s.get("final_accuracy", 0)),
            "rounds_completed": int(s.get("rounds_completed", 0)),
            "simulation_completed": bool(s.get("simulation_completed", False)),
        })
        for r in run["rounds"]:
            round_rows.append({**meta, **r})

    df_runs = pd.DataFrame(run_rows)
    df_rounds = pd.DataFrame(round_rows)

    if "round" in df_rounds.columns and "round_num" not in df_rounds.columns:
        df_rounds = df_rounds.rename(columns={"round": "round_num"})

    return df_runs, df_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generuj wykresy dla eksperymentu full_cross")
    parser.add_argument(
        "--results_root", default="results/full_cross",
        help="Katalog z wynikami (domyślnie: results/full_cross)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Katalog wyjściowy (domyślnie: <results_root>/figures)",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output) if args.output else results_root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Wczytywanie wyników z: {results_root}")
    runs = load_phase(results_root)
    print(f"Wczytano {len(runs)} uruchomień")
    if not runs:
        print("Brak zakończonych uruchomień. Kończę.")
        return

    df_runs, df_rounds = _build_dataframes(runs)
    print(f"df_runs: {df_runs.shape}, df_rounds: {df_rounds.shape}")

    _gen("heatmap_strategy_agg_grid.png", output_dir,
         lambda p: heatmap_strategy_aggregation_grid(df_runs, p))

    _gen("heatmap_accuracy_drop_grid.png", output_dir,
         lambda p: heatmap_accuracy_drop_grid(df_runs, p))
    
    for attack in ATTACK_ORDER:
        _gen(f"convergence_{attack}.png", output_dir,
             lambda p, a=attack: convergence_small_multiples(df_rounds, p, attack=a))

    for agg in ["fedavg", "fltrust"]:
        _gen(f"accuracy_vs_pct_{agg}.png", output_dir,
             lambda p, a=agg: accuracy_vs_byzantine_pct(df_runs, p, aggregation=a))

    representative_pairs = [
        ("random", "fedavg"),
        ("reputation", "fltrust"),
        ("adaptive_mab", "krum"),
        ("adaptive_mab", "bulyan"),
    ]
    for strategy, aggregation in representative_pairs:
        _gen(f"attack_bars_{strategy}_{aggregation}.png", output_dir,
             lambda p, s=strategy, a=aggregation: attack_comparison_bars(df_runs, p, s, a))

    _gen("selection_time_boxplot.png", output_dir,
         lambda p: selection_time_boxplot(df_rounds, p))

    _gen("memory_cost_bars.png", output_dir,
         lambda p: memory_cost_bars(df_rounds, p))

    _gen("tradeoff_scatter.png", output_dir,
         lambda p: tradeoff_scatter(df_runs, df_rounds, p))

    _gen("cohort_composition.png", output_dir,
         lambda p: cohort_composition_vs_robustness(df_rounds, p))

    print("  summary_table.csv / summary_table.tex ...")
    summary_table_csv(df_runs, df_rounds, output_dir)

    print(f"\nWszystkie pliki zapisane do: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<45} {size_kb:>5} KB")


def _gen(filename: str, output_dir: Path, fn) -> None:
    print(f"  {filename} ...")
    try:
        fn(output_dir / filename)
    except Exception as exc:
        print(f"    BŁĄD: {exc}")


if __name__ == "__main__":
    main()
