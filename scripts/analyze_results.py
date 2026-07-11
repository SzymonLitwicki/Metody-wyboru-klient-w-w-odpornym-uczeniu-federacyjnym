import argparse
import json
from pathlib import Path

from fl_lab.analysis.loader import group_by_strategy, load_phase
from fl_lab.analysis.plots import accuracy_bar_plot, convergence_plot
from fl_lab.analysis.statistics import pairwise_ttest, rank_strategies
from fl_lab.analysis.tables import export_csv, export_latex, main_results_table


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FL experiment results")
    parser.add_argument("--results_dir", required=True, help="Path to phase results directory")
    parser.add_argument("--output", required=True, help="Output directory for plots and tables")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_phase(results_dir)
    print(f"Loaded {len(runs)} completed runs from {results_dir}")
    if not runs:
        print("No completed runs found. Exiting.")
        return

    grouped = group_by_strategy(runs)
    print(f"Strategies: {sorted(grouped.keys())}")

    convergence_plot(grouped, output_dir / "convergence.png")
    accuracy_bar_plot(grouped, output_dir / "accuracy_bar.png")

    df = main_results_table(grouped)
    export_csv(df, output_dir / "main_results.csv")
    export_latex(df, output_dir / "main_results.tex")

    acc_grouped = {}
    for s, rs in grouped.items():
        accs = []
        for r in rs:
            acc = r["summary"].get("final_accuracy")
            if acc is None:
                print(f"Warning: run {r['run_dir']} missing final_accuracy, skipping.")
            else:
                accs.append(acc)
        if accs:
            acc_grouped[s] = accs
    ranking = rank_strategies(acc_grouped)
    print("\nStrategy ranking (by mean final accuracy):")
    for rank, (strategy, mean_acc, std_acc) in enumerate(ranking, 1):
        print(f"  {rank}. {strategy}: {mean_acc:.4f} ± {std_acc:.4f}")

    ttest_results = pairwise_ttest(acc_grouped)
    stats_out = {
        "ranking": [{"strategy": s, "mean_accuracy": m, "std_accuracy": sd}
                    for s, m, sd in ranking],
        "pairwise_ttest": {
            f"{s1}_vs_{s2}": v for (s1, s2), v in ttest_results.items()
        },
    }
    (output_dir / "statistics.json").write_text(json.dumps(stats_out, indent=2))

    print(f"\nOutputs written to {output_dir}")


if __name__ == "__main__":
    main()
