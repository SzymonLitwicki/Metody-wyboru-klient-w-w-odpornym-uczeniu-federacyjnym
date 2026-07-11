from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fl_lab.analysis.constants import STRATEGY_ORDER, AGGREGATION_ORDER, ATTACK_ORDER

sns.set_theme(style="whitegrid")

_STRATEGY_ORDER = STRATEGY_ORDER


def _strategy_sort_key(name: str) -> int:
    try:
        return _STRATEGY_ORDER.index(name)
    except ValueError:
        return len(_STRATEGY_ORDER)


def convergence_plot(grouped: dict[str, list[dict]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in sorted(grouped.keys(), key=_strategy_sort_key):
        runs = grouped[strategy]
        runs_with_rounds = [r for r in runs if r["rounds"]]
        if not runs_with_rounds:
            continue
        max_rounds = max(len(r["rounds"]) for r in runs_with_rounds)
        acc_matrix = np.full((len(runs_with_rounds), max_rounds), np.nan)
        for i, run in enumerate(runs_with_rounds):
            for j, row in enumerate(run["rounds"]):
                acc_matrix[i, j] = row["test_accuracy"]
        mean = np.nanmean(acc_matrix, axis=0)
        std = np.nanstd(acc_matrix, axis=0)
        rounds = np.arange(1, max_rounds + 1)
        ax.plot(rounds, mean, label=strategy)
        ax.fill_between(rounds, mean - std, mean + std, alpha=0.15)

    ax.set_xlabel("Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Convergence by Selection Strategy")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def accuracy_bar_plot(grouped: dict[str, list[dict]], output_path: Path) -> None:
    strategies = sorted(grouped.keys(), key=_strategy_sort_key)
    means = []
    stds = []
    for s in strategies:
        accs = [r["summary"]["final_accuracy"] for r in grouped[s] if "final_accuracy" in r["summary"]]
        means.append(float(np.mean(accs)))
        stds.append(float(np.std(accs)))

    order = sorted(range(len(strategies)), key=lambda i: means[i], reverse=True)
    strategies = [strategies[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strategies))
    ax.bar(x, means, yerr=stds, capsize=4, color=sns.color_palette("tab10", len(strategies)))
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Final Accuracy by Selection Strategy (mean ± std)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


_AGGREGATION_ORDER = AGGREGATION_ORDER
_ATTACK_ORDER = ATTACK_ORDER


def _build_accuracy_matrix(
    runs: list[dict],
    row_key: str,
    col_key: str,
    row_order: list[str],
    col_order: list[str],
) -> np.ndarray:
    data: dict[tuple[str, str], list[float]] = {}
    for run in runs:
        r = run["summary"][row_key]
        c = run["summary"][col_key]
        data.setdefault((r, c), []).append(run["summary"]["final_accuracy"])
    matrix = np.full((len(row_order), len(col_order)), np.nan)
    for i, r in enumerate(row_order):
        for j, c in enumerate(col_order):
            vals = data.get((r, c), [])
            if vals:
                matrix[i, j] = float(np.mean(vals))
    return matrix


def strategy_aggregation_heatmap(runs: list[dict], output_path) -> None:
    matrix = _build_accuracy_matrix(
        runs, "selection_strategy", "aggregation_method", _STRATEGY_ORDER, _AGGREGATION_ORDER
    )
    df = pd.DataFrame(matrix, index=_STRATEGY_ORDER, columns=_AGGREGATION_ORDER)
    fig, ax = plt.subplots(figsize=(11, 7))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0.0, vmax=1.0,
                linewidths=0.5, linecolor="white")
    ax.set_title("Mean Accuracy: Strategy × Aggregation (Phase 4)")
    ax.set_xlabel("Aggregation Method")
    ax.set_ylabel("Selection Strategy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def strategy_attack_heatmap(runs: list[dict], output_path) -> None:
    matrix = _build_accuracy_matrix(
        runs, "selection_strategy", "attack_type", _STRATEGY_ORDER, _ATTACK_ORDER
    )
    df = pd.DataFrame(matrix, index=_STRATEGY_ORDER, columns=_ATTACK_ORDER)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax, vmin=0.0, vmax=1.0,
                linewidths=0.5, linecolor="white")
    ax.set_title("Mean Accuracy: Strategy × Attack Type (Phase 3)")
    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Selection Strategy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def phase_comparison_bar(runs_by_phase: dict[str, list[dict]], output_path) -> None:
    strategies = _STRATEGY_ORDER
    phases = sorted(runs_by_phase.keys())
    x = np.arange(len(strategies))
    width = 0.8 / max(len(phases), 1)

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, phase in enumerate(phases):
        runs = runs_by_phase[phase]
        means, stds = [], []
        for s in strategies:
            vals = [r["summary"]["final_accuracy"] for r in runs
                    if r["summary"]["selection_strategy"] == s]
            means.append(float(np.mean(vals)) if vals else 0.0)
            stds.append(float(np.std(vals)) if vals else 0.0)
        ax.bar(x + i * width, means, width, label=phase, yerr=stds, capsize=3)

    ax.set_xticks(x + width * (len(phases) - 1) / 2)
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylabel("Mean Final Accuracy")
    ax.set_title("Strategy Performance Across Phases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def scaling_bar(runs: list[dict], output_path) -> None:
    strategies = _STRATEGY_ORDER
    n_variants = ["n20", "n100"]
    k_variants = ["k5", "k20"]

    def _collect(variant):

        return {
            s: [r["summary"]["final_accuracy"] for r in runs
                if r["run_name"].split("_")[1] == variant
                and r["summary"]["selection_strategy"] == s]
            for s in strategies
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(strategies))
    width = 0.35

    for ax, variants, title, xlabel in [
        (ax1, n_variants, "Effect of Pool Size (N)", "N variant"),
        (ax2, k_variants, "Effect of Selection Fraction (K)", "K variant"),
    ]:
        for i, variant in enumerate(variants):
            grouped = _collect(variant)
            means = [float(np.mean(grouped[s])) if grouped[s] else 0.0 for s in strategies]
            stds = [float(np.std(grouped[s])) if grouped[s] else 0.0 for s in strategies]
            ax.bar(x + i * width, means, width, label=variant, yerr=stds, capsize=3)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(strategies, rotation=30, ha="right")
        ax.set_ylabel("Mean Final Accuracy")
        ax.set_title(title)
        ax.legend(title=xlabel)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


