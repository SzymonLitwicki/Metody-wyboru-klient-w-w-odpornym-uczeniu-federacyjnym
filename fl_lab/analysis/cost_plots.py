# fl_lab/analysis/cost_plots.py
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fl_lab.analysis.constants import STRATEGY_ORDER

sns.set_theme(style="whitegrid")

_STRATEGY_ORDER = STRATEGY_ORDER


def _collect_per_strategy(runs: list[dict], metric: str) -> dict[str, list[float]]:
    data: dict[str, list[float]] = {}
    for run in runs:
        strat = run["summary"].get("selection_strategy", "unknown")
        values = [r[metric] for r in run["rounds"] if metric in r]
        data.setdefault(strat, []).extend(values)
    return data


def selection_time_boxplot(runs: list[dict], output_path: Path) -> None:
    data = _collect_per_strategy(runs, "selection_time_ms")
    strategies = [s for s in _STRATEGY_ORDER if s in data]
    values = [data[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, tick_labels=strategies, patch_artist=True,
               medianprops={"color": "black", "linewidth": 2})
    ax.set_ylabel("Selection time [ms]")
    ax.set_xlabel("Selection strategy")
    ax.set_title("Computational cost of client selection (per round)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_time_boxplot(runs: list[dict], output_path: Path) -> None:
    data = _collect_per_strategy(runs, "avg_train_time_ms")
    strategies = [s for s in _STRATEGY_ORDER if s in data]
    values = [data[s] for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, tick_labels=strategies, patch_artist=True,
               medianprops={"color": "black", "linewidth": 2})
    ax.set_ylabel("Mean client training time [ms]")
    ax.set_xlabel("Selection strategy")
    ax.set_title("Mean local training time per round")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def bytes_per_round_bar(runs: list[dict], output_path: Path) -> None:
    data: dict[str, float] = {}
    for run in runs:
        strat = run["summary"].get("selection_strategy", "unknown")
        vals = [r["bytes_per_round"] for r in run["rounds"] if "bytes_per_round" in r]
        if vals:
            data[strat] = float(np.mean(vals))
    strategies = [s for s in _STRATEGY_ORDER if s in data]
    mb_values = [data[s] / 1e6 for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(strategies))
    ax.bar(x, mb_values, color=sns.color_palette("tab10", len(strategies)))
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=20, ha="right")
    ax.set_ylabel("Data uploaded per round [MB]")
    ax.set_title("Communication cost per round (client → server upload)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
