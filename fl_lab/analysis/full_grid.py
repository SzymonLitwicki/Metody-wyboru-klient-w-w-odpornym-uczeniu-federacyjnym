from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from fl_lab.analysis.constants import (
    AGGREGATION_ORDER,
    ATTACK_ORDER,
    PCT_ORDER,
    STRATEGY_ORDER,
)

sns.set_theme(style="whitegrid")

_PCT_LABELS = {0.1: "10%", 0.2: "20%", 0.4: "40%"}
_ATTACK_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "label_flip": "Label flip",
    "sign_flip": "Sign flip",
}
_STRATEGY_LABELS = {
    "random": "Random",
    "cyclic": "Cyclic",
    "reputation": "Reputation",
    "adaptive_mab": "Adaptive MAB",
}
_AGG_LABELS = {
    "fedavg": "FedAvg",
    "trimmed_mean": "TrimMean",
    "krum": "Krum",
    "bulyan": "Bulyan",
    "fltrust": "FLTrust",
}


def _safe_pivot(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    index_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            np.full((len(index_order), len(col_order)), np.nan),
            index=index_order,
            columns=col_order,
        )
    piv = df.pivot_table(index=index, columns=columns, values=values, aggfunc="mean")
    return piv.reindex(index=index_order, columns=col_order)


def heatmap_strategy_aggregation_grid(df_runs: pd.DataFrame, output_path: Path) -> None:

    attacks = ATTACK_ORDER
    pcts = PCT_ORDER

    fig, axes = plt.subplots(len(attacks), len(pcts), figsize=(24, 20), constrained_layout=True)

    for i, attack in enumerate(attacks):
        for j, pct in enumerate(pcts):
            ax = axes[i, j]
            subset = df_runs[
                (df_runs["attack"] == attack) & (df_runs["byzantine_pct"] == pct)
            ]
            piv = _safe_pivot(
                subset, "strategy", "aggregation", "final_accuracy",
                STRATEGY_ORDER, AGGREGATION_ORDER,
            )

            if piv.isnull().all().all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(
                    f"{_ATTACK_LABELS.get(attack, attack)}\n{_PCT_LABELS.get(pct, pct)} byzantine",
                    fontsize=10,
                )
                continue

            display = piv.rename(index=_STRATEGY_LABELS, columns=_AGG_LABELS)
            sns.heatmap(
                display, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0.0, vmax=1.0, cbar=False,
                linewidths=0.4, linecolor="white", annot_kws={"size": 9},
            )
            ax.set_title(
                f"{_ATTACK_LABELS.get(attack, attack)}\n{_PCT_LABELS.get(pct, pct)} byzantine",
                fontsize=10,
            )
            ax.set_ylabel("Strategy" if j == 0 else "", fontsize=9)
            ax.set_xlabel("Aggregation" if i == len(attacks) - 1 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.55, pad=0.02)
    cbar.set_label("Test accuracy", fontsize=12)

    fig.suptitle(
        "Final accuracy: strategy × aggregation (full experiment grid)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmap_accuracy_drop_grid(df_runs: pd.DataFrame, output_path: Path) -> None:

    attacks = ATTACK_ORDER
    pcts = PCT_ORDER

    fig, axes = plt.subplots(len(attacks), len(pcts), figsize=(24, 20), constrained_layout=True)

    for i, attack in enumerate(attacks):
        for j, pct in enumerate(pcts):
            ax = axes[i, j]
            subset = df_runs[
                (df_runs["attack"] == attack) & (df_runs["byzantine_pct"] == pct)
            ]
            piv = _safe_pivot(
                subset, "strategy", "aggregation", "final_accuracy",
                STRATEGY_ORDER, AGGREGATION_ORDER,
            )

            if piv.isnull().all().all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(
                    f"{_ATTACK_LABELS.get(attack, attack)}\n{_PCT_LABELS.get(pct, pct)} byzantine",
                    fontsize=10,
                )
                continue

            best = float(np.nanmax(piv.values))
            drop_piv = piv - best

            display = drop_piv.rename(index=_STRATEGY_LABELS, columns=_AGG_LABELS)
            sns.heatmap(
                display, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1.0, vmax=0.0, cbar=False,
                linewidths=0.4, linecolor="white", annot_kws={"size": 9},
            )
            ax.set_title(
                f"{_ATTACK_LABELS.get(attack, attack)}\n{_PCT_LABELS.get(pct, pct)} byzantine",
                fontsize=10,
            )
            ax.set_ylabel("Strategy" if j == 0 else "", fontsize=9)
            ax.set_xlabel("Aggregation" if i == len(attacks) - 1 else "", fontsize=9)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.tick_params(axis="y", rotation=0, labelsize=8)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=-1, vmax=0))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.55, pad=0.02)
    cbar.set_label("Δ accuracy vs. best in panel", fontsize=12)

    fig.suptitle(
        "Accuracy drop vs. best configuration (per scenario)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def convergence_small_multiples(
    df_rounds: pd.DataFrame,
    output_path: Path,
    attack: str = "label_flip",
) -> None:

    round_col = "round_num" if "round_num" in df_rounds.columns else "round"
    subset = df_rounds[df_rounds["attack"] == attack] if "attack" in df_rounds.columns else df_rounds

    pct_colors = {0.1: "#1f77b4", 0.2: "#ff7f0e", 0.4: "#d62728"}

    fig, axes = plt.subplots(
        len(STRATEGY_ORDER), len(AGGREGATION_ORDER),
        figsize=(22, 16), sharey=True,
    )

    legend_handles, legend_labels = [], []

    for i, strategy in enumerate(STRATEGY_ORDER):
        for j, agg in enumerate(AGGREGATION_ORDER):
            ax = axes[i, j]
            panel = subset[
                (subset["strategy"] == strategy) & (subset["aggregation"] == agg)
            ]

            if panel.empty:
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10)
            else:
                for pct in PCT_ORDER:
                    pct_data = panel[panel["byzantine_pct"] == pct].sort_values(round_col)
                    if pct_data.empty:
                        continue
                    line, = ax.plot(
                        pct_data[round_col], pct_data["test_accuracy"],
                        color=pct_colors[pct], linewidth=1.5,
                    )
                    if i == 0 and j == 0:
                        legend_handles.append(line)
                        legend_labels.append(f"{_PCT_LABELS.get(pct, pct)} byzantine")

            ax.set_ylim(0, 1)

            if i == 0:
                ax.set_title(_AGG_LABELS.get(agg, agg), fontsize=9)
            if j == 0:
                ax.set_ylabel(_STRATEGY_LABELS.get(strategy, strategy), fontsize=8)
            else:
                ax.set_ylabel("")
            if i == len(STRATEGY_ORDER) - 1:
                ax.set_xlabel("Round", fontsize=8)
            ax.tick_params(labelsize=7)

    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="lower center", ncol=3, fontsize=10,
            bbox_to_anchor=(0.5, -0.01),
        )

    attack_label = _ATTACK_LABELS.get(attack, attack)
    fig.suptitle(f"Convergence curves (attack: {attack_label})", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def accuracy_vs_byzantine_pct(
    df_runs: pd.DataFrame,
    output_path: Path,
    aggregation: str = "fedavg",
) -> None:

    subset = df_runs[df_runs["aggregation"] == aggregation] if "aggregation" in df_runs.columns else df_runs

    fig, axes = plt.subplots(1, len(ATTACK_ORDER), figsize=(18, 5), sharey=True)
    palette = sns.color_palette("tab10", len(STRATEGY_ORDER))
    x_ticks = [int(p * 100) for p in sorted(PCT_ORDER)]

    for j, attack in enumerate(ATTACK_ORDER):
        ax = axes[j]
        attack_data = subset[subset["attack"] == attack] if "attack" in subset.columns else subset

        for k, strategy in enumerate(STRATEGY_ORDER):
            strat_data = attack_data[attack_data["strategy"] == strategy]
            if strat_data.empty:
                continue
            agg_data = (
                strat_data.groupby("byzantine_pct")["final_accuracy"]
                .mean()
                .reset_index()
                .sort_values("byzantine_pct")
            )
            pcts_int = [int(p * 100) for p in agg_data["byzantine_pct"]]
            ax.plot(
                pcts_int, agg_data["final_accuracy"],
                marker="o", color=palette[k],
                label=_STRATEGY_LABELS.get(strategy, strategy),
            )

        ax.set_title(_ATTACK_LABELS.get(attack, attack), fontsize=11)
        ax.set_xlabel("Byzantine share of pool [%]", fontsize=10)
        if j == 0:
            ax.set_ylabel("Final accuracy", fontsize=10)
        ax.set_xticks(x_ticks)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower left")

    agg_label = _AGG_LABELS.get(aggregation, aggregation)
    fig.suptitle(
        f"Impact of byzantine share on accuracy (aggregation: {agg_label})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def attack_comparison_bars(
    df_runs: pd.DataFrame,
    output_path: Path,
    strategy: str,
    aggregation: str,
) -> None:

    subset = df_runs[
        (df_runs["strategy"] == strategy) & (df_runs["aggregation"] == aggregation)
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    if subset.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        strat_l = _STRATEGY_LABELS.get(strategy, strategy)
        agg_l = _AGG_LABELS.get(aggregation, aggregation)
        fig.suptitle(f"No data: {strat_l} + {agg_l}")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    pct_colors = {0.1: "#1f77b4", 0.2: "#ff7f0e", 0.4: "#d62728"}
    n_attacks = len(ATTACK_ORDER)
    n_pcts = len(PCT_ORDER)
    bar_width = 0.8 / n_pcts
    x = np.arange(n_attacks)

    for k, pct in enumerate(PCT_ORDER):
        vals = []
        for attack in ATTACK_ORDER:
            v = subset[
                (subset["attack"] == attack) & (subset["byzantine_pct"] == pct)
            ]["final_accuracy"]
            vals.append(float(v.mean()) if not v.empty else 0.0)
        offset = (k - (n_pcts - 1) / 2) * bar_width
        ax.bar(
            x + offset, vals, bar_width,
            label=f"{_PCT_LABELS.get(pct, pct)} byzantine",
            color=pct_colors[pct], alpha=0.85,
        )

    attack_labels = [_ATTACK_LABELS.get(a, a) for a in ATTACK_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(attack_labels, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Final accuracy", fontsize=11)
    ax.set_ylim(0, 1)
    strat_l = _STRATEGY_LABELS.get(strategy, strategy)
    agg_l = _AGG_LABELS.get(aggregation, aggregation)
    ax.set_title(f"Attack comparison: {strat_l} + {agg_l}", fontsize=12)
    ax.legend(title="Byzantine share of pool", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def selection_time_boxplot(df_rounds: pd.DataFrame, output_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = [s for s in STRATEGY_ORDER if s in df_rounds.get("strategy", pd.Series()).values]
    if not strategies:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    values = [
        df_rounds[df_rounds["strategy"] == s]["selection_time_ms"].dropna().values
        for s in strategies
    ]
    labels = [_STRATEGY_LABELS.get(s, s) for s in strategies]

    bp = ax.boxplot(values, tick_labels=labels, patch_artist=True,
                    medianprops={"color": "black", "linewidth": 2})
    palette = sns.color_palette("tab10", len(strategies))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Selection time [ms]", fontsize=11)
    ax.set_xlabel("Selection strategy", fontsize=11)
    ax.set_title("Computational cost of client selection per round", fontsize=12)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def memory_cost_bars(df_rounds: pd.DataFrame, output_path: Path) -> None:

    fig, ax = plt.subplots(figsize=(11, 6))

    strategies = [s for s in STRATEGY_ORDER if s in df_rounds.get("strategy", pd.Series()).values]
    if not strategies:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    round_col = "round_num" if "round_num" in df_rounds.columns else "round"
    state_vals, peak_vals = [], []
    for s in strategies:
        sdata = df_rounds[df_rounds["strategy"] == s]
        last = sdata.nlargest(min(10, len(sdata)), round_col) if not sdata.empty else sdata
        state_vals.append(max(1.0, float(last["selector_state_bytes"].mean())))
        peak_vals.append(max(1.0, float(sdata["selection_peak_mem_bytes"].mean())))

    x = np.arange(len(strategies))
    width = 0.35
    labels = [_STRATEGY_LABELS.get(s, s) for s in strategies]

    bars1 = ax.bar(x - width / 2, state_vals, width,
                   label="Persistent selector state", color="#5B9BD5", alpha=0.85)
    bars2 = ax.bar(x + width / 2, peak_vals, width,
                   label="Peak working memory", color="#ED7D31", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Bytes (log scale)", fontsize=11)
    ax.set_title("Memory cost of selection strategies", fontsize=12)
    ax.legend(fontsize=10)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            label = f"{h/1024:.0f} KB" if h < 1e6 else f"{h/1e6:.1f} MB"
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.2,
                    label, ha="center", va="bottom", fontsize=7)


    ax.set_ylim(top=max(state_vals + peak_vals) * 8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def tradeoff_scatter(
    df_runs: pd.DataFrame,
    df_rounds: pd.DataFrame,
    output_path: Path,
) -> None:

    acc_by_s = df_runs.groupby("strategy")["final_accuracy"].mean()
    time_by_s = df_rounds.groupby("strategy")["selection_time_ms"].mean()
    mem_by_s = df_rounds.groupby("strategy")["selector_state_bytes"].mean()

    strategies = [s for s in STRATEGY_ORDER if s in acc_by_s.index]
    if not strategies:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    palette = sns.color_palette("tab10", len(strategies))

    def _pareto(costs: list[float], accs: list[float]) -> list[int]:
        pareto = []
        for i in range(len(costs)):
            dominated = any(
                j != i and costs[j] <= costs[i] and accs[j] >= accs[i]
                and (costs[j] < costs[i] or accs[j] > accs[i])
                for j in range(len(costs))
            )
            if not dominated:
                pareto.append(i)
        return pareto

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        (time_by_s, "Mean selection time [ms]", 1.0),
        (mem_by_s, "Mean selector state [KB]", 1 / 1024),
    ]
    for ax, (cost_series, xlabel, scale) in zip(axes, panels):
        costs = [float(cost_series.get(s, 0)) * scale for s in strategies]
        accs = [float(acc_by_s.get(s, 0)) for s in strategies]
        pareto_idx = set(_pareto(costs, accs))

        for k, (s, cost, acc) in enumerate(zip(strategies, costs, accs)):
            is_pareto = k in pareto_idx
            ax.scatter(cost, acc,
                       color=palette[k],
                       marker="*" if is_pareto else "o",
                       s=250 if is_pareto else 100,
                       zorder=5, label=_STRATEGY_LABELS.get(s, s))
            ax.annotate(_STRATEGY_LABELS.get(s, s), (cost, acc),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)

        ax.margins(x=0.2, y=0.15)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Mean final accuracy", fontsize=11)
        ax.set_title("Cost vs. quality trade-off\n(★ = Pareto-optimal)", fontsize=11)

    fig.suptitle("Trade-off analysis: strategy cost vs. accuracy", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cohort_composition_vs_robustness(
    df_rounds: pd.DataFrame,
    output_path: Path,
) -> None:

    required = {"aggregation", "n_byzantine_in_cohort", "test_accuracy", "byzantine_pct"}
    if not required.issubset(df_rounds.columns):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Missing required columns", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    subset = df_rounds[
        df_rounds["aggregation"].isin(["krum", "bulyan"]) &
        (df_rounds["byzantine_pct"] == 0.4)
    ].copy()

    if subset.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Brak danych (krum/bulyan, pct=40%)", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    colors = {"krum": "#2196F3", "bulyan": "#FF5722"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, agg in enumerate(["krum", "bulyan"]):
        ax = axes[ax_idx]
        agg_data = subset[subset["aggregation"] == agg].dropna(
            subset=["n_byzantine_in_cohort", "test_accuracy"]
        )

        if agg_data.empty:
            ax.text(0.5, 0.5, f"No data ({agg})", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(agg.upper())
            continue

        x = agg_data["n_byzantine_in_cohort"].values.astype(float)
        y = agg_data["test_accuracy"].values.astype(float)

        ax.scatter(x, y, color=colors[agg], alpha=0.35, s=15, label=agg.upper())

        if len(np.unique(x)) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color=colors[agg], linewidth=2,
                    linestyle="--", label="Linear trend")

        rho, pval = stats.spearmanr(x, y)
        ax.set_xlabel("Byzantine clients in cohort (K=10)", fontsize=11)
        if ax_idx == 0:
            ax.set_ylabel("Test accuracy", fontsize=11)
        ax.set_title(f"{agg.upper()}\nSpearman ρ = {rho:.3f}  (p = {pval:.3f})", fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Cohort composition vs. aggregation robustness (40% byzantine in pool, N=50)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summary_table_csv(
    df_runs: pd.DataFrame,
    df_rounds: pd.DataFrame,
    output_dir: Path,
) -> None:

    if df_runs.empty:
        return

    group_keys = ["strategy", "aggregation", "attack", "byzantine_pct"]
    available_keys = [k for k in group_keys if k in df_rounds.columns]

    if not df_rounds.empty and available_keys == group_keys:
        cost_agg = df_rounds.groupby(group_keys).agg(
            mean_selection_time_ms=("selection_time_ms", "mean"),
            mean_selector_state_bytes=("selector_state_bytes", "mean"),
            mean_peak_mem_bytes=("selection_peak_mem_bytes", "mean"),
        ).reset_index()
        df_out = df_runs.merge(cost_agg, on=group_keys, how="left")
    else:
        df_out = df_runs.copy()

    df_out = df_out.sort_values(
        ["attack", "byzantine_pct", "strategy", "aggregation"]
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_dir / "summary_table.csv", index=False)

    cols = group_keys + ["final_accuracy"]
    cols += [c for c in ["mean_selection_time_ms", "mean_selector_state_bytes"] if c in df_out.columns]
    latex_df = df_out[cols].copy()

    latex_df.columns = [c.replace("_", r"\_") for c in latex_df.columns]
    (output_dir / "summary_table.tex").write_text(
        latex_df.to_latex(index=False, float_format="%.4f")
    )
