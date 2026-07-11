from __future__ import annotations

import ast
import re
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
_PCT_LABELS = {0.1: "10%", 0.2: "20%", 0.4: "40%"}

_COLOR_BENIGN = "#1f77b4"
_COLOR_MAL = "#d62728"



def _parse_beta(s) -> tuple[float, float]:
    """Parsuj '[B, M]' → (B, M). Zwraca (nan, nan) przy niepowodzeniu."""
    if pd.isna(s) or str(s).strip() == "":
        return np.nan, np.nan
    try:
        vals = ast.literal_eval(str(s))
        if len(vals) >= 2:
            return float(vals[0]), float(vals[1])
    except Exception:
        pass
    nums = re.findall(r"[\d.]+", str(s))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return np.nan, np.nan



def build_client_df(runs: list[dict]) -> pd.DataFrame:
    dfs: list[pd.DataFrame] = []
    for run in runs:
        path = run.get("client_metrics_path")
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        s = run["summary"]
        df["strategy"] = s.get("selection_strategy")
        df["aggregation"] = s.get("aggregation_method")
        df["attack"] = s.get("attack_type")
        df["byzantine_pct"] = float(s.get("byzantine_pct", 0))
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result["is_malicious"] = result["is_malicious"].astype(bool)
    result["was_selected"] = result["was_selected"].astype(bool)

    if "beta_params" in result.columns:
        parsed = result["beta_params"].map(_parse_beta)
        result["beta_b"] = [v[0] for v in parsed]
        result["beta_m"] = [v[1] for v in parsed]

    return result


def _empty_panel(ax, msg: str = "No data") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=10, color="gray")
    ax.set_xticks([])
    ax.set_yticks([])


def _gini(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=float))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() - (n + 1) * arr.sum()) / (n * arr.sum()))


def byzantine_fraction_in_cohort_over_rounds(
    df_clients: pd.DataFrame,
    output_path: Path,
    attack: str = "label_flip",
    pct: float = 0.4,
    aggregation: str = "fedavg",
) -> None:

    subset = df_clients[
        (df_clients["attack"] == attack) &
        (df_clients["byzantine_pct"] == pct) &
        (df_clients["aggregation"] == aggregation)
    ]

    fig, ax = plt.subplots(figsize=(11, 5))

    if subset.empty:
        _empty_panel(ax, f"No data ({attack}, pct={pct}, agg={aggregation})")
    else:
        palette = sns.color_palette("tab10", len(STRATEGY_ORDER))
        for k, strategy in enumerate(STRATEGY_ORDER):
            strat = subset[subset["strategy"] == strategy]
            if strat.empty:
                continue
            per_round = (
                strat.groupby("round_num")["is_malicious"]
                .mean()
                .reset_index(name="byz_frac")
            )
            ax.plot(per_round["round_num"], per_round["byz_frac"],
                    color=palette[k], linewidth=2,
                    label=_STRATEGY_LABELS.get(strategy, strategy))

        ax.axhline(pct, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Nominal pool share ({int(pct*100)}%)")
        ax.set_xlabel("Round", fontsize=11)
        ax.set_ylabel("Byzantine share in cohort", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="upper right")

    attack_l = _ATTACK_LABELS.get(attack, attack)
    ax.set_title(
        f"Byzantine share in cohort: {attack_l}, {int(pct*100)}% byzantine in pool",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def byzantine_avoidance_summary_bars(
    df_clients: pd.DataFrame,
    output_path: Path,
    late_rounds_start: int = 31,
) -> None:
  
    late = df_clients[df_clients["round_num"] >= late_rounds_start]

    fig, axes = plt.subplots(1, len(PCT_ORDER), figsize=(19, 5), sharey=False)
    palette = sns.color_palette("tab10", len(STRATEGY_ORDER))
    n_attacks = len(ATTACK_ORDER)
    n_strats = len(STRATEGY_ORDER)
    bar_width = 0.8 / n_strats

    for j, pct in enumerate(PCT_ORDER):
        ax = axes[j]
        pct_data = late[late["byzantine_pct"] == pct]

        if pct_data.empty:
            _empty_panel(ax, f"No data (pct={pct})")
            ax.set_title(_PCT_LABELS.get(pct, pct))
            continue

        grp = (
            pct_data.groupby(["strategy", "attack"])["is_malicious"]
            .mean()
            .reset_index(name="byz_frac")
        )

        x = np.arange(n_attacks)
        for k, strategy in enumerate(STRATEGY_ORDER):
            vals = []
            for attack in ATTACK_ORDER:
                v = grp[
                    (grp["strategy"] == strategy) & (grp["attack"] == attack)
                ]["byz_frac"]
                vals.append(float(v.iloc[0]) if not v.empty else 0.0)
            offset = (k - (n_strats - 1) / 2) * bar_width
            ax.bar(x + offset, vals, bar_width, alpha=0.85, color=palette[k],
                   label=_STRATEGY_LABELS.get(strategy, strategy))

        ax.axhline(pct, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Nominal {int(pct*100)}%")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_ATTACK_LABELS.get(a, a) for a in ATTACK_ORDER],
            rotation=15, ha="right", fontsize=9,
        )
        ax.set_title(f"Byzantine pool: {_PCT_LABELS.get(pct, pct)}", fontsize=11)
        if j == 0:
            ax.set_ylabel(f"Mean byzantine share in cohort (rounds {late_rounds_start}–60)",
                          fontsize=10)
        if j == len(PCT_ORDER) - 1:
            ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Byzantine avoidance effectiveness in late training", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def reputation_evolution(
    df_clients: pd.DataFrame,
    output_path: Path,
    pct: float = 0.2,
) -> None:

    rep_data = df_clients[
        (df_clients["strategy"] == "reputation") &
        (df_clients["byzantine_pct"] == pct) &
        (df_clients["reputation_score"].notna())
    ]

    fig, axes = plt.subplots(1, len(ATTACK_ORDER), figsize=(18, 5), sharey=True)

    for j, attack in enumerate(ATTACK_ORDER):
        ax = axes[j]
        atk = rep_data[rep_data["attack"] == attack]

        if atk.empty or atk["reputation_score"].isna().all():
            _empty_panel(ax, "No reputation data")
            ax.set_title(_ATTACK_LABELS.get(attack, attack))
            continue

        for is_mal, color, label in [
            (False, _COLOR_BENIGN, "Honest"),
            (True, _COLOR_MAL, "Byzantine"),
        ]:
            group = atk[atk["is_malicious"] == is_mal]
            if group.empty:
                continue


            for cid in group["client_id"].unique():
                cdata = group[group["client_id"] == cid].sort_values("round_num")
                ax.plot(cdata["round_num"], cdata["reputation_score"],
                        color=color, alpha=0.08, linewidth=0.8)

            mean_r = (
                group.groupby("round_num")["reputation_score"]
                .mean()
                .reset_index()
            )
            ax.plot(mean_r["round_num"], mean_r["reputation_score"],
                    color=color, linewidth=2.5, label=label)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Round", fontsize=10)
        if j == 0:
            ax.set_ylabel("Reputation score", fontsize=10)
        ax.set_title(_ATTACK_LABELS.get(attack, attack), fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Reputation evolution of selected clients ({int(pct*100)}% byzantine)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def mab_beta_evolution(
    df_clients: pd.DataFrame,
    output_path: Path,
    pct: float = 0.2,
) -> None:

    mab_data = df_clients[
        (df_clients["strategy"] == "adaptive_mab") &
        (df_clients["byzantine_pct"] == pct) &
        (df_clients["beta_b"].notna())
    ].copy()

    if mab_data.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        _empty_panel(ax, "No adaptive_mab data with beta_params")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    mab_data["beta_ev"] = mab_data["beta_b"] / (mab_data["beta_b"] + mab_data["beta_m"]).replace(0, np.nan)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, len(ATTACK_ORDER), hspace=0.4, wspace=0.3)


    for j, attack in enumerate(ATTACK_ORDER):
        ax = fig.add_subplot(gs[0, j])
        atk = mab_data[mab_data["attack"] == attack]

        if atk.empty:
            _empty_panel(ax, "No data")
            ax.set_title(_ATTACK_LABELS.get(attack, attack))
            continue

        for is_mal, color, label in [
            (False, _COLOR_BENIGN, "Honest"),
            (True, _COLOR_MAL, "Byzantine"),
        ]:
            grp = atk[atk["is_malicious"] == is_mal]
            if grp.empty:
                continue
            mean_r = grp.groupby("round_num")["beta_ev"].mean().reset_index()
            ax.plot(mean_r["round_num"], mean_r["beta_ev"],
                    color=color, linewidth=2.5, label=label)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Round", fontsize=9)
        if j == 0:
            ax.set_ylabel("E[Beta] = B/(B+M)", fontsize=9)
        ax.set_title(_ATTACK_LABELS.get(attack, attack), fontsize=10)
        ax.legend(fontsize=8)


    for j, attack in enumerate(ATTACK_ORDER):
        ax = fig.add_subplot(gs[1, j])
        atk = mab_data[mab_data["attack"] == attack]
        if atk.empty:
            _empty_panel(ax)
            continue

        last_rnd = int(atk["round_num"].max())
        last = atk[atk["round_num"] == last_rnd]

        for is_mal, color, label in [
            (False, _COLOR_BENIGN, "Honest"),
            (True, _COLOR_MAL, "Byzantine"),
        ]:
            grp = last[last["is_malicious"] == is_mal]["beta_m"].dropna()
            if grp.empty:
                continue
            ax.hist(grp, bins=10, color=color, alpha=0.6, label=label, density=True)

        ax.set_xlabel("M_k (penalties) — final round", fontsize=9)
        if j == 0:
            ax.set_ylabel("Density", fontsize=9)
        ax.set_title(_ATTACK_LABELS.get(attack, attack), fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Beta-MAB: E[Beta] evolution and penalty distribution ({int(pct*100)}% byzantine)",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def gradient_norm_distributions(
    df_clients: pd.DataFrame,
    output_path: Path,
) -> None:

    sel = df_clients[df_clients["gradient_norm"].notna()]

    fig, axes = plt.subplots(1, len(ATTACK_ORDER), figsize=(16, 5), sharey=False)

    for j, attack in enumerate(ATTACK_ORDER):
        ax = axes[j]
        atk = sel[sel["attack"] == attack]

        if atk.empty:
            _empty_panel(ax)
            ax.set_title(_ATTACK_LABELS.get(attack, attack))
            continue

        norms_hon = atk[~atk["is_malicious"]]["gradient_norm"].values
        norms_mal = atk[atk["is_malicious"]]["gradient_norm"].values

        if len(norms_hon) == 0 or len(norms_mal) == 0:
            _empty_panel(ax, "One group missing")
            ax.set_title(_ATTACK_LABELS.get(attack, attack))
            continue

        p99 = np.percentile(np.concatenate([norms_hon, norms_mal]), 99)
        norms_hon_c = np.clip(norms_hon, 0, p99)
        norms_mal_c = np.clip(norms_mal, 0, p99)

        vdata = pd.DataFrame({
            "gradient_norm": np.concatenate([norms_hon_c, norms_mal_c]),
            "Type": (
                ["Honest"] * len(norms_hon_c) +
                ["Byzantine"] * len(norms_mal_c)
            ),
        })
        sns.violinplot(data=vdata, x="Type", y="gradient_norm",
                       hue="Type",
                       palette={"Honest": _COLOR_BENIGN, "Byzantine": _COLOR_MAL},
                       ax=ax, inner="box", cut=0, legend=False)


        try:
            stat, _ = stats.mannwhitneyu(norms_mal, norms_hon, alternative="two-sided")
            auc = stat / (len(norms_mal) * len(norms_hon))
            sep = max(auc, 1 - auc)
            auc_str = f"AUC = {sep:.3f}"
        except Exception:
            auc_str = ""

        ax.set_title(
            f"{_ATTACK_LABELS.get(attack, attack)}\n{auc_str}",
            fontsize=11,
        )
        ax.set_xlabel("")
        if j == 0:
            ax.set_ylabel("‖∇‖ (clipped at p99)", fontsize=10)

    fig.suptitle("Gradient norm distributions: byzantine vs. honest (pooled, all configurations)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)



def selection_count_distribution(
    df_clients: pd.DataFrame,
    output_path: Path,
    attack: str = "label_flip",
    pct: float = 0.2,
    aggregation: str = "fedavg",
) -> None:

    subset = df_clients[
        (df_clients["attack"] == attack) &
        (df_clients["byzantine_pct"] == pct) &
        (df_clients["aggregation"] == aggregation)
    ]

    fig, axes = plt.subplots(1, len(STRATEGY_ORDER), figsize=(20, 5), sharey=True)

    n_rounds = int(df_clients["round_num"].max()) if not df_clients.empty else 60
    K = 10
    N = 50
    expected = K * n_rounds / N

    for k, strategy in enumerate(STRATEGY_ORDER):
        ax = axes[k]
        strat = subset[subset["strategy"] == strategy]

        if strat.empty:
            _empty_panel(ax, f"No data\n{strategy}")
            ax.set_title(_STRATEGY_LABELS.get(strategy, strategy))
            continue

        counts = strat.groupby(["client_id", "is_malicious"]).size().reset_index(name="sel_count")
        counts = counts.sort_values("client_id")

        colors = [_COLOR_MAL if m else _COLOR_BENIGN for m in counts["is_malicious"]]
        ax.bar(counts["client_id"], counts["sel_count"], color=colors, width=0.7, alpha=0.8)
        ax.axhline(expected, color="gray", linestyle="--", linewidth=1.5,
                   label=f"Expected ({expected:.1f})")


        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color=_COLOR_BENIGN, label="Honest"),
            Patch(color=_COLOR_MAL, label="Byzantine"),
        ], fontsize=8, loc="upper right")

        ax.set_title(_STRATEGY_LABELS.get(strategy, strategy), fontsize=11)
        ax.set_xlabel("Client ID", fontsize=9)
        if k == 0:
            ax.set_ylabel(f"Selection count ({n_rounds} rounds)", fontsize=10)

    attack_l = _ATTACK_LABELS.get(attack, attack)
    fig.suptitle(
        f"Selection coverage: {attack_l}, {int(pct*100)}% byzantine in pool",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def selection_fairness_summary(
    df_clients: pd.DataFrame,
    output_path: Path,
) -> None:

    strategies = [s for s in STRATEGY_ORDER if s in df_clients.get("strategy", pd.Series()).values]

    if not strategies:
        fig, ax = plt.subplots(figsize=(8, 5))
        _empty_panel(ax)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    gini_all, gini_honest = [], []
    for strategy in strategies:
        strat = df_clients[df_clients["strategy"] == strategy]
        counts_all = strat.groupby("client_id").size().values
        counts_hon = (
            strat[~strat["is_malicious"]].groupby("client_id").size().values
        )
        gini_all.append(_gini(counts_all))
        gini_honest.append(_gini(counts_hon))

    x = np.arange(len(strategies))
    width = 0.35
    labels = [_STRATEGY_LABELS.get(s, s) for s in strategies]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, gini_all, width, label="All clients",
           color="#5B9BD5", alpha=0.85)
    ax.bar(x + width / 2, gini_honest, width, label="Honest only",
           color="#70AD47", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=11)
    ax.set_ylabel("Gini coefficient", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Selection coverage inequality per strategy\n"
                 "(Gini=0: uniform access; Gini→1: concentration on few clients)",
                 fontsize=11)
    ax.legend(fontsize=10)

    for i, (g_all, g_hon) in enumerate(zip(gini_all, gini_honest)):
        ax.text(i - width / 2, g_all + 0.01, f"{g_all:.2f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, g_hon + 0.01, f"{g_hon:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
