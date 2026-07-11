import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from scipy.stats import f_oneway, mannwhitneyu, wilcoxon


def rounds_to_target(round_list: list[dict], target: float = 0.5) -> int | None:
    for row in round_list:
        if row["test_accuracy"] >= target:
            return int(row["round"])
    return None


def rank_strategies(
    grouped: dict[str, list[float]],
) -> list[tuple[str, float, float]]:
    result = [
        (s, float(np.mean(accs)), float(np.std(accs)))
        for s, accs in grouped.items()
    ]
    return sorted(result, key=lambda x: x[1], reverse=True)


def pairwise_ttest(
    grouped: dict[str, list[float]],
) -> dict[tuple[str, str], dict]:

    strategies = [s for s, accs in grouped.items() if len(accs) >= 2]
    results: dict[tuple[str, str], dict] = {}
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i + 1:]:
            stat, p = stats.ttest_ind(grouped[s1], grouped[s2], equal_var=False)
            results[(s1, s2)] = {"statistic": float(stat), "p_value": float(p)}
    return results


def one_way_anova(grouped: dict[str, list[dict]]) -> dict:
    groups = [
        [r["summary"]["final_accuracy"] for r in runs]
        for runs in grouped.values()
        if len(runs) >= 2
    ]
    if len(groups) < 2:
        return {"f_statistic": float("nan"), "p_value": float("nan")}
    f_stat, p_val = f_oneway(*groups)
    return {"f_statistic": float(f_stat), "p_value": float(p_val)}


def pairwise_wilcoxon_signed_rank(
    accuracy_by_strategy: dict[str, list[float]],
) -> pd.DataFrame:

    rows = []
    strategies = list(accuracy_by_strategy.keys())
    for a, b in combinations(strategies, 2):
        x = accuracy_by_strategy[a]
        y = accuracy_by_strategy[b]
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            continue
        try:
            stat, p = wilcoxon(x, y, alternative="two-sided")
            rows.append({"strategy_a": a, "strategy_b": b, "W": stat, "p_value": round(p, 4)})
        except ValueError:
            # wilcoxon fails gdy wszystkie różnice są zerowe
            rows.append({"strategy_a": a, "strategy_b": b, "W": 0.0, "p_value": 1.0})
    return pd.DataFrame(rows)


def wilcoxon_pairwise(grouped: dict[str, list[dict]]) -> dict:

    results = {}
    names = [k for k, v in grouped.items() if len(v) >= 1]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            accs_a = [r["summary"]["final_accuracy"] for r in grouped[a]]
            accs_b = [r["summary"]["final_accuracy"] for r in grouped[b]]
            try:
                stat, p = mannwhitneyu(accs_a, accs_b, alternative="two-sided")
            except ValueError:
                stat, p = float("nan"), float("nan")
            results[(a, b)] = {"statistic": float(stat), "p_value": float(p)}
    return results
