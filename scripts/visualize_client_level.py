import argparse
from pathlib import Path

from fl_lab.analysis.constants import ATTACK_ORDER
from fl_lab.analysis.loader import load_phase
from fl_lab.analysis.client_level import (
    build_client_df,
    byzantine_avoidance_summary_bars,
    byzantine_fraction_in_cohort_over_rounds,
    gradient_norm_distributions,
    mab_beta_evolution,
    reputation_evolution,
    selection_count_distribution,
    selection_fairness_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Wykresy per-klient dla full_cross")
    parser.add_argument("--results_root", default="results/full_cross")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output) if args.output else results_root / "figures_client"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Wczytywanie wyników z: {results_root}")
    runs = load_phase(results_root)
    print(f"Wczytano {len(runs)} runów")
    if not runs:
        print("Brak danych. Kończę.")
        return

    print("Budowanie df_clients ...")
    df_clients = build_client_df(runs)
    if df_clients.empty:
        print("Brak plików client_metrics.csv. Kończę.")
        return
    print(f"df_clients: {df_clients.shape}")

    for attack in ATTACK_ORDER:
        for pct in [0.2, 0.4]:
            fname = f"byzantine_fraction_{attack}_pct{int(pct*100)}.png"
            _gen(fname, output_dir,
                 lambda p, a=attack, q=pct: byzantine_fraction_in_cohort_over_rounds(
                     df_clients, p, attack=a, pct=q, aggregation="fedavg"
                 ))

    _gen("byzantine_avoidance_summary.png", output_dir,
         lambda p: byzantine_avoidance_summary_bars(df_clients, p))

    for pct in [0.2, 0.4]:
        _gen(f"reputation_evolution_pct{int(pct*100)}.png", output_dir,
             lambda p, q=pct: reputation_evolution(df_clients, p, pct=q))

        _gen(f"mab_beta_evolution_pct{int(pct*100)}.png", output_dir,
             lambda p, q=pct: mab_beta_evolution(df_clients, p, pct=q))

    _gen("gradient_norm_distributions.png", output_dir,
         lambda p: gradient_norm_distributions(df_clients, p))

    for attack in ATTACK_ORDER:
        fname = f"selection_count_distribution_{attack}.png"
        _gen(fname, output_dir,
             lambda p, a=attack: selection_count_distribution(
                 df_clients, p, attack=a, pct=0.2, aggregation="fedavg"
             ))

    _gen("selection_fairness_summary.png", output_dir,
         lambda p: selection_fairness_summary(df_clients, p))

    print(f"\nWszystkie pliki zapisane do: {output_dir}")
    for f in sorted(output_dir.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name:<55} {size_kb:>5} KB")


def _gen(filename: str, output_dir: Path, fn) -> None:
    print(f"  {filename} ...")
    try:
        fn(output_dir / filename)
    except Exception as exc:
        print(f"    BŁĄD: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
