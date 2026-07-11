import argparse

from fl_lab.fl.simulation import run_simulation
from fl_lab.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single FL experiment")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    collector = run_simulation(cfg)
    print(f"Done. Final accuracy: {collector.final_accuracy():.4f}")


if __name__ == "__main__":
    main()
