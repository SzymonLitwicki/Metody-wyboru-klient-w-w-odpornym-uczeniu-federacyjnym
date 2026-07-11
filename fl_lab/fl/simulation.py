import csv
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flwr.common import Context
from flwr.server import ServerConfig
from flwr.simulation import start_simulation

from fl_lab.aggregations import get_aggregator
from fl_lab.attacks import get_attack
from fl_lab.data.cifar10 import load_cifar10
from fl_lab.data.distribution import (
    dirichlet_partition,
    iid_partition,
    pathological_partition,
)
from fl_lab.fl.client import FlClient
from fl_lab.fl.server import ComposedFlowerStrategy
from fl_lab.fl.weights import get_parameters
from fl_lab.metrics import MetricsCollector
from fl_lab.models import get_model
from fl_lab.strategies import get_selector
from fl_lab.strategies.base import SelectionHistory
from fl_lab.utils.config import ExperimentConfig
from fl_lab.utils.seed import derive_seed, set_all_seeds

_PARTITION_ID_KEY = "partition-id"
_SERVER_ROOT_FRACTION = 0.02

_ROUND_CSV_HEADER = [
    "round_num", "test_accuracy", "test_loss", "train_loss",
    "selected_clients", "n_byzantine_in_cohort", "byzantine_in_cohort_ids",
    "selection_time_ms", "avg_train_time_ms", "bytes_per_round",
    "selector_state_bytes", "selection_peak_mem_bytes",
]

_CLIENT_CSV_HEADER = [
    "round_num", "client_id", "is_malicious", "was_selected",
    "local_loss", "gradient_norm", "reputation_score", "beta_params",
]


def _lib_versions() -> dict:
    libs = ["torch", "flwr", "numpy", "pydantic"]
    versions = {}
    for lib in libs:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except Exception:
            versions[lib] = "unknown"
    return versions


def run_simulation(cfg: ExperimentConfig) -> MetricsCollector:
    set_all_seeds(cfg.seed)

    output_dir = Path(cfg.logging.output_dir) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds = load_cifar10(data_dir="./data/cifar10")

    root_rng = np.random.default_rng(derive_seed(cfg.seed, "server_root"))
    n_root = max(32, int(len(train_ds) * _SERVER_ROOT_FRACTION))
    all_train_idx = np.arange(len(train_ds))
    root_indices = root_rng.choice(all_train_idx, size=n_root, replace=False)
    client_indices = np.setdiff1d(all_train_idx, root_indices)

    _pin = torch.cuda.is_available()
    server_root_loader = DataLoader(
        Subset(train_ds, root_indices.tolist()),
        batch_size=cfg.fl.batch_size,
        shuffle=False,
        pin_memory=_pin,
    )

    client_ds = Subset(train_ds, client_indices.tolist())
    data_seed = derive_seed(cfg.seed, "data")
    if cfg.data.distribution == "iid":
        partitions = iid_partition(client_ds, cfg.fl.num_clients, seed=data_seed)
    elif cfg.data.distribution == "dirichlet":
        partitions = dirichlet_partition(
            client_ds, cfg.fl.num_clients, cfg.data.alpha, seed=data_seed
        )
    else:
        partitions = pathological_partition(
            client_ds, cfg.fl.num_clients, cfg.data.num_classes_per_client, seed=data_seed
        )

    byz_rng = np.random.default_rng(derive_seed(cfg.seed, "byz"))
    n_byzantine = int(cfg.fl.num_clients * cfg.attack.byzantine_pct)
    malicious_ids: set[int] = set(
        int(i)
        for i in byz_rng.choice(cfg.fl.num_clients, size=n_byzantine, replace=False)
    )
    attack = get_attack(cfg.attack.type)

    history = SelectionHistory()

    torch.manual_seed(derive_seed(cfg.seed, "init"))
    torch.cuda.manual_seed_all(derive_seed(cfg.seed, "init"))
    global_model = get_model(cfg.model.name)
    if torch.cuda.is_available():
        global_model = torch.compile(global_model)

    selector = get_selector(cfg.selection.strategy)
    aggregator = get_aggregator(cfg.aggregation.method)
    collector = MetricsCollector()
    test_loader = DataLoader(test_ds, batch_size=cfg.fl.batch_size, shuffle=False, pin_memory=_pin)

    round_csv_path = output_dir / "round_metrics.csv"
    client_csv_path = output_dir / "client_metrics.csv"

    round_csvfile = open(round_csv_path, "w", newline="", buffering=1)
    client_csvfile = open(client_csv_path, "w", newline="", buffering=1)
    round_writer = csv.writer(round_csvfile)
    client_writer = csv.writer(client_csvfile)
    round_writer.writerow(_ROUND_CSV_HEADER)
    client_writer.writerow(_CLIENT_CSV_HEADER)

    strategy = ComposedFlowerStrategy(
        cfg=cfg,
        model=global_model,
        selector=selector,
        aggregator=aggregator,
        history=history,
        collector=collector,
        test_loader=test_loader,
        server_root_loader=server_root_loader,
        malicious_ids=malicious_ids,
        output_dir=output_dir,
        round_csv_writer=round_writer,
        client_csv_writer=client_writer,
    )

    def client_fn(context: Context):
        cid_int = int(context.node_config[_PARTITION_ID_KEY])
        model = get_model(cfg.model.name)
        return FlClient(
            cid=cid_int,
            model=model,
            dataset=partitions[cid_int],
            cfg=cfg,
            attack=attack if cid_int in malicious_ids else None,
        ).to_client()

    try:
        start_simulation(
            client_fn=client_fn,
            num_clients=cfg.fl.num_clients,
            config=ServerConfig(num_rounds=cfg.fl.rounds),
            strategy=strategy,
            client_resources={"num_cpus": 2, "num_gpus": 0.2 if torch.cuda.is_available() else 0},
            ray_init_args={"num_cpus": os.cpu_count() or 4, "include_dashboard": False},
        )
    finally:
        round_csvfile.close()
        client_csvfile.close()

    cfg_dict = json.loads(cfg.model_dump_json())
    cfg_dict["_lib_versions"] = _lib_versions()
    cfg_dict["_malicious_ids"] = sorted(malicious_ids)
    (output_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    summary = {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "selection_strategy": cfg.selection.strategy,
        "aggregation_method": cfg.aggregation.method,
        "attack_type": cfg.attack.type,
        "byzantine_pct": cfg.attack.byzantine_pct,
        "n_byzantine_total": n_byzantine,
        "distribution": cfg.data.distribution,
        "alpha": cfg.data.alpha,
        "model": cfg.model.name,
        "final_accuracy": collector.final_accuracy(),
        "total_rounds": cfg.fl.rounds,
        "rounds_completed": len(collector.history),
        "simulation_completed": len(collector.history) == cfg.fl.rounds,
        "total_selection_time_ms": collector.total_selection_time_ms(),
    }
    (output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2))

    return collector
