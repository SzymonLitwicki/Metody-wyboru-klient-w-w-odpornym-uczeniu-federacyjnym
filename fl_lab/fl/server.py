import csv
import json
import tracemalloc
from pathlib import Path
from typing import IO, Optional, Union
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from fl_lab.aggregations.base import Aggregator, ClientUpdate, ServerState
from fl_lab.fl.weights import flat_to_params, get_parameters, params_to_flat, set_parameters
from fl_lab.metrics import MetricsCollector, RoundMetrics
from fl_lab.strategies.base import ClientSelector, SelectionHistory
from fl_lab.utils.config import ExperimentConfig
from fl_lab.utils.seed import derive_seed

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _selector_state_bytes(selector: ClientSelector) -> int:
    try:
        from pympler import asizeof
        return int(asizeof.asizeof(selector))
    except Exception:
        return 0


class ComposedFlowerStrategy(Strategy):
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: nn.Module,
        selector: ClientSelector,
        aggregator: Aggregator,
        history: SelectionHistory,
        collector: MetricsCollector,
        test_loader: DataLoader,
        server_root_loader: DataLoader | None = None,
        malicious_ids: set[int] | None = None,
        output_dir: Path | None = None,
        round_csv_writer: "csv.writer | None" = None,
        client_csv_writer: "csv.writer | None" = None,
    ) -> None:
        self._cfg = cfg
        self._model = model.to(_DEVICE)
        self._selector = selector
        self._aggregator = aggregator
        self._history = history
        self._collector = collector
        self._test_loader = test_loader
        self._server_root_loader = server_root_loader
        self._malicious_ids: set[int] = malicious_ids or set()
        self._output_dir = output_dir
        self._round_csv_writer = round_csv_writer
        self._client_csv_writer = client_csv_writer

        self._global_params: Parameters | None = None
        self._last_selected: list[int] = []
        self._last_train_loss: float = 0.0
        self._last_selection_time_ms: float = 0.0
        self._last_avg_train_time_ms: float = 0.0
        self._last_selection_peak_mem_bytes: int = 0
        self._last_client_metrics: list[dict] = []
        self._model_num_params: int = sum(p.numel() for p in model.parameters())
        self._cid_to_partition: dict[str, int] = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        self._global_params = ndarrays_to_parameters(get_parameters(self._model))
        return self._global_params

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        client_manager.wait_for(self._cfg.fl.num_clients, timeout=86400)
        all_proxies = client_manager.all()

        partition_to_proxy: dict[int, ClientProxy] = {}
        for proxy in all_proxies.values():
            pid = int(getattr(proxy, "partition_id", proxy.cid))
            partition_to_proxy[pid] = proxy
            self._cid_to_partition[proxy.cid] = pid

        client_ids = sorted(partition_to_proxy.keys())
        self._history.current_round = server_round

        sel_rng = np.random.default_rng(
            derive_seed(self._cfg.seed, f"sel-{server_round}")
        )

        tracemalloc.start()
        _sel_start = time.perf_counter()
        selected = self._selector.select(
            client_ids,
            k=self._cfg.fl.clients_per_round,
            history=self._history,
            rng=sel_rng,
        )
        self._last_selection_time_ms = (time.perf_counter() - _sel_start) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self._last_selection_peak_mem_bytes = int(peak)

        for cid in selected:
            self._history.selection_count[cid] = (
                self._history.selection_count.get(cid, 0) + 1
            )
            self._history.last_selected[cid] = server_round
        self._last_selected = selected

        fit_ins = FitIns(parameters=parameters, config={"current_round": server_round})
        return [(partition_to_proxy[cid], fit_ins) for cid in selected]

    def _compute_server_gradient(self) -> np.ndarray:
        assert self._server_root_loader is not None
        assert self._global_params is not None

        weights_before = params_to_flat(parameters_to_ndarrays(self._global_params))
        set_parameters(self._model, parameters_to_ndarrays(self._global_params))

        self._model.train()
        optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self._cfg.fl.learning_rate, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        for images, labels in self._server_root_loader:
            images, labels = images.to(_DEVICE), labels.long().to(_DEVICE)
            optimizer.zero_grad()
            criterion(self._model(images), labels).backward()
            optimizer.step()

        weights_after = params_to_flat(get_parameters(self._model))
        set_parameters(self._model, parameters_to_ndarrays(self._global_params))
        return weights_after - weights_before

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return self._global_params, {}

        assert self._global_params is not None
        ref_params = parameters_to_ndarrays(self._global_params)
        global_flat = params_to_flat(ref_params)

        updates: list[ClientUpdate] = []
        total_loss = 0.0
        loss_count = 0
        self._last_client_metrics = []

        for proxy, fit_res in results:
            cid = self._cid_to_partition.get(proxy.cid, int(proxy.cid))
            flat = params_to_flat(parameters_to_ndarrays(fit_res.parameters))
            metrics = {k: float(v) for k, v in fit_res.metrics.items()}
            updates.append(
                ClientUpdate(
                    client_id=cid,
                    weights=flat,
                    num_samples=fit_res.num_examples,
                    metrics=metrics,
                )
            )
            if "loss" in metrics:
                self._history.loss[cid] = metrics["loss"]
                total_loss += metrics["loss"]
                loss_count += 1
            if "grad_norm" in metrics:
                self._history.grad_norm[cid] = metrics["grad_norm"]

            rep_score = ""
            beta_params = ""
            if hasattr(self._selector, "_reputation"):
                rep_score = str(self._selector._reputation.get(cid, ""))
            if hasattr(self._selector, "_beta"):
                bm = self._selector._beta.get(cid)
                beta_params = json.dumps(bm) if bm is not None else ""

            self._last_client_metrics.append({
                "round_num": server_round,
                "client_id": cid,
                "is_malicious": int(cid in self._malicious_ids),
                "was_selected": 1,
                "local_loss": metrics.get("loss", ""),
                "gradient_norm": metrics.get("grad_norm", ""),
                "reputation_score": rep_score,
                "beta_params": beta_params,
            })

        train_times = [
            float(fit_res.metrics.get("train_time_ms", 0.0))
            for _, fit_res in results
            if fit_res.metrics.get("train_time_ms", 0.0) > 0
        ]
        self._last_avg_train_time_ms = float(np.mean(train_times)) if train_times else 0.0
        self._last_train_loss = total_loss / max(1, loss_count)

        trusted_data = (
            self._compute_server_gradient() if self._server_root_loader is not None else None
        )
        state = ServerState(
            global_weights=global_flat,
            round_num=server_round,
            trusted_data=trusted_data,
        )
        new_flat = self._aggregator.aggregate(updates, state, self._cfg.aggregation)
        new_params_list = flat_to_params(new_flat, ref_params)
        self._global_params = ndarrays_to_parameters(new_params_list)
        return self._global_params, {}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        set_parameters(self._model, parameters_to_ndarrays(parameters))
        criterion = nn.CrossEntropyLoss()

        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self._test_loader:
                images, labels = images.to(_DEVICE), labels.long().to(_DEVICE)
                outputs = self._model(images)
                loss_val = criterion(outputs, labels)
                total_loss += loss_val.item() * len(labels)
                correct += int((outputs.argmax(dim=1) == labels).sum())
                total += len(labels)

        test_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)

        if server_round > 0:
            bytes_per_round = (
                self._cfg.fl.clients_per_round * self._model_num_params * 4
            )
            byz_in_cohort = sorted(
                cid for cid in self._last_selected if cid in self._malicious_ids
            )
            selector_bytes = _selector_state_bytes(self._selector)

            m = RoundMetrics(
                round_num=server_round,
                test_accuracy=accuracy,
                test_loss=test_loss,
                selected_clients=list(self._last_selected),
                train_loss=self._last_train_loss,
                selection_time_ms=self._last_selection_time_ms,
                avg_train_time_ms=self._last_avg_train_time_ms,
                bytes_per_round=bytes_per_round,
                n_byzantine_in_cohort=len(byz_in_cohort),
                byzantine_in_cohort_ids=byz_in_cohort,
                selector_state_bytes=selector_bytes,
                selection_peak_mem_bytes=self._last_selection_peak_mem_bytes,
            )
            self._collector.record(m)

            if self._round_csv_writer is not None:
                self._round_csv_writer.writerow([
                    m.round_num,
                    f"{m.test_accuracy:.6f}",
                    f"{m.test_loss:.6f}",
                    f"{m.train_loss:.6f}",
                    json.dumps(m.selected_clients),
                    m.n_byzantine_in_cohort,
                    json.dumps(m.byzantine_in_cohort_ids),
                    f"{m.selection_time_ms:.4f}",
                    f"{m.avg_train_time_ms:.4f}",
                    m.bytes_per_round,
                    m.selector_state_bytes,
                    m.selection_peak_mem_bytes,
                ])

            if self._client_csv_writer is not None:
                for cm in self._last_client_metrics:
                    self._client_csv_writer.writerow([
                        cm["round_num"],
                        cm["client_id"],
                        cm["is_malicious"],
                        cm["was_selected"],
                        cm["local_loss"],
                        cm["gradient_norm"],
                        cm["reputation_score"],
                        cm["beta_params"],
                    ])

        return test_loss, {"accuracy": accuracy}
