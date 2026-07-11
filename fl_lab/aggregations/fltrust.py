import numpy as np

from fl_lab.aggregations.base import Aggregator, ClientUpdate, ServerState
from fl_lab.aggregations.fedavg import FedAvgAggregator
from fl_lab.utils.config import AggregationConfig


class FLTrustAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        if state.trusted_data is None:
            return FedAvgAggregator().aggregate(updates, state, cfg)

        server_grad = state.trusted_data
        server_norm = float(np.linalg.norm(server_grad))
        if server_norm == 0.0:
            return state.global_weights.copy()

        trust_scores: list[float] = []
        scaled_updates: list[np.ndarray] = []

        for u in updates:
            delta = u.weights - state.global_weights
            client_norm = float(np.linalg.norm(delta))
            if client_norm == 0.0:
                trust_scores.append(0.0)
                scaled_updates.append(np.zeros_like(delta))
                continue
            cos_sim = float(np.dot(delta, server_grad) / (client_norm * server_norm))
            trust = max(0.0, cos_sim)
            trust_scores.append(trust)
            scaled_updates.append(delta / client_norm * server_norm)

        total_trust = sum(trust_scores)
        if total_trust == 0.0:
            return state.global_weights.copy()

        result = np.zeros_like(state.global_weights, dtype=float)
        for trust, scaled in zip(trust_scores, scaled_updates):
            result += scaled * (trust / total_trust)
        return state.global_weights + result
