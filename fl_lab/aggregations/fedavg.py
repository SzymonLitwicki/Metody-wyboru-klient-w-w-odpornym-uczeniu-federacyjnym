import numpy as np

from fl_lab.aggregations.base import Aggregator, ClientUpdate, ServerState
from fl_lab.utils.config import AggregationConfig


class FedAvgAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        if not updates:
            raise ValueError("aggregate() called with an empty updates list")
        total = sum(u.num_samples for u in updates)
        result = np.zeros_like(updates[0].weights, dtype=float)
        for u in updates:
            result += u.weights * (u.num_samples / total)
        return result
