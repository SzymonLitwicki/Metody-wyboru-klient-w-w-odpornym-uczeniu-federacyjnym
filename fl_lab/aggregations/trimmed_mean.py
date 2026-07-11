import numpy as np

from fl_lab.aggregations.base import Aggregator, ClientUpdate, ServerState
from fl_lab.utils.config import AggregationConfig


class TrimmedMeanAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        weights = np.stack([u.weights for u in updates])
        n = len(updates)
        k = max(0, int(n * cfg.trim_ratio))
        sorted_weights = np.sort(weights, axis=0)
        if k > 0:
            trimmed = sorted_weights[k:-k]
        else:
            trimmed = sorted_weights
        return np.mean(trimmed, axis=0)
