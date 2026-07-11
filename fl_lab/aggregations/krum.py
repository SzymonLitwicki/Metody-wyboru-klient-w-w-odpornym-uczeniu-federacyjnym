import logging
import numpy as np

from fl_lab.aggregations.base import Aggregator, ClientUpdate, ServerState
from fl_lab.utils.config import AggregationConfig

logger = logging.getLogger(__name__)


def _krum_scores(weights: np.ndarray, neighbors: int) -> np.ndarray:
    norms_sq = np.sum(weights ** 2, axis=1)
    sq_dists = norms_sq[:, None] + norms_sq[None, :] - 2 * (weights @ weights.T)
    np.fill_diagonal(sq_dists, np.inf)
    sorted_dists = np.sort(sq_dists, axis=1)
    actual_neighbors = min(neighbors, sq_dists.shape[1] - 1)
    return np.sum(sorted_dists[:, :actual_neighbors], axis=1)


class KrumAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        weights = np.stack([u.weights for u in updates])
        n = len(updates)
        # Formal Krum condition: n >= 2f + 3.  When violated (high byzantine
        # pct), log a warning and degrade gracefully rather than crash.
        neighbors = max(1, n - cfg.krum_m - 2)
        if neighbors < 1:
            logger.warning(
                "Krum: n=%d, krum_m=%d → neighbors=%d < 1; formal guarantee "
                "violated (likely high byzantine_pct). Falling back to FedAvg.",
                n, cfg.krum_m, neighbors,
            )
            return np.mean(weights, axis=0)
        try:
            scores = _krum_scores(weights, neighbors)
            return weights[int(np.argmin(scores))]
        except Exception as exc:
            logger.warning("Krum aggregate failed (%s); falling back to FedAvg.", exc)
            return np.mean(weights, axis=0)


class MultiKrumAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        weights = np.stack([u.weights for u in updates])
        n = len(updates)
        neighbors = max(1, n - cfg.krum_m - 2)
        if neighbors < 1:
            logger.warning(
                "MultiKrum: n=%d, krum_m=%d → neighbors=%d < 1; falling back to FedAvg.",
                n, cfg.krum_m, neighbors,
            )
            return np.mean(weights, axis=0)
        try:
            scores = _krum_scores(weights, neighbors)
            m = max(1, n - cfg.krum_m)
            selected = np.argsort(scores)[:m]
            return np.mean(weights[selected], axis=0)
        except Exception as exc:
            logger.warning("MultiKrum aggregate failed (%s); falling back to FedAvg.", exc)
            return np.mean(weights, axis=0)
