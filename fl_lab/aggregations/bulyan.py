import logging
import numpy as np

from fl_lab.aggregations.base import Aggregator, ClientUpdate, ServerState
from fl_lab.aggregations.krum import _krum_scores
from fl_lab.utils.config import AggregationConfig

logger = logging.getLogger(__name__)


class BulyanAggregator(Aggregator):
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        n = len(updates)
        f = max(1, int(n * cfg.trim_ratio))
        # Formal Bulyan condition: n >= 4f + 3.  Degrade gracefully when violated.
        if n < 4 * f + 3:
            logger.warning(
                "Bulyan: n=%d, f=%d → n < 4f+3=%d; formal guarantee violated "
                "(likely high byzantine_pct in cohort). Running best-effort Bulyan.",
                n, f, 4 * f + 3,
            )

        beta = max(1, n - 2 * f)
        weights = np.stack([u.weights for u in updates])
        try:
            neighbors = max(1, n - f - 2)
            scores = _krum_scores(weights, neighbors)
            selected_idx = np.argsort(scores)[:beta]
            selected = weights[selected_idx]

            sorted_selected = np.sort(selected, axis=0)
            trim = f if (f > 0 and 2 * f < beta) else 0
            trimmed = sorted_selected[trim: len(sorted_selected) - trim] if trim > 0 else sorted_selected
            return np.mean(trimmed, axis=0)
        except Exception as exc:
            logger.warning("Bulyan aggregate failed (%s); falling back to FedAvg.", exc)
            return np.mean(weights, axis=0)
