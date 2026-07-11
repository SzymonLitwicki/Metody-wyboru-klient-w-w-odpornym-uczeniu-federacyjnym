from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fl_lab.utils.config import AggregationConfig


@dataclass
class ClientUpdate:
    client_id: int
    weights: np.ndarray
    num_samples: int
    metrics: dict[str, float]


@dataclass
class ServerState:
    global_weights: np.ndarray
    round_num: int
    trusted_data: np.ndarray | None = None


class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        updates: list[ClientUpdate],
        state: ServerState,
        cfg: AggregationConfig,
    ) -> np.ndarray:
        ...
