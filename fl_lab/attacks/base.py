from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fl_lab.utils.config import AttackConfig


@dataclass
class ClientDataset:
    indices: list[int]
    targets: list[int]


class Attack(ABC):
    @abstractmethod
    def apply(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        client_dataset: ClientDataset,
        cfg: AttackConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        ...
