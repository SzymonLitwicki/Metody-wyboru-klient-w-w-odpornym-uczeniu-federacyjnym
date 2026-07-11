from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelectionHistory:
    loss: dict[int, float] = field(default_factory=dict)
    grad_norm: dict[int, float] = field(default_factory=dict)
    selection_count: dict[int, int] = field(default_factory=dict)
    last_selected: dict[int, int] = field(default_factory=dict)
    class_distribution: dict[int, dict[int, float]] = field(default_factory=dict)
    current_round: int = 0


class ClientSelector(ABC):
    @abstractmethod
    def select(
        self,
        client_ids: list[int],
        k: int,
        history: SelectionHistory,
        rng: np.random.Generator,
    ) -> list[int]:
        ...
