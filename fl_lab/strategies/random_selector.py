import numpy as np

from fl_lab.strategies.base import ClientSelector, SelectionHistory


class RandomSelector(ClientSelector):
    def select(
        self,
        client_ids: list[int],
        k: int,
        history: SelectionHistory,
        rng: np.random.Generator,
    ) -> list[int]:
        idx = rng.choice(len(client_ids), size=k, replace=False)
        return [client_ids[i] for i in sorted(idx)]
