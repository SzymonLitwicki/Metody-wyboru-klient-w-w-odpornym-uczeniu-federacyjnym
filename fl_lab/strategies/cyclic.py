import numpy as np
from fl_lab.strategies.base import ClientSelector, SelectionHistory


class CyclicSelector(ClientSelector):

    def __init__(self, num_groups: int = 5) -> None:
        self.num_groups = num_groups
        self._groups: list[list[int]] | None = None

    def _build_groups(self, client_ids: list[int], rng: np.random.Generator) -> None:
        ids = list(client_ids)
        perm = rng.permutation(len(ids))
        shuffled = [ids[i] for i in perm]
        g = max(1, min(self.num_groups, len(ids)))
        self._groups = [shuffled[i::g] for i in range(g)]

    def select(
        self,
        client_ids: list[int],
        k: int,
        history: SelectionHistory,
        rng: np.random.Generator,
    ) -> list[int]:
        if self._groups is None:
            self._build_groups(client_ids, rng)

        round_num = max(0, history.current_round)
        active = self._groups[round_num % len(self._groups)]

        if len(active) <= k:
            chosen = list(active)
        else:
            idx = rng.choice(len(active), size=k, replace=False)
            chosen = [active[i] for i in idx]
        return sorted(chosen)

    @property
    def state_size_bytes(self) -> int:
        try:
            from pympler import asizeof
            return int(asizeof.asizeof(self._groups))
        except Exception:
            import sys
            return sys.getsizeof(self._groups)
