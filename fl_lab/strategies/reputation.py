import numpy as np
from fl_lab.strategies.base import ClientSelector, SelectionHistory


class ReputationSelector(ClientSelector):

    def __init__(
        self,
        init_reputation: float = 0.5,
        decay: float = 0.6,
        reward: float = 1.15,
        deviation_threshold: float = 2.0,
        exploration_floor: float = 0.05,
    ) -> None:
        self.init_reputation = init_reputation
        self.decay = decay
        self.reward = reward
        self.deviation_threshold = deviation_threshold
        self.exploration_floor = exploration_floor
        self._reputation: dict[int, float] = {}
        self._last_selected: list[int] = []

    def _update_reputation(self, history: SelectionHistory) -> None:
        if not self._last_selected:
            return
        norms = np.array(
            [history.grad_norm.get(cid, np.nan) for cid in self._last_selected],
            dtype=float,
        )
        if np.all(np.isnan(norms)):
            return
        med = np.nanmedian(norms)
        mad = np.nanmedian(np.abs(norms - med)) + 1e-9
        for cid, norm in zip(self._last_selected, norms):
            r = self._reputation.get(cid, self.init_reputation)
            if np.isnan(norm):
                continue
            deviation = abs(norm - med) / mad
            if deviation > self.deviation_threshold:
                r *= self.decay
            else:
                r = min(1.0, r * self.reward)
            self._reputation[cid] = float(np.clip(r, 1e-4, 1.0))

    def select(
        self,
        client_ids: list[int],
        k: int,
        history: SelectionHistory,
        rng: np.random.Generator,
    ) -> list[int]:
        k = min(k, len(client_ids))
        for cid in client_ids:
            self._reputation.setdefault(cid, self.init_reputation)

        self._update_reputation(history)

        scores = np.array(
            [self._reputation[cid] + self.exploration_floor for cid in client_ids],
            dtype=float,
        )
        total = scores.sum()
        if total <= 0 or not np.isfinite(total):
            idx = rng.choice(len(client_ids), size=k, replace=False)
            chosen = [client_ids[i] for i in idx]
        else:
            probs = scores / total
            idx = rng.choice(len(client_ids), size=k, replace=False, p=probs)
            chosen = [client_ids[i] for i in idx]

        self._last_selected = sorted(chosen)
        return self._last_selected

    @property
    def state_size_bytes(self) -> int:
        try:
            from pympler import asizeof
            return int(asizeof.asizeof(self._reputation))
        except Exception:
            import sys
            return sys.getsizeof(self._reputation)
