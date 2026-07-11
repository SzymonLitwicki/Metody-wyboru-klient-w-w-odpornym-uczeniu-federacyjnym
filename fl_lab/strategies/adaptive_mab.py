import numpy as np
from fl_lab.strategies.base import ClientSelector, SelectionHistory


class AdaptiveMabSelector(ClientSelector):

    def __init__(self, deviation_threshold: float = 2.0) -> None:
        self.deviation_threshold = deviation_threshold
        self._beta: dict[int, list[int]] = {}
        self._last_selected: list[int] = []

    def _distribute_rewards(self, history: SelectionHistory) -> None:
        if not self._last_selected:
            return
        norms = np.array(
            [history.grad_norm.get(cid, np.nan) for cid in self._last_selected],
            dtype=float,
        )
        if np.all(np.isnan(norms)):
            for cid in self._last_selected:
                self._beta.setdefault(cid, [1, 1])[0] += 1
            return
        med = np.nanmedian(norms)
        mad = np.nanmedian(np.abs(norms - med)) + 1e-9
        for cid, norm in zip(self._last_selected, norms):
            bm = self._beta.setdefault(cid, [1, 1])
            if not np.isnan(norm) and abs(norm - med) / mad > self.deviation_threshold:
                bm[1] += 1
            else:
                bm[0] += 1

    def select(
        self,
        client_ids: list[int],
        k: int,
        history: SelectionHistory,
        rng: np.random.Generator,
    ) -> list[int]:
        for cid in client_ids:
            self._beta.setdefault(cid, [1, 1])

        self._distribute_rewards(history)

        sampled = np.array(
            [rng.beta(self._beta[cid][0], self._beta[cid][1]) for cid in client_ids],
            dtype=float,
        )
        if len(client_ids) <= k:
            chosen = list(client_ids)
        else:
            top = np.argsort(sampled)[-k:]
            chosen = [client_ids[i] for i in top]

        self._last_selected = sorted(chosen)
        return self._last_selected

    @property
    def state_size_bytes(self) -> int:
        try:
            from pympler import asizeof
            return int(asizeof.asizeof(self._beta))
        except Exception:
            import sys
            return sys.getsizeof(self._beta)
