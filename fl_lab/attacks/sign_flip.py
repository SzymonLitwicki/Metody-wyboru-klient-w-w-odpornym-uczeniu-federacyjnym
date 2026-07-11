import numpy as np

from fl_lab.attacks.base import Attack, ClientDataset
from fl_lab.utils.config import AttackConfig


class SignFlipAttack(Attack):
    def apply(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        client_dataset: ClientDataset,
        cfg: AttackConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        delta = weights_after - weights_before
        return weights_before - delta
