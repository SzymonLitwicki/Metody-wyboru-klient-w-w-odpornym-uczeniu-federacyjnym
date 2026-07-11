import numpy as np

from fl_lab.attacks.base import Attack, ClientDataset
from fl_lab.utils.config import AttackConfig


class GaussianNoiseAttack(Attack):
    def apply(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        client_dataset: ClientDataset,
        cfg: AttackConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        noise = rng.normal(0.0, cfg.scale, size=weights_after.shape)
        return weights_after + noise
