import numpy as np

from fl_lab.attacks.base import Attack, ClientDataset
from fl_lab.utils.config import AttackConfig


class LabelFlipAttack(Attack):
    """Pass-through attack: actual label flipping happens in client data loading."""

    def apply(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        client_dataset: ClientDataset,
        cfg: AttackConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return weights_after.copy()
