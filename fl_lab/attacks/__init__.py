import numpy as np

from fl_lab.attacks.base import Attack, ClientDataset
from fl_lab.attacks.gaussian_noise import GaussianNoiseAttack
from fl_lab.attacks.label_flip import LabelFlipAttack
from fl_lab.attacks.sign_flip import SignFlipAttack
from fl_lab.utils.config import AttackConfig


class _NoAttack(Attack):
    def apply(
        self,
        weights_before: np.ndarray,
        weights_after: np.ndarray,
        client_dataset: ClientDataset,
        cfg: AttackConfig,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return weights_after.copy()


_REGISTRY: dict[str, type[Attack]] = {
    "none": _NoAttack,
    "label_flip": LabelFlipAttack,
    "sign_flip": SignFlipAttack,
    "gaussian_noise": GaussianNoiseAttack,
}


def get_attack(name: str) -> Attack:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown attack {name!r}. Expected one of: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
