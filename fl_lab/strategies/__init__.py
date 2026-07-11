from fl_lab.strategies.adaptive_mab import AdaptiveMabSelector
from fl_lab.strategies.base import ClientSelector
from fl_lab.strategies.cyclic import CyclicSelector
from fl_lab.strategies.random_selector import RandomSelector
from fl_lab.strategies.reputation import ReputationSelector

_REGISTRY: dict[str, type[ClientSelector]] = {
    "random": RandomSelector,
    "cyclic": CyclicSelector,
    "reputation": ReputationSelector,
    "adaptive_mab": AdaptiveMabSelector,
}


def get_selector(name: str) -> ClientSelector:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown selector {name!r}. Expected one of: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()
