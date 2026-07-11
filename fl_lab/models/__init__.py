import torch.nn as nn

from fl_lab.models.simple_cnn import SimpleCNN

_REGISTRY: dict[str, type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
}


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Expected one of: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](num_classes=num_classes)
