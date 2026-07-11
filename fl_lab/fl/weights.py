import numpy as np
import torch
import torch.nn as nn


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    model_params = list(model.parameters())
    if len(model_params) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(model_params)} tensors, "
            f"got {len(parameters)}"
        )
    for param, arr in zip(model_params, parameters):
        param.data.copy_(torch.from_numpy(arr))


def params_to_flat(parameters: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([p.ravel().astype(np.float32) for p in parameters])


def flat_to_params(flat: np.ndarray, reference: list[np.ndarray]) -> list[np.ndarray]:
    expected = sum(r.size for r in reference)
    if flat.size != expected:
        raise ValueError(f"flat has {flat.size} elements, reference expects {expected}")
    result: list[np.ndarray] = []
    offset = 0
    for ref in reference:
        n = ref.size
        result.append(flat[offset : offset + n].reshape(ref.shape).astype(ref.dtype))
        offset += n
    return result
