from typing import Any
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from flwr.client import NumPyClient

from fl_lab.attacks.base import Attack, ClientDataset
from fl_lab.attacks.label_flip import LabelFlipAttack
from fl_lab.fl.weights import flat_to_params, get_parameters, params_to_flat, set_parameters
from fl_lab.utils.config import ExperimentConfig
from fl_lab.utils.seed import derive_seed

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _LabelFlippedDataset(Dataset):

    _N_CLASSES = 10

    def __init__(self, subset: Subset) -> None:
        self._subset = subset

    def __len__(self) -> int:
        return len(self._subset)

    def __getitem__(self, idx: int) -> tuple:
        image, label = self._subset[idx]
        return image, self._N_CLASSES - 1 - int(label)


class FlClient(NumPyClient):
    def __init__(
        self,
        cid: int,
        model: nn.Module,
        dataset: Subset,
        cfg: ExperimentConfig,
        attack: Attack | None,
    ) -> None:
        self._cid = cid
        self._model = model
        self._dataset = dataset
        self._cfg = cfg
        self._attack = attack

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self._model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        set_parameters(self._model, parameters)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        self.set_parameters(parameters)
        weights_before = get_parameters(self._model)

        round_num = int(config.get("current_round", 0))

        torch.manual_seed(derive_seed(self._cfg.seed, f"fit-{self._cid}-r{round_num}"))
        np.random.seed(derive_seed(self._cfg.seed, f"fit-np-{self._cid}-r{round_num}") % (2**31))

        dataset: Dataset = self._dataset
        if isinstance(self._attack, LabelFlipAttack):
            dataset = _LabelFlippedDataset(self._dataset)

        _loader_seed = derive_seed(self._cfg.seed, f"loader-{self._cid}-r{round_num}")
        _generator = torch.Generator()
        _generator.manual_seed(_loader_seed)

        loader = DataLoader(dataset, batch_size=self._cfg.fl.batch_size, shuffle=True,
                            generator=_generator, pin_memory=_DEVICE.type == "cuda")
        self._model.to(_DEVICE)
        optimizer = torch.optim.SGD(
            self._model.parameters(), lr=self._cfg.fl.learning_rate, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        _train_start = time.perf_counter()
        self._model.train()
        total_loss = 0.0
        n_batches = 0
        for _ in range(self._cfg.fl.local_epochs):
            for images, labels in loader:
                images, labels = images.to(_DEVICE), labels.long().to(_DEVICE)
                optimizer.zero_grad()
                loss = criterion(self._model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        weights_after = get_parameters(self._model)
        _train_time_ms = (time.perf_counter() - _train_start) * 1000.0

        if self._attack is not None:
            flat_before = params_to_flat(weights_before)
            flat_after = params_to_flat(weights_after)
            atk_rng = np.random.default_rng(derive_seed(self._cfg.seed, f"atk-{self._cid}-r{round_num}"))
            inner_ds = self._dataset.dataset
            if hasattr(inner_ds, "indices"):
                actual_indices = [inner_ds.indices[i] for i in self._dataset.indices]
                targets_source = inner_ds.dataset
            else:
                actual_indices = list(self._dataset.indices)
                targets_source = inner_ds
            ds = ClientDataset(
                indices=actual_indices,
                targets=[int(targets_source.targets[i]) for i in actual_indices],
            )
            flat_poisoned = self._attack.apply(
                flat_before, flat_after, ds, self._cfg.attack, atk_rng
            )
            weights_after = flat_to_params(flat_poisoned, weights_before)
            set_parameters(self._model, weights_after)

        grad_norm = float(
            np.linalg.norm(
                np.concatenate([(a - b).ravel() for a, b in zip(weights_after, weights_before)])
            )
        )
        return get_parameters(self._model), len(self._dataset), {
            "loss": avg_loss,
            "grad_norm": grad_norm,
            "train_time_ms": _train_time_ms,
        }

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        self.set_parameters(parameters)
        self._model.to(_DEVICE)
        loader = DataLoader(self._dataset, batch_size=self._cfg.fl.batch_size,
                            pin_memory=_DEVICE.type == "cuda")
        criterion = nn.CrossEntropyLoss()

        self._model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(_DEVICE), labels.long().to(_DEVICE)
                outputs = self._model(images)
                total_loss += criterion(outputs, labels).item() * len(labels)
                correct += int((outputs.argmax(dim=1) == labels).sum())
                total += len(labels)

        loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        return loss, len(self._dataset), {"accuracy": accuracy}
