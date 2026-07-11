from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


_FROZEN = ConfigDict(frozen=True, extra="forbid")


class DataConfig(BaseModel):
    model_config = _FROZEN

    dataset: Literal["cifar10"] = "cifar10"
    distribution: Literal["iid", "dirichlet", "pathological"] = "dirichlet"
    alpha: float = Field(default=0.5, gt=0.0)
    num_classes_per_client: int = Field(default=2, ge=1)


class ModelConfig(BaseModel):
    model_config = _FROZEN

    name: Literal["simple_cnn"] = "simple_cnn"


class FLConfig(BaseModel):
    model_config = _FROZEN

    num_clients: int = Field(default=50, ge=2)
    clients_per_round: int = Field(default=10, ge=1)
    rounds: int = Field(default=100, ge=1)
    local_epochs: int = Field(default=5, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    batch_size: int = Field(default=32, ge=1)

    @model_validator(mode="after")
    def _check_clients_per_round(self) -> "FLConfig":
        if self.clients_per_round > self.num_clients:
            raise ValueError(
                f"clients_per_round ({self.clients_per_round}) "
                f"must not exceed num_clients ({self.num_clients})"
            )
        return self


class AttackConfig(BaseModel):
    model_config = _FROZEN

    type: Literal["none", "label_flip", "sign_flip", "gaussian_noise"] = "label_flip"
    byzantine_pct: float = Field(default=0.2, ge=0.0, le=1.0)
    scale: float = Field(default=1.0)


class SelectionConfig(BaseModel):
    model_config = _FROZEN

    strategy: Literal["random", "cyclic", "reputation", "adaptive_mab"] = "random"


class AggregationConfig(BaseModel):
    model_config = _FROZEN

    method: Literal["fedavg", "krum", "trimmed_mean", "bulyan", "fltrust"] = "fedavg"
    krum_m: int = Field(default=1, ge=1)
    trim_ratio: float = Field(default=0.1, ge=0.0, lt=0.5)


class LoggingConfig(BaseModel):
    model_config = _FROZEN

    output_dir: str = "results"
    log_every: int = Field(default=1, ge=1)


class ExperimentConfig(BaseModel):
    model_config = _FROZEN

    name: str = Field(default="exp")
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    fl: FLConfig = Field(default_factory=FLConfig)
    attack: AttackConfig = Field(default_factory=AttackConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig.model_validate(raw)


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_overrides(
    path: str | Path, overrides: dict | None = None
) -> ExperimentConfig:
    with Path(path).open("r") as f:
        raw = yaml.safe_load(f) or {}
    if overrides is not None:
        raw = deep_merge(raw, overrides)
    return ExperimentConfig.model_validate(raw)
