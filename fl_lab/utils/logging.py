import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


_CLIENT_FIELDS = ["round", "client_id", "loss", "grad_norm", "selected", "malicious"]


class RoundLogger:
    def __init__(self, path: str | Path, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self._ensure_header()

    def _ensure_header(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def record(self, row: dict[str, Any]) -> None:
        extra = set(row) - set(self.fieldnames)
        if extra:
            raise ValueError(f"Unknown fields in row: {sorted(extra)}")
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)


class ClientLogger:
    def __init__(self, path: str | Path) -> None:
        self._logger = RoundLogger(path, fieldnames=_CLIENT_FIELDS)

    def record(
        self,
        round_num: int,
        client_id: int,
        loss: float,
        grad_norm: float,
        selected: bool,
        malicious: bool,
    ) -> None:
        self._logger.record(
            {
                "round": round_num,
                "client_id": client_id,
                "loss": loss,
                "grad_norm": grad_norm,
                "selected": selected,
                "malicious": malicious,
            }
        )


def save_config(config: BaseModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2))


def save_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
