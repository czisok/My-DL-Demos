from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def human_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class NoamScheduler:
    """Learning-rate schedule from the original Transformer paper."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        factor: float = 1.0,
        step_num: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = step_num

    def _rate(self, step: int) -> float:
        step = max(step, 1)
        return self.factor * self.d_model**-0.5 * min(
            step**-0.5, step * self.warmup_steps**-1.5
        )

    def step(self) -> float:
        self.step_num += 1
        rate = self._rate(self.step_num)
        for group in self.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def state_dict(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.d_model = int(state["d_model"])
        self.warmup_steps = int(state["warmup_steps"])
        self.factor = float(state["factor"])
        self.step_num = int(state["step_num"])


@dataclass
class RunningMetrics:
    loss_sum: float = 0.0
    correct_tokens: int = 0
    token_count: int = 0
    batches: int = 0

    def update(
        self, loss: float, logits: torch.Tensor, targets: torch.Tensor, pad_id: int
    ) -> None:
        valid = targets.ne(pad_id)
        predictions = logits.argmax(dim=-1)
        valid_count = int(valid.sum().item())
        self.loss_sum += loss * valid_count
        self.correct_tokens += int((predictions.eq(targets) & valid).sum().item())
        self.token_count += valid_count
        self.batches += 1

    def compute(self) -> dict[str, float]:
        average_loss = self.loss_sum / max(self.token_count, 1)
        return {
            "loss": average_loss,
            "perplexity": math.exp(min(average_loss, 20.0)),
            "token_accuracy": self.correct_tokens / max(self.token_count, 1),
        }


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload: dict[str, Any]) -> None:
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
