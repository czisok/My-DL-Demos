from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import sacrebleu
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PAD_ID, SentencePieceTokenizer
from .inference import greedy_decode
from .model import Transformer
from .utils import (
    JsonlLogger,
    NoamScheduler,
    RunningMetrics,
    human_time,
)


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple:
    return (
        batch["source_ids"].to(device, non_blocking=True),
        batch["target_ids"].to(device, non_blocking=True),
    )


def train_one_epoch(
    model: Transformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    accumulation_steps: int,
    grad_clip: float,
    log_every: int,
    epoch: int,
    start_time: float,
    logger: JsonlLogger,
) -> dict[str, float]:
    model.train()
    metrics = RunningMetrics()
    optimizer.zero_grad(set_to_none=True)
    current_lr = 0.0

    for step, batch in enumerate(loader, start=1):
        source_ids, target_ids = _move_batch(batch, device)
        decoder_input = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=scaler.is_enabled(),
        ):
            logits = model(source_ids, decoder_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
            )
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()
        should_update = (
            step % accumulation_steps == 0 or step == len(loader)
        )
        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            current_lr = scheduler.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        metrics.update(float(loss.item()), logits.detach(), labels, PAD_ID)
        if step % log_every == 0 or step == len(loader):
            values = metrics.compute()
            elapsed = human_time(time.perf_counter() - start_time)
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={values['loss']:.4f} "
                f"ppl={values['perplexity']:.2f} "
                f"acc={values['token_accuracy']:.4f} "
                f"lr={current_lr:.3e} elapsed={elapsed}",
                flush=True,
            )
            logger.log(
                {
                    "phase": "train_step",
                    "epoch": epoch,
                    "step": step,
                    "learning_rate": current_lr,
                    "elapsed_seconds": time.perf_counter() - start_time,
                    **values,
                }
            )
    return metrics.compute()


@torch.inference_mode()
def evaluate_loss(
    model: Transformer,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    metrics = RunningMetrics()
    for batch in loader:
        source_ids, target_ids = _move_batch(batch, device)
        decoder_input = target_ids[:, :-1]
        labels = target_ids[:, 1:]
        logits = model(source_ids, decoder_input)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        metrics.update(float(loss.item()), logits, labels, PAD_ID)
    return metrics.compute()


@torch.inference_mode()
def evaluate_bleu(
    model: Transformer,
    loader: DataLoader,
    target_tokenizer: SentencePieceTokenizer,
    device: torch.device,
    max_length: int,
    max_samples: int,
) -> float:
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    for batch in loader:
        source_ids = batch["source_ids"].to(device)
        generated = greedy_decode(
            model, source_ids, max_new_tokens=max_length - 1
        )
        predictions.extend(
            target_tokenizer.decode(row.tolist()) for row in generated
        )
        references.extend(batch["target_texts"])
        if len(predictions) >= max_samples:
            break
    predictions = predictions[:max_samples]
    references = references[:max_samples]
    return float(
        sacrebleu.corpus_bleu(
            predictions, [references], tokenize="zh"
        ).score
    )


def save_checkpoint(
    path: str | Path,
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    config: dict[str, Any],
    epoch: int,
    best_bleu: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": model.export_config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "epoch": epoch,
            "best_bleu": best_bleu,
        },
        path,
    )
