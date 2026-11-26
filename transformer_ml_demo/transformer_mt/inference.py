from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from .data import BOS_ID, EOS_ID, PAD_ID, SentencePieceTokenizer
from .model import Transformer, TransformerConfig


@torch.inference_mode()
def greedy_decode(
    model: Transformer,
    source_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    model.eval()
    memory, source_mask = model.encode(source_ids)
    generated = torch.full(
        (source_ids.size(0), 1),
        BOS_ID,
        dtype=torch.long,
        device=source_ids.device,
    )
    finished = torch.zeros(
        source_ids.size(0), dtype=torch.bool, device=source_ids.device
    )
    for _ in range(max_new_tokens):
        decoder_output = model.decode(generated, memory, source_mask)
        logits = model.output_projection(decoder_output[:, -1])
        next_ids = logits.argmax(dim=-1)
        next_ids = torch.where(
            finished, torch.full_like(next_ids, EOS_ID), next_ids
        )
        generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
        finished |= next_ids.eq(EOS_ID)
        if bool(finished.all()):
            break
    return generated


def encode_source_batch(
    texts: Iterable[str],
    tokenizer: SentencePieceTokenizer,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    sequences = [tokenizer.encode(text, max_length) for text in texts]
    width = max(map(len, sequences))
    batch = torch.full((len(sequences), width), PAD_ID, dtype=torch.long)
    for row, sequence in enumerate(sequences):
        batch[row, : len(sequence)] = torch.tensor(sequence)
    return batch.to(device)


def translate_texts(
    model: Transformer,
    texts: list[str],
    source_tokenizer: SentencePieceTokenizer,
    target_tokenizer: SentencePieceTokenizer,
    device: torch.device,
    max_length: int,
) -> list[str]:
    source_ids = encode_source_batch(
        texts, source_tokenizer, max_length, device
    )
    generated = greedy_decode(model, source_ids, max_new_tokens=max_length - 1)
    return [
        target_tokenizer.decode(sequence.tolist()) for sequence in generated
    ]


def load_translation_checkpoint(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[
    Transformer,
    SentencePieceTokenizer,
    SentencePieceTokenizer,
    dict,
]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = Transformer(TransformerConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    tokenizer_dir = checkpoint_path.parent / "tokenizers"
    config = checkpoint["config"]
    source_lang = config["data"]["source_lang"]
    target_lang = config["data"]["target_lang"]
    source_tokenizer = SentencePieceTokenizer(
        tokenizer_dir / f"{source_lang}.model"
    )
    target_tokenizer = SentencePieceTokenizer(
        tokenizer_dir / f"{target_lang}.model"
    )
    return model, source_tokenizer, target_tokenizer, config
