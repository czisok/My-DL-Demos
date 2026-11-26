from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Iterable

import sentencepiece as spm
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text).strip()


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path) -> None:
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, max_length: int) -> list[int]:
        pieces = self.processor.encode(normalize_text(text), out_type=int)
        return [BOS_ID, *pieces[: max_length - 2], EOS_ID]

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = [
            int(token_id)
            for token_id in token_ids
            if int(token_id) not in {PAD_ID, BOS_ID, EOS_ID}
        ]
        return self.processor.decode(ids).strip()


def _write_tokenizer_corpus(
    dataset: Dataset, language: str, output_path: Path
) -> None:
    with output_path.open("w", encoding="utf-8") as file:
        for example in dataset:
            text = normalize_text(example["translation"][language])
            if text:
                file.write(text.replace("\n", " ") + "\n")


def train_tokenizer(
    dataset: Dataset,
    language: str,
    output_dir: Path,
    vocab_size: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{language}.model"
    if model_path.exists():
        return model_path

    corpus_path = output_dir / f"{language}_tokenizer_corpus.txt"
    _write_tokenizer_corpus(dataset, language, corpus_path)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(output_dir / language),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        hard_vocab_limit=False,
        shuffle_input_sentence=True,
        input_sentence_size=(
            0 if len(dataset) <= 100 else min(len(dataset), 1_000_000)
        ),
    )
    corpus_path.unlink(missing_ok=True)
    return model_path


def _valid_example(
    example: dict[str, Any], source_lang: str, target_lang: str, min_length: int
) -> bool:
    translation = example["translation"]
    source = normalize_text(translation[source_lang])
    target = normalize_text(translation[target_lang])
    return len(source) >= min_length and len(target) >= min_length


def _limit(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None or maximum <= 0 or maximum >= len(dataset):
        return dataset
    return dataset.select(range(maximum))


def load_parallel_data(config: dict[str, Any], seed: int) -> DatasetDict:
    data = load_dataset(config["dataset_name"], config["dataset_config"])
    source_lang = config["source_lang"]
    target_lang = config["target_lang"]
    min_length = int(config.get("min_sentence_length", 1))

    processed = DatasetDict()
    split_limits = {
        "train": config.get("max_train_samples"),
        "validation": config.get("max_valid_samples"),
        "test": config.get("max_test_samples"),
    }
    for split, maximum in split_limits.items():
        if split not in data:
            continue
        candidate = data[split]
        if split == "train":
            candidate = candidate.shuffle(seed=seed)
        if maximum is not None and maximum > 0:
            candidate_count = min(
                len(candidate), max(int(maximum) * 2, int(maximum) + 100)
            )
            candidate = candidate.select(range(candidate_count))
        clean = candidate.filter(
            lambda row: _valid_example(row, source_lang, target_lang, min_length)
        )
        processed[split] = _limit(clean, maximum)
    return processed


class TranslationCollator:
    def __init__(
        self,
        source_tokenizer: SentencePieceTokenizer,
        target_tokenizer: SentencePieceTokenizer,
        source_lang: str,
        target_lang: str,
        max_length: int,
    ) -> None:
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_length = max_length

    @staticmethod
    def _pad(sequences: list[list[int]]) -> torch.Tensor:
        max_length = max(map(len, sequences))
        output = torch.full(
            (len(sequences), max_length), PAD_ID, dtype=torch.long
        )
        for row, sequence in enumerate(sequences):
            output[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        return output

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        source_texts = [
            normalize_text(example["translation"][self.source_lang])
            for example in examples
        ]
        target_texts = [
            normalize_text(example["translation"][self.target_lang])
            for example in examples
        ]
        source_ids = [
            self.source_tokenizer.encode(text, self.max_length)
            for text in source_texts
        ]
        target_ids = [
            self.target_tokenizer.encode(text, self.max_length)
            for text in target_texts
        ]
        return {
            "source_ids": self._pad(source_ids),
            "target_ids": self._pad(target_ids),
            "source_texts": source_texts,
            "target_texts": target_texts,
        }


def build_data_pipeline(
    data_config: dict[str, Any],
    output_dir: str | Path,
    seed: int,
    batch_size: int,
) -> tuple[
    dict[str, DataLoader],
    SentencePieceTokenizer,
    SentencePieceTokenizer,
    DatasetDict,
]:
    datasets = load_parallel_data(data_config, seed)
    tokenizer_dir = Path(output_dir) / "tokenizers"
    source_path = train_tokenizer(
        datasets["train"],
        data_config["source_lang"],
        tokenizer_dir,
        int(data_config["vocab_size"]),
    )
    target_path = train_tokenizer(
        datasets["train"],
        data_config["target_lang"],
        tokenizer_dir,
        int(data_config["vocab_size"]),
    )
    source_tokenizer = SentencePieceTokenizer(source_path)
    target_tokenizer = SentencePieceTokenizer(target_path)
    collator = TranslationCollator(
        source_tokenizer,
        target_tokenizer,
        data_config["source_lang"],
        data_config["target_lang"],
        int(data_config["max_length"]),
    )

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=int(data_config.get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
            collate_fn=collator,
        )
        for split, dataset in datasets.items()
    }
    return loaders, source_tokenizer, target_tokenizer, datasets
