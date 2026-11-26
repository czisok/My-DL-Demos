from __future__ import annotations

import argparse
import sys

from transformer_mt.inference import (
    load_translation_checkpoint,
    translate_texts,
)
from transformer_mt.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate English text to Chinese with a trained model."
    )
    parser.add_argument(
        "--checkpoint", default="outputs/quickstart/best.pt"
    )
    parser.add_argument(
        "--text",
        action="append",
        help="English sentence. Repeat this option for batch translation.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = args.text
    if not texts:
        if sys.stdin.isatty():
            texts = [input("English> ").strip()]
        else:
            texts = [line.strip() for line in sys.stdin if line.strip()]
    if not texts:
        raise SystemExit("No input text was provided.")

    device = resolve_device(args.device)
    model, source_tokenizer, target_tokenizer, config = (
        load_translation_checkpoint(args.checkpoint, device)
    )
    max_length = args.max_length or int(config["data"]["max_length"])
    translations = translate_texts(
        model,
        texts,
        source_tokenizer,
        target_tokenizer,
        device,
        max_length,
    )
    for source, target in zip(texts, translations, strict=True):
        print(f"EN: {source}")
        print(f"ZH: {target}")


if __name__ == "__main__":
    main()
