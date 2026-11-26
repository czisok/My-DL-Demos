from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from transformer_mt.data import PAD_ID, build_data_pipeline
from transformer_mt.model import Transformer, TransformerConfig
from transformer_mt.training import (
    evaluate_bleu,
    evaluate_loss,
    save_checkpoint,
    train_one_epoch,
)
from transformer_mt.utils import (
    JsonlLogger,
    NoamScheduler,
    human_time,
    load_config,
    resolve_device,
    save_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a from-scratch Transformer on OPUS-100 en-zh."
    )
    parser.add_argument("--config", default="configs/quickstart.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--skip-bleu",
        action="store_true",
        help="Skip autoregressive BLEU evaluation to speed up smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    device = resolve_device(config["training"]["device"])
    print(f"device={device}", flush=True)
    loaders, source_tokenizer, target_tokenizer, datasets = build_data_pipeline(
        config["data"],
        output_dir,
        int(config["seed"]),
        int(config["training"]["batch_size"]),
    )
    print(
        "dataset_sizes="
        + ", ".join(f"{name}:{len(data)}" for name, data in datasets.items()),
        flush=True,
    )
    print(
        f"vocab_sizes=source:{source_tokenizer.vocab_size},"
        f"target:{target_tokenizer.vocab_size}",
        flush=True,
    )

    model_config = TransformerConfig(
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
        max_length=int(config["data"]["max_length"]),
        pad_id=PAD_ID,
        **config["model"],
    )
    model = Transformer(model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={parameter_count:,}", flush=True)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=model_config.d_model,
        warmup_steps=int(config["training"]["warmup_steps"]),
        factor=float(config["training"]["learning_rate"]),
    )
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=float(config["training"]["label_smoothing"]),
    )
    use_amp = (
        bool(config["training"].get("mixed_precision", True))
        and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 1
    best_bleu = float("-inf")

    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_bleu = float(checkpoint.get("best_bleu", best_bleu))
        print(f"resumed_from={args.resume} start_epoch={start_epoch}", flush=True)

    logger = JsonlLogger(output_dir / "metrics.jsonl")
    start_time = time.perf_counter()
    epochs = int(config["training"]["epochs"])
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
            accumulation_steps=int(
                config["training"]["gradient_accumulation_steps"]
            ),
            grad_clip=float(config["training"]["grad_clip"]),
            log_every=int(config["training"]["log_every"]),
            epoch=epoch,
            start_time=start_time,
            logger=logger,
        )
        valid_metrics = evaluate_loss(
            model, loaders["validation"], criterion, device
        )
        bleu = float("nan")
        evaluate_every = int(config["training"]["evaluate_bleu_every"])
        if not args.skip_bleu and epoch % evaluate_every == 0:
            bleu = evaluate_bleu(
                model,
                loaders["validation"],
                target_tokenizer,
                device,
                max_length=int(config["data"]["max_length"]),
                max_samples=int(config["training"]["bleu_max_samples"]),
            )
            if bleu > best_bleu:
                best_bleu = bleu
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    scheduler,
                    config,
                    epoch,
                    best_bleu,
                )

        elapsed_seconds = time.perf_counter() - start_time
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_ppl={valid_metrics['perplexity']:.2f} "
            f"valid_acc={valid_metrics['token_accuracy']:.4f} "
            f"bleu={bleu:.2f} elapsed={human_time(elapsed_seconds)}",
            flush=True,
        )
        logger.log(
            {
                "phase": "epoch",
                "epoch": epoch,
                "elapsed_seconds": elapsed_seconds,
                "bleu": bleu,
                "train": train_metrics,
                "validation": valid_metrics,
            }
        )
        if epoch % int(config["training"]["save_every"]) == 0:
            save_checkpoint(
                output_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_bleu,
            )

    if args.skip_bleu:
        save_checkpoint(
            output_dir / "best.pt",
            model,
            optimizer,
            scheduler,
            config,
            epochs,
            best_bleu,
        )
    print(
        f"training_complete checkpoint={output_dir / 'best.pt'} "
        f"elapsed={human_time(time.perf_counter() - start_time)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
