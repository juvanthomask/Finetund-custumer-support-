#!/usr/bin/env python3
"""Train a QLoRA adapter on the prepared customer support dataset.

Why these choices matter:
- QLoRA keeps memory use low enough for a free T4 GPU.
- We use a chat-style prompt template so the model learns the desired support
  response format instead of memorizing raw CSV columns.
- We save only adapter weights, which keeps the project lightweight and easy to
  iterate on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Customer Query:
{input}

### Response:
{response}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TinyLlama with QLoRA.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/qlora_config.yaml"),
        help="Path to the YAML hyperparameter config.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing train.jsonl and val.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for checkpoints, logs, and the final adapter.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit for quick local smoke tests.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional limit for quick local smoke tests.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(path: Path, max_samples: int | None = None):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def add_text_field(dataset):
    def _format_batch(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = []
        for instruction, user_input, response in zip(
            batch["instruction"], batch["input"], batch["response"]
        ):
            texts.append(
                PROMPT_TEMPLATE.format(
                    instruction=instruction,
                    input=user_input,
                    response=response,
                )
            )
        return {"text": texts}

    return dataset.map(_format_batch, batched=True)


def choose_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    return torch.float32


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    final_adapter_dir = output_dir / "final_adapter"

    cuda_available = torch.cuda.is_available()
    compute_dtype = choose_dtype(cfg["qlora"]["bnb_4bit_compute_dtype"])
    use_4bit = bool(cfg["qlora"]["load_in_4bit"]) and cuda_available

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["qlora"]["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(cfg["qlora"]["use_nested_quant"]),
        )
        print("CUDA detected: using 4-bit QLoRA.")
    else:
        print("CUDA not available: falling back to full-precision loading on CPU.")

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": "auto" if cuda_available else None,
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], **model_kwargs)
    model.config.use_cache = False

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg["lora"]["r"]),
        lora_alpha=int(cfg["lora"]["lora_alpha"]),
        lora_dropout=float(cfg["lora"]["lora_dropout"]),
        target_modules=list(cfg["lora"]["target_modules"]),
        bias=cfg["lora"]["bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    train_path = args.data_dir / "train.jsonl"
    val_path = args.data_dir / "val.jsonl"
    train_dataset = load_jsonl_dataset(train_path, args.max_train_samples)
    eval_dataset = load_jsonl_dataset(val_path, args.max_eval_samples)

    # Keep the raw columns because the formatting function builds the final prompt text.
    train_dataset = add_text_field(train_dataset)
    eval_dataset = add_text_field(eval_dataset)

    trainable, total = count_parameters(model)
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable share: {100 * trainable / total:.4f}%")

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=float(cfg["training"]["epochs"]),
        per_device_train_batch_size=int(cfg["training"]["batch_size"]),
        per_device_eval_batch_size=int(cfg["training"]["batch_size"]),
        gradient_accumulation_steps=int(cfg["training"]["gradient_accumulation_steps"]),
        learning_rate=float(cfg["training"]["learning_rate"]),
        warmup_ratio=float(cfg["training"]["warmup_ratio"]),
        lr_scheduler_type=cfg["training"]["lr_scheduler"],
        logging_steps=int(cfg["output"]["logging_steps"]),
        save_steps=int(cfg["output"]["save_steps"]),
        eval_steps=int(cfg["output"]["eval_steps"]),
        save_strategy="steps",
        evaluation_strategy="steps",
        load_best_model_at_end=bool(cfg["output"]["load_best_model_at_end"]),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cuda_available,
        bf16=False,
        report_to=[],
        optim="paged_adamw_8bit" if cuda_available else "adamw_torch",
        save_total_limit=int(cfg["output"]["save_total_limit"]),
        remove_unused_columns=False,
        logging_dir=str(output_dir / "logs"),
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=int(cfg["training"]["max_seq_length"]),
        packing=False,
    )

    trainer.train()

    # Persist the training history so the loss curve can be inspected later.
    history_path = output_dir / "training_logs.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2, default=str)

    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    print(f"Saved final adapter to: {final_adapter_dir}")
    print(f"Saved training logs to: {history_path}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"File error: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - entry point
        print(f"Training failed: {exc}")
        raise SystemExit(1) from exc
