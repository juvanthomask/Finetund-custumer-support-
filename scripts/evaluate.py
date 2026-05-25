#!/usr/bin/env python3
"""Evaluate the fine-tuned adapter against the base model.

Why this design:
- We compare against the same base model to measure whether the adapter is
  actually helping rather than just generating fluent text.
- IID and OOD splits answer two different questions: does the model learn the
  dataset, and does it behave sensibly on edge cases?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import yaml
from bert_score import score as bertscore
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Customer Query:
{input}

### Response:"""
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned adapter.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "qlora_config.yaml",
        help="Path to the YAML hyperparameter config.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "eval",
        help="Directory containing iid_eval.jsonl and ood_eval.jsonl.",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "final_adapter",
        help="Directory containing the saved LoRA adapter.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "eval_results.json",
        help="Where to store the evaluation results.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional sample cap for fast smoke tests.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=96,
        help="Maximum number of tokens to generate per response.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    dataset = load_dataset("json", data_files=str(path), split="train")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dataset[i] for i in range(len(dataset))]


def build_prompt(example: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example["input"],
    )


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_base_components(model_name: str):
    device = choose_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_finetuned_model(model_name: str, adapter_dir: Path):
    base_model, tokenizer = load_base_components(model_name)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    device = choose_device()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated[0][encoded["input_ids"].shape[-1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    return text


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_1, rouge_2, rouge_l = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge_1.append(scores["rouge1"].fmeasure)
        rouge_2.append(scores["rouge2"].fmeasure)
        rouge_l.append(scores["rougeL"].fmeasure)

    _, _, bert_f1 = bertscore(
        predictions,
        references,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )

    return {
        "rouge_1": float(mean(rouge_1) if rouge_1 else 0.0),
        "rouge_2": float(mean(rouge_2) if rouge_2 else 0.0),
        "rouge_l": float(mean(rouge_l) if rouge_l else 0.0),
        "bertscore_f1": float(bert_f1.mean().item() if len(bert_f1) else 0.0),
        "avg_response_length": float(mean(len(pred.split()) for pred in predictions) if predictions else 0.0),
    }


def evaluate_split(model, tokenizer, records: list[dict[str, Any]], max_new_tokens: int):
    predictions, references = [], []
    for example in records:
        prompt = build_prompt(example)
        prediction = generate_response(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        predictions.append(prediction)
        references.append(example["response"])

    return predictions, references, compute_metrics(predictions, references)


def print_table(results: dict[str, Any]) -> None:
    rows = []
    for split_name, split_results in results.items():
        for metric_name in ["rouge_1", "rouge_2", "rouge_l", "bertscore_f1", "avg_response_length"]:
            rows.append(
                (
                    split_name,
                    metric_name,
                    split_results["baseline"][metric_name],
                    split_results["finetuned"][metric_name],
                    split_results["finetuned"][metric_name] - split_results["baseline"][metric_name],
                )
            )

    header = f"{'Split':<10} {'Metric':<18} {'Baseline':>12} {'Finetuned':>12} {'Delta':>12}"
    print("\n" + header)
    print("-" * len(header))
    for split_name, metric_name, baseline, finetuned, delta in rows:
        print(f"{split_name:<10} {metric_name:<18} {baseline:12.4f} {finetuned:12.4f} {delta:12.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    iid_path = args.data_dir / "iid_eval.jsonl"
    ood_path = args.data_dir / "ood_eval.jsonl"
    iid_records = load_records(iid_path, args.max_samples)
    ood_records = load_records(ood_path, args.max_samples)

    base_model, tokenizer = load_base_components(cfg["model_name"])
    finetuned_model, _ = load_finetuned_model(cfg["model_name"], args.adapter_dir)

    results: dict[str, Any] = {}
    for split_name, records in [("iid_eval", iid_records), ("ood_eval", ood_records)]:
        _, _, baseline_metrics = evaluate_split(
            base_model,
            tokenizer,
            records,
            max_new_tokens=args.max_new_tokens,
        )
        _, _, finetuned_metrics = evaluate_split(
            finetuned_model,
            tokenizer,
            records,
            max_new_tokens=args.max_new_tokens,
        )
        results[split_name] = {
            "baseline": baseline_metrics,
            "finetuned": finetuned_metrics,
        }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_table(results)
    print(f"\nSaved evaluation results to: {args.output_file}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        print(f"File error: {exc}")
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - entry point
        print(f"Evaluation failed: {exc}")
        raise SystemExit(1) from exc
