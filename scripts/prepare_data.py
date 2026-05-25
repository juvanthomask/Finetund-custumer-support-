#!/usr/bin/env python3
"""Prepare the customer support dataset for LoRA/QLoRA fine-tuning.

Why this exists:
- Fine-tuning works best when the raw dataset is converted into a consistent
  instruction format.
- Stratified splits keep category proportions roughly stable across train,
  validation, and IID evaluation.
- A separate OOD file gives us a small sanity-check set for edge cases that do
  not look like the training distribution.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_INSTRUCTION = (
    "You are a helpful customer support agent. Answer the following customer query."
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare JSONL splits for training.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing the downloaded Kaggle CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory where processed JSONL files will be written.",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional explicit path to the raw CSV. If omitted, the first CSV in raw-dir is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def find_csv(raw_dir: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"CSV not found: {explicit_path}")
        return explicit_path

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Download the Kaggle dataset there first."
        )
    return csv_files[0]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_columns = {"instruction", "category", "intent", "response"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    df = df.copy()
    df["instruction"] = df["instruction"].map(clean_text)
    df["category"] = df["category"].map(clean_text)
    df["intent"] = df["intent"].map(clean_text)
    df["response"] = df["response"].map(clean_text)
    df = df[df["instruction"].ne("") & df["response"].ne("") & df["category"].ne("")].reset_index(
        drop=True
    )
    return df


def build_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.itertuples(index=False):
        records.append(
            {
                "instruction": BASE_INSTRUCTION,
                "input": row.instruction,
                "response": row.response,
                "category": row.category,
                "intent": row.intent,
            }
        )
    return records


def stratified_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError("Need at least 3 rows to create train/val/eval splits.")

    # First hold out 20% for validation + IID eval.
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["category"],
    )

    # Split the held-out 20% into two 10% portions.
    val_df, iid_eval_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["category"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), iid_eval_df.reset_index(
        drop=True
    )


def ood_examples() -> list[dict]:
    # These are deliberately outside the training distribution: mixed intents,
    # angry phrasing, vague requests, and policy-adjacent questions.
    examples = [
        ("I am furious. My order is late and nobody is responding.", "I’m sorry for the delay. I can help check the order status and next steps.", "shipping", "late_delivery"),
        ("Can you change the address and also refund the extra charge?", "I can help with both. Please share the order number and I’ll check whether the address change is still possible and review the charge.", "orders", "multi_request"),
        ("This is ridiculous. I want my money back right now.", "I’m sorry for the frustration. Please share the order details so I can review the refund eligibility.", "billing", "refund_request"),
        ("Where is it?", "Could you share your order number so I can look up the shipment status?", "shipping", "vague_status"),
        ("My tracking stopped updating three days ago and the package was supposed to arrive yesterday.", "I’m sorry about that. I can check the tracking status and see whether the carrier has an updated scan.", "shipping", "tracking_issue"),
        ("I got charged twice for one purchase.", "I understand the concern. Please share the order number so I can investigate the duplicate charge.", "billing", "duplicate_charge"),
        ("My account is locked and I cannot reset my password.", "I can help with that. Let’s verify the account details and then we can reset access.", "account", "locked_account"),
        ("Do you have a human agent? I need someone to call me.", "I can help here. Please describe the issue and I’ll guide you through the next steps or escalation options.", "support", "human_request"),
        ("I want to cancel, but I already received part of the shipment.", "Please share the order number. I can check whether cancellation, return, or replacement is the right next step.", "orders", "cancel_partial"),
        ("The coupon did not work and now the total changed.", "I can help review the promo code issue. Please send the code and the order details.", "billing", "promo_issue"),
        ("My subscription renewed today even though I meant to cancel last week.", "I can review the renewal and cancellation timing. Please share the account email so I can check the subscription history.", "subscriptions", "renewal_cancellation"),
        ("I’m missing one item from a bundled order.", "I’m sorry about the missing item. Please share the order number and I’ll help review the shipment contents.", "shipping", "missing_item"),
        ("Your app keeps crashing whenever I try to pay.", "I’m sorry for the trouble. Please share your device and payment step so I can help isolate the issue.", "technical", "payment_crash"),
        ("Can I return something that was a gift?", "Yes, I can check the return policy for gifted items if you share the order details.", "returns", "gift_return"),
        ("I need an invoice with my company tax ID.", "I can help with the invoice request. Please share the order number and billing details.", "billing", "invoice_request"),
        ("My package says delivered but nothing is here.", "I’m sorry about that. I can help investigate the delivery scan and advise on the next step.", "shipping", "delivered_not_received"),
        ("Stop sending me marketing emails.", "I can help with email preferences. Please confirm the email address on the account.", "account", "unsubscribe"),
        ("Why was my return rejected?", "I can review the return decision. Please share the order number and the return reason if available.", "returns", "return_rejected"),
        ("I changed my mind after ordering two hours ago.", "I can check whether the order is still cancellable if you share the order number.", "orders", "cancellation_window"),
        ("Your support page says one thing and the checkout says another.", "I can help clarify the mismatch. Please share the page or message you saw and I’ll review it with you.", "technical", "policy_conflict"),
        ("Can you apply the discount retroactively?", "I can check whether a price adjustment is available. Please share the order number and the coupon details.", "billing", "retro_discount"),
        ("The item arrived broken and the box was wet.", "I’m sorry for the damage. Please share photos and the order number so I can help with a replacement or refund review.", "returns", "damaged_item"),
        ("I entered the wrong email and now I can’t log in.", "I can help update the account email after verifying ownership. Please share the current order or account details.", "account", "wrong_email"),
        ("Do you have stock in blue, size medium, and can you ship express?", "I can check stock and shipping options if you share the product name or SKU.", "inventory", "stock_and_shipping"),
        ("I already contacted support yesterday and nothing changed.", "I’m sorry for the follow-up. Please share the ticket number so I can review the previous case.", "support", "case_followup"),
        ("My refund says completed, but the money is not in my bank.", "I can help check refund timing and bank processing. Please share the order number.", "billing", "refund_pending"),
        ("Why do I need to verify my identity again?", "I can explain the verification step and help you complete it if you prefer.", "account", "identity_verification"),
        ("This order was never supposed to be split into two shipments.", "I can review why the order was split and whether any consolidation is possible.", "shipping", "split_shipment"),
        ("I want to talk about privacy and my stored data.", "I can help with privacy questions and point you to the account data request process.", "policy", "privacy_request"),
        ("The chatbot keeps repeating the same answer.", "I’m sorry for the loop. Please tell me the exact issue and I’ll try a different approach.", "support", "bot_loop"),
    ]

    records = []
    for query, response, category, intent in examples:
        records.append(
            {
                "instruction": BASE_INSTRUCTION,
                "input": query,
                "response": response,
                "category": category,
                "intent": intent,
            }
        )
    return records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def avg_length(records: list[dict], key: str) -> float:
    if not records:
        return 0.0
    total = sum(len(record[key].split()) for record in records)
    return total / len(records)


def print_stats(split_name: str, records: list[dict]) -> None:
    category_counts = Counter(record["category"] for record in records)
    print(f"\n{split_name}: {len(records)} samples")
    print(f"  avg input length: {avg_length(records, 'input'):.2f} words")
    print(f"  avg response length: {avg_length(records, 'response'):.2f} words")
    print("  category distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"    - {category}: {count}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    csv_path = find_csv(args.raw_dir, args.csv_path)
    print(f"Loading dataset from: {csv_path}")
    df = load_dataset(csv_path)

    train_df, val_df, iid_eval_df = stratified_split(df, seed=args.seed)
    train_records = build_records(train_df)
    val_records = build_records(val_df)
    iid_eval_records = build_records(iid_eval_df)
    ood_records = ood_examples()

    output_dir = args.output_dir
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "val.jsonl", val_records)
    write_jsonl(output_dir / "eval" / "iid_eval.jsonl", iid_eval_records)
    write_jsonl(output_dir / "eval" / "ood_eval.jsonl", ood_records)

    all_records = train_records + val_records + iid_eval_records
    print("\nDataset statistics")
    print("==================")
    print(f"Total raw samples: {len(df)}")
    print(f"Processed samples: {len(all_records)}")
    print_stats("Train", train_records)
    print_stats("Validation", val_records)
    print_stats("IID eval", iid_eval_records)
    print_stats("OOD eval", ood_records)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - command line entry point
        print(f"Error: {exc}")
        raise SystemExit(1) from exc
