#!/usr/bin/env python3
"""Train DistilBERT, run Groq baseline, and export evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from groq import Groq
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LABELS = ("strategy_tip", "hot_take", "reaction")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}

BASELINE_PROMPT = """You are a classifier for r/leagueoflegends discourse. Assign exactly one label per post.

Labels:
- strategy_tip — Actionable gameplay, build, matchup, or macro advice with specific in-game reasoning.
- hot_take — Bold opinion about champions, balance, or players; confident tone with little or no supporting evidence.
- reaction — Immediate emotional response to a play, patch, or match moment; expresses feeling, not argument.

Decision rules:
- Opinion plus one bare stat without matchup or item context → hot_take.
- Actionable advice naming abilities, items, wave states, or timings → strategy_tip.
- Live-moment hype or venting without a balance claim → reaction.

Post:
{text}

Respond with only one word: strategy_tip, hot_take, or reaction."""


def parse_label(response: str) -> str | None:
    text = response.strip().lower()
    for label in LABELS:
        if label in text:
            return label
    return None


def run_baseline(
    texts: list[str],
    client: Groq,
    cache_path: Path | None = None,
    max_retries: int = 8,
) -> list[str]:
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    predictions: list[str] = []
    for index, text in enumerate(texts, start=1):
        if text in cache:
            predictions.append(cache[text])
            continue

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    temperature=0,
                    messages=[
                        {"role": "user", "content": BASELINE_PROMPT.format(text=text)},
                    ],
                )
                label = parse_label(response.choices[0].message.content or "")
                prediction = label or "hot_take"
                break
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if "429" in message or "rate_limit" in message.lower():
                    wait_seconds = 90
                    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", message)
                    if match:
                        minutes = int(match.group(1) or 0)
                        seconds = float(match.group(2))
                        wait_seconds = int(minutes * 60 + seconds) + 5
                    print(f"Rate limited at {index}/{len(texts)}; waiting {wait_seconds}s...")
                    if cache_path:
                        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
                    time.sleep(wait_seconds)
                    if attempt == max_retries - 1:
                        raise
                    continue
                raise

        predictions.append(prediction)
        cache[text] = prediction
        if cache_path:
            cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        if index % 5 == 0:
            print(f"Baseline {index}/{len(texts)}")
        time.sleep(0.5)
    return predictions


def metrics_dict(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=list(LABELS), average="macro"),
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
            }
            for label in LABELS
        },
    }


def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    def tokenize(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    return dataset.map(tokenize, batched=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "labeled_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    frame = pd.read_csv(args.data)
    frame = frame[frame["label"].isin(LABELS)].reset_index(drop=True)
    print(f"Loaded {len(frame)} labeled rows")

    train_df, holdout_df = train_test_split(
        frame,
        test_size=0.30,
        random_state=42,
        stratify=frame["label"],
    )
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=0.50,
        random_state=42,
        stratify=holdout_df["label"],
    )
    print(f"Split sizes train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    label_counts = train_df["label"].value_counts().to_dict()
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (len(LABELS) * label_counts.get(label, 1)) for label in LABELS],
        dtype=torch.float,
    )
    print("Class weights:", dict(zip(LABELS, class_weights.tolist())))

    baseline_metrics = None
    baseline_predictions: list[str] = []
    if not args.skip_baseline:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("Set GROQ_API_KEY for baseline evaluation")
        client = Groq(api_key=api_key)
        baseline_predictions = run_baseline(test_df["text"].tolist(), client)
        baseline_metrics = metrics_dict(test_df["label"].tolist(), baseline_predictions)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def to_dataset(df: pd.DataFrame) -> Dataset:
        return Dataset.from_dict(
            {
                "text": df["text"].tolist(),
                "label": [LABEL2ID[label] for label in df["label"].tolist()],
            }
        )

    train_dataset = tokenize_dataset(to_dataset(train_df), tokenizer)
    val_dataset = tokenize_dataset(to_dataset(val_df), tokenizer)
    test_dataset = tokenize_dataset(to_dataset(test_df), tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "model"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        logging_steps=20,
        report_to=[],
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro"),
        }

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning DistilBERT...")
    trainer.train()

    predictions = trainer.predict(test_dataset)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    finetuned_predictions = [ID2LABEL[pred_id] for pred_id in pred_ids]
    finetuned_metrics = metrics_dict(test_df["label"].tolist(), finetuned_predictions)

    cm = confusion_matrix(
        test_df["label"].tolist(),
        finetuned_predictions,
        labels=list(LABELS),
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right")
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    cm_path = args.output_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    wrong_examples = []
    for text, true_label, pred_label in zip(
        test_df["text"],
        test_df["label"],
        finetuned_predictions,
        strict=True,
    ):
        if true_label != pred_label:
            wrong_examples.append(
                {
                    "text": text,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                }
            )

    sample_rows = []
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1)
    for text, pred_label, prob_row in zip(
        test_df["text"].head(5),
        finetuned_predictions[:5],
        probs[:5],
        strict=True,
    ):
        confidence = float(prob_row[LABEL2ID[pred_label]])
        sample_rows.append(
            {
                "text": text,
                "predicted_label": pred_label,
                "confidence": round(confidence, 4),
            }
        )

    results = {
        "split_sizes": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
        },
        "baseline": baseline_metrics,
        "finetuned": finetuned_metrics,
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": cm.tolist(),
        },
        "wrong_examples": wrong_examples[:10],
        "sample_classifications": sample_rows,
        "hyperparameters": {
            "base_model": "distilbert-base-uncased",
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
        },
    }

    results_path = args.output_dir / "evaluation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    export_dir = args.output_dir / "finetuned_model"
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    print(f"Wrote {results_path}")
    print(f"Wrote {cm_path}")
    print("Fine-tuned accuracy:", finetuned_metrics["accuracy"])
    if baseline_metrics:
        print("Baseline accuracy:", baseline_metrics["accuracy"])


if __name__ == "__main__":
    main()
