#!/usr/bin/env python3
"""Print sample classifications for demo video recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "finetuned_model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_results.json",
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    samples = results.get("sample_classifications", [])
    wrong = results.get("wrong_examples", [])[:2]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    print("=== SAMPLE CLASSIFICATIONS (for demo) ===\n")
    for row in samples:
        print(f"TEXT: {row['text'][:200]}...")
        print(f"PREDICTED: {row['predicted_label']} ({row['confidence']:.1%})\n")

    print("=== MISCLASSIFICATION EXAMPLES ===\n")
    for row in wrong:
        encoded = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        pred = model.config.id2label[pred_id]
        conf = float(probs[pred_id])
        print(f"TEXT: {row['text'][:200]}...")
        print(f"TRUE: {row['true_label']} | PREDICTED: {pred} ({conf:.1%})\n")


if __name__ == "__main__":
    main()
