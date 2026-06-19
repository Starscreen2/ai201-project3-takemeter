#!/usr/bin/env python3
"""Run Groq baseline only and merge into evaluation_results.json."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from train_eval import BASELINE_PROMPT, LABELS, metrics_dict, parse_label, run_baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "labeled_dataset.csv",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "evaluation_results.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "baseline_cache.json",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY")

    frame = pd.read_csv(args.data)
    frame = frame[frame["label"].isin(LABELS)].reset_index(drop=True)
    _, holdout_df = train_test_split(
        frame, test_size=0.30, random_state=42, stratify=frame["label"]
    )
    _, test_df = train_test_split(
        holdout_df, test_size=0.50, random_state=42, stratify=holdout_df["label"]
    )

    client = Groq(api_key=api_key)
    predictions = run_baseline(test_df["text"].tolist(), client, cache_path=args.cache)
    baseline_metrics = metrics_dict(test_df["label"].tolist(), predictions)

    results = {}
    if args.results.exists():
        results = json.loads(args.results.read_text(encoding="utf-8"))
    results["baseline"] = baseline_metrics

    args.results.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Baseline accuracy:", baseline_metrics["accuracy"])
    print(f"Updated {args.results}")


if __name__ == "__main__":
    main()
