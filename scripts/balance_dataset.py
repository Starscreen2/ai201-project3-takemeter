#!/usr/bin/env python3
"""Fetch strategy-heavy LoL comments and rebalance labeled dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from groq import Groq

LABELS = ("strategy_tip", "hot_take", "reaction")
PULLPUSH = "https://api.pullpush.io/reddit/search/comment/"
KEYWORDS = (
    "against",
    "build",
    "rush",
    "matchup",
    "wave",
    "item",
    "level",
    "combo",
    "gank",
    "trade",
    "cs",
    "objective",
)
MIN_LEN = 40
MAX_LEN = 512

SYSTEM_PROMPT = """You classify r/leagueoflegends posts/comments into exactly one label.

Labels:
- strategy_tip: Actionable gameplay, build, matchup, or macro advice with specific in-game reasoning.
- hot_take: Bold opinion about champions, balance, or players with confident tone and little or no supporting evidence.
- reaction: Immediate emotional response to a play, patch, or match moment; expresses feeling, not argument.

Respond with JSON only: {"label": "<one of strategy_tip|hot_take|reaction>", "notes": "<short reason>"}"""


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def fetch_keyword_comments(subreddit: str, keyword: str, size: int = 40) -> list[dict]:
    params = {
        "subreddit": subreddit,
        "q": keyword,
        "size": size,
        "sort": "desc",
        "sort_type": "created_utc",
    }
    response = requests.get(PULLPUSH, params=params, timeout=30)
    response.raise_for_status()
    rows = []
    for comment in response.json().get("data", []):
        body = clean(comment.get("body", ""))
        if body in {"[deleted]", "[removed]"}:
            continue
        if MIN_LEN <= len(body) <= MAX_LEN:
            rows.append(
                {
                    "text": body,
                    "source": f"pullpush_q:{keyword}",
                    "reddit_id": comment.get("id", ""),
                }
            )
    return rows


def classify(client: Groq, text: str) -> tuple[str, str]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    payload = json.loads(response.choices[0].message.content)
    label = payload["label"].strip()
    notes = payload.get("notes", "").strip()
    if label not in LABELS:
        raise ValueError(label)
    return label, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "labeled_dataset.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "labeled_dataset.csv",
    )
    parser.add_argument("--per-label", type=int, default=70)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY")

    client = Groq(api_key=api_key)
    existing = list(csv.DictReader(args.input.open(encoding="utf-8")))
    seen_ids = {row.get("reddit_id", "") for row in existing}
    seen_text = {row["text"] for row in existing}

    extra_rows: list[dict] = []
    for keyword in KEYWORDS:
        for row in fetch_keyword_comments("leagueoflegends", keyword):
            if row["reddit_id"] in seen_ids or row["text"] in seen_text:
                continue
            label, notes = classify(client, row["text"])
            row.update({"label": label, "notes": notes, "pre_labeled": "true"})
            extra_rows.append(row)
            seen_ids.add(row["reddit_id"])
            seen_text.add(row["text"])
            time.sleep(0.35)
        time.sleep(0.5)

    combined = existing + extra_rows
    by_label: dict[str, list[dict]] = {label: [] for label in LABELS}
    for row in combined:
        if row["label"] in by_label:
            by_label[row["label"]].append(row)

    random.seed(42)
    balanced: list[dict] = []
    for label in LABELS:
        pool = by_label[label]
        random.shuffle(pool)
        balanced.extend(pool[: args.per_label])

    fieldnames = ["text", "label", "notes", "pre_labeled", "reddit_id", "source"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(balanced)

    counts = {label: sum(1 for row in balanced if row["label"] == label) for label in LABELS}
    print(f"Wrote {len(balanced)} balanced rows to {args.output}")
    print(counts)


if __name__ == "__main__":
    main()
