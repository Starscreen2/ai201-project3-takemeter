#!/usr/bin/env python3
"""Pre-label raw Reddit text with Groq for human review."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

LABELS = ("strategy_tip", "hot_take", "reaction")

SYSTEM_PROMPT = """You classify r/leagueoflegends posts/comments into exactly one label.

Labels:
- strategy_tip: Actionable gameplay, build, matchup, or macro advice with specific in-game reasoning.
- hot_take: Bold opinion about champions, balance, or players with confident tone and little or no supporting evidence.
- reaction: Immediate emotional response to a play, patch, or match moment; expresses feeling, not argument.

Edge rules:
- If a post mixes opinion and evidence, label strategy_tip only when removing the opinion still leaves actionable or verifiable game info (patch mechanics, matchup, stats, item timings). Otherwise hot_take.
- Short hype or venting without reasoning is reaction, even if it names a champion.
- Memes or jokes with no gameplay claim are reaction.

Respond with JSON only: {"label": "<one of strategy_tip|hot_take|reaction>", "notes": "<short reason>"}"""


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
    label = payload.get("label", "").strip()
    notes = payload.get("notes", "").strip()
    if label not in LABELS:
        raise ValueError(f"Invalid label: {label}")
    return label, notes


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-label TakeMeter dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw_posts.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "labeled_dataset.csv",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY in your environment")

    client = Groq(api_key=api_key)

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if args.limit:
        rows = rows[: args.limit]

    labeled: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        text = row["text"]
        for attempt in range(3):
            try:
                label, notes = classify(client, text)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    raise
                print(f"Retry {index} after error: {exc}")
                time.sleep(1.5)
        labeled.append(
            {
                "text": text,
                "label": label,
                "notes": notes,
                "pre_labeled": "true",
                "reddit_id": row.get("reddit_id", ""),
                "source": row.get("source", ""),
            }
        )
        if index % 10 == 0:
            print(f"Labeled {index}/{len(rows)}")
        time.sleep(0.35)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["text", "label", "notes", "pre_labeled", "reddit_id", "source"]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    counts: dict[str, int] = {label: 0 for label in LABELS}
    for row in labeled:
        counts[row["label"]] += 1
    print(f"Wrote {len(labeled)} rows to {args.output}")
    print("Label distribution:", counts)


if __name__ == "__main__":
    main()
