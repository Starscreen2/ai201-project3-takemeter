#!/usr/bin/env python3
"""Apply human review corrections to labeled_dataset.csv."""

from __future__ import annotations

import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "labeled_dataset.csv"

# Substring match -> corrected label (reviewed against planning.md rules)
CORRECTIONS: list[tuple[str, str, str]] = [
    ("[Item Concept] Galeforce Rework", "hot_take", "Rework concept post, not current-game advice"),
    ("Would Rite of Ruin on Yunara be good?", "hot_take", "Speculative build question without actionable reasoning"),
    ("League and TFT separation", "reaction", "Client/install question, not gameplay advice"),
    ("That's your gotcha? You think at extreme tiers", "hot_take", "Matchmaking opinion, not gameplay tip"),
    ("What is your fps? I know there's a rubber banding bug", "reaction", "Technical client bug report"),
    ("Check your DNS settings and set it to cloudfare", "reaction", "Client troubleshooting, not in-game strategy"),
    ("There is a way to simply do it. Send me a dm", "reaction", "Help request about client setup"),
    ("I think it shows enemy buff timers, which could theoretically", "hot_take", "Opinion about overlay tools"),
    ("How to properly learn and master Olaf", "reaction", "Asking for a learning roadmap, not giving advice"),
    ("Should i have flashed the Syndra Q?", "reaction", "Replay review question"),
    ("How do you deal with Sivir/Karma as a Yunara", "reaction", "Asking for advice after a loss"),
    ("How can I Improve my knowledge in mechanics", "reaction", "General help question"),
    ("Shouldnt minions aggro Malphite here?", "reaction", "Replay confusion question"),
    ("I want to precise that the effect of the \"/mute all\" movement", "hot_take", "Community/meta opinion about muting"),
    ("Hypothetical* if you were to pick a champ and once you move to lane", "hot_take", "Hypothetical balance thought experiment"),
    ("You need that regardless since statistics as in math are often misleading", "hot_take", "Opinion on stats sites/overlays"),
    ("I did say it's reasonable that if everything was banned I do think build/rune helper", "hot_take", "Opinion on overlay fairness"),
    ("That's your gotcha?", "hot_take", "Matchmaking/EOMM argument"),
    ("I would say this season is the season I enjoy the most", "reaction", "Personal enjoyment vent, not balance claim"),
    ("Built stuff isn't unfair imo, runes and items", "hot_take", "Overlay fairness opinion"),
]


def main() -> None:
    rows = list(csv.DictReader(DATA.open(encoding="utf-8")))
    fieldnames = list(rows[0].keys()) if rows else []
    changed = 0

    for row in rows:
        row["pre_labeled"] = "false"
        for needle, new_label, note in CORRECTIONS:
            if needle in row["text"] and row["label"] != new_label:
                row["label"] = new_label
                row["notes"] = f"Review fix: {note}"
                changed += 1
                break

    with DATA.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter

    counts = Counter(row["label"] for row in rows)
    print(f"Reviewed {len(rows)} rows; corrected {changed} labels")
    print(dict(counts))


if __name__ == "__main__":
    main()
