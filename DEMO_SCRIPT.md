# Demo video script (~3–5 min)

Use after Colab training. Run `python app.py` or classify in the notebook with confidence scores visible.

## 1. Intro (30 sec)

- "TakeMeter classifies r/leagueoflegends posts into strategy tips, hot takes, and reactions."
- Show repo README evaluation section on screen.

## 2. Live classifications (90 sec)

Run `python scripts/demo_classify.py` and screen-record the output.

Suggested examples from the test set:

| Post excerpt | Label | Confidence |
|--------------|-------|------------|
| "My fanart of Aurora from LoL..." | reaction | 40.5% |
| Riot social-feature rant (long paragraph) | strategy_tip (wrong) | 36.4% |
| "What about a grand-ma character, R?" | reaction | 38.8% |

## 3. Correct prediction deep-dive (45 sec)

Use the fanart post (`reaction`, 40.5%). Explain: "Low confidence, but sharing creative work is emotional/community response, not gameplay advice or a balance argument."

## 4. Wrong prediction deep-dive (45 sec)

Use the Riot social-feature rant (true `hot_take`, predicted `strategy_tip`). Explain: "The model saw long argumentative text with causal reasoning and over-generalized to strategy_tip."

## 5. Evaluation walkthrough (45 sec)

- Baseline vs fine-tuned accuracy
- One per-class F1 row
- Confusion matrix — which pair confuses most
- One sentence reflection on intended vs learned behavior

## 6. Outro (15 sec)

- "Fine-tuning beat the baseline by X points" (use real number)
- Link to GitHub repo
