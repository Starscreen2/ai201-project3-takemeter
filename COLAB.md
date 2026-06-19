# Groq zero-shot baseline prompt (Colab Section 5)

Paste this into the classification prompt cell. The model must output **only** the label name.

## Local alternative

You can also run the full pipeline locally (no Colab GPU required on Apple Silicon):

```bash
cd TakeMeter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=your_key
python scripts/train_eval.py          # baseline + fine-tune
# OR if Groq daily limit exhausted:
python scripts/train_eval.py --skip-baseline
python scripts/run_baseline_only.py   # run when quota resets
```

Outputs: `evaluation_results.json`, `confusion_matrix.png`, `finetuned_model/`

---

```
You are a classifier for r/leagueoflegends discourse. Assign exactly one label per post.

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

Respond with only one word: strategy_tip, hot_take, or reaction.
```

## Colab checklist

### Milestone 4 — Baseline
1. Copy the [TakeMeter starter notebook](https://colab.research.google.com/) from the course portal.
2. Runtime → Change runtime type → **T4 GPU**.
3. Colab Secrets → add `GROQ_API_KEY`.
4. Run **Section 1**: set label map to `{"strategy_tip": 0, "hot_take": 1, "reaction": 2}` and upload `data/labeled_dataset.csv`.
5. Run **Section 2**: verify 70/15/15 split and label counts.
6. Run **Section 5**: paste the prompt above; save baseline accuracy and per-class metrics.

### Milestone 5 — Fine-tune
1. Run **Section 3** (DistilBERT fine-tune). Defaults: 3 epochs, lr `2e-5`, batch size 16.
2. Run **Section 4** (test evaluation + confusion matrix image).
3. Run **Section 6** (baseline vs fine-tuned comparison).
4. Download `evaluation_results.json` and `confusion_matrix.png` into this repo root.

If the runtime resets, re-run Sections 1, 2, and 5 before Section 6.

## Label map for Section 1

```python
label2id = {
    "strategy_tip": 0,
    "hot_take": 1,
    "reaction": 2,
}
id2label = {v: k for k, v in label2id.items()}
```
