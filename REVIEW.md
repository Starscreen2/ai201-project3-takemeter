# Label review checklist

The dataset in `data/labeled_dataset.csv` was **pre-labeled by Groq** (`pre_labeled=true`). Before submitting, review every row:

1. Open `data/labeled_dataset.csv` in Excel, Google Sheets, or a CSV editor.
2. For each row, read `text` and confirm `label` matches [planning.md](planning.md) definitions.
3. Update `notes` when you override a label with your reasoning.
4. Optional: set `pre_labeled` to `false` after review.

## Quick filter tips

- Sort by `label` and scan `strategy_tip` first (smallest class, highest error risk).
- Search for posts with stats but no matchup detail → usually `hot_take`.
- Short hype/vent posts → usually `reaction`.

## Current distribution (pre-review)

| Label | Count | % |
|-------|------:|--:|
| strategy_tip | 30 | 13.2% |
| hot_take | 77 | 33.8% |
| reaction | 121 | 53.1% |
| **Total** | **228** | 100% |

No label exceeds 70%. Total exceeds the 200 minimum. Consider adding more `strategy_tip` examples if you have time.

## Three difficult examples to verify

1. **"Yasuo mid is unplayable this patch."** → `hot_take` (no mechanics cited)
2. **"Why is Riot buffing Yone again this is ridiculous."** → `hot_take` (balance judgment)
3. **"That Baron steal was insane."** → `reaction` (moment hype)

If your dataset contains these patterns, apply the same rules consistently.
