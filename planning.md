# TakeMeter — planning.md

> Written before data collection. Updated during annotation and before any stretch features.

---

## Community

**Chosen community:** r/leagueoflegends

League of Legends is one of the largest gaming subreddits, with constant discussion about patches, champion balance, esports moments, and ranked gameplay. The discourse is text-heavy and varies sharply in quality: some threads offer matchup-specific build advice and wave-management tips, while others are pure emotional reactions to a Baron steal or bold balance opinions stated without evidence.

This community is a strong fit for TakeMeter because regulars already distinguish "actual advice" from "reddit analyst" hot takes and live-game reactions. The labels map onto distinctions players make when scrolling post-game threads, not abstract "good vs bad" quality.

---

## Labels

Three mutually exclusive labels grounded in how r/leagueoflegends participants talk:

### `strategy_tip`

Actionable gameplay, build, matchup, or macro advice with specific in-game reasoning.

**Clear examples:**
1. "Against Zed mid, take W at level 2 and shove the first two waves so you can base for Seeker's before his level 6 all-in window."
2. "If your jungler is pathing bot-side on spawn, hold the crash on wave 3 instead of freezing — you lose tempo if you reset before crab."

**Uncertain example:** "Yasuo mid is unplayable this patch." → Usually `hot_take` unless the post also names patch-specific mechanics (windwall CD, base AD changes) that support the claim.

### `hot_take`

Bold opinion about champions, balance, or players — confident tone, little or no supporting evidence.

**Clear examples:**
1. "Jungle diff is the only reason this game is unplayable below Masters. Iron junglers int every game."
2. "Riot ruined ADC by making every item give ability haste — the role has no identity anymore."

**Uncertain example:** "LeBlanc is the most broken mid in 14.12 because her one-shot combo is unfair." → `hot_take` (emotional framing, no matchup or item detail).

### `reaction`

Immediate emotional response to a play, patch, or match moment — expresses feeling, not argument.

**Clear examples:**
1. "I just watched Faker flash into five people and somehow survive. What was that."
2. "My entire team ff'd at 15 while I was 8/0. I am done with ranked tonight."

**Uncertain example:** "That Baron steal was insane." → `reaction` (moment-focused hype, no gameplay lesson).

---

## Hard Edge Cases

### 1. Opinion + one stat (strategy_tip vs hot_take)

**Ambiguous post:** "LeBron-style take: Jinx is overrated — her win rate in Emerald+ is below 49%."

**Could be:** `hot_take` (accusatory framing) or `strategy_tip` (cites a stat).

**Decision rule:** If the stat is decorative — one number without matchup, item, or macro context — label `hot_take`. If the post uses the stat as part of a reasoning chain (e.g., comparing to another ADC's win rate, patch change, or build path), label `strategy_tip`.

**Annotation decision for LoL variant:** "Jinx is overrated, her Emerald+ win rate is 48.9%." → `hot_take`.

### 2. Short champion complaints (hot_take vs reaction)

**Ambiguous post:** "Why is Riot buffing Yone again this is ridiculous."

**Could be:** `hot_take` (balance opinion) or `reaction` (venting).

**Decision rule:** If the post asserts a balance judgment ("buffing/nerfing is wrong"), label `hot_take`. If it is pure frustration about a specific moment with no claim about balance, label `reaction`.

### 3. Esports hype with player names (reaction vs hot_take)

**Ambiguous post:** "Chovy will never win worlds and it's embarrassing."

**Decision rule:** Player reputation claims without evidence → `hot_take`. Live-match hype ("WHAT A PLAY") → `reaction`.

### Difficult examples encountered during annotation

1. **"Built stuff isn't unfair imo, runes and items are both things riot already assists you with..."** — `hot_take` vs `strategy_tip`. Labeled `hot_take` (opinion on overlays, not matchup advice).

2. **"Teamfights don't work without objectives"** — `strategy_tip` vs `hot_take`. Labeled `strategy_tip` (macro principle).

3. **"EKKO: Ekko needs Locke's E takedown reset..."** — `hot_take` vs `strategy_tip`. Labeled `hot_take` (rework wishlist, not current-game advice).

---

## Data Collection Plan

**Source:** Public posts and top-level comments from r/leagueoflegends via Reddit's public JSON endpoints (`/hot`, `/new`, `/top`, comment threads) and, when blocked, Arctic Shift (posts) + PullPush (comments) archives. No private or authenticated content.

**Target size:** 220 raw examples collected, 200+ retained after filtering duplicates and too-short text.

**Per-label target:** ~67+ examples each (33% per label). No single label above 70% of the final dataset.

**If underrepresented after 200 examples:**
- Search additional listings (`rising`, `controversial`) and manually target posts that match the scarce label.
- Prefer comments in patch-note threads for `strategy_tip`, balance rant threads for `hot_take`, and live-game/esports threads for `reaction`.

**CSV format:** `text`, `label`, `notes`, `pre_labeled`, `reddit_id`, `source`. Single file; Colab splits 70/15/15.

---

## Evaluation Metrics

| Metric | Why it matters for this task |
|--------|------------------------------|
| **Overall accuracy** | Simple headline comparison between fine-tuned model and Groq baseline on the same locked test set. |
| **Per-class precision, recall, F1** | Subjective classes are imbalanced; accuracy alone hides a model that always predicts `reaction`. |
| **Macro-F1** | Equal weight per label so minority classes are not ignored. |
| **Confusion matrix** | Shows directional errors (e.g., `strategy_tip` → `hot_take`), which maps directly to label-boundary problems. |

Accuracy alone is insufficient because a 3-class task with 50% `reaction` posts could reach ~50% accuracy by always guessing the majority class. Macro-F1 and per-class recall reveal whether all three discourse types are learned.

---

## Definition of Success

**Useful classifier threshold:**
- Fine-tuned **macro-F1 ≥ 0.55** on the held-out test set.
- Fine-tuned model **beats Groq zero-shot baseline by ≥ 5 percentage points** on overall accuracy.
- No class with **F1 < 0.40** (if one class is near zero, the boundary or data for that label needs rework).

**"Good enough" for a real community tool:** Moderators could use it to surface `strategy_tip` posts in patch threads with ≥70% precision, accepting lower recall. Not production-ready for auto-moderation — discourse labels are too contextual for that without more data and inter-annotator agreement.

---

## AI Tool Plan

### Label stress-testing (before annotation)

Used Claude/Cursor to generate boundary posts between `strategy_tip` and `hot_take`. Examples that exposed weak boundaries:

| Generated post | Issue | Resolution |
|----------------|-------|------------|
| "Gwen top is broken — she wins every extended trade after level 6." | Opinion vs matchup claim | Rule: needs item/ability timing detail for `strategy_tip` |
| "Just rush Bramble into Steelcaps and Gwen can't trade." | Clear strategy | Kept as `strategy_tip` example |
| "Patch 14.12 ruined jungle, change my mind." | Meme-format hot take | `hot_take` |
| "WHAT A BARON STEAL" | Pure reaction | `reaction` |
| "Azir is useless without a coordinated team." | Generalization | `hot_take` unless team-comp context is specific |

Definitions were tightened before annotating 200 examples.

### Annotation assistance

**Tool:** Groq `llama-3.3-70b-versatile` via `scripts/prelabel.py`.

**Workflow:** Fetch raw text → LLM assigns label + short note → human reviews every row in `labeled_dataset.csv` and corrects mistakes. Column `pre_labeled=true` tracks AI-assisted rows for README disclosure.

**Human override policy:** If I disagree with the pre-label, I change `label` and update `notes` with my reasoning. I do not bulk-accept without reading.

### Failure analysis (after training)

**Plan:** Export misclassified test examples from Colab → paste into an AI tool → ask for systematic patterns (sarcasm, short posts, label-pair confusion) → manually verify each pattern against 5+ examples before writing README analysis.

**Look for:** `strategy_tip` vs `hot_take` confusion on one-stat posts; `reaction` vs `hot_take` on balance vents; over-confidence on short text.

---

## Label Stress-Test Log (Milestone 1)

Boundary posts generated and classified under final rules:

1. "Stop building Liandry's on Brand support, Luden's procs faster in lane trades." → `strategy_tip`
2. "Brand support is griefing every game." → `hot_take`
3. "I screamed when he stole elder at 1 HP." → `reaction`
4. "K'Sante is unkillable with the new items, riot please." → `hot_take`
5. "Into K'Sante top, rush Lord Dominik's second and short-trade with W active." → `strategy_tip`
6. "That pentakill was cinema." → `reaction`
7. "ADC role is dead, change my mind." → `hot_take`
8. "Wave 3 crash bot when jungler starts krugs → level 2 gank timing." → `strategy_tip`

All eight classify cleanly under the final decision rules.
