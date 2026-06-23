# YouTube Thumbnail Scorer — Methodology & History

## Goal

Build a tool that takes a YouTube thumbnail and returns a score from **0 to 20**, reflecting how likely it is to drive above-average views relative to the channel's baseline.

---

## Data

### Scraping
- Videos scraped from ~18 YouTube channels across two niches: **AI/Tech** and **Business**
- ~1,243 videos total, thumbnails downloaded locally
- Performance metric: `performance_score = views / subscribers` — a ratio that normalises for channel size

### Vision API Labelling (`auto_label_vision.py`)
Google Cloud Vision API was run on every thumbnail to extract:
| Feature | Values |
|---|---|
| `visage_present` | True / False |
| `nb_personnes` | 0 / 1 / 2 / 3+ |
| `expression` | Neutre / Sourire / Surprise / Intense / Aucun |
| `texte_present` | True / False |
| `texte_contenu` | Raw text string |
| `couleur_dominante` | Neutre / Chaud / Froid |
| `contraste` | Faible / Moyen / Élevé |
| `fond` | Uni / Chargé / Flou |

---

## Scoring Versions

### v1 — Simple rule-based (`score_thumbnails.py` original)
**Approach:** Assign fixed point values per feature, sum to 20.

**Problem:** Double-counted face + nb_personnes (6 pts for "1 person with face"). Weights were guesses, not data-driven. Also gave 1 pt to "0 people" which was noise.

---

### v2 — Ridge regression (`score_regression.py`)
**Approach:** Train a Ridge regression model to predict `log(performance_score)`, then scale predictions to 0–20.

**Problem:** Weights were counter-intuitive. `high_contrast = -0.33` (negative despite all top-50 having it), `intense_expression = +0.45` (top weight, which makes no sense). Cross-channel raw performance as the target confused the model. Cross-validation R² was low.

---

### v3 — GradientBoostingClassifier (`score_classifier.py`)
**Approach:** Classify each video as "top 25% performer" or not. Use the predicted probability × 20 as the score.

**Result:** Cross-validation ROC-AUC = **0.508** ≈ 0.5 — indistinguishable from random. The classifier failed because:
- Dataset too small for a tree-based ensemble to generalise
- The target (top 25% across all channels) is dominated by channel identity, not thumbnail features

**Symptom:** A 2.4M-view viral video got 3.0/20; a low-performing video got 19.8/20.

---

### v4 — Empirical multiplier weights (`score_thumbnails.py` v4)
**Approach:** A prior session identified raw performance multipliers per feature (text=9.5x, bg_busy=5.5x, face=2.8x). Assign points proportional to log(multiplier), sum to 20.

**Problem:** These multipliers were computed as **cross-channel averages**, so they captured channel-type differences, not thumbnail quality. Pearson correlation with raw performance: **+0.04** (flat).

---

### v5 — Within-channel validated (`score_thumbnails.py` current)
**Approach:**
1. Z-score `performance_score` within each channel → `perf_norm`
   - Removes channel-size confound
   - Now measures: "did this video beat its own channel's average?"
2. Compute mean `perf_norm` for each feature present vs. absent
3. Run Ridge regression (RidgeCV, alpha=50) on all features vs `perf_norm`
4. Use only features that are **consistently positive in both analyses**

**Key findings from within-channel analysis:**

| Feature | WC-Normalised diff | Direction |
|---|---|---|
| Text present | **+0.192** | Strong positive |
| 1 person | +0.065 | Positive |
| Neutral expression | +0.064 | Positive |
| Cold colors | +0.054 | Positive |
| Smile expression | ~+0.065 | Positive |
| **Busy background** | **-0.079** | **Negative — was confounded** |
| **High contrast** | **-0.117** | **Negative — was confounded** |

**Why busy background flipped:** The raw cross-channel analysis found 5.5x multiplier for busy backgrounds. But after within-channel normalisation it turned negative. This means: channels that tend to use busy backgrounds simply perform worse overall — the background wasn't driving anything, it was a marker of channel type.

**Why high contrast flipped:** Same reason. Channels with typically high-contrast thumbnails are a certain content type that underperforms within its own baseline.

**Final score formula (max = 20):**

| Feature | Max pts | Logic |
|---|---|---|
| Text (length-scaled) | 0–10 | Long (>50 chars)=10, Medium (>20)=8.5, Short=7, None=0 |
| Persons / Face | 0–4 | 1 person=4, 2 people=2, 3+=0, 0=0 |
| Expression | 0–3 | Neutral=3, Smile=2.5, else=0 |
| Colors | 0–3 | Cold=3, Neutral=1.5, Warm=0 |

**Validation:**
- Score bands vs `perf_norm`: 0–5 → −0.16, 5–10 → −0.12, 10–15 → −0.06, 15–20 → +0.05
- Monotonically correct: higher score = above-average performance within channel
- Within-channel Pearson: **+0.092** (vs +0.04 for v4)

---

## Fundamental Limitation

Thumbnail features explain only a small fraction of variance in performance. Most of the variance comes from:
- Content topic and virality
- Upload timing
- YouTube algorithm promotion
- Channel authority and audience loyalty

The score is a **best-practice guide** based on what the data supports, not a predictor of individual video success. A high score means the thumbnail follows patterns statistically associated with above-average performance — it does not guarantee results.

---

## Files

| File | Purpose |
|---|---|
| `scraper.py` | YouTube API scraping |
| `auto_label_vision.py` | Google Vision API labelling |
| `score_thumbnails.py` | Current scorer (v5) |
| `score_regression.py` | v2 Ridge regression (archived) |
| `score_classifier.py` | v3 GradientBoosting (archived) |
| `data/videos_enriched.csv` | Base dataset with Vision API labels |
| `data/videos_scored_v5.csv` | Dataset with current `thumbnail_score` column |
| `data/thumbnail_weights_v5.json` | Scoring weights + validation metrics |
| `notebooks/01_dataset_exploration.ipynb` | Exploration and visualisation |
