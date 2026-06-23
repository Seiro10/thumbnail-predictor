"""
Thumbnail Scoring System v5 — Within-channel validated weights

Key methodology:
  Performance scores are z-scored WITHIN each channel before analysis,
  removing the channel-size confound. Feature weights come from both
  per-feature mean-difference analysis and Ridge regression on those
  normalized scores.

Validated signals (within-channel normalized diff):
  Text present:   +0.192  ← strongest by far
  1 person:       +0.065
  Expr neutral:   +0.064
  Cold colors:    +0.054
  Expr smile:     ~+0.065 (regression parity with neutral)

Debunked signals (looked good cross-channel, confounded):
  Busy background: -0.079  (channels using it just perform worse overall)
  High contrast:   -0.117  (same confound)

Score breakdown (max 20):
  Text       0-10  (proportional to length: long=10, medium=8.5, short=7)
  Persons    0-4   (1=4, 2=2, 3+=0, 0=0)
  Expression 0-3   (neutral=3, smile=2.5, else=0)
  Colors     0-3   (cold=3, neutral=1.5, warm=0)
"""
import pandas as pd
import numpy as np
import json

df = pd.read_csv('data/videos_enriched.csv')
df['text_length'] = df['texte_contenu'].fillna('').str.len()


def score_thumbnail(row):
    score = 0.0

    # TEXT (0-10 pts) — within-channel diff +0.192, strongest signal
    tl = row['text_length']
    if row['texte_present']:
        if tl > 50:
            score += 10.0
        elif tl > 20:
            score += 8.5
        else:
            score += 7.0

    # PERSONS / FACE (0-4 pts) — within-channel diff +0.065
    nb = row['nb_personnes']
    if nb == 1:
        score += 4.0
    elif nb == 2:
        score += 2.0

    # EXPRESSION (0-3 pts) — neutral & smile both positive
    expr = row['expression']
    if expr == 'Neutre':
        score += 3.0
    elif expr == 'Sourire':
        score += 2.5

    # COLORS (0-3 pts) — cold positive, neutral mildly positive, warm=0
    color = row['couleur_dominante']
    if color == 'Froid':
        score += 3.0
    elif color == 'Neutre':
        score += 1.5

    return round(score, 1)


df['thumbnail_score'] = df.apply(score_thumbnail, axis=1)

# Within-channel normalized performance for validation
df['perf_norm'] = df.groupby('channel_name')['performance_score'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

print("=" * 65)
print("THUMBNAIL SCORING v5 — Within-channel validated")
print("=" * 65)
print(f"\nScore range:  {df['thumbnail_score'].min():.1f} – {df['thumbnail_score'].max():.1f} / 20")
print(f"Mean:         {df['thumbnail_score'].mean():.1f}")
print(f"Median:       {df['thumbnail_score'].median():.1f}")

# Pearson on raw performance
corr_raw  = df['thumbnail_score'].corr(df['performance_score'])
# Pearson on within-channel normalized performance
corr_norm = df['thumbnail_score'].corr(df['perf_norm'])
print(f"\nCorrelation (raw performance):               {corr_raw:+.4f}")
print(f"Correlation (within-channel normalized):     {corr_norm:+.4f}  ← correct metric")

# Score band validation
print("\nValidation by score band (within-channel normalized avg):")
print(f"{'Band':<8} {'WC-Norm Avg':>14} {'n':>6}")
print("-" * 32)
for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 20)]:
    sub = df[(df['thumbnail_score'] >= lo) & (df['thumbnail_score'] < hi)]
    if len(sub):
        print(f"{lo:2d}-{hi:<4d}   {sub['perf_norm'].mean():+12.4f}   {len(sub):6d}")

# Top 10
print("\nTop 10 scoring thumbnails:")
cols = ['title', 'channel_name', 'thumbnail_score', 'perf_norm', 'performance_score']
top10 = df.nlargest(10, 'thumbnail_score')[cols]
for _, r in top10.iterrows():
    title = r['title'][:38] + '..' if len(r['title']) > 38 else r['title']
    print(f"  {r['thumbnail_score']:5.1f}/20 | wc={r['perf_norm']:+.3f} | {r['channel_name']:15s} | {title}")

# Save
df.to_csv('data/videos_scored_v5.csv', index=False)

weights = {
    "text_short":    7.0,
    "text_medium":   8.5,
    "text_long":     10.0,
    "one_person":    4.0,
    "two_people":    2.0,
    "expr_neutral":  3.0,
    "expr_smile":    2.5,
    "color_cold":    3.0,
    "color_neutral": 1.5,
    "model_type":    "within_channel_validated_v5",
    "max_score":     20.0,
    "corr_raw":      float(corr_raw),
    "corr_norm":     float(corr_norm),
    "note": (
        "bg_busy and contrast_high excluded: appeared strong in raw analysis "
        "but both turned NEGATIVE after within-channel normalization — "
        "they were channel-type confounds, not real thumbnail signals."
    )
}
with open('data/thumbnail_weights_v5.json', 'w') as f:
    json.dump(weights, f, indent=2)

print(f"\nSaved: data/videos_scored_v5.csv")
print(f"Saved: data/thumbnail_weights_v5.json")

print("""
┌─────────────────────┬────────┬────────────────────────────────────────┐
│ Feature             │ Pts    │ Values                                 │
├─────────────────────┼────────┼────────────────────────────────────────┤
│ Text present        │ 0-10   │ Long(>50)=10, Medium(>20)=8.5, Short=7 │
│ Persons / Face      │ 0-4    │ 1 person=4, 2 people=2, else=0        │
│ Expression          │ 0-3    │ Neutral=3, Smile=2.5, else=0          │
│ Colors              │ 0-3    │ Cold=3, Neutral=1.5, Warm=0           │
├─────────────────────┼────────┼────────────────────────────────────────┤
│ TOTAL               │ 0-20   │                                        │
│ Excluded (confound) │  —     │ Busy bg, High contrast                 │
└─────────────────────┴────────┴────────────────────────────────────────┘
""")
