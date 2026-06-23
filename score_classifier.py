"""
Thumbnail Scoring System v3 - Classification-based
Predicts probability of being a "top performer" thumbnail
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')

# Load enriched dataset
df = pd.read_csv('data/videos_enriched.csv')

print("🔧 Feature Engineering...")
print("="*60)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

features = pd.DataFrame()

# Binary features
features['has_text'] = df['texte_present'].astype(int)
features['has_face'] = df['visage_present'].astype(int)
features['one_person'] = (df['nb_personnes'] == 1).astype(int)
features['two_people'] = (df['nb_personnes'] == 2).astype(int)
features['many_people'] = (df['nb_personnes'] >= 3).astype(int)
features['no_people'] = (df['nb_personnes'] == 0).astype(int)

# Expression one-hot
features['expr_neutral'] = (df['expression'] == 'Neutre').astype(int)
features['expr_smile'] = (df['expression'] == 'Sourire').astype(int)
features['expr_surprise'] = (df['expression'] == 'Surprise').astype(int)
features['expr_intense'] = (df['expression'] == 'Intense').astype(int)

# Color one-hot
features['color_neutral'] = (df['couleur_dominante'] == 'Neutre').astype(int)
features['color_warm'] = (df['couleur_dominante'] == 'Chaud').astype(int)
features['color_cold'] = (df['couleur_dominante'] == 'Froid').astype(int)

# Background one-hot
features['bg_busy'] = (df['fond'] == 'Chargé').astype(int)
features['bg_plain'] = (df['fond'] == 'Uni').astype(int)
features['bg_blur'] = (df['fond'] == 'Flou').astype(int)

# Contrast
features['high_contrast'] = (df['contraste'] == 'Élevé').astype(int)

# Text length features
df['text_length'] = df['texte_contenu'].fillna('').str.len()
features['text_short'] = ((df['text_length'] > 0) & (df['text_length'] <= 20)).astype(int)
features['text_medium'] = ((df['text_length'] > 20) & (df['text_length'] <= 50)).astype(int)
features['text_long'] = (df['text_length'] > 50).astype(int)

# Face confidence
features['face_conf_high'] = (df['face_confidence'] >= 0.9).astype(int)
features['face_conf_medium'] = ((df['face_confidence'] >= 0.7) & (df['face_confidence'] < 0.9)).astype(int)

print(f"Created {len(features.columns)} features")

# =============================================================================
# CREATE TARGET: Top 25% performer
# =============================================================================

# Define "top performer" as top 25% by performance score
threshold = df['performance_score'].quantile(0.75)
y = (df['performance_score'] >= threshold).astype(int)

print(f"\nTop performer threshold: {threshold:.6f}")
print(f"Top performers: {y.sum()} ({y.mean()*100:.1f}%)")

# =============================================================================
# TRAIN CLASSIFIER
# =============================================================================

print("\n" + "="*60)
print("🤖 Training Gradient Boosting Classifier...")
print("="*60)

X = features.values

# Use Gradient Boosting (handles non-linear relationships)
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
print(f"\nCross-validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

accuracy_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std()*2:.4f})")

# Fit final model
model.fit(X, y)

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*60)
print("📊 FEATURE IMPORTANCE (Top 15)")
print("="*60)

importance = pd.DataFrame({
    'feature': features.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(importance.head(15).iterrows()):
    bar = '█' * int(row['importance'] * 50)
    print(f"  {row['feature']:20s} {row['importance']:.3f} {bar}")

# =============================================================================
# CALCULATE SCORES
# =============================================================================

print("\n" + "="*60)
print("🎯 Calculating Thumbnail Scores...")
print("="*60)

# Get probability of being a top performer
probabilities = model.predict_proba(X)[:, 1]

# Scale to 0-20
df['thumbnail_score_v3'] = (probabilities * 20).round(1)

# Statistics
print(f"\nScore distribution:")
print(f"  Min:    {df['thumbnail_score_v3'].min():.1f}")
print(f"  Max:    {df['thumbnail_score_v3'].max():.1f}")
print(f"  Mean:   {df['thumbnail_score_v3'].mean():.1f}")
print(f"  Median: {df['thumbnail_score_v3'].median():.1f}")

# Validation
correlation = df['thumbnail_score_v3'].corr(df['performance_score'])
print(f"\n📈 Correlation with actual performance: {correlation:.4f}")

# Score band analysis
print("\n✅ VALIDATION BY SCORE BAND:")
print(f"{'Score Band':<12} {'Avg Perf':>12} {'% Top Perf':>12} {'Count':>8}")
print("-" * 48)
for low, high in [(0, 5), (5, 10), (10, 15), (15, 20)]:
    subset = df[(df['thumbnail_score_v3'] >= low) & (df['thumbnail_score_v3'] < high)]
    if len(subset) > 0:
        avg_perf = subset['performance_score'].mean()
        pct_top = (subset['performance_score'] >= threshold).mean() * 100
        print(f"{low:2d}-{high:2d}        {avg_perf:12.6f} {pct_top:11.1f}% {len(subset):8d}")

# =============================================================================
# EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("🏆 TOP 10 SCORING THUMBNAILS")
print("="*60)
top10 = df.nlargest(10, 'thumbnail_score_v3')[['title', 'channel_name', 'thumbnail_score_v3', 'performance_score']]
for _, row in top10.iterrows():
    title = row['title'][:40] + '...' if len(row['title']) > 40 else row['title']
    print(f"  {row['thumbnail_score_v3']:5.1f}/20 | perf={row['performance_score']:.4f} | {title}")

print("\n" + "="*60)
print("📉 BOTTOM 10 SCORING THUMBNAILS")
print("="*60)
bottom10 = df.nsmallest(10, 'thumbnail_score_v3')[['title', 'channel_name', 'thumbnail_score_v3', 'performance_score']]
for _, row in bottom10.iterrows():
    title = row['title'][:40] + '...' if len(row['title']) > 40 else row['title']
    print(f"  {row['thumbnail_score_v3']:5.1f}/20 | perf={row['performance_score']:.4f} | {title}")

# =============================================================================
# SAVE
# =============================================================================

# Save model feature importance as weights
weights = dict(zip(features.columns, model.feature_importances_))
weights['model_type'] = 'GradientBoostingClassifier'
weights['roc_auc'] = float(cv_scores.mean())
weights['threshold_top_performer'] = float(threshold)

with open('data/thumbnail_weights_v3.json', 'w') as f:
    json.dump(weights, f, indent=2)

df.to_csv('data/videos_scored_v3.csv', index=False)

print(f"\n💾 Saved:")
print(f"  - data/videos_scored_v3.csv")
print(f"  - data/thumbnail_weights_v3.json")

# =============================================================================
# SCORING INTERPRETATION
# =============================================================================

print("\n" + "="*60)
print("📋 SCORE INTERPRETATION")
print("="*60)
print("""
  Score 0-5:   Poor thumbnail - missing key elements
  Score 5-10:  Below average - some elements present
  Score 10-15: Good thumbnail - most best practices followed
  Score 15-20: Excellent - high probability of top performance

  Note: Score represents probability of being in top 25% performers,
  scaled to 0-20. A score of 15 means ~75% probability of top performance.
""")
