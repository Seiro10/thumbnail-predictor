"""
Thumbnail Scoring System v2 - Regression-based with penalty factors
Uses actual performance data to learn optimal weights
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load enriched dataset
df = pd.read_csv('data/videos_enriched.csv')

print("🔧 Feature Engineering with Penalty Factors...")
print("="*60)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

features = pd.DataFrame()

# --- POSITIVE FEATURES ---

# Text presence (binary)
features['has_text'] = df['texte_present'].astype(int)

# Face presence (binary)
features['has_face'] = df['visage_present'].astype(int)

# Exactly 1 person (optimal)
features['one_person'] = (df['nb_personnes'] == 1).astype(int)

# High contrast
features['high_contrast'] = (df['contraste'] == 'Élevé').astype(int)

# Busy background (correlates with performance)
features['busy_background'] = (df['fond'] == 'Chargé').astype(int)

# Neutral expression (best performer)
features['neutral_expression'] = (df['expression'] == 'Neutre').astype(int)

# Neutral colors (best performer)
features['neutral_colors'] = (df['couleur_dominante'] == 'Neutre').astype(int)

# Smile (second best expression)
features['smile'] = (df['expression'] == 'Sourire').astype(int)

# --- PENALTY FEATURES (negative impact expected) ---

# Too many people (3+) - cluttered
features['too_many_people'] = (df['nb_personnes'] >= 3).astype(int)

# No face at all
features['no_face'] = (~df['visage_present']).astype(int)

# No text
features['no_text'] = (~df['texte_present']).astype(int)

# Plain/boring background
features['plain_background'] = (df['fond'] == 'Uni').astype(int)

# Warm colors (worst performer)
features['warm_colors'] = (df['couleur_dominante'] == 'Chaud').astype(int)

# Intense expression (low performer)
features['intense_expression'] = (df['expression'] == 'Intense').astype(int)

# Text length penalty (too much text = cluttered)
df['text_length'] = df['texte_contenu'].fillna('').str.len()
features['excessive_text'] = (df['text_length'] > 50).astype(int)

# Multiple faces but no clear subject (2+ people)
features['multiple_faces'] = (df['nb_personnes'] >= 2).astype(int)

# Low confidence face detection (blurry/unclear face)
features['low_face_confidence'] = ((df['face_confidence'] > 0) & (df['face_confidence'] < 0.8)).astype(int)

print(f"Created {len(features.columns)} features:")
print(f"  Positive: has_text, has_face, one_person, high_contrast, busy_background,")
print(f"            neutral_expression, neutral_colors, smile")
print(f"  Penalty:  too_many_people, no_face, no_text, plain_background,")
print(f"            warm_colors, intense_expression, excessive_text,")
print(f"            multiple_faces, low_face_confidence")

# =============================================================================
# REGRESSION MODEL
# =============================================================================

print("\n" + "="*60)
print("🤖 Training Regression Model...")
print("="*60)

# Target: log-transform performance score to handle skew
y = np.log1p(df['performance_score'] * 1000)  # Scale up and log transform

# Remove any NaN
mask = ~(features.isna().any(axis=1) | y.isna())
X = features[mask]
y = y[mask]

print(f"\nTraining samples: {len(X)}")

# Use Ridge regression (handles multicollinearity)
model = Ridge(alpha=1.0)

# Cross-validation score
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Fit final model
model.fit(X, y)

# =============================================================================
# EXTRACT WEIGHTS
# =============================================================================

print("\n" + "="*60)
print("📊 LEARNED FEATURE WEIGHTS")
print("="*60)

weights = pd.DataFrame({
    'feature': features.columns,
    'weight': model.coef_
}).sort_values('weight', ascending=False)

print("\n🟢 POSITIVE IMPACT (boost score):")
positive = weights[weights['weight'] > 0]
for _, row in positive.iterrows():
    print(f"  +{row['weight']:6.3f}  {row['feature']}")

print("\n🔴 NEGATIVE IMPACT (penalty):")
negative = weights[weights['weight'] <= 0]
for _, row in negative.iterrows():
    print(f"  {row['weight']:7.3f}  {row['feature']}")

# =============================================================================
# CREATE SCORING FUNCTION
# =============================================================================

print("\n" + "="*60)
print("🎯 Creating Final Scoring System...")
print("="*60)

# Calculate raw scores
raw_scores = model.predict(features)

# Normalize to 0-20 scale
scaler = MinMaxScaler(feature_range=(0, 20))
df['thumbnail_score_v2'] = scaler.fit_transform(raw_scores.reshape(-1, 1)).flatten()

# Round to 1 decimal
df['thumbnail_score_v2'] = df['thumbnail_score_v2'].round(1)

# Statistics
print(f"\nScore distribution:")
print(f"  Min:    {df['thumbnail_score_v2'].min():.1f}")
print(f"  Max:    {df['thumbnail_score_v2'].max():.1f}")
print(f"  Mean:   {df['thumbnail_score_v2'].mean():.1f}")
print(f"  Median: {df['thumbnail_score_v2'].median():.1f}")

# Validation: correlation with actual performance
correlation = df['thumbnail_score_v2'].corr(df['performance_score'])
print(f"\n📈 Correlation with actual performance: {correlation:.4f}")

# Compare score bands
print("\n✅ VALIDATION BY SCORE BAND:")
for low, high in [(0, 5), (5, 10), (10, 15), (15, 20)]:
    subset = df[(df['thumbnail_score_v2'] >= low) & (df['thumbnail_score_v2'] < high)]
    if len(subset) > 0:
        avg_perf = subset['performance_score'].mean()
        print(f"  Score {low:2d}-{high:2d}: avg_performance = {avg_perf:.6f} (n={len(subset):4d})")

# =============================================================================
# SAVE WEIGHTS FOR PRODUCTION USE
# =============================================================================

# Create weight dictionary for easy use
weight_dict = dict(zip(features.columns, model.coef_))
weight_dict['intercept'] = model.intercept_
weight_dict['score_min'] = raw_scores.min()
weight_dict['score_max'] = raw_scores.max()

# Save to file
import json
with open('data/thumbnail_weights.json', 'w') as f:
    json.dump(weight_dict, f, indent=2)

print(f"\n💾 Weights saved to: data/thumbnail_weights.json")

# =============================================================================
# TOP & BOTTOM EXAMPLES
# =============================================================================

print("\n" + "="*60)
print("🏆 TOP 10 SCORING THUMBNAILS")
print("="*60)
top10 = df.nlargest(10, 'thumbnail_score_v2')[['title', 'channel_name', 'thumbnail_score_v2', 'performance_score']]
top10['title'] = top10['title'].str[:35] + '...'
for _, row in top10.iterrows():
    print(f"  {row['thumbnail_score_v2']:5.1f}/20 | perf={row['performance_score']:.4f} | {row['title']}")

print("\n" + "="*60)
print("📉 BOTTOM 10 SCORING THUMBNAILS")
print("="*60)
bottom10 = df.nsmallest(10, 'thumbnail_score_v2')[['title', 'channel_name', 'thumbnail_score_v2', 'performance_score']]
bottom10['title'] = bottom10['title'].str[:35] + '...'
for _, row in bottom10.iterrows():
    print(f"  {row['thumbnail_score_v2']:5.1f}/20 | perf={row['performance_score']:.4f} | {row['title']}")

# Save final dataset
df.to_csv('data/videos_scored_v2.csv', index=False)
print(f"\n💾 Final dataset saved to: data/videos_scored_v2.csv")

# =============================================================================
# SCORING FORMULA SUMMARY
# =============================================================================

print("\n" + "="*60)
print("📋 FINAL SCORING FORMULA")
print("="*60)

print("""
Score = normalize_to_20(
    # Positive factors
    + {has_text:.2f} × has_text
    + {has_face:.2f} × has_face
    + {one_person:.2f} × one_person
    + {busy_background:.2f} × busy_background
    + {neutral_expression:.2f} × neutral_expression
    + {neutral_colors:.2f} × neutral_colors
    + {smile:.2f} × smile
    + {high_contrast:.2f} × high_contrast

    # Penalty factors
    {too_many_people:.2f} × too_many_people
    {no_face:.2f} × no_face
    {no_text:.2f} × no_text
    {plain_background:.2f} × plain_background
    {warm_colors:.2f} × warm_colors
    {intense_expression:.2f} × intense_expression
    {excessive_text:.2f} × excessive_text
    {multiple_faces:.2f} × multiple_faces
    {low_face_confidence:.2f} × low_face_confidence
)
""".format(**weight_dict))
