"""
Thumbnail score model — Keras
Architecture:
  EfficientNetB0 embeddings (1280-dim, frozen)  ┐
  Vision API tabular features (~18-dim)          ├─ concat → Dense head → score
  Channel context features (~3-dim)              ┘

Target: within-channel z-scored performance (perf_norm)
Output: saved Keras model to model/thumbnail_scorer/
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import json, joblib
from pathlib import Path

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Load embeddings ───────────────────────────────────────────────────────────
emb_matrix = np.load('data/efficientnet_embeddings.npy').astype(np.float32)
emb_index  = pd.read_csv('data/embedding_index.csv')
emb_map    = dict(zip(emb_index['video_id'], range(len(emb_index))))

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/videos_enriched.csv')
df['text_length'] = df['texte_contenu'].fillna('').str.len()
df = df[df['video_id'].isin(emb_map)].reset_index(drop=True)

# Within-channel normalized target
df['perf_norm'] = df.groupby('channel_name')['performance_score'].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-9)
)

print(f"Training samples: {len(df)}")

# ── Tabular features ──────────────────────────────────────────────────────────
tab = pd.DataFrame()

# Vision API features
tab['has_text']      = df['texte_present'].astype(float)
tab['text_short']    = ((df['text_length'] > 0)  & (df['text_length'] <= 20)).astype(float)
tab['text_medium']   = ((df['text_length'] > 20) & (df['text_length'] <= 50)).astype(float)
tab['text_long']     = (df['text_length'] > 50).astype(float)
tab['one_person']    = (df['nb_personnes'] == 1).astype(float)
tab['two_people']    = (df['nb_personnes'] == 2).astype(float)
tab['many_people']   = (df['nb_personnes'] >= 3).astype(float)
tab['bg_busy']       = (df['fond'] == 'Chargé').astype(float)
tab['bg_blur']       = (df['fond'] == 'Flou').astype(float)
tab['color_neutral'] = (df['couleur_dominante'] == 'Neutre').astype(float)
tab['color_cold']    = (df['couleur_dominante'] == 'Froid').astype(float)
tab['color_warm']    = (df['couleur_dominante'] == 'Chaud').astype(float)
tab['contrast_high'] = (df['contraste'] == 'Élevé').astype(float)
tab['contrast_med']  = (df['contraste'] == 'Moyen').astype(float)
tab['expr_neutral']  = (df['expression'] == 'Neutre').astype(float)
tab['expr_smile']    = (df['expression'] == 'Sourire').astype(float)
tab['face_conf']     = df['face_confidence'].fillna(0).astype(float)

# Channel context
df['niche_ai'] = (df['niche'] == 'AI/Tech').astype(float)
tab['niche_ai']       = df['niche_ai']
tab['log_subs']       = np.log1p(df['subscriber_count']).astype(float)
tab['channel_avg_perf'] = df.groupby('channel_name')['performance_score'].transform('mean').astype(float)

TAB_DIM = len(tab.columns)
print(f"Tabular features: {TAB_DIM} — {list(tab.columns)}")

# Assemble arrays
X_emb = np.array([emb_matrix[emb_map[vid]] for vid in df['video_id']], dtype=np.float32)
X_tab = tab.values.astype(np.float32)
y     = df['perf_norm'].values.astype(np.float32)

# Scale tabular features
scaler = StandardScaler()
X_tab  = scaler.fit_transform(X_tab).astype(np.float32)

# ── Model definition ──────────────────────────────────────────────────────────
def build_model(emb_dim=1280, tab_dim=TAB_DIM, l2=1e-4):
    emb_input = keras.Input(shape=(emb_dim,), name='embedding')
    tab_input = keras.Input(shape=(tab_dim,),  name='tabular')

    # Embedding branch — compress
    x = layers.Dense(256, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(l2))(emb_input)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu',
                      kernel_regularizer=keras.regularizers.l2(l2))(x)

    # Tabular branch
    t = layers.Dense(32, activation='relu')(tab_input)

    # Merge
    merged = layers.Concatenate()([x, t])
    out = layers.Dense(32, activation='relu')(merged)
    out = layers.Dropout(0.3)(out)
    out = layers.Dense(1, name='score')(out)

    model = keras.Model(inputs=[emb_input, tab_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

# ── 5-fold cross-validation ───────────────────────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
fold_corrs = []

print("\n=== 5-Fold Cross-Validation ===")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_emb)):
    model = build_model()
    model.fit(
        [X_emb[tr_idx], X_tab[tr_idx]], y[tr_idx],
        validation_data=([X_emb[val_idx], X_tab[val_idx]], y[val_idx]),
        epochs=60,
        batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
        verbose=0
    )
    preds = model.predict([X_emb[val_idx], X_tab[val_idx]], verbose=0).flatten()
    corr  = np.corrcoef(preds, y[val_idx])[0, 1]
    fold_corrs.append(corr)
    print(f"  Fold {fold+1}: corr={corr:+.4f}")

print(f"\nMean correlation: {np.mean(fold_corrs):+.4f} ± {np.std(fold_corrs):.4f}")

# ── Train final model on all data ────────────────────────────────────────────
print("\nTraining final model on full dataset...")
final_model = build_model()
final_model.fit(
    [X_emb, X_tab], y,
    epochs=80,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    verbose=0
)

# Final score on 0-20 scale
raw_preds = final_model.predict([X_emb, X_tab], verbose=0).flatten()
# Clip and scale: map [-3σ, +3σ] → [0, 20]
score_min, score_max = raw_preds.mean() - 3*raw_preds.std(), raw_preds.mean() + 3*raw_preds.std()
scores_20 = np.clip((raw_preds - score_min) / (score_max - score_min) * 20, 0, 20)

df['thumbnail_score_nn'] = np.round(scores_20, 1)
corr_final = np.corrcoef(scores_20, df['perf_norm'].values)[0, 1]
print(f"Final model within-channel corr: {corr_final:+.4f}")
print(f"Score range: {scores_20.min():.1f} – {scores_20.max():.1f}")
print(f"Score mean:  {scores_20.mean():.1f}")

# ── Save ──────────────────────────────────────────────────────────────────────
Path('model').mkdir(exist_ok=True)
final_model.save('model/thumbnail_scorer.keras')
joblib.dump(scaler, 'model/tabular_scaler.pkl')

meta = {
    'tab_features':  list(tab.columns),
    'emb_dim':       1280,
    'tab_dim':       TAB_DIM,
    'score_min':     float(score_min),
    'score_max':     float(score_max),
    'cv_corr_mean':  float(np.mean(fold_corrs)),
    'cv_corr_std':   float(np.std(fold_corrs)),
    'final_corr':    float(corr_final),
    'n_train':       len(df),
}
with open('model/meta.json', 'w') as f:
    json.dump(meta, f, indent=2)

df[['video_id','channel_name','thumbnail_score_nn','perf_norm','performance_score']].to_csv(
    'data/videos_scored_nn.csv', index=False)

print("\nSaved:")
print("  model/thumbnail_scorer.keras")
print("  model/tabular_scaler.pkl")
print("  model/meta.json")
print("  data/videos_scored_nn.csv")
