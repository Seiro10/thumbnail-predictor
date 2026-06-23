"""
Export the trained Keras model as a TF SavedModel with a serving signature
that accepts a flat dict of all features (embedding + tabular) and returns
the 0-20 thumbnail score.

This works with Vertex AI's pre-built TF2 serving container — no custom
Docker image needed.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import keras
import joblib
import json
from pathlib import Path

# ── Load artifacts ─────────────────────────────────────────────────────────
model  = keras.models.load_model('model/thumbnail_scorer.keras')
scaler = joblib.load('model/tabular_scaler.pkl')
with open('model/meta.json') as f:
    meta = json.load(f)

TAB_COLS  = meta['tab_features']
SCORE_MIN = tf.constant(meta['score_min'], dtype=tf.float32)
SCORE_MAX = tf.constant(meta['score_max'], dtype=tf.float32)

# Encode scaler params as constants so they're baked into the SavedModel
scaler_mean  = tf.constant(scaler.mean_.astype(np.float32))
scaler_scale = tf.constant(scaler.scale_.astype(np.float32))

# ── Wrap in a Module with a serving function ───────────────────────────────
class ThumbnailScorerModule(tf.Module):
    def __init__(self, keras_model):
        super().__init__()
        self.model = keras_model

    @tf.function(input_signature=[{
        'embedding': tf.TensorSpec(shape=[None, 1280], dtype=tf.float32, name='embedding'),
        **{col: tf.TensorSpec(shape=[None, 1], dtype=tf.float32, name=col) for col in TAB_COLS}
    }])
    def serve(self, inputs):
        emb = inputs['embedding']

        # Rebuild tabular matrix from individual feature tensors
        tab_cols = [inputs[c] for c in TAB_COLS]
        tab_raw  = tf.concat(tab_cols, axis=1)

        # Apply scaler (baked in as constants)
        tab_scaled = (tab_raw - scaler_mean) / scaler_scale

        raw_preds = self.model([emb, tab_scaled], training=False)
        raw_preds = tf.squeeze(raw_preds, axis=1)

        scores = tf.clip_by_value(
            (raw_preds - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 20.0,
            0.0, 20.0
        )
        return {'thumbnail_score': scores}

module = ThumbnailScorerModule(model)

# ── Export ─────────────────────────────────────────────────────────────────
export_dir = 'model/saved_model'
tf.saved_model.save(
    module,
    export_dir,
    signatures={'serving_default': module.serve}
)

print(f"SavedModel exported to: {export_dir}")
print("Serving signature inputs:")
for k in ['embedding'] + TAB_COLS:
    print(f"  {k}: shape [batch, 1 or 1280], float32")
print("Serving signature output:")
print("  thumbnail_score: shape [batch], float32 in [0, 20]")

# ── Upload to GCS ─────────────────────────────────────────────────────────
import subprocess
result = subprocess.run(
    ['gcloud', 'storage', 'cp', '-r', export_dir,
     'gs://thumbnail-predictor-models/saved_model/'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print("\nUploaded to: gs://thumbnail-predictor-models/saved_model/")
else:
    print(f"Upload error: {result.stderr}")
