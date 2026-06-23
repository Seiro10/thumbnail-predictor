"""
Custom prediction routine for Vertex AI batch prediction.

Each input instance (one row in the JSONL batch file):
{
  "video_id": "abc123",
  "embedding": [1280 floats],   ← EfficientNetB0 embedding
  "tabular": {                   ← Vision API + channel metrics
    "has_text": 1,
    "text_short": 0,
    "text_medium": 1,
    "text_long": 0,
    "one_person": 1,
    "two_people": 0,
    "many_people": 0,
    "bg_busy": 0,
    "bg_blur": 0,
    "color_neutral": 1,
    "color_cold": 0,
    "color_warm": 0,
    "contrast_high": 1,
    "contrast_med": 0,
    "expr_neutral": 1,
    "expr_smile": 0,
    "face_conf": 0.95,
    "niche_ai": 1,
    "log_subs": 11.49,
    "channel_avg_perf": 0.0032
  }
}

Output:
{
  "video_id": "abc123",
  "thumbnail_score": 14.2,
  "raw_pred": 0.37
}
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import json
import joblib
import keras


class ThumbnailPredictor:
    def __init__(self, model_dir: str):
        self.model  = keras.models.load_model(f"{model_dir}/thumbnail_scorer.keras")
        self.scaler = joblib.load(f"{model_dir}/tabular_scaler.pkl")
        with open(f"{model_dir}/meta.json") as f:
            self.meta = json.load(f)

    def predict(self, instances: list[dict]) -> list[dict]:
        tab_cols = self.meta['tab_features']

        embeddings = np.array([inst['embedding'] for inst in instances], dtype=np.float32)
        tab_raw    = np.array(
            [[inst['tabular'][c] for c in tab_cols] for inst in instances],
            dtype=np.float32
        )
        tab_scaled = self.scaler.transform(tab_raw).astype(np.float32)

        raw_preds = self.model.predict([embeddings, tab_scaled], verbose=0).flatten()

        # Scale to 0-20
        s_min = self.meta['score_min']
        s_max = self.meta['score_max']
        scores = np.clip((raw_preds - s_min) / (s_max - s_min) * 20, 0, 20)

        return [
            {
                'video_id':        inst.get('video_id', ''),
                'thumbnail_score': round(float(s), 1),
                'raw_pred':        round(float(r), 4),
            }
            for inst, s, r in zip(instances, scores, raw_preds)
        ]
