"""
Local inference — no Vertex batch job.
Loads model once at startup, scores a thumbnail in ~1s.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model
import keras
import joblib
import json
from pathlib import Path
from google.cloud import vision

IMG_SIZE   = 224
MODEL_DIR  = Path(__file__).parent.parent / 'model'

# ── Load once at startup ──────────────────────────────────────────────────────
print("Loading EfficientNetB0...")
_base = EfficientNetB0(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3))
_base.trainable = False
_inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
_x   = preprocess_input(_inp)
_x   = _base(_x, training=False)
_x   = layers.GlobalAveragePooling2D()(_x)
EXTRACTOR = Model(_inp, _x)

print("Loading Keras scorer...")
SCORER  = keras.models.load_model(str(MODEL_DIR / 'thumbnail_scorer.keras'))
SCALER  = joblib.load(str(MODEL_DIR / 'tabular_scaler.pkl'))
with open(MODEL_DIR / 'meta.json') as f:
    META = json.load(f)

TAB_COLS  = META['tab_features']
SCORE_MIN = META['score_min']
SCORE_MAX = META['score_max']

print("All models loaded.")


# ── Vision API feature extraction ─────────────────────────────────────────────
def _vision_features(image_bytes: bytes) -> dict:
    client = vision.ImageAnnotatorClient()
    image  = vision.Image(content=image_bytes)

    faces  = client.face_detection(image=image).face_annotations
    texts  = client.text_detection(image=image).text_annotations
    props  = client.image_properties(image=image).image_properties_annotation
    labels = client.label_detection(image=image).label_annotations

    nb_faces     = len(faces)
    text_content = texts[0].description[:200] if texts else ''
    text_len     = len(text_content)

    # Colour
    color = 'Neutre'
    if props.dominant_colors.colors:
        c = props.dominant_colors.colors[0].color
        r, g, b = c.red, c.green, c.blue
        if r > g and r > b and r > 150:
            color = 'Chaud'
        elif (b > r and b > g and b > 150) or (g > r and g > b and g > 150):
            color = 'Froid'

    # Contrast
    n_colors  = len(props.dominant_colors.colors)
    contraste = 'Élevé' if n_colors > 8 else ('Faible' if n_colors < 4 else 'Moyen')

    # Background
    label_texts = [l.description.lower() for l in labels]
    fond = ('Uni' if any(bg in ' '.join(label_texts)
                         for bg in ['sky', 'wall', 'ceiling', 'floor'])
            else ('Chargé' if len(labels) > 7 else 'Uni'))

    # Expression
    expression = 'Aucun'
    if faces:
        face = faces[0]
        emotions = {
            'joy':      face.joy_likelihood,
            'surprise': face.surprise_likelihood,
            'anger':    face.anger_likelihood,
            'sorrow':   face.sorrow_likelihood,
        }
        top = max(emotions.items(), key=lambda x: x[1])
        if top[1] >= 4:
            expression = {'joy': 'Sourire', 'surprise': 'Surprise'}.get(
                top[0], 'Intense')
        elif nb_faces > 0:
            expression = 'Neutre'

    face_conf = faces[0].detection_confidence if faces else 0.0

    return {
        'nb_faces':     nb_faces,
        'text_content': text_content,
        'text_len':     text_len,
        'color':        color,
        'contraste':    contraste,
        'fond':         fond,
        'expression':   expression,
        'face_conf':    face_conf,
    }


def _build_tab_row(vision: dict, subscriber_count: int,
                   channel_avg_perf: float, niche: str) -> dict:
    tl = vision['text_len']
    return {
        'has_text':         float(tl > 0),
        'text_short':       float(0 < tl <= 20),
        'text_medium':      float(20 < tl <= 50),
        'text_long':        float(tl > 50),
        'one_person':       float(vision['nb_faces'] == 1),
        'two_people':       float(vision['nb_faces'] == 2),
        'many_people':      float(vision['nb_faces'] >= 3),
        'bg_busy':          float(vision['fond'] == 'Chargé'),
        'bg_blur':          0.0,
        'color_neutral':    float(vision['color'] == 'Neutre'),
        'color_cold':       float(vision['color'] == 'Froid'),
        'color_warm':       float(vision['color'] == 'Chaud'),
        'contrast_high':    float(vision['contraste'] == 'Élevé'),
        'contrast_med':     float(vision['contraste'] == 'Moyen'),
        'expr_neutral':     float(vision['expression'] == 'Neutre'),
        'expr_smile':       float(vision['expression'] == 'Sourire'),
        'face_conf':        float(vision['face_conf']),
        'niche_ai':         float(niche == 'AI/Tech'),
        'log_subs':         float(np.log1p(subscriber_count)),
        'channel_avg_perf': float(channel_avg_perf),
    }


# ── Public scoring function ───────────────────────────────────────────────────
def score(image_bytes: bytes, subscriber_count: int,
          channel_avg_perf: float, niche: str = 'AI/Tech') -> dict:

    # 1. Vision API
    v = _vision_features(image_bytes)

    # 2. EfficientNet embedding
    img_tensor = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img_tensor = tf.image.resize(img_tensor, [IMG_SIZE, IMG_SIZE])
    emb = EXTRACTOR(tf.expand_dims(img_tensor, 0), training=False).numpy()

    # 3. Tabular features
    tab_row = _build_tab_row(v, subscriber_count, channel_avg_perf, niche)
    tab_vec = np.array([[tab_row[c] for c in TAB_COLS]], dtype=np.float32)
    tab_scaled = SCALER.transform(tab_vec).astype(np.float32)

    # 4. Predict
    raw = SCORER.predict([emb, tab_scaled], verbose=0)[0][0]
    score_20 = float(np.clip((raw - SCORE_MIN) / (SCORE_MAX - SCORE_MIN) * 20, 0, 20))

    # 5. Per-dimension breakdown (rule-based, always readable)
    tl = v['text_len']
    text_pts  = (10.0 if tl > 50 else 8.5 if tl > 20 else 7.0) if tl > 0 else 0.0
    face_pts  = (4.0 if v['nb_faces'] == 1 else 2.0 if v['nb_faces'] == 2 else 0.0)
    expr_pts  = (3.0 if v['expression'] == 'Neutre' else
                 2.5 if v['expression'] == 'Sourire' else 0.0)
    color_pts = (3.0 if v['color'] == 'Froid' else
                 1.5 if v['color'] == 'Neutre' else 0.0)

    return {
        'score':      round(score_20, 1),
        'score_max':  20,
        'dimensions': {
            'text':       {'score': text_pts,  'max': 10, 'value': v['text_content']},
            'face':       {'score': face_pts,  'max': 4,  'value': v['nb_faces']},
            'expression': {'score': expr_pts,  'max': 3,  'value': v['expression']},
            'colors':     {'score': color_pts, 'max': 3,  'value': v['color']},
        },
        'vision': {
            'text_present':  tl > 0,
            'text_length':   tl,
            'text_content':  v['text_content'],
            'nb_faces':      v['nb_faces'],
            'expression':    v['expression'],
            'color':         v['color'],
            'background':    v['fond'],
            'contrast':      v['contraste'],
            'face_conf':     round(v['face_conf'], 3),
        },
    }
