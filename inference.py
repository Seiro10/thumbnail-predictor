"""
End-to-end inference pipeline for scoring new thumbnails.

Usage:
  python inference.py --thumbnails path/to/thumb1.jpg path/to/thumb2.jpg \
                      --channel_name "My Channel" \
                      --subscriber_count 50000 \
                      --channel_avg_perf 0.003 \
                      --niche AI/Tech

What it does:
  1. Runs Vision API on each thumbnail
  2. Extracts EfficientNetB0 embedding
  3. Builds JSONL batch file and uploads to GCS
  4. Submits Vertex AI Batch Prediction job
  5. Downloads results from GCS and prints scores
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import json
import math
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model
from google.cloud import vision, storage, aiplatform
from pathlib import Path

# ── Config (from environment / .env) ─────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT    = os.environ['GCP_PROJECT']
LOCATION   = os.getenv('GCP_REGION', 'europe-west1')
MODEL_ID   = os.environ['VERTEX_MODEL_ID']
GCS_BUCKET = os.environ['GCS_BUCKET']
IMG_SIZE   = 224

# ── EfficientNet extractor ────────────────────────────────────────────────────
def build_extractor():
    base = EfficientNetB0(weights='imagenet', include_top=False,
                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = preprocess_input(inp)
    x   = base(x, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    return Model(inp, x)

# ── Vision API helpers ────────────────────────────────────────────────────────
def get_vision_features(image_path: str, vision_client) -> dict:
    with open(image_path, 'rb') as f:
        content = f.read()
    image = vision.Image(content=content)

    faces  = vision_client.face_detection(image=image).face_annotations
    texts  = vision_client.text_detection(image=image).text_annotations
    props  = vision_client.image_properties(image=image).image_properties_annotation
    labels = vision_client.label_detection(image=image).label_annotations

    nb_faces = len(faces)
    text_content = texts[0].description[:200] if texts else ''
    text_len = len(text_content)

    # Dominant color
    main_color = {'Neutre': 0, 'Chaud': 0, 'Froid': 0}
    if props.dominant_colors.colors:
        c = props.dominant_colors.colors[0].color
        r, g, b = c.red, c.green, c.blue
        if r > g and r > b and r > 150:
            main_color = 'Chaud'
        elif (b > r and b > g and b > 150) or (g > r and g > b and g > 150):
            main_color = 'Froid'
        else:
            main_color = 'Neutre'

    # Contrast
    n_colors = len(props.dominant_colors.colors)
    contraste = 'Élevé' if n_colors > 8 else ('Faible' if n_colors < 4 else 'Moyen')

    # Background
    label_texts = [l.description.lower() for l in labels]
    if any(bg in ' '.join(label_texts) for bg in ['sky', 'wall', 'ceiling', 'floor']):
        fond = 'Uni'
    elif len(labels) > 7:
        fond = 'Chargé'
    else:
        fond = 'Uni'

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
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        if top_emotion[1] >= 4:
            expression = {'joy': 'Sourire', 'surprise': 'Surprise'}.get(
                top_emotion[0], 'Intense')
        elif nb_faces > 0:
            expression = 'Neutre'

    face_conf = faces[0].detection_confidence if faces else 0.0

    return {
        'has_text':      float(len(texts) > 1),
        'text_short':    float(0 < text_len <= 20),
        'text_medium':   float(20 < text_len <= 50),
        'text_long':     float(text_len > 50),
        'one_person':    float(nb_faces == 1),
        'two_people':    float(nb_faces == 2),
        'many_people':   float(nb_faces >= 3),
        'bg_busy':       float(fond == 'Chargé'),
        'bg_blur':       0.0,
        'color_neutral': float(main_color == 'Neutre'),
        'color_cold':    float(main_color == 'Froid'),
        'color_warm':    float(main_color == 'Chaud'),
        'contrast_high': float(contraste == 'Élevé'),
        'contrast_med':  float(contraste == 'Moyen'),
        'expr_neutral':  float(expression == 'Neutre'),
        'expr_smile':    float(expression == 'Sourire'),
        'face_conf':     float(face_conf),
    }

# ── Main pipeline ─────────────────────────────────────────────────────────────
def score_thumbnails(
    thumbnail_paths: list[str],
    channel_name: str,
    subscriber_count: int,
    channel_avg_perf: float,
    niche: str = 'AI/Tech',
    job_display_name: str | None = None,
) -> pd.DataFrame:

    print(f"Scoring {len(thumbnail_paths)} thumbnail(s)...")

    # 1. Build extractor
    print("Loading EfficientNetB0...")
    extractor = build_extractor()

    # 2. Vision API client
    vision_client = vision.ImageAnnotatorClient()

    # 3. Build instances
    instances = []
    niche_ai = float(niche == 'AI/Tech')
    log_subs = math.log1p(subscriber_count)

    for path in thumbnail_paths:
        print(f"  Processing {Path(path).name}...")

        # Embedding
        img = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        emb = extractor(tf.expand_dims(img, 0), training=False).numpy()[0].tolist()

        # Vision API features
        tab = get_vision_features(path, vision_client)
        tab['niche_ai']         = niche_ai
        tab['log_subs']         = log_subs
        tab['channel_avg_perf'] = channel_avg_perf

        instances.append({
            'video_id':  Path(path).stem,
            'embedding': emb,
            'tabular':   tab,
        })

    # 4. Build flat format for TF Serving signature
    # Each instance must match the SavedModel's serve() input spec
    flat_instances = []
    for inst in instances:
        flat = {'embedding': [inst['embedding']]}
        for k, v in inst['tabular'].items():
            flat[k] = [[v]]
        flat_instances.append({'instance': flat, '_video_id': inst['video_id']})

    # 5. Write JSONL to GCS
    run_id = int(time.time())
    gcs_input  = f'gs://{GCS_BUCKET}/batch_jobs/{run_id}/input.jsonl'
    gcs_output = f'gs://{GCS_BUCKET}/batch_jobs/{run_id}/output/'

    storage_client = storage.Client(project=PROJECT)
    bucket = storage_client.bucket(GCS_BUCKET)
    blob   = bucket.blob(f'batch_jobs/{run_id}/input.jsonl')

    jsonl_content = '\n'.join(
        json.dumps({'instance': fi['instance']}) for fi in flat_instances
    )
    blob.upload_from_string(jsonl_content, content_type='application/jsonl')
    print(f"Input uploaded: {gcs_input}")

    # 6. Submit Vertex Batch Prediction job
    aiplatform.init(project=PROJECT, location=LOCATION)
    model = aiplatform.Model(f'projects/{PROJECT}/locations/{LOCATION}/models/{MODEL_ID}')

    job = model.batch_predict(
        job_display_name=job_display_name or f'thumbnail-score-{run_id}',
        gcs_source=gcs_input,
        gcs_destination_prefix=gcs_output,
        instances_format='jsonl',
        predictions_format='jsonl',
        machine_type='n1-standard-4',
        sync=True,
    )
    print(f"Batch job complete: {job.display_name}")

    # 7. Read results from GCS
    prefix = f'batch_jobs/{run_id}/output/'
    blobs  = list(storage_client.list_blobs(GCS_BUCKET, prefix=prefix))
    output_blob = next((b for b in blobs if b.name.endswith('.jsonl')), None)

    results = []
    if output_blob:
        for line in output_blob.download_as_text().strip().split('\n'):
            pred = json.loads(line).get('prediction', {})
            results.append(pred)

    # Merge video_ids back
    df_out = pd.DataFrame(results)
    if 'thumbnail_score' in df_out.columns:
        df_out['video_id'] = [fi['_video_id'] for fi in flat_instances[:len(results)]]

    return df_out


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thumbnails',        nargs='+', required=True)
    parser.add_argument('--channel_name',      required=True)
    parser.add_argument('--subscriber_count',  type=int,   required=True)
    parser.add_argument('--channel_avg_perf',  type=float, required=True,
                        help='Channel average performance_score (views/subs)')
    parser.add_argument('--niche', default='AI/Tech', choices=['AI/Tech', 'Business'])
    args = parser.parse_args()

    results = score_thumbnails(
        thumbnail_paths   = args.thumbnails,
        channel_name      = args.channel_name,
        subscriber_count  = args.subscriber_count,
        channel_avg_perf  = args.channel_avg_perf,
        niche             = args.niche,
    )

    print("\n" + "="*55)
    print("THUMBNAIL SCORES")
    print("="*55)
    for _, row in results.iterrows():
        print(f"  {row.get('video_id','?'):30s}  {row.get('thumbnail_score', '?'):>5}/20")
