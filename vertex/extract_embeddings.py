"""
Extract EfficientNetB0 embeddings for all thumbnails.
Frozen ImageNet weights → 1280-dim feature vector per image.
Saved to data/clip_embeddings.npy + data/embedding_index.csv
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, Model
from pathlib import Path
from tqdm import tqdm

IMG_SIZE = 224
BATCH_SIZE = 32
THUMB_DIR = Path('data/thumbnails')
OUT_EMB   = Path('data/efficientnet_embeddings.npy')
OUT_IDX   = Path('data/embedding_index.csv')

# ── Build frozen feature extractor ───────────────────────────────────────────
base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = preprocess_input(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
extractor = Model(inputs, x)

print(f"Feature extractor: EfficientNetB0 → {extractor.output_shape[-1]}-dim embeddings")

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/videos_enriched.csv')
video_ids = df['video_id'].tolist()
print(f"Videos: {len(video_ids)}")

def load_image(path):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

# ── Extract in batches ────────────────────────────────────────────────────────
embeddings = []
valid_ids  = []

for i in tqdm(range(0, len(video_ids), BATCH_SIZE), desc='Extracting'):
    batch_ids = video_ids[i:i+BATCH_SIZE]
    batch_imgs = []
    batch_valid = []

    for vid in batch_ids:
        path = THUMB_DIR / f"{vid}.jpg"
        if not path.exists():
            continue
        try:
            img = load_image(path)
            batch_imgs.append(img)
            batch_valid.append(vid)
        except Exception:
            continue

    if not batch_imgs:
        continue

    batch_tensor = tf.stack(batch_imgs)
    embs = extractor(batch_tensor, training=False).numpy()
    embeddings.extend(embs)
    valid_ids.extend(batch_valid)

embeddings = np.array(embeddings, dtype=np.float32)
print(f"\nExtracted {len(embeddings)} embeddings — shape: {embeddings.shape}")

np.save(OUT_EMB, embeddings)
pd.DataFrame({'video_id': valid_ids}).to_csv(OUT_IDX, index=False)

print(f"Saved: {OUT_EMB}")
print(f"Saved: {OUT_IDX} ({len(valid_ids)} valid thumbnails)")
