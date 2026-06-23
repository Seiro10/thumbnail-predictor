# YouTube Thumbnail Scorer

A machine learning tool that scores YouTube thumbnails from **0 to 20** based on what actually drives views on your channel — trained on real performance data, deployed on Google Vertex AI.

Built with Google Cloud Vision API, EfficientNetB0 (Keras), and Vertex AI Batch Prediction.

---

## How it works

1. **Scrape** your channel's video history with the YouTube Data API
2. **Label** each thumbnail using Google Cloud Vision API (faces, text, colours, background)
3. **Extract** visual embeddings using a frozen EfficientNetB0 (ImageNet pretrained)
4. **Train** a Keras regression model on within-channel normalised performance scores
5. **Deploy** to Vertex AI — batch prediction only, ~$0.05 per job
6. **Score** new thumbnails before uploading

---

## Prerequisites

- Python 3.11+
- A Google Cloud project with billing enabled
- `gcloud` CLI installed and authenticated (`gcloud auth application-default login`)
- APIs enabled: YouTube Data API v3, Cloud Vision API, Vertex AI, Artifact Registry, Cloud Storage

---

## Getting Started

### 1. Clone and install

```bash
git clone https://github.com/your-username/thumbnail-predictor.git
cd thumbnail-predictor

python -m venv pyenv
source pyenv/bin/activate       # Windows: pyenv\Scripts\activate

pip install tensorflow keras scikit-learn google-cloud-vision \
            google-cloud-aiplatform google-cloud-storage \
            pandas numpy tqdm python-dotenv joblib
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
API_KEY_YOUTUBE=AIzaSy...          # YouTube Data API v3 key
GCP_PROJECT=my-project-id          # Your GCP project ID
GCP_REGION=europe-west1            # Vertex AI region
GCS_BUCKET=my-bucket-name          # Will be created in step 5
VERTEX_MODEL_ID=                   # Fill in after step 7
```

### 3. Enable required GCP APIs

```bash
gcloud services enable \
  youtube.googleapis.com \
  vision.googleapis.com \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  storage.googleapis.com \
  --project=$GCP_PROJECT
```

### 4. Scrape your channel data

Edit `scraper.py` to add your channel IDs, then run:

```bash
python scraper.py
```

This creates `data/videos.csv` and downloads thumbnails to `data/thumbnails/`.

### 5. Label thumbnails with Vision API

> **Cost:** ~$1.50 per 1,000 images (a 200-video channel = ~$0.30)

```bash
python auto_label_vision.py
```

Creates `data/auto_labels.csv` with face, text, colour, and background features per thumbnail.

### 6. Create GCS bucket and upload model

```bash
gcloud storage buckets create gs://$GCS_BUCKET \
  --project=$GCP_PROJECT \
  --location=$GCP_REGION
```

### 7. Extract embeddings and train the model

```bash
# Extract EfficientNetB0 embeddings (~5 min on CPU)
python vertex/extract_embeddings.py

# Train the Keras regression model (~10 min on CPU)
python vertex/train.py
```

Cross-validation correlation will be printed at the end. Typical range: +0.10 to +0.25 depending on your channel data.

### 8. Export and deploy to Vertex AI

```bash
# Export as TF SavedModel and upload to GCS
python vertex/export_savedmodel.py

# Register in Vertex AI Model Registry
python vertex/register_model.py
```

Copy the **Model ID** printed at the end into your `.env`:
```env
VERTEX_MODEL_ID=123456789012345678
```

### 9. Score a new thumbnail

```bash
python inference.py \
  --thumbnails my_thumbnail.jpg \
  --channel_name "Your Channel" \
  --subscriber_count 50000 \
  --channel_avg_perf 0.0035 \
  --niche AI/Tech
```

`channel_avg_perf` is your channel's average `views / subscribers` ratio. You can find it from `data/videos.csv` after scraping.

Output:
```
THUMBNAIL SCORES
═══════════════════════════════════════════════
  my_thumbnail.jpg              14.2/20
```

---

## Score interpretation

| Score | Meaning |
|---|---|
| 0–5 | Below average — likely missing text or a clear subject |
| 5–10 | Underperforms its channel baseline on average |
| 10–15 | Around channel average |
| **15–20** | **Above channel average — optimised thumbnail** |

The score reflects what has historically worked **on your specific channel**, not generic YouTube advice.

---

## Monthly retraining

As new videos accumulate real performance data, retrain the model to incorporate them:

```bash
python vertex/retrain_pipeline.py
```

The pipeline only promotes the new model if its cross-validation correlation improves by at least 0.005. Safe to run on a schedule (Cloud Scheduler + Cloud Run, or a simple cron job).

---

## Cost reference

| Step | Cost |
|---|---|
| Vision API labelling (per 1,000 images) | ~$1.50 |
| EfficientNet embedding extraction | $0 (runs locally) |
| Model training | $0 (runs locally) |
| GCS storage (model + data) | < $0.01/month |
| Vertex model registration | $0 |
| Batch prediction job | ~$0.05 per job |

A typical end-to-end setup for a 500-video channel costs **under $5**.

---

## Project structure

```
thumbnail-predictor/
├── scraper.py                    # YouTube API scraping
├── auto_label_vision.py          # Vision API feature extraction
├── score_thumbnails.py           # Rule-based scorer v5 (baseline)
├── inference.py                  # End-to-end CLI: thumbnail → score
│
├── vertex/
│   ├── extract_embeddings.py     # EfficientNetB0 embedding extraction
│   ├── train.py                  # Keras model training + cross-validation
│   ├── export_savedmodel.py      # Export + upload to GCS
│   ├── predictor.py              # Prediction logic
│   ├── serve.py                  # Flask wrapper (optional custom container)
│   ├── Dockerfile                # Custom container (optional)
│   └── retrain_pipeline.py       # Monthly retraining + auto-promotion
│
├── data/
│   ├── videos.csv                # Scraped video metadata
│   ├── auto_labels.csv           # Vision API labels
│   ├── videos_enriched.csv       # Merged dataset
│   └── thumbnail_weights_v5.json # Rule-based scorer weights
│
├── model/
│   └── meta.json                 # Model metadata + validation metrics
│
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── .env.example                  # Environment variable template
├── .gitignore
├── METHODOLOGY.md                # Technical methodology
├── FULL_STORY.md                 # Full narrative of every decision made
└── README.md                     # This file
```

---

## Key design decisions

| Decision | Chosen | Why |
|---|---|---|
| Performance metric | views ÷ subscribers | Normalises for channel size |
| Feature extraction | Google Vision API | One API, all features, ~$8 for 1,200 images |
| Image model | EfficientNetB0 frozen | Keras-native, no PyTorch, works with 1,000+ samples |
| Deployment | Vertex AI batch | Pay-per-use (~$0.05/job) vs online endpoint ($360+/month) |
| Normalisation | Within-channel z-score | Removes channel-size confound |

See `FULL_STORY.md` for the full explanation of every choice, every dead end, and the statistics behind each decision.

---

## Requirements

```
tensorflow>=2.20
keras>=3.14
scikit-learn>=1.9
google-cloud-vision>=3.0
google-cloud-aiplatform>=1.60
google-cloud-storage>=2.0
pandas>=2.0
numpy>=1.26
tqdm
python-dotenv
joblib
```
