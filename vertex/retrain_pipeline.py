"""
Vertex AI Pipeline — monthly retraining loop.

Steps:
  1. Pull latest performance data from GCS (enriched CSV)
  2. Re-extract embeddings for any new thumbnails
  3. Retrain the model (same architecture as train.py)
  4. Evaluate: only promote if new model CV-corr > current model CV-corr
  5. Upload new SavedModel to GCS and register new version in Vertex

Run manually:
  python vertex/retrain_pipeline.py

Or schedule via Cloud Scheduler + Cloud Run (see README).
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import subprocess
import numpy as np
from google.cloud import aiplatform, storage
from pathlib import Path
from datetime import datetime

import os
from dotenv import load_dotenv
load_dotenv()

PROJECT    = os.environ['GCP_PROJECT']
LOCATION   = os.getenv('GCP_REGION', 'europe-west1')
GCS_BUCKET = os.environ['GCS_BUCKET']

def get_current_model_corr() -> float:
    client = storage.Client(project=PROJECT)
    bucket = client.bucket(GCS_BUCKET)
    blob   = bucket.blob('model/meta.json')
    meta   = json.loads(blob.download_as_text())
    return meta.get('cv_corr_mean', 0.0)

def retrain_and_evaluate() -> dict:
    result = subprocess.run(
        ['python', 'vertex/train.py'],
        capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )
    if result.returncode != 0:
        raise RuntimeError(f"Training failed:\n{result.stderr}")
    print(result.stdout)

    with open('model/meta.json') as f:
        return json.load(f)

def upload_new_version(version_tag: str):
    subprocess.run([
        'gcloud', 'storage', 'cp', '-r',
        'model/saved_model',
        f'gs://{GCS_BUCKET}/saved_model_{version_tag}/'
    ], check=True)

    aiplatform.init(project=PROJECT, location=LOCATION)
    model = aiplatform.Model.upload(
        display_name=f'thumbnail-scorer-{version_tag}',
        artifact_uri=f'gs://{GCS_BUCKET}/saved_model_{version_tag}/saved_model',
        serving_container_image_uri=(
            'europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest'
        ),
        description=f'Retrained {version_tag}. CV-corr from meta.json.',
    )
    print(f"New model registered: {model.resource_name}")
    return model

def main():
    print("="*55)
    print("THUMBNAIL SCORER — RETRAINING PIPELINE")
    print("="*55)

    current_corr = get_current_model_corr()
    print(f"\nCurrent model CV-corr: {current_corr:+.4f}")

    print("\nRetraining...")
    new_meta = retrain_and_evaluate()
    new_corr = new_meta['cv_corr_mean']
    print(f"New model CV-corr:     {new_corr:+.4f}")

    if new_corr > current_corr + 0.005:
        version_tag = datetime.now().strftime('%Y%m%d')
        print(f"\nImprovement detected (+{new_corr - current_corr:.4f}). Promoting {version_tag}...")
        upload_new_version(version_tag)

        # Update canonical meta on GCS
        client = storage.Client(project=PROJECT)
        bucket = client.bucket(GCS_BUCKET)
        bucket.blob('model/meta.json').upload_from_string(
            json.dumps(new_meta, indent=2), content_type='application/json'
        )
        print("Done. New version is live.")
    else:
        print(f"\nNo significant improvement. Current model retained.")

if __name__ == '__main__':
    main()
