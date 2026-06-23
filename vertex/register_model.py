"""
Register the uploaded SavedModel in Vertex AI Model Registry.
Run after export_savedmodel.py has uploaded the model to GCS.

Prints the Model ID — copy it into your .env as VERTEX_MODEL_ID.
"""
import os
from dotenv import load_dotenv
load_dotenv()

from google.cloud import aiplatform

PROJECT    = os.environ['GCP_PROJECT']
LOCATION   = os.getenv('GCP_REGION', 'europe-west1')
GCS_BUCKET = os.environ['GCS_BUCKET']

aiplatform.init(project=PROJECT, location=LOCATION)

model = aiplatform.Model.upload(
    display_name='thumbnail-scorer-v1',
    artifact_uri=f'gs://{GCS_BUCKET}/saved_model/saved_model',
    serving_container_image_uri=(
        'europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-15:latest'
    ),
    description='EfficientNetB0 + tabular features → thumbnail score 0-20',
)

print(f"\nModel registered successfully.")
print(f"  Display name:  {model.display_name}")
print(f"  Resource name: {model.resource_name}")
print(f"\nCopy this Model ID into your .env:")
print(f"  VERTEX_MODEL_ID={model.name}")
