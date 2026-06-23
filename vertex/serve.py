"""
Flask server wrapping ThumbnailPredictor for Vertex AI batch prediction container.
Vertex sends POST /predict with {"instances": [...]} and expects {"predictions": [...]}.
"""
import os
import json
from flask import Flask, request, jsonify
from predictor import ThumbnailPredictor

app = Flask(__name__)

MODEL_DIR = os.environ.get('MODEL_DIR', '/app/model')
predictor = ThumbnailPredictor(MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json(force=True)
    instances = body.get('instances', [])
    predictions = predictor.predict(instances)
    return jsonify({'predictions': predictions})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})
