FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow==2.21.0 \
    keras==3.14.1 \
    scikit-learn==1.9.0 \
    joblib \
    google-cloud-storage \
    flask \
    gunicorn

COPY predictor.py .
COPY serve.py .

ENV MODEL_DIR=/app/model
ENV PORT=8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "serve:app"]
