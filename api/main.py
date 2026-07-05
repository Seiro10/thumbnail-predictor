"""
FastAPI backend for the thumbnail scorer.
Runs on http://localhost:8000

POST /score
  multipart/form-data:
    - file: image (jpg/png/webp)
    - subscriber_count: int
    - channel_avg_perf: float
    - niche: "AI/Tech" | "Business"

GET /health  → {"status": "ok"}
"""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

load_dotenv()

from scorer import score as run_score

app = FastAPI(title='Thumbnail Scorer API')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3001'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/score')
async def score_thumbnail(
    file:             UploadFile = File(...),
    subscriber_count: int        = Form(default=100_000),
    channel_avg_perf: float      = Form(default=0.003),
    niche:            str        = Form(default='AI/Tech'),
):
    if file.content_type not in ('image/jpeg', 'image/png', 'image/webp'):
        raise HTTPException(400, 'Only JPEG, PNG or WebP images are accepted.')

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(400, 'Image must be under 10 MB.')

    try:
        result = run_score(
            image_bytes      = image_bytes,
            subscriber_count = subscriber_count,
            channel_avg_perf = channel_avg_perf,
            niche            = niche,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    return result


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
