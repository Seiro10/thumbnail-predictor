#!/bin/bash
# Start both the FastAPI backend and the Next.js frontend

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Thumbnail Scorer..."
echo ""

# Backend
source "$ROOT/pyenv/bin/activate"
cd "$ROOT"
echo "[API]  Starting FastAPI on http://localhost:8000"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Frontend
cd "$ROOT/frontend"
echo "[APP]  Starting Next.js on http://localhost:3000"
npm run dev &
FE_PID=$!

echo ""
echo "  API  → http://localhost:8000/health"
echo "  App  → http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop both servers."

trap "kill $API_PID $FE_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
