#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR/Frontend"
npm run build

cd "$ROOT_DIR/Backend"
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
