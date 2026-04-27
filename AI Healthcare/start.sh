#!/usr/bin/env bash
set -e

BACKEND_DIR="$(cd "$(dirname "$0")/backend" && pwd)"
VENV_DIR="$BACKEND_DIR/venv"
PORT="${PORT:-8000}"

echo ""
echo "  PneumoniaDetect"
echo "  ---------------"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo ""
  echo "  Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install / sync dependencies
echo ""
echo "  Checking dependencies..."
pip install -q -r "$BACKEND_DIR/requirements.txt"

# Start the server
echo ""
echo "  Starting server on http://localhost:$PORT"
echo "  API docs at  http://localhost:$PORT/docs"
echo ""
echo "  Press Ctrl+C to stop."
echo ""

cd "$BACKEND_DIR"
uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
