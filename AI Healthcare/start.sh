#!/usr/bin/env bash
# Start the FastAPI backend.
# The venv must already exist — run ./setup_home.sh once before this script.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$HERE/backend"
PORT="${PORT:-8000}"

# Prefer the project-level venv created by setup_home.sh; fall back to the
# legacy backend/venv layout for backwards compatibility.
if   [ -d "$HERE/venv" ]; then         VENV_DIR="$HERE/venv"
elif [ -d "$BACKEND_DIR/venv" ]; then  VENV_DIR="$BACKEND_DIR/venv"
else
  echo "No venv found. Run ./setup_home.sh first." >&2
  exit 1
fi

echo ""
echo "  PneumoniaDetect"
echo "  ---------------"
echo "  Venv:  $VENV_DIR"
echo "  Port:  $PORT"
echo ""

# On Windows (Git Bash/MSYS) the venv layout is Scripts/, on Linux/Mac it's bin/
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
  ACTIVATE="$VENV_DIR/Scripts/activate"
else
  ACTIVATE="$VENV_DIR/bin/activate"
fi
# shellcheck disable=SC1090
source "$ACTIVATE"

echo "  Starting server on http://localhost:$PORT"
echo "  API docs at        http://localhost:$PORT/docs"
echo "  Press Ctrl+C to stop."
echo ""

cd "$BACKEND_DIR"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
