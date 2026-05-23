#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/tesis-demo}"
BACKEND_DIR="${PROJECT_DIR}/backend"
VENV_DIR="${VENV_DIR:-$HOME/tesis-demo-venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

source "${VENV_DIR}/bin/activate"
cd "${BACKEND_DIR}"

exec uvicorn main:app --host "${HOST}" --port "${PORT}"
