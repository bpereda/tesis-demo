#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/tesis-demo}"
BACKEND_DIR="${BACKEND_DIR:-${PROJECT_DIR}/backend}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"
DEMO_DATA_DIR="${DEMO_DATA_DIR:-/workspace/demo_data}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV_DIR}/bin/activate" ]]; then
  source "${VENV_DIR}/bin/activate"
fi

export DEMO_DATA_DIR
cd "${BACKEND_DIR}"

exec uvicorn main:app --host "${HOST}" --port "${PORT}"
