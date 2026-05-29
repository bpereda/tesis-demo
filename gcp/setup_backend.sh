#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/tesis-demo}"
BACKEND_DIR="${BACKEND_DIR:-${PROJECT_DIR}/backend}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/venv}"

if [ ! -d "${BACKEND_DIR}" ]; then
  echo "Backend directory not found: ${BACKEND_DIR}" >&2
  echo "Copy tesis-demo to the VM first, for example:" >&2
  echo "  gcloud compute scp --recurse tesis-demo tesis-demo-gpu:/workspace/tesis-demo --zone us-central1-a" >&2
  exit 1
fi

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r "${BACKEND_DIR}/requirements.txt"

echo
echo "Backend dependencies installed in ${VENV_DIR}"
echo "Check GPU from Python with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python - <<'PY'"
echo "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
echo "PY"
