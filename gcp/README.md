# GCP GPU VM Demo

This is the simplest reliable path for the thesis defense: run the FastAPI backend on a Compute Engine VM with one NVIDIA T4 GPU, then access it from your laptop through an SSH tunnel.

## 1. Create The VM

Install and authenticate the Google Cloud CLI on your laptop, then choose your project and zone:

```bash
cd tesis-demo
export PROJECT_ID="your-gcp-project-id"
export ZONE="us-central1-a"
./gcp/create_gpu_vm.sh
```

Defaults:

- VM name: `tesis-demo-gpu`
- GPU: one `nvidia-tesla-t4`
- Machine: `n1-standard-4`
- Boot disk: `200GB`
- Image family: `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`

If the zone has no T4 capacity or your quota is missing, try another zone or request GPU quota.

## 2. Copy The Project

From your laptop:

```bash
gcloud compute scp --recurse tesis-demo tesis-demo-gpu:~/tesis-demo --zone "$ZONE"
```

Copy model assets into the VM project:

```text
~/tesis-demo/backend/models/sam3.pt
~/tesis-demo/backend/models/calib_YYYYMMDD_HHMMSS.npz
~/tesis-demo/backend/models/modelo_final2.joblib
```

`modelo_final2.joblib` is already included in this local scaffold. The SAM checkpoint and calibration files still need to be added. Calibration is selected automatically from the timestamp in the `.bag` name.

## 3. Install Backend Dependencies

SSH into the VM:

```bash
gcloud compute ssh tesis-demo-gpu --zone "$ZONE"
```

On the VM:

```bash
cd ~/tesis-demo
chmod +x gcp/*.sh
./gcp/setup_backend.sh
```

Check the GPU:

```bash
nvidia-smi
source ~/tesis-demo-venv/bin/activate
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
PY
```

## 4. Smoke Test The CLI

Copy a short `.bag` file to the VM, then run a short frame-limited test:

```bash
cd ~/tesis-demo/backend
source ~/tesis-demo-venv/bin/activate
python -m pipeline.run_pipeline \
  --bag /path/to/input.bag \
  --out jobs/test_001 \
  --sam-model models/sam3.pt \
  --calib-dir models \
  --yield-model models/modelo_final2.joblib \
  --max-frames 30
```

If that works, remove `--max-frames 30` for the full demo.

## 5. Run FastAPI

On the VM:

```bash
cd ~/tesis-demo
./gcp/run_api.sh
```

Keep the API bound to `127.0.0.1` and use an SSH tunnel from your laptop:

```bash
gcloud compute ssh tesis-demo-gpu --zone "$ZONE" -- -L 8000:127.0.0.1:8000
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Upload the `.bag` through `POST /upload`.

## 6. Stop Costs

When you finish testing:

```bash
gcloud compute instances stop tesis-demo-gpu --zone "$ZONE"
```

Delete it when you no longer need it:

```bash
gcloud compute instances delete tesis-demo-gpu --zone "$ZONE"
```

## Troubleshooting

### `ImportError: libGL.so.1` From `cv2`

This usually means Python is importing the GUI OpenCV package (`opencv-python`) instead of the server-safe package (`opencv-python-headless`), or the virtual environment is not active.

On the VM:

```bash
cd ~/tesis-demo/backend
source ~/tesis-demo-venv/bin/activate
which python
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless
python -m pip install opencv-python-headless
python - <<'PY'
import cv2
print(cv2.__version__)
print(cv2.__file__)
PY
```

Then run the pipeline with `python`, not the system `python3`:

```bash
python -m pipeline.run_pipeline \
  --bag ~/demo_data/20240304_210426_qr_168_167.bag \
  --out jobs/test_001 \
  --sam-model models/sam3.pt \
  --calib-dir models \
  --yield-model models/modelo_final2.joblib \
  --max-frames 30
```
