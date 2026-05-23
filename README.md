# Tesis Grape Yield Demo

Minimal first milestone for the defense demo: run the RealSense computer vision pipeline from the command line, then expose it through FastAPI.

## Layout

```text
tesis-demo/
  backend/
    main.py
    pipeline/
      run_pipeline.py
      extract_rgb.py
      segmentation.py
      tracking.py
      metrics.py
      predict.py
    models/
      sam3.pt
      calib_YYYYMMDD_HHMMSS.npz
      modelo_final2.joblib
    jobs/
```

## Local CLI

Install dependencies in a Python environment with GPU-enabled PyTorch/Ultralytics if segmentation should use CUDA.

If you need to create the calibration file from a RealSense bag that contains camera info and transform topics:

```bash
cd tesis-demo/backend
python -m pipeline.calibration \
  --bag /path/to/calibration_or_compact.bag \
  --out models/calibration.npz
```

```bash
cd tesis-demo/backend
pip install -r requirements.txt
python -m pipeline.run_pipeline \
  --bag /path/to/input.bag \
  --out jobs/test_001 \
  --sam-model models/sam3.pt \
  --calib-dir models \
  --yield-model models/modelo_final2.joblib
```

For a quick smoke test on a short slice:

```bash
python -m pipeline.run_pipeline \
  --bag /path/to/input.bag \
  --out jobs/test_001 \
  --sam-model models/sam3.pt \
  --calib-dir models \
  --max-frames 30
```

## FastAPI

```bash
cd tesis-demo/backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Upload:

```bash
curl -F "file=@/path/to/input.bag" http://localhost:8000/upload
```

The server stores each run in `backend/jobs/{job_id}` and returns `predicted_weight`, `detected_clusters`, `total_estimated_volume_cm3`, and paths to the generated outputs.

## GCP GPU VM

For the defense demo, use the VM-first guide in [gcp/README.md](gcp/README.md). It creates a Compute Engine VM with one NVIDIA T4 GPU, copies this project to the VM, runs the CLI smoke test, then serves FastAPI through an SSH tunnel.

## Model and Calibration Files

Place these files before running a full demo:

- `backend/models/sam3.pt`: SAM3 segmentation checkpoint.
- `backend/models/calib_YYYYMMDD_HHMMSS.npz`: calibration with `fxD`, `fyD`, `cxD`, `cyD`, `Wd`, `Hd`, `fxC`, `fyC`, `cxC`, `cyC`, `Wc`, `Hc`, and `T_depth_to_color`. The timestamp should match the timestamp inside the `.bag` filename. The pipeline also accepts `calib__YYYYMMDD_HHMMSS.npz` and `calib_from_compact_YYYYMMDD_HHMMSS.npz`.
- `backend/models/modelo_final2.joblib`: optional trained regression model. It expects `mask_count`, `mask_area_m2_sum`, `mask_area_m2_p75`, `mask_area_m2_std`, and `liters_totales`. If missing, the pipeline still returns metrics and `predicted_weight: null`.
# tesis-demo
