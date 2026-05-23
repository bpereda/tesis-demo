from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from pipeline.run_pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_SAM_MODEL = MODELS_DIR / "sam3.pt"
DEFAULT_YIELD_MODEL = MODELS_DIR / "modelo_final2.joblib"

app = FastAPI(title="Tesis Vineyard Yield Demo")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_bag(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.endswith(".bag"):
        raise HTTPException(status_code=400, detail="Please upload a .bag file.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    bag_path = job_dir / Path(file.filename).name

    try:
        with bag_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        result = run_pipeline(
            bag=bag_path,
            out=job_dir,
            sam_model=DEFAULT_SAM_MODEL,
            calib_dir=MODELS_DIR,
            yield_model=DEFAULT_YIELD_MODEL if DEFAULT_YIELD_MODEL.exists() else None,
        )
        result["job_id"] = job_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
