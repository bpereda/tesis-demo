from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pipeline.run_pipeline import run_pipeline


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "frontend"
JOBS_DIR = BASE_DIR / "jobs"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_SAM_MODEL = MODELS_DIR / "sam3.pt"
DEFAULT_YIELD_MODEL = MODELS_DIR / "modelo_final2.joblib"

app = FastAPI(title="Tesis Vineyard Yield Demo")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/upload")
async def upload_bag(
    file: UploadFile = File(...),
    max_frames: int = Form(-1),
) -> dict:
    if not file.filename or not file.filename.endswith(".bag"):
        raise HTTPException(status_code=400, detail="Please upload a .bag file.")
    if max_frames == 0 or max_frames < -1:
        raise HTTPException(status_code=400, detail="max_frames must be -1 or a positive integer.")

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
            max_frames=max_frames,
        )
        result["job_id"] = job_id
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/jobs/{job_id}/{file_path:path}")
def get_job_file(job_id: str, file_path: str) -> FileResponse:
    job_dir = (JOBS_DIR / job_id).resolve()
    target = (job_dir / file_path).resolve()
    if not str(target).startswith(str(job_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)
