from __future__ import annotations

import shutil
import threading
import uuid
import json
from datetime import datetime, timezone
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
REAL_WEIGHTS_FILE = MODELS_DIR / "real_weights.json"
DEMO_DATA_DIR = Path.home() / "demo_data"

app = FastAPI(title="Demo de estimacion de cosecha en vinedos")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

JOB_STATUS: dict[str, dict] = {}
JOB_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_job_status(status_job_id: str, **updates) -> None:
    with JOB_LOCK:
        current = JOB_STATUS.setdefault(status_job_id, {})
        current.update(updates)
        current["updated_at"] = _now_iso()


def _load_real_weights() -> list[dict]:
    if not REAL_WEIGHTS_FILE.exists():
        return []
    payload = json.loads(REAL_WEIGHTS_FILE.read_text(encoding="utf-8"))
    return payload.get("entries", [])


def _find_real_weight(reference_key: str | None) -> dict | None:
    if not reference_key:
        return None
    for entry in _load_real_weights():
        if entry.get("key") == reference_key:
            return entry
    return None


def _attach_real_weight(result: dict, reference_key: str | None) -> None:
    reference = _find_real_weight(reference_key)
    if reference is None:
        return

    predicted = result.get("predicted_weight")
    real_weight = reference["real_weight_kg"]
    comparison = {
        "reference_key": reference["key"],
        "primary_id": reference["primary_id"],
        "paired_id": reference.get("paired_id"),
        "aliases": reference.get("aliases", []),
        "real_weight_kg": real_weight,
    }
    if predicted is not None:
        error_kg = float(predicted) - real_weight
        comparison.update(
            {
                "error_kg": error_kg,
                "absolute_error_kg": abs(error_kg),
                "error_percent": abs(error_kg) / real_weight * 100 if real_weight else None,
            }
        )
    result["real_weight_comparison"] = comparison


def _run_job(job_id: str, bag_path: Path, job_dir: Path, max_frames: int, reference_key: str | None) -> None:
    def progress(stage: str, percent: int, message: str) -> None:
        _set_job_status(job_id, state="running", stage=stage, percent=percent, message=message)

    try:
        progress("extract", 10, "Iniciando el análisis en la VM.")
        result = run_pipeline(
            bag=bag_path,
            out=job_dir,
            sam_model=DEFAULT_SAM_MODEL,
            calib_dir=MODELS_DIR,
            yield_model=DEFAULT_YIELD_MODEL if DEFAULT_YIELD_MODEL.exists() else None,
            max_frames=max_frames,
            progress_callback=progress,
        )
        result["job_id"] = job_id
        _attach_real_weight(result, reference_key)
        (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        _set_job_status(
            job_id,
            state="complete",
            stage="complete",
            percent=100,
            message="La estimacion de peso y las imagenes del analisis estan listas.",
            result=result,
        )
    except Exception as exc:
        _set_job_status(job_id, state="error", stage="error", percent=0, message=str(exc), error=str(exc))


def _validate_max_frames(max_frames: int) -> None:
    if max_frames == 0 or max_frames < -1:
        raise HTTPException(status_code=400, detail="max_frames debe ser -1 o un entero positivo.")


def _start_job(bag_path: Path, max_frames: int, queued_message: str, reference_key: str | None = None) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    _set_job_status(
        job_id,
        state="queued",
        stage="upload",
        percent=5,
        message=queued_message,
        created_at=_now_iso(),
        job_id=job_id,
        bag=str(bag_path),
        reference_key=reference_key,
    )
    thread = threading.Thread(target=_run_job, args=(job_id, bag_path, job_dir, max_frames, reference_key), daemon=True)
    thread.start()
    return {"job_id": job_id, "state": "queued"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/demo-files")
def list_demo_files() -> dict:
    if not DEMO_DATA_DIR.exists():
        return {"demo_data_dir": str(DEMO_DATA_DIR), "files": []}

    files = []
    for path in sorted(DEMO_DATA_DIR.glob("*.bag")):
        if path.is_file():
            files.append({"name": path.name, "path": str(path), "size_bytes": path.stat().st_size})
    return {"demo_data_dir": str(DEMO_DATA_DIR), "files": files}


@app.get("/real-weights")
def list_real_weights() -> dict:
    entries = _load_real_weights()
    return {"unit": "kg", "entries": entries}


@app.post("/upload")
async def upload_bag(
    file: UploadFile = File(...),
    max_frames: int = Form(-1),
    reference_key: str | None = Form(None),
) -> dict:
    if not file.filename or not file.filename.endswith(".bag"):
        raise HTTPException(status_code=400, detail="Debe subir un archivo .bag.")
    _validate_max_frames(max_frames)

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    bag_path = job_dir / Path(file.filename).name

    try:
        with bag_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        _set_job_status(job_id, job_id=job_id, bag=str(bag_path), reference_key=reference_key)
        thread = threading.Thread(target=_run_job, args=(job_id, bag_path, job_dir, max_frames, reference_key), daemon=True)
        _set_job_status(
            job_id,
            state="queued",
            stage="upload",
            percent=5,
            message="Video cargado. Esperando el inicio del analisis.",
            created_at=_now_iso(),
        )
        thread.start()
        return {"job_id": job_id, "state": "queued"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/run-demo-file")
async def run_demo_file(
    filename: str = Form(...),
    max_frames: int = Form(-1),
    reference_key: str | None = Form(None),
) -> dict:
    _validate_max_frames(max_frames)
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo de demo invalido.")

    bag_path = (DEMO_DATA_DIR / filename).resolve()
    demo_root = DEMO_DATA_DIR.resolve()
    if not str(bag_path).startswith(str(demo_root)) or not bag_path.exists() or bag_path.suffix != ".bag":
        raise HTTPException(status_code=404, detail="No se encontro el archivo .bag de demo.")

    return _start_job(bag_path, max_frames, "Usando video .bag ya disponible en la VM.", reference_key)


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str) -> dict:
    with JOB_LOCK:
        status = JOB_STATUS.get(job_id)
        if status is None:
            result_path = JOBS_DIR / job_id / "result.json"
            if result_path.exists():
                return {
                    "job_id": job_id,
                    "state": "complete",
                    "stage": "complete",
                    "percent": 100,
                    "message": "Resultado completado cargado desde disco.",
                    "result": __import__("json").loads(result_path.read_text(encoding="utf-8")),
                }
            raise HTTPException(status_code=404, detail="Trabajo no encontrado.")
        return dict(status)


@app.get("/jobs/{job_id}/{file_path:path}")
def get_job_file(job_id: str, file_path: str) -> FileResponse:
    job_dir = (JOBS_DIR / job_id).resolve()
    target = (job_dir / file_path).resolve()
    if not str(target).startswith(str(job_dir)):
        raise HTTPException(status_code=400, detail="Ruta de archivo invalida.")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return FileResponse(target)
