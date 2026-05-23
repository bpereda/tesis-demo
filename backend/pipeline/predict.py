from __future__ import annotations

from pathlib import Path


FEATURE_COLUMNS = [
    "mask_count",
    "mask_area_m2_sum",
    "mask_area_m2_p75",
    "mask_area_m2_std",
    "liters_totales",
]


def load_yield_model(model_path: str | Path):
    import joblib

    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_weight(model_path: str | Path | None, aggregate: dict) -> float | None:
    import pandas as pd

    if not model_path:
        return None

    model = load_yield_model(model_path)
    if model is None:
        return None

    model_features = list(getattr(model, "feature_names_in_", FEATURE_COLUMNS))
    row = {column: aggregate.get(column, 0.0) for column in model_features}
    frame = pd.DataFrame([row], columns=model_features)
    pred = model.predict(frame)
    return float(pred[0])
