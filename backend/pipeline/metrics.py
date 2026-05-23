from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def depth_to_color_depthmap(
    depth_m: np.ndarray,
    fx_d: float,
    fy_d: float,
    cx_d: float,
    cy_d: float,
    fx_c: float,
    fy_c: float,
    cx_c: float,
    cy_c: float,
    t_depth_to_color: np.ndarray,
    h_c: int,
    w_c: int,
) -> np.ndarray:
    h_d, w_d = depth_m.shape
    depth_on_color = np.zeros((h_c, w_c), dtype=np.float32)
    zbuf = np.full((h_c, w_c), np.inf, dtype=np.float32)
    vd, ud = np.indices((h_d, w_d), dtype=np.float32)
    z = depth_m.astype(np.float32)
    valid = z > 0
    if not np.any(valid):
        return depth_on_color

    udv = ud[valid]
    vdv = vd[valid]
    zv = z[valid]
    xd = (udv - cx_d) * zv / fx_d
    yd = (vdv - cy_d) * zv / fy_d
    pd3 = np.stack([xd, yd, zv, np.ones_like(zv, dtype=np.float32)], axis=1)
    pc3 = (t_depth_to_color @ pd3.T).T
    x_c = pc3[:, 0]
    y_c = pc3[:, 1]
    z_c = pc3[:, 2]
    ok = z_c > 0
    if not np.any(ok):
        return depth_on_color

    x_c = x_c[ok]
    y_c = y_c[ok]
    z_c = z_c[ok]
    u_c = np.round((x_c * fx_c / z_c) + cx_c).astype(np.int32)
    v_c = np.round((y_c * fy_c / z_c) + cy_c).astype(np.int32)
    inside = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c)

    for u, v, z_val in zip(u_c[inside], v_c[inside], z_c[inside]):
        if z_val < zbuf[v, u]:
            zbuf[v, u] = z_val
            depth_on_color[v, u] = z_val
    return depth_on_color


def compute_area_volume_one_mask_per_grape(
    reps_df: pd.DataFrame,
    one_mask_per_grape_npz: str | Path,
    depth_npz: str | Path,
    calib_npz: str | Path,
    out_csv: str | Path,
    min_mask_pixels: int = 150,
) -> pd.DataFrame:
    one_mask_per_grape_npz = Path(one_mask_per_grape_npz).expanduser().resolve()
    depth_npz = Path(depth_npz).expanduser().resolve()
    calib_npz = Path(calib_npz).expanduser().resolve()
    out_csv = Path(out_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    for path in [one_mask_per_grape_npz, depth_npz, calib_npz]:
        if not path.exists():
            raise FileNotFoundError(path)

    pack = np.load(str(one_mask_per_grape_npz), allow_pickle=True)
    masks_dict = pack["masks"].item()
    dep_npz = np.load(str(depth_npz), allow_pickle=True)
    calib = np.load(str(calib_npz), allow_pickle=True)

    fx_d, fy_d, cx_d, cy_d = map(float, [calib["fxD"], calib["fyD"], calib["cxD"], calib["cyD"]])
    fx_c, fy_c, cx_c, cy_c = map(float, [calib["fxC"], calib["fyC"], calib["cxC"], calib["cyC"]])
    t_depth_to_color = calib["T_depth_to_color"].astype(np.float32)
    w_d, h_d = int(calib["Wd"]), int(calib["Hd"])
    w_c, h_c = int(calib["Wc"]), int(calib["Hc"])

    rows = []
    skipped = {
        "missing_mask": 0,
        "missing_depth": 0,
        "tiny_mask": 0,
        "bad_shapes": 0,
        "no_valid_depth": 0,
        "too_few_depth_points": 0,
    }

    for _, row in reps_df.iterrows():
        tid = int(row["track_id"])
        frame = int(row["chosen_frame"])
        mask = masks_dict.get(str(tid))
        if mask is None:
            skipped["missing_mask"] += 1
            continue

        mask = np.asarray(mask).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[0]
        if mask.ndim != 2:
            skipped["bad_shapes"] += 1
            continue

        mask_bool = mask.astype(bool)
        area_px = int(mask_bool.sum())
        if area_px < min_mask_pixels:
            skipped["tiny_mask"] += 1
            continue

        depth_key = str(frame)
        if depth_key not in dep_npz.files:
            skipped["missing_depth"] += 1
            continue

        depth_m = dep_npz[depth_key].astype(np.float32)
        if depth_m.shape != (h_d, w_d):
            skipped["bad_shapes"] += 1
            continue

        depth_on_color = depth_to_color_depthmap(
            depth_m,
            fx_d,
            fy_d,
            cx_d,
            cy_d,
            fx_c,
            fy_c,
            cx_c,
            cy_c,
            t_depth_to_color,
            h_c,
            w_c,
        )
        if depth_on_color.shape != mask.shape:
            skipped["bad_shapes"] += 1
            continue

        vals = depth_on_color[mask_bool]
        vals = vals[(vals > 0) & np.isfinite(vals)]
        if vals.size == 0:
            skipped["no_valid_depth"] += 1
            mean_z = None
            area_m2 = None
            vol_cm3 = None
            thickness_m = None
            z_valid_ratio = 0.0
        else:
            mean_z = float(vals.mean())
            z_valid_ratio = float(vals.size / max(1, area_px))
            area_m2 = float(np.sum((vals * vals) / (fx_c * fy_c)))
            vol_cm3 = None
            thickness_m = None

            z = vals.copy()
            z_med = np.median(z)
            mad = np.median(np.abs(z - z_med))
            if mad > 0:
                z = z[np.abs(z - z_med) <= 3.5 * mad]
            if z.size < 20:
                skipped["too_few_depth_points"] += 1
            else:
                thickness_m = float(np.percentile(z, 90) - np.percentile(z, 10))
                if thickness_m > 0:
                    vol_cm3 = float(area_m2 * thickness_m * 1e6)

        rows.append(
            {
                "track_id": tid,
                "chosen_frame": frame,
                "mask_area_px": area_px,
                "z_valid_ratio": z_valid_ratio,
                "mask_area_m2": area_m2,
                "mean_depth_m": mean_z,
                "thickness_m": thickness_m,
                "volume_cm3_proxy": vol_cm3,
            }
        )

    print("Skipped:", skipped)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("track_id").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print("[OK] Saved metrics:", out_csv)
    return df


def aggregate_metrics(metrics_df: pd.DataFrame) -> dict:
    if metrics_df.empty:
        return {
            "mask_count": 0,
            "mask_area_m2_sum": 0.0,
            "mask_area_m2_p75": 0.0,
            "mask_area_m2_std": 0.0,
            "liters_totales": 0.0,
            "detected_clusters": 0,
            "total_estimated_volume_cm3": 0.0,
            "mean_depth_m": None,
            "total_mask_area_px": 0,
        }

    mask_area_m2 = metrics_df["mask_area_m2"].dropna()
    total_volume_cm3 = float(metrics_df["volume_cm3_proxy"].fillna(0).sum())

    return {
        "mask_count": int(metrics_df["track_id"].nunique()),
        "mask_area_m2_sum": float(mask_area_m2.sum()) if not mask_area_m2.empty else 0.0,
        "mask_area_m2_p75": float(mask_area_m2.quantile(0.75)) if not mask_area_m2.empty else 0.0,
        "mask_area_m2_std": float(mask_area_m2.std()) if len(mask_area_m2) > 1 else 0.0,
        "liters_totales": float(total_volume_cm3 / 1000.0),
        "detected_clusters": int(metrics_df["track_id"].nunique()),
        "total_estimated_volume_cm3": total_volume_cm3,
        "mean_depth_m": float(metrics_df["mean_depth_m"].dropna().mean()) if metrics_df["mean_depth_m"].notna().any() else None,
        "total_mask_area_px": int(metrics_df["mask_area_px"].fillna(0).sum()),
    }
