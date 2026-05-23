from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def box_iou_xyxy(a, b) -> float:
    x_a = max(a[0], b[0])
    y_a = max(a[1], b[1])
    x_b = min(a[2], b[2])
    y_b = min(a[3], b[3])
    inter = max(0, x_b - x_a) * max(0, y_b - y_a)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    den = area_a + area_b - inter
    return float(inter / den) if den > 0 else 0.0


def track_detections_simple(
    per_frame_csv: str | Path,
    out_tracked_csv: str | Path,
    iou_th: float = 0.15,
    max_center_dist: float = 80.0,
    max_age: int = 15,
) -> pd.DataFrame:
    df = pd.read_csv(per_frame_csv)
    out_tracked_csv = Path(out_tracked_csv).expanduser().resolve()
    out_tracked_csv.parent.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        df["track_id"] = []
        df.to_csv(out_tracked_csv, index=False)
        print("[WARN] No detections. Saved empty:", out_tracked_csv)
        return df

    df = df.sort_values(["frame", "det_i"]).reset_index(drop=True)
    next_tid = 0
    tracks = {}
    tracked_rows = []

    for frame, group in df.groupby("frame", sort=True):
        dets = group.to_dict("records")
        assigned = [-1] * len(dets)
        used_tracks = set()
        candidates = []

        for det_idx, det in enumerate(dets):
            det_box = [det["x1"], det["y1"], det["x2"], det["y2"]]
            det_cx, det_cy = det["cx"], det["cy"]
            for tid, track in tracks.items():
                if frame - track["last_frame"] > max_age:
                    continue
                iou = box_iou_xyxy(det_box, track["last_box"])
                dist = np.hypot(det_cx - track["last_cx"], det_cy - track["last_cy"])
                if iou >= iou_th or dist <= max_center_dist:
                    candidates.append((iou - 0.001 * dist, tid, det_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])
        for _score, tid, det_idx in candidates:
            if assigned[det_idx] != -1 or tid in used_tracks:
                continue
            assigned[det_idx] = tid
            used_tracks.add(tid)

        for det_idx, det in enumerate(dets):
            if assigned[det_idx] == -1:
                tid = next_tid
                next_tid += 1
                assigned[det_idx] = tid
                used_tracks.add(tid)

            tid = assigned[det_idx]
            tracks[tid] = {
                "last_box": [det["x1"], det["y1"], det["x2"], det["y2"]],
                "last_cx": float(det["cx"]),
                "last_cy": float(det["cy"]),
                "last_frame": int(frame),
            }

            out_row = dict(det)
            out_row["track_id"] = int(tid)
            tracked_rows.append(out_row)

        dead = [tid for tid, track in tracks.items() if frame - track["last_frame"] > max_age]
        for tid in dead:
            tracks.pop(tid, None)

    out = pd.DataFrame(tracked_rows).sort_values(["frame", "track_id", "det_i"]).reset_index(drop=True)
    out.to_csv(out_tracked_csv, index=False)
    print("[OK] tracked CSV:", out_tracked_csv)
    print("unique track_id:", out["track_id"].nunique())
    return out


def choose_representative_per_track(
    tracked_df: pd.DataFrame,
    iqr_k: float = 1.5,
    min_frames: int = 3,
) -> pd.DataFrame:
    reps = []
    if tracked_df.empty:
        return pd.DataFrame(columns=["track_id", "chosen_frame", "chosen_det_i", "mask_area_px", "conf"])

    for tid, group in tracked_df.groupby("track_id"):
        if len(group) < min_frames:
            continue

        areas = group["mask_area_px"].astype(float).values
        q1 = np.percentile(areas, 25)
        q3 = np.percentile(areas, 75)
        iqr = q3 - q1
        group2 = group[(group["mask_area_px"] >= q1 - iqr_k * iqr) & (group["mask_area_px"] <= q3 + iqr_k * iqr)]
        if len(group2) == 0:
            group2 = group

        med = float(np.median(group2["mask_area_px"].values))
        group2 = group2.copy()
        group2["dist_to_median"] = (group2["mask_area_px"].astype(float) - med).abs()
        pick = group2.sort_values(["dist_to_median", "conf"], ascending=[True, False]).iloc[0]

        reps.append(
            {
                "track_id": int(tid),
                "chosen_frame": int(pick["frame"]),
                "chosen_det_i": int(pick["det_i"]),
                "mask_area_px": int(pick["mask_area_px"]),
                "conf": float(pick["conf"]),
            }
        )

    reps_df = pd.DataFrame(reps)
    if not reps_df.empty:
        reps_df = reps_df.sort_values("track_id").reset_index(drop=True)
    print("[OK] tracks kept:", len(reps_df))
    return reps_df


def save_representative_masks_npz(
    reps_df: pd.DataFrame,
    per_frame_npz_path: str | Path,
    out_npz_path: str | Path,
) -> str:
    per_frame_npz_path = Path(per_frame_npz_path).expanduser().resolve()
    out_npz_path = Path(out_npz_path).expanduser().resolve()
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    pack = np.load(str(per_frame_npz_path), allow_pickle=True)
    masks_dict = pack["masks"].item() if "masks" in pack.files else {k: pack[k] for k in pack.files}
    representative_masks = {}
    representative_meta = {}
    missing = 0

    for _, row in reps_df.iterrows():
        tid = int(row["track_id"])
        frame_key = str(int(row["chosen_frame"]))
        det_i = int(row["chosen_det_i"])
        if frame_key not in masks_dict or det_i < 0 or det_i >= masks_dict[frame_key].shape[0]:
            missing += 1
            continue
        representative_masks[str(tid)] = masks_dict[frame_key][det_i].astype(np.uint8)
        representative_meta[str(tid)] = {
            "frame": int(row["chosen_frame"]),
            "det_i": det_i,
            "area_px": int(row["mask_area_px"]),
            "conf": float(row["conf"]),
        }

    np.savez_compressed(str(out_npz_path), masks=representative_masks, meta=representative_meta)
    print("[OK] Saved representative masks:", out_npz_path)
    if missing:
        print("[WARN] missing masks:", missing)
    return str(out_npz_path)
