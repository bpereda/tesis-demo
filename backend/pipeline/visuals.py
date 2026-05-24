from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _track_color(track_id: int) -> tuple[int, int, int]:
    palette = [
        (56, 126, 184),
        (77, 175, 74),
        (152, 78, 163),
        (255, 127, 0),
        (228, 26, 28),
        (166, 86, 40),
        (247, 129, 191),
        (153, 153, 153),
    ]
    return palette[int(track_id) % len(palette)]


def _read_frame(video_path: Path, frame_idx: int):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def save_rgb_preview(color_mp4: str | Path, out_jpg: str | Path, frame_idx: int = 0) -> Path | None:
    import cv2

    color_mp4 = Path(color_mp4).expanduser().resolve()
    out_jpg = Path(out_jpg).expanduser().resolve()
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    frame = _read_frame(color_mp4, frame_idx)
    if frame is None:
        return None
    cv2.imwrite(str(out_jpg), frame)
    return out_jpg


def save_tracking_overlays(
    color_mp4: str | Path,
    sam_npz: str | Path,
    tracked_csv: str | Path,
    out_dir: str | Path,
    max_images: int = 4,
) -> list[Path]:
    import cv2

    color_mp4 = Path(color_mp4).expanduser().resolve()
    sam_npz = Path(sam_npz).expanduser().resolve()
    tracked_csv = Path(tracked_csv).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tracked = pd.read_csv(tracked_csv)
    if tracked.empty:
        return []

    pack = np.load(str(sam_npz), allow_pickle=True)
    masks_by_frame = pack["masks"].item() if "masks" in pack.files else {k: pack[k] for k in pack.files}
    frames = sorted(int(frame) for frame in tracked["frame"].unique() if str(int(frame)) in masks_by_frame)
    if not frames:
        return []

    if len(frames) > max_images:
        indices = np.linspace(0, len(frames) - 1, max_images).round().astype(int)
        frames = [frames[i] for i in indices]

    paths = []
    for frame_idx in frames:
        frame = _read_frame(color_mp4, frame_idx)
        if frame is None:
            continue

        overlay = frame.copy()
        rows = tracked[tracked["frame"] == frame_idx].copy()
        masks = masks_by_frame[str(frame_idx)]

        for _, row in rows.iterrows():
            det_i = int(row["det_i"])
            track_id = int(row["track_id"])
            if det_i < 0 or det_i >= masks.shape[0]:
                continue

            color = _track_color(track_id)
            mask = masks[det_i].astype(bool)
            tint = np.zeros_like(frame)
            tint[:, :] = color
            overlay[mask] = cv2.addWeighted(overlay[mask], 0.45, tint[mask], 0.55, 0)

            x1, y1, x2, y2 = [int(round(row[c])) for c in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"ID {track_id}",
                (max(0, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        out_path = out_dir / f"tracking_overlay_frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(out_path), overlay)
        paths.append(out_path)

    return paths


def save_representative_contact_sheet(
    one_mask_npz: str | Path,
    out_jpg: str | Path,
    max_items: int = 24,
    tile_size: int = 144,
) -> Path | None:
    import cv2

    one_mask_npz = Path(one_mask_npz).expanduser().resolve()
    out_jpg = Path(out_jpg).expanduser().resolve()
    out_jpg.parent.mkdir(parents=True, exist_ok=True)

    pack = np.load(str(one_mask_npz), allow_pickle=True)
    if "masks" not in pack.files:
        return None
    masks = pack["masks"].item()
    if not masks:
        return None

    items = list(masks.items())[:max_items]
    cols = min(6, max(1, len(items)))
    rows = int(np.ceil(len(items) / cols))
    sheet = np.full((rows * tile_size, cols * tile_size, 3), 250, dtype=np.uint8)

    for idx, (track_id, mask) in enumerate(items):
        mask = np.asarray(mask).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[0]
        ys, xs = np.where(mask > 0)
        if xs.size == 0:
            continue

        crop = mask[max(0, ys.min() - 8) : ys.max() + 9, max(0, xs.min() - 8) : xs.max() + 9]
        crop = (crop > 0).astype(np.uint8) * 255
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        crop_rgb[crop > 0] = (122, 63, 115)

        h, w = crop_rgb.shape[:2]
        scale = min((tile_size - 30) / max(1, w), (tile_size - 42) / max(1, h))
        resized = cv2.resize(crop_rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_NEAREST)

        r = idx // cols
        c = idx % cols
        y0 = r * tile_size + 28
        x0 = c * tile_size + (tile_size - resized.shape[1]) // 2
        sheet[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        cv2.putText(
            sheet,
            f"ID {track_id}",
            (c * tile_size + 10, r * tile_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (38, 43, 39),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_jpg), sheet)
    return out_jpg
