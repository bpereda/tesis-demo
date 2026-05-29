from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def validate_sam_checkpoint(model_path: Path) -> None:
    size_bytes = model_path.stat().st_size
    if size_bytes == 0:
        raise RuntimeError(f"El checkpoint SAM esta vacio: {model_path}")

    head = model_path.read_bytes()[:256]
    if head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            "El checkpoint SAM parece ser un pointer de Git LFS, no el modelo real. "
            f"Revisar/copiar nuevamente: {model_path}"
        )

    if size_bytes < 1_000_000:
        size_kb = size_bytes / 1024
        raise RuntimeError(
            f"El checkpoint SAM pesa solo {size_kb:.1f} KB. "
            "Probablemente esta incompleto o se copio mal. "
            f"Revisar/copiar nuevamente: {model_path}"
        )


def mask_centroid(mask_u8: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask_u8.astype(bool))
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def box_from_mask(mask_u8: np.ndarray) -> list[float] | None:
    ys, xs = np.where(mask_u8.astype(bool))
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def split_mask_into_pixel_components(mask_u8: np.ndarray, min_pixels: int = 150) -> list[np.ndarray]:
    mask_u8 = (mask_u8 > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8)
    parts = []
    for lab in range(1, num_labels):
        comp = (labels == lab).astype(np.uint8)
        if int(comp.sum()) >= int(min_pixels):
            parts.append(comp)
    return parts


def run_sam3_per_frame(
    in_mp4: str | Path,
    model_path: str | Path,
    prompts: tuple[str, ...] | list[str],
    out_csv: str | Path,
    out_npz: str | Path,
    device: str = "0",
    half: bool = True,
    conf: float = 0.25,
    min_mask_pixels: int = 150,
    max_frames: int = -1,
    every_n: int = 1,
) -> pd.DataFrame:
    from ultralytics.models.sam import SAM3SemanticPredictor

    in_mp4 = Path(in_mp4).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    out_csv = Path(out_csv).expanduser().resolve()
    out_npz = Path(out_npz).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if not in_mp4.exists():
        raise FileNotFoundError(f"Video not found: {in_mp4}")
    if not model_path.exists():
        raise FileNotFoundError(f"SAM model not found: {model_path}")
    validate_sam_checkpoint(model_path)

    predictor = SAM3SemanticPredictor(
        overrides={
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "half": half,
            "device": device,
            "save": False,
            "verbose": False,
        }
    )

    cap = cv2.VideoCapture(str(in_mp4))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {in_mp4}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = []
    masks_dict = {}
    boxes_dict = {}
    confs_dict = {}
    frame_idx = 0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_idx >= max_frames:
            break
        if every_n > 1 and frame_idx % every_n != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        result = predictor(text=list(prompts))[0]
        det_masks, det_boxes, det_confs = [], [], []

        if result.masks is not None and result.boxes is not None and len(result.boxes) > 0:
            masks = (result.masks.data > 0.5).to("cpu").numpy().astype(np.uint8)
            confs = result.boxes.conf.cpu().numpy().astype(np.float32)
            mask_h, mask_w = masks.shape[1], masks.shape[2]
            need_resize = (mask_h != height) or (mask_w != width)

            for i in range(min(len(confs), masks.shape[0])):
                mask = masks[i]
                if need_resize:
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                for comp in split_mask_into_pixel_components(mask, min_pixels=min_mask_pixels):
                    area = int(comp.sum())
                    box = box_from_mask(comp)
                    center = mask_centroid(comp)
                    if area < min_mask_pixels or box is None or center is None:
                        continue

                    det_masks.append(comp.astype(np.uint8))
                    det_boxes.append(box)
                    det_confs.append(float(confs[i]))
                    rows.append(
                        {
                            "frame": int(frame_idx),
                            "det_i": int(len(det_masks) - 1),
                            "conf": float(confs[i]),
                            "mask_area_px": int(area),
                            "cx": float(center[0]),
                            "cy": float(center[1]),
                            "x1": float(box[0]),
                            "y1": float(box[1]),
                            "x2": float(box[2]),
                            "y2": float(box[3]),
                        }
                    )

        if det_masks:
            masks_dict[str(frame_idx)] = np.stack(det_masks, axis=0).astype(np.uint8)
            boxes_dict[str(frame_idx)] = np.array(det_boxes, dtype=np.float32)
            confs_dict[str(frame_idx)] = np.array(det_confs, dtype=np.float32)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print("processed frames:", frame_idx)

    cap.release()
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    np.savez_compressed(str(out_npz), masks=masks_dict, boxes=boxes_dict, confs=confs_dict)
    print("[OK] per-frame CSV:", out_csv)
    print("[OK] per-frame NPZ:", out_npz)
    return df
