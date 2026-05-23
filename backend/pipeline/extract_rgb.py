from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


COLOR_TOPIC = "/device_0/sensor_1/Color_0/image/data"
DEPTH_TOPIC = "/device_0/sensor_0/Depth_0/image/data"


def _read_ros1_string(buf: bytes, offset: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    value = buf[offset : offset + n].decode("utf-8", errors="replace")
    offset += n
    return value, offset


def parse_sensor_msgs_image_ros1(raw: bytes) -> tuple[int, int, str, int, bytes]:
    off = 0
    off += 4
    off += 8
    _, off = _read_ros1_string(raw, off)

    height, width = struct.unpack_from("<II", raw, off)
    off += 8
    encoding, off = _read_ros1_string(raw, off)
    off += 1
    (step,) = struct.unpack_from("<I", raw, off)
    off += 4
    (data_len,) = struct.unpack_from("<I", raw, off)
    off += 4
    data = raw[off : off + data_len]
    return int(height), int(width), encoding, int(step), data


def decode_color_image(height: int, width: int, encoding: str, data: bytes) -> np.ndarray:
    enc = encoding.lower()
    if enc == "rgb8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "bgr8":
        return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    if enc == "rgba8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        img = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported color encoding: {encoding}")


def decode_depth_to_m_from_raw(raw: bytes) -> tuple[np.ndarray, tuple[int, int, str, int]]:
    height, width, encoding, step, data = parse_sensor_msgs_image_ros1(raw)
    if encoding.lower() not in {"16uc1", "mono16"}:
        raise ValueError(f"Unsupported depth encoding: {encoding}")
    depth_u16 = np.frombuffer(data, dtype=np.uint16).reshape(height, width)
    depth_m = depth_u16.astype(np.float32) / 1000.0
    return depth_m, (height, width, encoding, step)


def infer_fps_from_timestamps_ns(ts_list: list[int], fallback: float = 30.0) -> float:
    if len(ts_list) < 10:
        return float(fallback)
    ts = np.array(ts_list, dtype=np.int64)
    dt = np.diff(ts) / 1e9
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        return float(fallback)
    med = float(np.median(dt))
    if med <= 0:
        return float(fallback)
    return float(np.clip(1.0 / med, 1.0, 120.0))


def extract_color_mp4_from_rosbag(
    bag_path: str | Path,
    out_mp4: str | Path,
    out_timestamps_csv: str | Path,
    color_topic: str = COLOR_TOPIC,
    max_frames: int = -1,
    every_n: int = 1,
    fps_fallback: float = 30.0,
    fps_warmup: int = 60,
) -> None:
    bag_path = Path(bag_path).expanduser().resolve()
    out_mp4 = Path(out_mp4).expanduser().resolve()
    out_timestamps_csv = Path(out_timestamps_csv).expanduser().resolve()

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_timestamps_csv.parent.mkdir(parents=True, exist_ok=True)

    ts_rows = []
    ts_for_fps: list[int] = []
    writer = None
    fps = float(fps_fallback)
    frame_idx = 0
    written = 0

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == color_topic]
        if not conns:
            raise RuntimeError(f"Color topic not found in bag: {color_topic}")

        for _conn, ts, raw in reader.messages(connections=conns):
            if max_frames > 0 and frame_idx >= max_frames:
                break
            if every_n > 1 and frame_idx % every_n != 0:
                frame_idx += 1
                continue

            height, width, encoding, _step, data = parse_sensor_msgs_image_ros1(raw)
            bgr = decode_color_image(height, width, encoding, data)

            if writer is None:
                ts_for_fps.append(int(ts))
                if len(ts_for_fps) >= min(fps_warmup, 10):
                    fps = infer_fps_from_timestamps_ns(ts_for_fps, fallback=fps_fallback)
                h, w = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_mp4),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(fps),
                    (w, h),
                )
            elif len(ts_for_fps) < fps_warmup:
                ts_for_fps.append(int(ts))

            writer.write(bgr)
            ts_rows.append({"frame": int(frame_idx), "timestamp_ns": int(ts)})
            written += 1
            frame_idx += 1

    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError("No RGB frames extracted. Check topic name and bag content.")

    pd.DataFrame(ts_rows).to_csv(out_timestamps_csv, index=False)
    print(f"[OK] RGB extract: frames={written} | fps≈{fps:.2f}")


def save_depth_frames_meters_npz_from_bag(
    bag_path: str | Path,
    rgb_timestamps_csv: str | Path,
    out_depth_npz: str | Path,
    depth_topic: str = DEPTH_TOPIC,
    ts_col: str = "timestamp_ns",
    frame_col: str = "frame",
    max_frames: int = -1,
) -> None:
    bag_path = Path(bag_path).expanduser().resolve()
    rgb_timestamps_csv = Path(rgb_timestamps_csv).expanduser().resolve()
    out_depth_npz = Path(out_depth_npz).expanduser().resolve()
    out_depth_npz.parent.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")
    if not rgb_timestamps_csv.exists():
        raise FileNotFoundError(f"Timestamp CSV not found: {rgb_timestamps_csv}")

    tsdf = pd.read_csv(rgb_timestamps_csv)
    if frame_col not in tsdf.columns or ts_col not in tsdf.columns:
        raise ValueError(f"Timestamp CSV must include '{frame_col}' and '{ts_col}'")

    rgb = tsdf[[frame_col, ts_col]].copy()
    rgb[frame_col] = rgb[frame_col].astype(int)
    rgb[ts_col] = rgb[ts_col].astype(np.int64)
    rgb = rgb.sort_values(ts_col).reset_index(drop=True)
    if max_frames > 0:
        rgb = rgb.head(max_frames)

    rgb_targets = list(zip(rgb[frame_col].tolist(), rgb[ts_col].tolist()))
    if not rgb_targets:
        raise RuntimeError("No RGB timestamps to align.")

    depth_store = {}
    target_i = 0
    prev_ts = None
    prev_raw = None

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == depth_topic]
        if not conns:
            raise RuntimeError(f"Depth topic not found in bag: {depth_topic}")

        for _conn, ts, raw in reader.messages(connections=conns):
            ts = int(ts)
            while target_i < len(rgb_targets):
                frame_id, t_rgb = rgb_targets[target_i]
                if prev_ts is None or ts < t_rgb:
                    break

                chosen_raw = prev_raw if abs(t_rgb - prev_ts) <= abs(ts - t_rgb) else raw
                depth_m, _meta = decode_depth_to_m_from_raw(chosen_raw)
                depth_store[str(frame_id)] = depth_m
                target_i += 1

                if target_i % 50 == 0:
                    print(f"aligned depth frames: {target_i}/{len(rgb_targets)}")
                if target_i >= len(rgb_targets):
                    break

            if target_i >= len(rgb_targets):
                break
            prev_ts, prev_raw = ts, raw

    np.savez_compressed(str(out_depth_npz), **depth_store)
    print("[OK] Saved depth NPZ:", out_depth_npz)
    print("[OK] Frames saved:", len(depth_store), "/", len(rgb_targets))
