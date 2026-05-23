from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEPTH_INFO_TOPIC = "/device_0/sensor_0/Depth_0/info/camera_info"
COLOR_INFO_TOPIC = "/device_0/sensor_1/Color_0/info/camera_info"
DEPTH_TF_TOPIC = "/device_0/sensor_0/Depth_0/tf/0"
COLOR_TF_TOPIC = "/device_0/sensor_1/Color_0/tf/0"


def get_first_deserialized(bag_path: Path, topic: str):
    from rosbags.highlevel import AnyReader

    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise RuntimeError(f"Topic not found in bag: {topic}")
        for conn, ts, raw in reader.messages(connections=conns):
            return int(ts), reader.deserialize(raw, conn.msgtype)
    raise RuntimeError(f"No messages for topic: {topic}")


def intrinsics_from_ci(camera_info) -> tuple[float, float, float, float, int, int]:
    k = np.array(camera_info.K, dtype=np.float32).reshape(3, 3)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    return fx, fy, cx, cy, int(camera_info.width), int(camera_info.height)


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_msg_to_t(msg) -> np.ndarray:
    tx, ty, tz = float(msg.translation.x), float(msg.translation.y), float(msg.translation.z)
    qx, qy, qz, qw = float(msg.rotation.x), float(msg.rotation.y), float(msg.rotation.z), float(msg.rotation.w)
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = quat_to_rot(qx, qy, qz, qw)
    t[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return t


def extract_calibration(
    bag: str | Path,
    out: str | Path,
    depth_info_topic: str = DEPTH_INFO_TOPIC,
    color_info_topic: str = COLOR_INFO_TOPIC,
    depth_tf_topic: str = DEPTH_TF_TOPIC,
    color_tf_topic: str = COLOR_TF_TOPIC,
) -> Path:
    bag = Path(bag).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    if not bag.exists():
        raise FileNotFoundError(f"Bag not found: {bag}")

    t_depth_ci, ci_depth = get_first_deserialized(bag, depth_info_topic)
    t_color_ci, ci_color = get_first_deserialized(bag, color_info_topic)
    fx_d, fy_d, cx_d, cy_d, w_d, h_d = intrinsics_from_ci(ci_depth)
    fx_c, fy_c, cx_c, cy_c, w_c, h_c = intrinsics_from_ci(ci_color)

    t_depth_tf, tf_depth = get_first_deserialized(bag, depth_tf_topic)
    t_color_tf, tf_color = get_first_deserialized(bag, color_tf_topic)
    t_depth = transform_msg_to_t(tf_depth)
    t_color = transform_msg_to_t(tf_color)
    t_depth_to_color = np.linalg.inv(t_color) @ t_depth
    t_color_to_depth = np.linalg.inv(t_depth_to_color)

    np.savez(
        out,
        fxD=fx_d,
        fyD=fy_d,
        cxD=cx_d,
        cyD=cy_d,
        Wd=w_d,
        Hd=h_d,
        fxC=fx_c,
        fyC=fy_c,
        cxC=cx_c,
        cyC=cy_c,
        Wc=w_c,
        Hc=h_c,
        T_depth=t_depth,
        T_color=t_color,
        T_depth_to_color=t_depth_to_color,
        T_color_to_depth=t_color_to_depth,
        t_depth_ci=t_depth_ci,
        t_color_ci=t_color_ci,
        t_depth_tf=t_depth_tf,
        t_color_tf=t_color_tf,
    )
    print("Saved calibration:", out)
    print("Depth size:", w_d, h_d, "| Color size:", w_c, h_c)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract depth/color calibration from a RealSense .bag.")
    parser.add_argument("--bag", required=True, help="Input bag containing camera info and tf topics.")
    parser.add_argument("--out", required=True, help="Output calibration .npz.")
    args = parser.parse_args()
    extract_calibration(args.bag, args.out)


if __name__ == "__main__":
    main()
