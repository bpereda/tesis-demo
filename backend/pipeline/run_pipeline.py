from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from pathlib import Path


def _as_relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def infer_calibration_path(bag: str | Path, calib_dir: str | Path) -> Path:
    bag = Path(bag).expanduser().resolve()
    calib_dir = Path(calib_dir).expanduser().resolve()
    match = re.search(r"(\d{8}_\d{6})", bag.name)
    if not match:
        raise ValueError(
            f"Could not infer calibration timestamp from bag name '{bag.name}'. "
            "Expected a name containing YYYYMMDD_HHMMSS, for example 20240304_210426."
        )

    timestamp = match.group(1)
    candidates = [
        calib_dir / f"calib_{timestamp}.npz",
        calib_dir / f"calib__{timestamp}.npz",
        calib_dir / f"calib_from_compact_{timestamp}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    available = sorted(p.name for p in calib_dir.glob("*.npz"))[:20]
    raise FileNotFoundError(
        f"No calibration file found for timestamp {timestamp} in {calib_dir}. "
        f"Tried: {', '.join(p.name for p in candidates)}. "
        f"Available .npz files: {available}"
    )


def run_pipeline(
    bag: str | Path,
    out: str | Path,
    sam_model: str | Path,
    calib: str | Path | None = None,
    calib_dir: str | Path | None = None,
    yield_model: str | Path | None = None,
    prompts: tuple[str, ...] = ("grape cluster",),
    device: str = "0",
    half: bool = True,
    max_frames: int = -1,
    every_n: int = 1,
    min_frames_per_track: int = 3,
    progress_callback: Callable[[str, int, str], None] | None = None,
) -> dict:
    from .extract_rgb import extract_color_mp4_from_rosbag, save_depth_frames_meters_npz_from_bag
    from .metrics import aggregate_metrics, compute_area_volume_one_mask_per_grape
    from .predict import predict_weight
    from .segmentation import run_sam3_per_frame
    from .tracking import choose_representative_per_track, save_representative_masks_npz, track_detections_simple
    from .visuals import save_representative_contact_sheet, save_rgb_preview, save_tracking_overlays

    bag = Path(bag).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    sam_model = Path(sam_model).expanduser().resolve()
    if calib is None:
        if calib_dir is None:
            raise ValueError("Either calib or calib_dir must be provided.")
        calib = infer_calibration_path(bag, calib_dir)
    else:
        calib = Path(calib).expanduser().resolve()
    yield_model_path = Path(yield_model).expanduser().resolve() if yield_model else None

    if not bag.exists():
        raise FileNotFoundError(f"Bag not found: {bag}")
    if bag.suffix != ".bag":
        raise ValueError(f"Expected a .bag file, got: {bag.name}")

    out.mkdir(parents=True, exist_ok=True)
    color_mp4 = out / "color.mp4"
    timestamps_csv = out / "color_timestamps.csv"
    sam_csv = out / "sam3_per_frame.csv"
    sam_npz = out / "sam3_per_frame.npz"
    tracked_csv = out / "per_frame_tracked.csv"
    reps_csv = out / "representative_tracks.csv"
    one_mask_npz = out / "one_mask_per_grape.npz"
    depth_npz = out / "depth_frames_meters.npz"
    metrics_csv = out / "one_grape_area_volume.csv"
    rgb_preview_jpg = out / "rgb_preview.jpg"
    overlays_dir = out / "overlays"
    representative_sheet_jpg = out / "representative_masks.jpg"
    result_json = out / "result.json"

    def progress(stage: str, percent: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(stage, percent, message)

    progress("extract", 15, "Extracting RGB frames from the RealSense bag.")
    extract_color_mp4_from_rosbag(
        bag_path=bag,
        out_mp4=color_mp4,
        out_timestamps_csv=timestamps_csv,
        max_frames=max_frames,
        every_n=every_n,
    )
    save_rgb_preview(color_mp4, rgb_preview_jpg)
    progress("segment", 35, "Running grape-cluster segmentation.")
    run_sam3_per_frame(
        in_mp4=color_mp4,
        model_path=sam_model,
        prompts=prompts,
        out_csv=sam_csv,
        out_npz=sam_npz,
        device=device,
        half=half,
        max_frames=max_frames,
        every_n=1,
    )
    progress("track", 58, "Tracking detections and assigning stable cluster IDs.")
    tracked_df = track_detections_simple(
        per_frame_csv=sam_csv,
        out_tracked_csv=tracked_csv,
        iou_th=0.12,
        max_center_dist=120,
        max_age=20,
    )
    reps_df = choose_representative_per_track(tracked_df, min_frames=min_frames_per_track)
    reps_df.to_csv(reps_csv, index=False)
    overlay_paths = save_tracking_overlays(color_mp4, sam_npz, tracked_csv, overlays_dir)
    save_representative_masks_npz(reps_df, sam_npz, one_mask_npz)
    save_representative_contact_sheet(one_mask_npz, representative_sheet_jpg)
    progress("metrics", 76, "Aligning depth and computing geometric metrics.")
    save_depth_frames_meters_npz_from_bag(bag, timestamps_csv, depth_npz, max_frames=max_frames)
    metrics_df = compute_area_volume_one_mask_per_grape(reps_df, one_mask_npz, depth_npz, calib, metrics_csv)

    progress("predict", 92, "Loading the trained model and predicting yield.")
    aggregate = aggregate_metrics(metrics_df)
    predicted_weight = predict_weight(yield_model_path, aggregate)
    result = {
        "job_dir": str(out),
        "bag": str(bag),
        "calibration": str(calib),
        "predicted_weight": predicted_weight,
        "detected_clusters": aggregate["detected_clusters"],
        "total_estimated_volume_cm3": aggregate["total_estimated_volume_cm3"],
        "model_features": {
            "mask_count": aggregate["mask_count"],
            "mask_area_m2_sum": aggregate["mask_area_m2_sum"],
            "mask_area_m2_p75": aggregate["mask_area_m2_p75"],
            "mask_area_m2_std": aggregate["mask_area_m2_std"],
            "liters_totales": aggregate["liters_totales"],
        },
        "mean_depth_m": aggregate["mean_depth_m"],
        "outputs": {
            "color_video": _as_relative(color_mp4, out),
            "timestamps_csv": _as_relative(timestamps_csv, out),
            "segmentation_csv": _as_relative(sam_csv, out),
            "segmentation_npz": _as_relative(sam_npz, out),
            "tracked_csv": _as_relative(tracked_csv, out),
            "representative_tracks_csv": _as_relative(reps_csv, out),
            "representative_masks_npz": _as_relative(one_mask_npz, out),
            "depth_npz": _as_relative(depth_npz, out),
            "metrics_csv": _as_relative(metrics_csv, out),
            "rgb_preview": _as_relative(rgb_preview_jpg, out),
            "representative_masks_preview": _as_relative(representative_sheet_jpg, out),
            "result_json": _as_relative(result_json, out),
        },
        "visuals": {
            "rgb_preview": _as_relative(rgb_preview_jpg, out),
            "tracking_overlays": [_as_relative(path, out) for path in overlay_paths],
            "representative_masks": _as_relative(representative_sheet_jpg, out),
        },
    }
    result_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the RealSense grape yield pipeline on one .bag file.")
    parser.add_argument("--bag", required=True, help="Input Intel RealSense .bag recording.")
    parser.add_argument("--out", required=True, help="Output job directory, e.g. jobs/test_001.")
    parser.add_argument("--sam-model", required=True, help="Path to SAM3 segmentation model, e.g. models/sam3.pt.")
    calib_group = parser.add_mutually_exclusive_group(required=True)
    calib_group.add_argument("--calib", help="Path to one calibration .npz with depth/color intrinsics and transform.")
    calib_group.add_argument("--calib-dir", help="Directory with calibration files named calib_YYYYMMDD_HHMMSS.npz.")
    parser.add_argument("--yield-model", default=None, help="Optional trained yield/weight model .joblib/.pkl.")
    parser.add_argument("--prompt", action="append", default=None, help="Text prompt for SAM3. Can be repeated.")
    parser.add_argument("--device", default="0", help="Torch device for segmentation: 0, cpu, cuda:0, etc.")
    parser.add_argument("--no-half", action="store_true", help="Disable half precision.")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit frames for a fast demo smoke test.")
    parser.add_argument("--every-n", type=int, default=1, help="Extract every Nth RGB frame.")
    parser.add_argument("--min-frames-per-track", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prompts = tuple(args.prompt) if args.prompt else ("grape cluster",)
    result = run_pipeline(
        bag=args.bag,
        out=args.out,
        sam_model=args.sam_model,
        calib=args.calib,
        calib_dir=args.calib_dir,
        yield_model=args.yield_model,
        prompts=prompts,
        device=args.device,
        half=not args.no_half,
        max_frames=args.max_frames,
        every_n=args.every_n,
        min_frames_per_track=args.min_frames_per_track,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
