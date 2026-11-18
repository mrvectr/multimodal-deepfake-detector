import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from visual.rppg_roi_geometry import (
    create_roi_masks,
    validate_roi_masks,
    extract_masked_rgb,
    visualize_roi_masks,
)

REGIONS = ("forehead", "left_cheek", "right_cheek")
LANDMARK_COLUMNS = [f"{axis}_{i}" for axis in ("x", "y") for i in range(68)]


def load_openface_csv(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True, encoding="utf-8-sig")
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    missing = [column for column in LANDMARK_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "The selected CSV does not contain all 68 OpenFace landmarks. "
            f"Missing examples: {missing[:8]}. "
            f"CSV columns found: {list(df.columns[:25])}"
        )

    if "success" in df.columns:
        success = pd.to_numeric(df["success"], errors="coerce").fillna(0).astype(int).eq(1)
    else:
        success = pd.Series(True, index=df.index)

    if "confidence" in df.columns:
        confidence = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
    else:
        confidence = pd.Series(1.0, index=df.index)

    return df, success, confidence


def row_landmarks(row):
    points = np.array(
        [[row[f"x_{i}"], row[f"y_{i}"]] for i in range(68)],
        dtype=np.float32,
    )
    if not np.isfinite(points).all():
        raise ValueError("Landmark row contains NaN or infinite coordinates")
    return points


def build_frame_lookup(df):
    """OpenFace's standard frame column is 1-based; use it when present."""
    if "frame" not in df.columns:
        return None

    numbers = pd.to_numeric(df["frame"], errors="coerce")
    lookup = {}
    for row_index, frame_number in numbers.items():
        if pd.notna(frame_number):
            lookup[int(frame_number)] = int(row_index)
    return lookup


def _mean_rgb_record(frame, mask):
    values = extract_masked_rgb(frame, mask)
    return values


def extract_video(
    video_path,
    csv_path,
    output_dir,
    use_skin_filter=False,
    min_confidence=0.0,
    min_roi_pixels=50,
    preview=True,
    preview_stride=1,
):
    video_path = Path(video_path)
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"OpenFace CSV not found: {csv_path}")
    if preview_stride < 1:
        raise ValueError("preview_stride must be >= 1")
    if min_roi_pixels < 1:
        raise ValueError("min_roi_pixels must be >= 1")

    df, success_flags, confidences = load_openface_csv(csv_path)
    frame_lookup = build_frame_lookup(df)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir.mkdir(parents=True, exist_ok=True)
    signals_dir = output_dir / "rgb_signals"
    previews_dir = output_dir / "roi_previews"
    signals_dir.mkdir(parents=True, exist_ok=True)
    if preview:
        previews_dir.mkdir(parents=True, exist_ok=True)

    stem = video_path.stem
    rgb_csv_path = signals_dir / f"{stem}_rgb.csv"
    preview_path = previews_dir / f"{stem}_roi.mp4"

    writer = None
    if preview:
        output_fps = fps or 25.0
        writer = cv2.VideoWriter(
            str(preview_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create preview video: {preview_path}")

    fields = [
        "frame_index", "timestamp_seconds", "openface_row",
        "face_detected", "confidence", "roi_error",
    ]
    for region in REGIONS:
        fields.extend([
            f"{region}_r", f"{region}_g", f"{region}_b",
            f"{region}_pixels", f"{region}_valid",
        ])

    total_frames = 0
    usable_face_frames = 0
    rejected_face_frames = 0
    roi_failure_frames = 0
    region_valid = {name: 0 for name in REGIONS}
    region_rejected = {name: 0 for name in REGIONS}
    region_pixel_sum = {name: 0 for name in REGIONS}
    preview_frames_written = 0

    try:
        with open(rgb_csv_path, "w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
            csv_writer.writeheader()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index = total_frames
                timestamp = frame_index / fps if fps else np.nan
                row_index = (
                    frame_lookup.get(frame_index + 1)
                    if frame_lookup is not None
                    else (frame_index if frame_index < len(df) else None)
                )

                record = {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "openface_row": "" if row_index is None else row_index,
                    "face_detected": 0,
                    "confidence": np.nan,
                    "roi_error": "",
                }
                for region in REGIONS:
                    record.update({
                        f"{region}_r": np.nan,
                        f"{region}_g": np.nan,
                        f"{region}_b": np.nan,
                        f"{region}_pixels": 0,
                        f"{region}_valid": 0,
                    })

                annotated = frame
                if row_index is None:
                    rejected_face_frames += 1
                else:
                    confidence = float(confidences.iloc[row_index])
                    face_ok = bool(success_flags.iloc[row_index]) and confidence >= min_confidence
                    record["confidence"] = confidence
                    record["face_detected"] = int(face_ok)

                    if not face_ok:
                        rejected_face_frames += 1
                    else:
                        usable_face_frames += 1
                        try:
                            landmarks = row_landmarks(df.iloc[row_index])
                            roi_data = create_roi_masks(
                                landmarks, frame, use_skin_filter=use_skin_filter
                            )
                            # This is the SINGLE validation result used by CSV and both JSON reports.
                            validation = validate_roi_masks(
                                roi_data, min_pixels=min_roi_pixels
                            )

                            any_region_invalid = False
                            for region in REGIONS:
                                mask = roi_data["masks"][region]
                                rgb = _mean_rgb_record(frame, mask)
                                pixel_count = rgb["pixel_count"]
                                is_valid = bool(validation[region]["valid"]) and pixel_count >= min_roi_pixels

                                record[f"{region}_pixels"] = pixel_count
                                record[f"{region}_valid"] = int(is_valid)
                                region_pixel_sum[region] += pixel_count

                                if is_valid:
                                    record[f"{region}_r"] = rgb["r"]
                                    record[f"{region}_g"] = rgb["g"]
                                    record[f"{region}_b"] = rgb["b"]
                                    region_valid[region] += 1
                                else:
                                    # Invalid regions are deliberately NaN, not usable RGB samples.
                                    region_rejected[region] += 1
                                    any_region_invalid = True

                            if any_region_invalid:
                                roi_failure_frames += 1

                            if writer is not None:
                                annotated = visualize_roi_masks(frame, roi_data, validation)

                        except (ValueError, cv2.error, IndexError, KeyError) as exc:
                            roi_failure_frames += 1
                            record["roi_error"] = str(exc)
                            for region in REGIONS:
                                region_rejected[region] += 1

                csv_writer.writerow(record)

                if writer is not None:
                    # Keep every frame in the preview so its duration matches the source.
                    if frame_index % preview_stride == 0 and annotated is not frame:
                        writer.write(annotated)
                    else:
                        writer.write(frame)
                    preview_frames_written += 1

                total_frames += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    region_report = {}
    for region in REGIONS:
        valid_count = region_valid[region]
        rejected_count = region_rejected[region]
        denominator = valid_count + rejected_count
        region_report[region] = {
            "valid_frames": valid_count,
            "rejected_frames": rejected_count,
            "valid_percentage": round(100.0 * valid_count / denominator, 2) if denominator else 0.0,
            "average_roi_pixels": round(region_pixel_sum[region] / denominator, 2) if denominator else 0.0,
            "minimum_roi_pixels": min_roi_pixels,
        }

    valid_face_pct = 100.0 * usable_face_frames / total_frames if total_frames else 0.0
    report = {
        "status": "success" if total_frames > 0 else "failed",
        "video": str(video_path),
        "openface_csv": str(csv_path),
        "rgb_csv": str(rgb_csv_path),
        "roi_preview": str(preview_path) if preview else None,
        "fps": fps,
        "frame_width": width,
        "frame_height": height,
        "reported_video_frames": reported_video_frames,
        "total_frames": total_frames,
        "openface_rows": int(len(df)),
        "valid_frames": usable_face_frames,
        "rejected_frames": rejected_face_frames,
        "valid_percentage": round(valid_face_pct, 2),
        "roi_failure_frames": roi_failure_frames,
        "confidence_threshold": min_confidence,
        "minimum_roi_pixels": min_roi_pixels,
        "skin_filter_enabled": bool(use_skin_filter),
        "frame_matching": "1-based OpenFace frame column" if frame_lookup is not None else "CSV row order",
        "regions": region_report,
        "validation_note": (
            "CSV validity flags and both JSON reports are computed from the same per-frame "
            "ROI validation results. Invalid ROI RGB values are written as NaN."
        ),
    }

    # Write identical report content to both names for compatibility with existing scripts.
    for report_name in (f"{stem}_summary.json", f"{stem}_report.json"):
        with open(output_dir / report_name, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, allow_nan=False)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-frame RGB signals from forehead and cheek ROIs."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--openface-csv", required=True)
    parser.add_argument("--output", default="rppg_output")
    parser.add_argument("--skin-filter", action="store_true")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-roi-pixels", type=int, default=50)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-stride", type=int, default=1)
    args = parser.parse_args()

    result = extract_video(
        video_path=args.video,
        csv_path=args.openface_csv,
        output_dir=args.output,
        use_skin_filter=args.skin_filter,
        min_confidence=args.min_confidence,
        min_roi_pixels=args.min_roi_pixels,
        preview=not args.no_preview,
        preview_stride=args.preview_stride,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
