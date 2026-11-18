"""Run one batch of OpenFace AU/2D-landmark extraction.

Each run creates exactly one batch TXT report and one batch summary CSV.
Raw OpenFace CSVs remain organized in a separate folder per input video.
No per-video text reports are generated.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENFACE_EXE = Path(
    r"C:\Tools\openface\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
)
DEFAULT_VIDEO_DIR = PROJECT_ROOT / "dataset" / "test_data" / "videos"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "fau"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "outputs" / "reports" / "batches"
VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}

# Ensure imports work when this script is launched from the project root.
sys.path.insert(0, str(PROJECT_ROOT))

from visual.openface_runner import process_video, validate_openface  # noqa: E402


def discover_videos(video_dir: Path, recursive: bool = False) -> list[Path]:
    """Find supported video files in the input directory."""
    if not video_dir.is_dir():
        raise NotADirectoryError(
            f"Video directory does not exist: {video_dir}"
        )

    candidates = video_dir.rglob("*") if recursive else video_dir.iterdir()
    videos = sorted(
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        raise RuntimeError(f"No supported video files found in: {video_dir}")

    return videos


def write_batch_summary_csv(
    summary_path: Path,
    summary: dict[str, Any],
) -> None:
    """Write exactly one aggregate row describing this batch."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def format_batch_report(
    *,
    batch_id: str,
    started_at: str,
    finished_at: str,
    duration_seconds: float,
    video_dir: Path,
    executable: Path,
    output_root: Path,
    total_videos: int,
    successful_videos: int,
    failed_videos: list[dict[str, str]],
    total_frames: int,
    valid_rows: int,
    invalid_rows: int,
    tracking_success_frames: int,
    tracking_failed_frames: int,
    mean_confidence: float | None,
    batch_summary_csv: Path,
) -> str:
    """Create one human-readable report for the complete batch."""
    failed_names = (
        ", ".join(item["video"] for item in failed_videos)
        if failed_videos else "None"
    )
    confidence_text = (
        f"{mean_confidence:.4f}"
        if mean_confidence is not None else "Not available"
    )

    lines = [
        "=" * 76,
        "                 VISUAL STREAM - FAU BATCH REPORT",
        "=" * 76,
        f"Batch ID                  : {batch_id}",
        f"Started                   : {started_at}",
        f"Finished                  : {finished_at}",
        f"Batch duration            : {duration_seconds:.2f} seconds",
        "",
        "[1] BATCH CONFIGURATION",
        "-" * 76,
        f"Input video directory     : {video_dir}",
        f"OpenFace executable       : {executable}",
        f"OpenFace working directory: {executable.parent}",
        f"Raw CSV output directory  : {output_root}",
        "Requested features        : Action Units + 2D facial landmarks",
        "Required CSV columns      : timestamp, AU06_r, AU12_r, x_0, y_0",
        "",
        "[2] BATCH RESULTS",
        "-" * 76,
        f"Videos discovered         : {total_videos}",
        f"Videos successful         : {successful_videos}",
        f"Videos failed             : {len(failed_videos)}",
        f"Total CSV data rows       : {total_frames}",
        f"Valid measurement rows    : {valid_rows}",
        f"Invalid measurement rows  : {invalid_rows}",
        f"Successful tracking frames: {tracking_success_frames}",
        f"Failed tracking frames    : {tracking_failed_frames}",
        f"Mean tracking confidence  : {confidence_text}",
        "",
        "[3] FAILED INPUTS",
        "-" * 76,
        failed_names,
    ]

    if failed_videos:
        lines.append("")
        lines.append("Failure details:")
        for item in failed_videos:
            lines.append(f"- {item['video']}: {item['error']}")

    lines.extend([
        "",
        "[4] OUTPUTS",
        "-" * 76,
        f"Raw per-video CSV files    : {output_root}",
        f"Batch summary CSV         : {batch_summary_csv}",
        "This run produces one batch report; no per-video text reports.",
        "",
        "=" * 76,
        "                       END OF BATCH REPORT",
        "=" * 76,
    ])

    return "\n".join(lines)


def run_batch(
    executable_path: Path,
    video_dir: Path,
    output_root: Path,
    report_root: Path,
    timeout_seconds: int = 600,
    recursive: bool = False,
    batch_id: str | None = None,
) -> dict[str, Any]:
    """Process all discovered videos and create one report for this batch."""
    started_clock = time.monotonic()
    started_datetime = datetime.now().astimezone()
    batch_id = batch_id or started_datetime.strftime("FAU_BATCH_%Y%m%d_%H%M%S")

    # Fail early for configuration errors, before starting a batch.
    executable = validate_openface(executable_path)
    videos = discover_videos(video_dir, recursive=recursive)

    output_root.mkdir(parents=True, exist_ok=True)
    batch_report_dir = report_root / batch_id
    batch_report_dir.mkdir(parents=True, exist_ok=False)

    report_path = batch_report_dir / "batch_report.txt"
    summary_path = batch_report_dir / "batch_summary.csv"

    successful_videos = 0
    failed_videos: list[dict[str, str]] = []
    total_frames = 0
    valid_rows = 0
    invalid_rows = 0
    tracking_success_frames = 0
    tracking_failed_frames = 0
    confidence_weighted_sum = 0.0
    confidence_count = 0

    print("\n" + "=" * 76)
    print("              VISUAL STREAM - FAU BATCH PROCESSING")
    print("=" * 76)
    print(f"Batch ID     : {batch_id}")
    print(f"Videos found : {len(videos)}")
    print(f"OpenFace     : {executable}")
    print("=" * 76)

    for index, video in enumerate(videos, start=1):
        print(f"\n[{index}/{len(videos)}] Processing: {video.name}")

        try:
            result = process_video(
                executable_path=executable,
                video_path=video,
                output_root=output_root,
                timeout_seconds=timeout_seconds,
            )

            validation = result["validation"]
            successful_videos += 1
            total_frames += validation["total_rows"]
            valid_rows += validation["valid_rows"]
            invalid_rows += validation["invalid_rows"]
            tracking_success_frames += validation["tracking_success_frames"]
            tracking_failed_frames += validation["tracking_failed_frames"]

            mean_confidence = validation["mean_confidence"]
            if mean_confidence is not None:
                confidence_weighted_sum += (
                    mean_confidence * validation["confidence_count"]
                )
                confidence_count += validation["confidence_count"]

            print(
                f"  SUCCESS | rows={validation['total_rows']} "
                f"| valid={validation['valid_rows']} "
                f"| tracking_success={validation['tracking_success_frames']}"
            )

        except Exception as error:
            failed_videos.append({
                "video": video.name,
                "error": f"{type(error).__name__}: {error}",
            })
            print(f"  FAILED  | {type(error).__name__}: {error}")

    finished_datetime = datetime.now().astimezone()
    duration_seconds = time.monotonic() - started_clock
    finished_at = finished_datetime.isoformat(timespec="seconds")
    started_at = started_datetime.isoformat(timespec="seconds")

    batch_mean_confidence = (
        confidence_weighted_sum / confidence_count
        if confidence_count else None
    )

    # Exactly one aggregate row is written for the complete batch.
    summary = {
        "batch_id": batch_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": round(duration_seconds, 2),
        "input_directory": str(video_dir.resolve()),
        "openface_executable": str(executable),
        "raw_output_directory": str(output_root.resolve()),
        "total_videos": len(videos),
        "successful_videos": successful_videos,
        "failed_videos": len(failed_videos),
        "total_csv_rows": total_frames,
        "valid_measurement_rows": valid_rows,
        "invalid_measurement_rows": invalid_rows,
        "tracking_success_frames": tracking_success_frames,
        "tracking_failed_frames": tracking_failed_frames,
        "mean_tracking_confidence": (
            round(batch_mean_confidence, 6)
            if batch_mean_confidence is not None else ""
        ),
        "failed_video_names": "; ".join(
            item["video"] for item in failed_videos
        ),
    }

    write_batch_summary_csv(summary_path, summary)

    report_text = format_batch_report(
        batch_id=batch_id,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration_seconds,
        video_dir=video_dir.resolve(),
        executable=executable,
        output_root=output_root.resolve(),
        total_videos=len(videos),
        successful_videos=successful_videos,
        failed_videos=failed_videos,
        total_frames=total_frames,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        tracking_success_frames=tracking_success_frames,
        tracking_failed_frames=tracking_failed_frames,
        mean_confidence=batch_mean_confidence,
        batch_summary_csv=summary_path.resolve(),
    )

    report_path.write_text(report_text, encoding="utf-8")

    print("\n" + report_text)
    print(f"\nBatch report saved: {report_path.resolve()}")
    print(f"Batch summary saved: {summary_path.resolve()}")

    return {
        "batch_id": batch_id,
        "total_videos": len(videos),
        "successful_videos": successful_videos,
        "failed_videos": len(failed_videos),
        "report_path": report_path.resolve(),
        "summary_path": summary_path.resolve(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one batch of OpenFace AU and 2D landmark extraction."
    )
    parser.add_argument(
        "--openface-exe",
        type=Path,
        default=DEFAULT_OPENFACE_EXE,
        help="Path to OpenFace FeatureExtraction.exe.",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory containing input videos.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for raw per-video OpenFace CSV outputs.",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=DEFAULT_REPORT_ROOT,
        help="Parent directory for one report folder per batch.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per video in seconds (default: 600).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Also search for videos in nested folders.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Optional custom batch identifier.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        result = run_batch(
            executable_path=args.openface_exe,
            video_dir=args.video_dir,
            output_root=args.output_root,
            report_root=args.report_root,
            timeout_seconds=args.timeout,
            recursive=args.recursive,
            batch_id=args.batch_id,
        )
    except Exception as error:
        print(f"BATCH ABORTED: {type(error).__name__}: {error}")
        return 1

    return 0 if result["failed_videos"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
