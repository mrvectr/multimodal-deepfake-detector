
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
}

MANIFEST_FIELDS = [
    "filename",
    "path",
    "status",
    "fps",
    "frame_count",
    "duration_seconds",
    "width",
    "height",
    "codec",
    "error",
]


def decode_fourcc(value: float) -> str:
    """Convert OpenCV FOURCC integer into a readable codec name."""

    code = int(value)

    if code == 0:
        return "Unknown"

    return "".join(
        chr((code >> (8 * i)) & 0xFF)
        for i in range(4)
    ).strip()


def inspect_video(video_path: Path) -> dict[str, Any]:
    """Extract basic metadata and check video decodability."""

    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {video_path}"
        )

    try:
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        codec = decode_fourcc(
            capture.get(cv2.CAP_PROP_FOURCC)
        )

        # Basic metadata validation
        if fps <= 0 or frame_count <= 0:
            raise RuntimeError(
                "Invalid FPS or frame count."
            )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Invalid video resolution."
            )

        # Verify that at least one frame can be decoded.
        success, frame = capture.read()

        if not success or frame is None:
            raise RuntimeError(
                "Video opened, but frame decoding failed."
            )

        duration = frame_count / fps

        return {
            "filename": video_path.name,
            "path": str(video_path.resolve()),
            "status": "valid",
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "duration_seconds": round(duration, 3),
            "width": width,
            "height": height,
            "codec": codec,
            "error": "",
        }

    finally:
        capture.release()


def create_video_manifest(
    input_directory: Path,
    output_csv: Path,
    recursive: bool = False,
) -> None:
    """Inspect all supported videos and generate a CSV manifest."""

    input_directory = Path(input_directory)
    output_csv = Path(output_csv)

    if not input_directory.is_dir():
        raise NotADirectoryError(
            f"Video directory not found: {input_directory}"
        )

    iterator = (
        input_directory.rglob("*")
        if recursive
        else input_directory.iterdir()
    )

    video_files = sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        raise RuntimeError(
            f"No supported videos found in {input_directory}"
        )

    records = []

    for video_path in video_files:

        print(f"Inspecting: {video_path.name}")

        try:
            record = inspect_video(video_path)

        except Exception as error:
            record = {
                "filename": video_path.name,
                "path": str(video_path.resolve()),
                "status": "invalid",
                "fps": "",
                "frame_count": "",
                "duration_seconds": "",
                "width": "",
                "height": "",
                "codec": "",
                "error": str(error),
            }

        records.append(record)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=MANIFEST_FIELDS,
        )

        writer.writeheader()
        writer.writerows(records)

    valid_count = sum(
        record["status"] == "valid"
        for record in records
    )

    print("\n========== VIDEO INSPECTION ==========")
    print(f"Total videos : {len(records)}")
    print(f"Valid videos : {valid_count}")
    print(f"Invalid      : {len(records) - valid_count}")
    print(f"Manifest     : {output_csv}")
    print("======================================")


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Inspect videos and generate a metadata manifest."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input videos.",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/fau/video_manifest.csv"),
        help="Destination for the video manifest.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search video files in subdirectories too.",
    )

    args = parser.parse_args()

    create_video_manifest(
        input_directory=args.input_dir,
        output_csv=args.output_csv,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
