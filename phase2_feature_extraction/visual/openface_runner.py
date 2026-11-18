from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REQUIRED_COLUMNS = (
    "timestamp",
    "AU06_r",
    "AU12_r",
    "x_0",
    "y_0",
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".mkv"}


def validate_openface(executable_path: str | Path) -> Path:
    """Return the absolute OpenFace executable path after basic validation."""
    executable = Path(executable_path).expanduser()

    if not executable.exists():
        raise FileNotFoundError(
            f"OpenFace executable not found: {executable}"
        )

    if not executable.is_file():
        raise ValueError(
            f"Expected an executable file, got: {executable}"
        )

    if executable.suffix.lower() != ".exe":
        raise ValueError(
            f"Expected a Windows .exe file, got: {executable.name}"
        )

    return executable.resolve()


def build_command(
    executable_path: str | Path,
    video_path: str | Path,
    output_dir: str | Path,
) -> list[str]:
    """Build an OpenFace command without invoking the executable."""
    executable = Path(executable_path).expanduser().resolve()
    video = Path(video_path).expanduser()
    output = Path(output_dir).expanduser()

    if not executable.is_file():
        raise FileNotFoundError(
            f"OpenFace executable not found: {executable}"
        )

    if not video.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video}"
        )

    output.mkdir(parents=True, exist_ok=True)

    return [
        str(executable),
        "-f", str(video.resolve()),
        "-out_dir", str(output.resolve()),
        "-aus",
        "-2Dfp",
    ]


def run_extraction(
    command: list[str],
    timeout_seconds: int = 600,
    working_directory: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run OpenFace and return its completed-process result."""
    if not command:
        raise ValueError("Command cannot be empty.")

    if timeout_seconds <= 0:
        raise ValueError("Timeout must be greater than zero.")

    cwd = (
        str(Path(working_directory).expanduser().resolve())
        if working_directory is not None
        else None
    )

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            shell=False,
        )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Executable could not be found: {command[0]}"
        ) from error
    except PermissionError as error:
        raise PermissionError(
            f"Permission denied while executing: {command[0]}"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"OpenFace execution exceeded {timeout_seconds} seconds."
        ) from error

    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        if not details:
            details = "OpenFace returned a non-zero exit code without details."
        raise RuntimeError(
            f"OpenFace failed with return code {result.returncode}:\n{details}"
        )

    return result


def validate_output(
    csv_path: str | Path,
    required_columns: tuple[str, ...] = DEFAULT_REQUIRED_COLUMNS,
) -> dict[str, Any]:
    """Validate an OpenFace CSV and return basic frame/tracking statistics.

    AU intensity and landmark values are checked for numeric, finite values.
    Tracking confidence and success are summarized when those optional columns
    are present in the OpenFace output.
    """
    output_file = Path(csv_path).expanduser()

    if not output_file.exists():
        raise FileNotFoundError(
            f"OpenFace output not found: {output_file}"
        )
    if not output_file.is_file():
        raise ValueError(f"Expected a CSV file: {output_file}")
    if output_file.stat().st_size == 0:
        raise ValueError("Output CSV is empty.")

    total_rows = 0
    valid_rows = 0
    tracking_success_frames = 0
    tracking_failed_frames = 0
    confidence_values: list[float] = []

    with output_file.open(
        "r", newline="", encoding="utf-8-sig"
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if not reader.fieldnames:
            raise ValueError("CSV has no column headers.")

        # Keep a mapping from normalized names to original CSV header names.
        header_map = {
            header.strip(): header
            for header in reader.fieldnames
            if header and header.strip()
        }
        columns = set(header_map)

        missing_columns = set(required_columns) - columns
        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        for row in reader:
            total_rows += 1
            normalized_row = {
                key.strip(): value.strip() if isinstance(value, str) else value
                for key, value in row.items()
                if key is not None
            }

            # Summarize optional tracking fields independently of AU/landmark
            # validity so tracking counts describe all CSV rows.
            if "confidence" in columns:
                try:
                    confidence = float(normalized_row["confidence"])
                    if math.isfinite(confidence):
                        confidence_values.append(confidence)
                except (ValueError, TypeError, KeyError):
                    pass

            if "success" in columns:
                success_value = normalized_row.get("success", "")
                try:
                    success_number = float(success_value)
                except (ValueError, TypeError):
                    success_number = None

                if success_number == 1.0:
                    tracking_success_frames += 1
                elif success_number == 0.0:
                    tracking_failed_frames += 1

            try:
                values = [
                    float(normalized_row[column])
                    for column in required_columns
                ]
            except (ValueError, TypeError, KeyError):
                continue

            if not all(math.isfinite(value) for value in values):
                continue

            valid_rows += 1

    if total_rows == 0:
        raise ValueError("CSV contains no data rows.")
    if valid_rows == 0:
        raise ValueError(
            "CSV contains no valid AU and landmark measurements."
        )

    return {
        "csv_path": str(output_file.resolve()),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": total_rows - valid_rows,
        "tracking_success_frames": tracking_success_frames,
        "tracking_failed_frames": tracking_failed_frames,
        "mean_confidence": (
            sum(confidence_values) / len(confidence_values)
            if confidence_values else None
        ),
        "confidence_count": len(confidence_values),
        "columns": sorted(columns),
        "status": "valid",
    }


def process_video(
    executable_path: str | Path,
    video_path: str | Path,
    output_root: str | Path,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Run OpenFace for one video and return data for the batch aggregator.

    This function does not create a per-video text report.
    """
    executable = validate_openface(executable_path)
    video = Path(video_path).expanduser()

    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")

    video = video.resolve()
    output_dir = Path(output_root).expanduser() / video.stem

    command = build_command(
        executable_path=executable,
        video_path=video,
        output_dir=output_dir,
    )

    expected_csv = output_dir / f"{video.stem}.csv"

    # Prevent a stale CSV from a previous run being mistaken for fresh output.
    if expected_csv.exists() and expected_csv.is_file():
        expected_csv.unlink()

    # OpenFace uses model paths relative to its installation directory.
    result = run_extraction(
        command,
        timeout_seconds=timeout_seconds,
        working_directory=executable.parent,
    )

    validation = validate_output(
        expected_csv,
        required_columns=DEFAULT_REQUIRED_COLUMNS,
    )

    return {
        "video": str(video),
        "csv_path": str(expected_csv.resolve()),
        "return_code": result.returncode,
        "validation": validation,
        "status": "SUCCESS",
    }
