
import csv
import subprocess
import sys

from pathlib import Path
from unittest.mock import patch


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))

from visual.openface_runner import (
    process_video,
    format_extraction_report,
    save_extraction_report,
)


# ============================================================
# MOCK OPENFACE EXECUTION
# ============================================================

def fake_extraction(command, timeout_seconds=600):

    output_dir = Path(
        command[command.index("-out_dir") + 1]
    )

    video_path = Path(
        command[command.index("-f") + 1]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_csv = output_dir / f"{video_path.stem}.csv"

    # Generate mock OpenFace-style CSV
    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "AU06_r",
            "AU12_r",
            "x_0",
            "y_0"
        ])

        writer.writerow([0.00, 0.12, 0.08, 120.5, 140.2])
        writer.writerow([0.04, 0.25, 0.31, 121.0, 141.0])
        writer.writerow([0.08, 0.65, 0.82, 122.0, 142.0])

    return subprocess.CompletedProcess(
        args=command,
        returncode=0,
        stdout="Mock extraction successful",
        stderr=""
    )


# ============================================================
# TEST EXECUTION
# ============================================================

def main():

    video_path = (
        PROJECT_ROOT
        / "dataset"
        / "test_data"
        / "videos"
        / "s2_l_bbim3a.mov"
    )

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Test video not found: {video_path}"
        )

    # Replace actual OpenFace execution with mock execution
    with patch(
        "visual.openface_runner.run_extraction",
        side_effect=fake_extraction
    ):

        report = process_video(

            executable_path=sys.executable,

            video_path=video_path,

            output_root=PROJECT_ROOT / "outputs" / "fau",

            timeout_seconds=600,

            test_id="TEST-001",

            execution_mode="MOCK"
        )

    # Format report
    report_text = format_extraction_report(report)

    # Display in terminal
    print("\n")
    print(report_text)

    # Save report
    report_file = save_extraction_report(
        report,
        PROJECT_ROOT
        / "outputs"
        / "reports"
        / "visual_stream_test_report.txt"
    )

    print(f"\nReport saved successfully: {report_file}")


if __name__ == "__main__":
    main()
