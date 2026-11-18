from pathlib import Path
import argparse
import json

from rppg_signal_processing import process_file


def main():
    parser = argparse.ArgumentParser(description="Run POS rPPG processing on one RGB trace CSV")
    parser.add_argument("--input", required=True, help="Path to *_rgb.csv")
    parser.add_argument("--output", default="outputs/rppg/processed")
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args()

    report = process_file(args.input, args.output, fps=args.fps)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
