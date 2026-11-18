"""
POS-based rPPG signal processing for RGB traces produced by the ROI extractor.

Input columns expected:
  timestamp_seconds
  forehead_r/g/b, left_cheek_r/g/b, right_cheek_r/g/b
  optional per-region validity columns: forehead_valid, left_cheek_valid,
  right_cheek_valid

The implementation intentionally refuses to estimate heart rate from recordings
that are too short or have insufficient valid samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, detrend, sosfiltfilt, welch

REGIONS = ("forehead", "left_cheek", "right_cheek")


def _interpolate_short_gaps(values: np.ndarray, valid: np.ndarray,
                            max_gap_seconds: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only short internal gaps; do not fill long or edge gaps."""
    x = np.asarray(values, dtype=float).copy()
    good = np.asarray(valid, dtype=bool) & np.isfinite(x)
    n = len(x)
    if n == 0:
        return x, good

    max_gap = max(0, int(round(max_gap_seconds * fps)))
    i = 0
    while i < n:
        if good[i]:
            i += 1
            continue
        start = i
        while i < n and not good[i]:
            i += 1
        end = i
        gap_len = end - start
        if start > 0 and end < n and gap_len <= max_gap:
            x[start:end] = np.linspace(x[start - 1], x[end], gap_len + 2)[1:-1]
            good[start:end] = True
    return x, good


def _prepare_rgb(frame: pd.DataFrame, region: str, fps: float,
                 max_gap_seconds: float, min_valid_fraction: float) -> tuple[np.ndarray | None, dict[str, Any]]:
    cols = [f"{region}_{c}" for c in ("r", "g", "b")]
    if not all(c in frame.columns for c in cols):
        return None, {"status": "missing_rgb_columns", "valid_fraction": 0.0}

    rgb = frame[cols].to_numpy(dtype=float)
    flag_col = f"{region}_valid"
    if flag_col in frame.columns:
        flags = frame[flag_col].fillna(0).to_numpy(dtype=float) > 0
    else:
        flags = np.ones(len(frame), dtype=bool)

    finite = np.isfinite(rgb).all(axis=1)
    valid = flags & finite
    valid_fraction = float(valid.mean()) if len(valid) else 0.0

    if valid_fraction < min_valid_fraction:
        return None, {
            "status": "insufficient_valid_frames",
            "valid_fraction": valid_fraction,
            "valid_frames": int(valid.sum()),
        }

    prepared = np.empty_like(rgb, dtype=float)
    for channel in range(3):
        prepared[:, channel], channel_valid = _interpolate_short_gaps(
            rgb[:, channel], valid, max_gap_seconds, fps
        )
        if not channel_valid.all():
            return None, {
                "status": "long_or_edge_gap",
                "valid_fraction": valid_fraction,
                "valid_frames": int(valid.sum()),
            }

    return prepared, {
        "status": "ready",
        "valid_fraction": valid_fraction,
        "valid_frames": int(valid.sum()),
    }


def pos_signal(rgb: np.ndarray, fps: float, window_seconds: float = 1.6,
               overlap: float = 0.5) -> np.ndarray:
    """
    Extract a POS pulse signal using overlapping windows.

    POS projection:
      S1 = G - B
      S2 = -2R + G + B
      h  = S1 + std(S1)/std(S2) * S2
    """
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape (frames, 3)")
    if len(rgb) < 3:
        raise ValueError("Not enough RGB samples")

    win = max(3, int(round(window_seconds * fps)))
    win = min(win, len(rgb))
    step = max(1, int(round(win * (1.0 - overlap))))
    output = np.zeros(len(rgb), dtype=float)
    weights = np.zeros(len(rgb), dtype=float)

    starts = list(range(0, max(1, len(rgb) - win + 1), step))
    last_start = max(0, len(rgb) - win)
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        end = min(len(rgb), start + win)
        block = rgb[start:end].copy()
        means = block.mean(axis=0)
        if np.any(np.abs(means) < 1e-8):
            continue
        normalized = block / means - 1.0
        r, g, b = normalized.T
        s1 = g - b
        s2 = -2.0 * r + g + b
        std2 = float(np.std(s2))
        if std2 < 1e-8:
            continue
        alpha = float(np.std(s1)) / std2
        h = s1 + alpha * s2
        h = h - h.mean()
        output[start:end] += h
        weights[start:end] += 1.0

    usable = weights > 0
    if not usable.all():
        output[~usable] = np.nan
    output[usable] /= weights[usable]
    if not np.isfinite(output).all():
        raise ValueError("POS produced uncovered samples")
    return output


def bandpass_signal(signal: np.ndarray, fps: float, low_hz: float = 0.7,
                    high_hz: float = 4.0, order: int = 3) -> np.ndarray:
    nyquist = fps / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Bandpass must satisfy 0 < low < high < Nyquist ({nyquist:.3f} Hz)"
        )
    x = detrend(np.asarray(signal, dtype=float), type="linear")
    sos = butter(order, [low_hz, high_hz], btype="bandpass",
                 fs=fps, output="sos")
    padlen = 3 * (2 * len(sos) + 1)
    if len(x) <= padlen:
        raise ValueError("Signal is too short for zero-phase bandpass filtering")
    return sosfiltfilt(sos, x)


def estimate_heart_rate(signal: np.ndarray, fps: float, min_duration_seconds: float,
                        low_hz: float, high_hz: float) -> dict[str, Any]:
    duration = len(signal) / fps
    if duration < min_duration_seconds:
        return {
            "status": "insufficient_duration",
            "duration_seconds": duration,
            "minimum_duration_seconds": min_duration_seconds,
            "heart_rate_bpm": None,
        }

    frequencies, power = welch(signal, fs=fps, nperseg=min(len(signal), int(fps * 8)))
    band = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not band.any() or not np.isfinite(power[band]).any():
        return {"status": "no_valid_spectral_peak", "heart_rate_bpm": None}

    band_freqs = frequencies[band]
    band_power = power[band]
    peak_index = int(np.argmax(band_power))
    peak_hz = float(band_freqs[peak_index])
    return {
        "status": "estimated",
        "heart_rate_bpm": round(peak_hz * 60.0, 2),
        "peak_frequency_hz": round(peak_hz, 4),
        "duration_seconds": round(duration, 3),
        "method": "Welch PSD peak",
    }


def process_file(input_csv: str | Path, output_dir: str | Path, fps: float | None = None,
                 min_duration_seconds: float = 10.0, low_hz: float = 0.7,
                 high_hz: float = 4.0, window_seconds: float = 1.6,
                 overlap: float = 0.5, max_gap_seconds: float = 0.25,
                 min_valid_fraction: float = 0.80) -> dict[str, Any]:
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_csv, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    if "timestamp_seconds" not in frame.columns:
        raise ValueError("Input CSV is missing timestamp_seconds")

    timestamps = pd.to_numeric(frame["timestamp_seconds"], errors="coerce").to_numpy()
    if len(timestamps) < 3 or not np.isfinite(timestamps).all():
        raise ValueError("Invalid or insufficient timestamps")
    deltas = np.diff(timestamps)
    positive_deltas = deltas[deltas > 0]
    if len(positive_deltas) == 0:
        raise ValueError("Timestamps must increase")
    measured_fps = 1.0 / float(np.median(positive_deltas))
    sample_fps = float(fps) if fps is not None else measured_fps
    if sample_fps <= 0:
        raise ValueError("FPS must be positive")

    processed: dict[str, np.ndarray] = {}
    region_reports: dict[str, Any] = {}
    for region in REGIONS:
        rgb, report = _prepare_rgb(
            frame, region, sample_fps, max_gap_seconds, min_valid_fraction
        )
        if rgb is None:
            region_reports[region] = report
            continue
        try:
            raw_pos = pos_signal(rgb, sample_fps, window_seconds, overlap)
            filtered = bandpass_signal(raw_pos, sample_fps, low_hz, high_hz)
            processed[region] = filtered
            region_reports[region] = {
                **report,
                "status": "processed",
                "pos_window_seconds": window_seconds,
            }
        except ValueError as exc:
            region_reports[region] = {**report, "status": "processing_failed", "reason": str(exc)}

    if not processed:
        raise RuntimeError("No ROI produced a usable POS signal")

    # Equal-weight average across successfully processed regions.
    combined = np.mean(np.vstack(list(processed.values())), axis=0)
    result = pd.DataFrame({
        "frame_index": frame["frame_index"] if "frame_index" in frame else np.arange(len(frame)),
        "timestamp_seconds": timestamps,
    })
    for region in REGIONS:
        result[f"{region}_pos_filtered"] = processed.get(
            region, np.full(len(frame), np.nan)
        )
    result["combined_pos_filtered"] = combined
    result.to_csv(output_dir / "rppg_processed_signals.csv", index=False)

    hr = estimate_heart_rate(combined, sample_fps, min_duration_seconds, low_hz, high_hz)
    report = {
        "input_csv": str(input_csv),
        "frame_count": len(frame),
        "duration_seconds": round(float(timestamps[-1] - timestamps[0]), 3),
        "fps_used": round(sample_fps, 4),
        "fps_measured_from_timestamps": round(measured_fps, 4),
        "filter_band_hz": [low_hz, high_hz],
        "regions": region_reports,
        "heart_rate": hr,
        "notes": [
            "POS output is a pulse-related signal, not a clinical measurement.",
            "Heart-rate estimation is suppressed below the configured minimum duration.",
            "The combined signal is an equal-weight average of successfully processed ROIs.",
        ],
    }
    with open(output_dir / "rppg_processing_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for region, values in processed.items():
            axes[0].plot(timestamps, values, label=region, linewidth=1)
        axes[0].set_title("Filtered POS signal by ROI")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend()
        axes[0].grid(True, alpha=0.25)
        axes[1].plot(timestamps, combined, label="combined", linewidth=1)
        axes[1].set_title("Combined rPPG signal")
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Amplitude")
        axes[1].grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "rppg_signal_plot.png", dpi=160)
        plt.close(fig)
    except ImportError:
        report["plot_note"] = "matplotlib not installed; plot was not generated"
        with open(output_dir / "rppg_processing_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="POS-based rPPG signal processing")
    parser.add_argument("--input", required=True, help="RGB trace CSV from ROI extraction")
    parser.add_argument("--output", default="outputs/rppg/processed", help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS; default is inferred from timestamps")
    parser.add_argument("--min-duration", type=float, default=10.0, help="Minimum duration for HR estimate")
    parser.add_argument("--low-hz", type=float, default=0.7)
    parser.add_argument("--high-hz", type=float, default=4.0)
    parser.add_argument("--window-seconds", type=float, default=1.6)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-gap-seconds", type=float, default=0.25)
    parser.add_argument("--min-valid-fraction", type=float, default=0.80)
    args = parser.parse_args()

    report = process_file(
        args.input, args.output, args.fps, args.min_duration,
        args.low_hz, args.high_hz, args.window_seconds, args.overlap,
        args.max_gap_seconds, args.min_valid_fraction
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
