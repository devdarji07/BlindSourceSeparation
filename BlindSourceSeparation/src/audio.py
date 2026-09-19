"""Amplitude statistics and in-memory audio encoding (no disk I/O)."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass
class AmplitudeStats:
    """Descriptive amplitude statistics for a mono signal."""

    minimum: float
    maximum: float
    mean: float
    std: float
    rms: float
    peak_db: float


def compute_amplitude_stats(data: np.ndarray) -> AmplitudeStats:
    """Compute min/max/mean/std/RMS/peak-dB for a mono float signal."""
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    peak_db = 20 * np.log10(peak) if peak > 1e-12 else float("-inf")
    return AmplitudeStats(
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        mean=float(np.mean(data)),
        std=float(np.std(data)),
        rms=float(np.sqrt(np.mean(np.square(data)))),
        peak_db=peak_db,
    )


def to_wav_bytes(data: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float signal as in-memory 16-bit PCM WAV bytes (for st.audio / downloads)."""
    buffer = io.BytesIO()
    # Clip defensively: upstream normalization keeps us in range, but downstream
    # processing (mixing, ICA) can produce signals that briefly exceed [-1, 1].
    safe_data = np.clip(data, -1.0, 1.0)
    sf.write(buffer, safe_data, sample_rate, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    return buffer.read()

