"""
Audio loading, validation, and harmonization for the BSS/ICA pipeline.

This module is intentionally decoupled from Streamlit: it operates on
file-like objects and returns plain dataclasses, so it can be unit
tested or reused outside the app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Sequence

import librosa
import numpy as np
import soundfile as sf


@dataclass
class AudioSignal:
    """A single mono audio signal plus the metadata needed downstream."""

    name: str
    data: np.ndarray            # 1-D float64 array in [-1, 1] after normalization
    sample_rate: int
    original_duration: float    # seconds, measured before any trimming

    @property
    def duration(self) -> float:
        """Current duration in seconds (reflects trimming, if any)."""
        return len(self.data) / self.sample_rate

    @property
    def n_samples(self) -> int:
        return len(self.data)


class AudioLoadError(Exception):
    """Raised when uploaded audio cannot be used for blind source separation."""


class AudioLoader:
    """
    Loads raw WAV uploads and prepares them for mixing/ICA.

    Pipeline
    --------
    1. ``load_file``      : bytes -> mono float64 AudioSignal, with basic
                             per-file validation (readable, non-empty, long enough).
    2. ``validate_count``   : enforce the 2-3 source constraint for a batch.
    3. ``harmonize_batch`` : make a batch mutually compatible -- common
                             sample rate (via resampling), common length
                             (via trimming to the shortest), and peak
                             normalization.
    """

    MIN_SIGNALS = 2
    MAX_SIGNALS = 3
    MIN_DURATION_SEC = 1.0

    def load_file(self, file: BinaryIO, name: str) -> AudioSignal:
        """Read one file-like object into a validated, mono AudioSignal."""
        try:
            data, sr = sf.read(file, dtype="float64", always_2d=False)
        except Exception as exc:  # noqa: BLE001 - surface any decode failure to the user
            raise AudioLoadError(f"'{name}' could not be read as audio: {exc}") from exc

        if data.size == 0:
            raise AudioLoadError(f"'{name}' contains no audio samples.")

        if data.ndim > 1:
            # Downmix stereo/multi-channel input to mono by averaging channels.
            data = data.mean(axis=1)

        duration = len(data) / sr
        if duration < self.MIN_DURATION_SEC:
            raise AudioLoadError(
                f"'{name}' is only {duration:.2f}s long; "
                f"minimum required duration is {self.MIN_DURATION_SEC:.0f}s."
            )

        return AudioSignal(name=name, data=data.astype(np.float64), sample_rate=sr, original_duration=duration)

    def validate_count(self, count: int) -> None:
        """Enforce that a batch has between MIN_SIGNALS and MAX_SIGNALS files."""
        if not (self.MIN_SIGNALS <= count <= self.MAX_SIGNALS):
            raise AudioLoadError(
                f"Please upload between {self.MIN_SIGNALS} and {self.MAX_SIGNALS} "
                f"WAV files (got {count})."
            )

    def harmonize_batch(self, signals: Sequence[AudioSignal]) -> list[AudioSignal]:
        """
        Make a batch of signals mutually compatible for mixing and ICA:

        1. Resample every signal to the *lowest* sample rate present in the
           batch (avoids upsampling artifacts and keeps things fast).
        2. Trim every signal to the length of the shortest signal, so the
           mixing matrix can be applied as simple matrix multiplication.
        3. Peak-normalize every signal to [-1, 1] for a level playing field.
        """
        target_sr = min(s.sample_rate for s in signals)

        resampled: list[AudioSignal] = []
        for s in signals:
            if s.sample_rate != target_sr:
                new_data = librosa.resample(s.data, orig_sr=s.sample_rate, target_sr=target_sr)
                resampled.append(AudioSignal(s.name, new_data, target_sr, s.original_duration))
            else:
                resampled.append(s)

        min_len = min(s.n_samples for s in resampled)
        trimmed = [
            AudioSignal(s.name, s.data[:min_len], s.sample_rate, s.original_duration)
            for s in resampled
        ]

        normalized = [
            AudioSignal(s.name, self._peak_normalize(s.data), s.sample_rate, s.original_duration)
            for s in trimmed
        ]
        return normalized

    @staticmethod
    def _peak_normalize(data: np.ndarray) -> np.ndarray:
        """Scale so the maximum absolute amplitude is 1.0. Leaves silence untouched."""
        peak = np.max(np.abs(data))
        if peak < 1e-12:
            return data
        return data / peak
