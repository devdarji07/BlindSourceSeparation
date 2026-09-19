"""
Signal mixing: build a random mixing matrix and apply it to a set of
independent source signals to simulate microphones at a cocktail party.

Model
-----
Given `n` independent sources stacked as rows, ``S`` (n_sources, n_samples),
and a square mixing matrix ``A`` (n_mics, n_sources), the instantaneous
linear mixing model is::

    X = A @ S

where each row of ``X`` is what one microphone actually records: a
weighted sum of every source. This module only builds ``A`` and applies
it; ICA (Milestone 4) is the inverse problem of recovering ``S`` from
``X`` alone.
"""

from __future__ import annotations

import numpy as np


class MixingError(Exception):
    """Raised when a mixing matrix or source stack is unusable."""


def stack_sources(signals: list) -> np.ndarray:
    """Stack a list of AudioSignal objects into an (n_sources, n_samples) matrix."""
    return np.vstack([s.data for s in signals])


def generate_mixing_matrix(
    n_sources: int,
    rng: np.random.Generator,
    min_reciprocal_condition: float = 1e-3,
    max_attempts: int = 100,
) -> np.ndarray:
    """
    Generate a random, well-conditioned square mixing matrix.

    Entries are drawn with magnitude in [0.2, 1.0] and a random sign, so
    every microphone picks up a meaningful (non-near-zero) contribution
    from every source. Near-singular draws (which would make the
    separation problem numerically unstable regardless of the algorithm)
    are rejected and redrawn.
    """
    for _ in range(max_attempts):
        magnitude = rng.uniform(0.2, 1.0, size=(n_sources, n_sources))
        sign = rng.choice([-1.0, 1.0], size=(n_sources, n_sources))
        matrix = magnitude * sign
        if 1.0 / np.linalg.cond(matrix) > min_reciprocal_condition:
            return matrix
    raise MixingError(f"Could not generate a well-conditioned {n_sources}x{n_sources} mixing matrix.")


def mix_signals(sources: np.ndarray, mixing_matrix: np.ndarray) -> np.ndarray:
    """
    Apply a mixing matrix to a stack of source signals: ``X = A @ S``.

    sources        : (n_sources, n_samples)
    mixing_matrix  : (n_mics, n_sources), square for this project
    returns        : (n_mics, n_samples) mixed signals, one row per microphone
    """
    if mixing_matrix.shape[0] != mixing_matrix.shape[1]:
        raise MixingError("Mixing matrix must be square.")
    if mixing_matrix.shape[1] != sources.shape[0]:
        raise MixingError(
            f"Mixing matrix expects {mixing_matrix.shape[1]} sources, got {sources.shape[0]}."
        )
    return mixing_matrix @ sources


def normalize_rows_peak(signal_matrix: np.ndarray) -> np.ndarray:
    """Peak-normalize each row independently to [-1, 1] (for playback/plotting only)."""
    peaks = np.max(np.abs(signal_matrix), axis=1, keepdims=True)
    peaks[peaks < 1e-12] = 1.0
    return signal_matrix / peaks
