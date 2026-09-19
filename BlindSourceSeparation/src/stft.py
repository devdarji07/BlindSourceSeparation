"""
Short-Time Fourier Transform (STFT) and inverse (ISTFT), from scratch.

The frequency-domain ICA pipeline (`src/fdica.py`) needs a per-bin
time-frequency representation of each microphone signal; this module
provides that (framing -> window -> FFT, and the inverse via
overlap-add), rather than reaching for `scipy.signal.stft`.

Windowing tradeoff (documented since it's a real design knob, not just an
implementation detail): a longer `n_fft` gives finer frequency resolution
-- important because the per-bin instantaneous-mixing approximation
(`src/fdica.py` module docstring) only holds when each bin is narrow
enough that the room impulse response looks like a flat gain across it --
but a longer window also means fewer, longer time frames, so each bin's
FastICA has less independent data to estimate statistics from, and a
longer window is a worse match to the assumption that the mixing filter
is constant over the analysis window. `hop_length = n_fft // 4` (75%
overlap) with a Hann window is a conventional default that keeps the
overlap-add reconstruction exact (see `TestPerfectReconstruction`).
"""

from __future__ import annotations

import numpy as np


def hann_window(n_fft: int) -> np.ndarray:
    """Periodic Hann window, i.e. `scipy.signal.windows.hann(n_fft, sym=False)` -- required for COLA at 75% hop."""
    n = np.arange(n_fft)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / n_fft)


def stft(signal: np.ndarray, n_fft: int = 1024, hop_length: int | None = None) -> np.ndarray:
    """
    Single-channel STFT: frame the signal, window each frame, take the
    real-input FFT of each.

    Zero-pads the signal on both ends by `n_fft - hop_length` so the
    first and last samples get full window coverage (standard "centered"
    STFT convention) -- `istft` undoes this padding when given `length`.

    Returns
    -------
    (n_freq_bins, n_frames) complex array, n_freq_bins = n_fft // 2 + 1.
    """
    if hop_length is None:
        hop_length = n_fft // 4
    window = hann_window(n_fft)

    pad = n_fft - hop_length
    padded = np.concatenate([np.zeros(pad), signal, np.zeros(pad + n_fft)])

    n_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        padded,
        shape=(n_fft, n_frames),
        strides=(padded.strides[0], padded.strides[0] * hop_length),
        writeable=False,
    )
    windowed = frames * window[:, None]
    return np.fft.rfft(windowed, axis=0)


def istft(spec: np.ndarray, hop_length: int | None = None, length: int | None = None) -> np.ndarray:
    """
    Inverse STFT via weighted overlap-add: inverse-FFT each frame, window
    it again, and accumulate at its hop offset, normalizing by the sum of
    squared windows at each sample (the standard overlap-add
    normalization that makes this exact when the window/hop pair
    satisfies COLA -- see `TestPerfectReconstruction`).

    spec   : (n_freq_bins, n_frames) complex, as returned by `stft`.
    length : if given, trim the output back to this many samples (undoes
             the centering pad `stft` adds); otherwise the padded length
             is returned.
    """
    n_freq_bins, n_frames = spec.shape
    n_fft = 2 * (n_freq_bins - 1)
    if hop_length is None:
        hop_length = n_fft // 4
    window = hann_window(n_fft)

    frames = np.fft.irfft(spec, n=n_fft, axis=0) * window[:, None]

    pad = n_fft - hop_length
    out_len = (n_frames - 1) * hop_length + n_fft
    out = np.zeros(out_len)
    win_sum = np.zeros(out_len)
    for t in range(n_frames):
        start = t * hop_length
        out[start : start + n_fft] += frames[:, t]
        win_sum[start : start + n_fft] += window**2

    nonzero = win_sum > 1e-10
    out[nonzero] /= win_sum[nonzero]

    out = out[pad : out_len - pad]  # undo stft's centering pad
    if length is not None:
        if len(out) < length:
            out = np.concatenate([out, np.zeros(length - len(out))])
        else:
            out = out[:length]
    return out


def stft_multichannel(signals: np.ndarray, n_fft: int = 1024, hop_length: int | None = None) -> np.ndarray:
    """STFT of every row of `signals` (n_channels, n_samples) -> (n_channels, n_freq_bins, n_frames)."""
    specs = [stft(signals[c], n_fft=n_fft, hop_length=hop_length) for c in range(signals.shape[0])]
    return np.stack(specs, axis=0)


def istft_multichannel(specs: np.ndarray, hop_length: int | None = None, length: int | None = None) -> np.ndarray:
    """Inverse of `stft_multichannel`: (n_channels, n_freq_bins, n_frames) -> (n_channels, n_samples)."""
    return np.stack(
        [istft(specs[c], hop_length=hop_length, length=length) for c in range(specs.shape[0])], axis=0
    )
