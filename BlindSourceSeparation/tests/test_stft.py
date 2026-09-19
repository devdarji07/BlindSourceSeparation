"""Unit tests for src/stft.py -- mainly the perfect-reconstruction property."""

from __future__ import annotations

import numpy as np
import pytest

from src.stft import hann_window, istft, istft_multichannel, stft, stft_multichannel


class TestPerfectReconstruction:
    @pytest.mark.parametrize("n_fft,hop_length", [(1024, 256), (512, 128), (256, 64), (1024, 512)])
    def test_reconstructs_random_signal(self, n_fft, hop_length):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        spec = stft(x, n_fft=n_fft, hop_length=hop_length)
        y = istft(spec, hop_length=hop_length, length=len(x))
        np.testing.assert_allclose(y, x, atol=1e-10)

    def test_reconstructs_pure_tone(self):
        sr = 16000
        t = np.arange(sr) / sr
        x = np.sin(2 * np.pi * 440 * t)
        spec = stft(x, n_fft=1024, hop_length=256)
        y = istft(spec, hop_length=256, length=len(x))
        np.testing.assert_allclose(y, x, atol=1e-9)

    def test_multichannel_round_trip(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(3, 4000))
        specs = stft_multichannel(x, n_fft=512, hop_length=128)
        y = istft_multichannel(specs, hop_length=128, length=x.shape[1])
        assert y.shape == x.shape
        np.testing.assert_allclose(y, x, atol=1e-9)


class TestShapesAndWindow:
    def test_hann_window_endpoints_and_symmetry(self):
        w = hann_window(8)
        assert w[0] == pytest.approx(0.0, abs=1e-12)
        # Periodic (asymmetric) Hann: w[n] for n=1..7 mirrors around the
        # implicit sample at n=8, i.e. w[1:] is palindromic on its own.
        np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-12)

    def test_freq_bin_count(self):
        spec = stft(np.zeros(4000), n_fft=1024, hop_length=256)
        assert spec.shape[0] == 1024 // 2 + 1

    def test_dc_signal_concentrates_in_bin_zero(self):
        x = np.ones(4000)
        spec = stft(x, n_fft=1024, hop_length=256)
        mid_frame = spec[:, spec.shape[1] // 2]
        assert np.argmax(np.abs(mid_frame)) == 0
