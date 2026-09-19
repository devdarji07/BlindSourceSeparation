"""
Unit tests for src/separation_metrics.py, against known analytic cases
(not just "it runs").
"""

from __future__ import annotations

import numpy as np
import pytest

from src.separation_metrics import evaluate_separation, si_sdr


class TestSiSdrAnalytic:
    """
    Hand-derived cases using
        reference = [1, -1, 1, -1]   (zero mean, ||r||^2 = 4)
        noise     = [1, -1, -1, 1]   (zero mean, ||n||^2 = 4, orthogonal to reference)
    so `alpha = <r, r + n> / <r, r>` works out to a clean value by
    construction, and SI-SDR can be checked against 10*log10(signal/noise)
    directly rather than re-deriving the implementation.
    """

    reference = np.array([1.0, -1.0, 1.0, -1.0])
    noise = np.array([1.0, -1.0, -1.0, 1.0])

    def test_orthogonal_noise_gives_exact_ratio(self):
        # alpha = <r, r+n>/<r,r> = (||r||^2 + <r,n>)/||r||^2 = 1 since <r,n> = 0.
        # => e_target = r, e_res = n, SI-SDR = 10*log10(||r||^2/||n||^2) = 0 dB.
        estimate = self.reference + self.noise
        result = si_sdr(self.reference, estimate, eps=1e-12)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_scaled_noise_matches_manual_snr(self):
        # Same construction with a scaled noise term: alpha is still 1
        # (noise stays orthogonal to reference under scaling), residual
        # energy scales by k^2, so SI-SDR = -20*log10(k) relative to the
        # unit-noise case.
        k = 3.0
        estimate = self.reference + k * self.noise
        expected = 10 * np.log10(np.dot(self.reference, self.reference) / (k**2 * np.dot(self.noise, self.noise)))
        result = si_sdr(self.reference, estimate, eps=1e-12)
        assert result == pytest.approx(expected, abs=1e-6)

    def test_pure_orthogonal_noise_is_very_negative(self):
        # No reference component at all (alpha = 0): SI-SDR should be
        # deeply negative, bounded only by eps.
        result = si_sdr(self.reference, self.noise, eps=1e-8)
        assert result < -50.0

    @pytest.mark.parametrize("scale", [0.1, 1.0, 100.0, -5.0])
    def test_scale_and_sign_invariance(self, scale):
        # Any nonzero scalar multiple of the reference (including sign
        # flip) is a perfect reconstruction up to that ambiguity: alpha
        # absorbs it exactly, leaving (near) zero residual. What's left
        # is bounded only by the eps floor, which is what the next test
        # characterizes precisely -- here we just check it stays high
        # over a moderate scale range where eps is negligible next to
        # signal energy.
        rng = np.random.default_rng(0)
        reference = rng.normal(size=200)
        estimate = scale * reference
        result = si_sdr(reference, estimate, eps=1e-8)
        assert result > 80.0

    def test_eps_floor_scales_with_signal_amplitude(self):
        # Perfect reconstruction (estimate = c * reference) makes e_res
        # exactly 0 in exact arithmetic, so with the eps floor the
        # closed form is 10*log10((c^2 * ||r_centered||^2 + eps) / eps)
        # -- i.e. SI-SDR is *not* perfectly scale-invariant right at the
        # eps floor, it grows by ~20*log10(c) as c grows. This test
        # documents that closed form rather than asserting a fixed bound.
        rng = np.random.default_rng(0)
        reference = rng.normal(size=200)
        centered = reference - reference.mean()
        energy = np.dot(centered, centered)
        eps = 1e-8
        for c in (0.01, 1.0, 50.0):
            expected = 10 * np.log10((c**2 * energy + eps) / eps)
            result = si_sdr(reference, c * reference, eps=eps)
            assert result == pytest.approx(expected, rel=1e-4)

    def test_zero_mean_preprocessing_ignores_dc_offset(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(size=500)
        estimate = reference.copy()
        with_offset = si_sdr(reference + 10.0, estimate - 5.0, eps=1e-8)
        without_offset = si_sdr(reference, estimate, eps=1e-8)
        assert with_offset == pytest.approx(without_offset, abs=1e-6)


class TestEvaluateSeparation:
    def test_identical_sources_score_high_on_every_metric(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(size=(2, 4000))
        report = evaluate_separation(reference, reference.copy())

        assert list(report.permutation) == [0, 1]
        assert report.mean_si_sdr_db > 50.0
        assert report.mean_sdr_db > 50.0
        assert report.mean_sir_db > 50.0
        assert report.mean_sar_db > 50.0

    def test_permuted_and_sign_flipped_sources_still_recovered(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=(2, 4000))
        # Reverse the row order and flip the sign of the first component --
        # exactly ICA's permutation/sign ambiguity, no distortion added.
        estimated = np.vstack([-reference[1], reference[0]])

        report = evaluate_separation(reference, estimated)

        assert list(report.permutation) == [1, 0]
        assert report.mean_si_sdr_db > 50.0

    def test_uncorrelated_estimate_scores_poorly(self):
        rng = np.random.default_rng(3)
        reference = rng.normal(size=(2, 4000))
        unrelated = rng.normal(size=(2, 4000))  # independent draw, not derived from reference

        report = evaluate_separation(reference, unrelated)

        assert report.mean_si_sdr_db < 0.0
