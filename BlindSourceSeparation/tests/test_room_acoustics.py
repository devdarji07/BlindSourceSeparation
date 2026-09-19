"""
Unit tests for src/room_acoustics.py against known analytic/physical cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.room_acoustics import (
    RoomAcousticsError,
    absorption_to_rt60,
    convolve_source_to_mic,
    generate_rir,
    render_convolutive_mixture,
    rt60_to_absorption,
)

ROOM = (4.0, 5.0, 3.0)
SOURCE = (1.0, 1.0, 1.5)
MIC = (3.0, 3.5, 1.5)
FS = 16000


def _measured_rt60(rir: np.ndarray, fs: int) -> float | None:
    """T30-style RT60 estimate via Schroeder backward integration, extrapolated from -5/-35 dB."""
    energy = rir[::-1] ** 2
    edc = np.cumsum(energy)[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-20)

    def crossing(level: float) -> float:
        idx = np.argmax(edc_db < level)
        return idx / fs

    t5, t35 = crossing(-5), crossing(-35)
    if t35 <= t5:
        return None
    slope = (-35 - (-5)) / (t35 - t5)
    return -60 / slope


class TestAbsorptionRt60Conversion:
    def test_round_trip(self):
        rt60 = 0.4
        absorption = rt60_to_absorption(rt60, ROOM)
        assert absorption_to_rt60(absorption, ROOM) == pytest.approx(rt60, rel=1e-9)

    def test_unachievable_rt60_raises(self):
        # An extremely short RT60 for this room implies alpha > 1, which is unphysical.
        with pytest.raises(RoomAcousticsError):
            rt60_to_absorption(0.001, ROOM)

    def test_invalid_absorption_raises(self):
        with pytest.raises(RoomAcousticsError):
            absorption_to_rt60(1.5, ROOM)


class TestGenerateRirAnechoic:
    """absorption=1.0 -> beta=0 -> every reflected image vanishes, leaving only the direct path."""

    def test_single_image_direct_path_only(self):
        result = generate_rir(ROOM, SOURCE, MIC, FS, absorption=1.0, duration_sec=0.05)
        assert result.n_images == 1

    def test_direct_path_delay_and_energy_conserved(self):
        # Free-field amplitude 1/distance, split by linear-interpolation fractional
        # delay across exactly two samples -- their weighted sum must reconstruct
        # the exact analytic amplitude (energy isn't lost or gained by interpolation).
        result = generate_rir(ROOM, SOURCE, MIC, FS, absorption=1.0, duration_sec=0.05)
        distance = np.linalg.norm(np.array(SOURCE) - np.array(MIC))
        expected_delay = distance / 343.0 * FS
        expected_amplitude = 1.0 / distance

        floor_idx = int(np.floor(expected_delay))
        two_sample_sum = result.rir[floor_idx] + result.rir[floor_idx + 1]
        assert two_sample_sum == pytest.approx(expected_amplitude, rel=1e-6)
        assert np.count_nonzero(result.rir) == 2


class TestGenerateRirReverberant:
    @pytest.mark.parametrize("target_rt60", [0.2, 0.4])
    def test_measured_rt60_within_tolerance_of_target(self, target_rt60):
        # Sabine's formula is a statistical/diffuse-field estimate; a single
        # source-mic image-method RIR won't match it exactly, but should be
        # in the right ballpark for a moderately sized, non-degenerate room.
        result = generate_rir(ROOM, SOURCE, MIC, FS, rt60=target_rt60)
        measured = _measured_rt60(result.rir, FS)
        assert measured is not None
        assert measured == pytest.approx(target_rt60, rel=0.25)

    def test_more_absorption_gives_shorter_measured_rt60(self):
        live = generate_rir(ROOM, SOURCE, MIC, FS, absorption=0.1)
        dead = generate_rir(ROOM, SOURCE, MIC, FS, absorption=0.6)
        assert _measured_rt60(dead.rir, FS) < _measured_rt60(live.rir, FS)

    def test_exactly_one_of_absorption_or_rt60_required(self):
        with pytest.raises(RoomAcousticsError):
            generate_rir(ROOM, SOURCE, MIC, FS)
        with pytest.raises(RoomAcousticsError):
            generate_rir(ROOM, SOURCE, MIC, FS, absorption=0.3, rt60=0.3)

    def test_point_outside_room_rejected(self):
        with pytest.raises(RoomAcousticsError):
            generate_rir(ROOM, (10.0, 1.0, 1.0), MIC, FS, absorption=0.3)


class TestConvolveSourceToMic:
    def test_impulse_rir_is_identity(self):
        rng = np.random.default_rng(0)
        source = rng.normal(size=500)
        rir = np.zeros(10)
        rir[0] = 1.0
        result = convolve_source_to_mic(source, rir)
        assert result.shape == source.shape
        np.testing.assert_allclose(result, source)

    def test_output_length_matches_source(self):
        rir = np.array([0.5, 0.3, 0.1])
        source = np.ones(20)
        result = convolve_source_to_mic(source, rir)
        assert len(result) == len(source)


class TestRenderConvolutiveMixture:
    def test_shape_and_mic_source_count_mismatch_raises(self):
        rng = np.random.default_rng(0)
        sources = rng.normal(size=(2, 2000))
        with pytest.raises(RoomAcousticsError):
            render_convolutive_mixture(
                sources,
                ROOM,
                source_positions=[SOURCE],  # only 1, but 2 sources given
                mic_positions=[MIC, (2.0, 2.0, 1.5)],
                sample_rate=FS,
                absorption=0.3,
            )

    def test_mixture_shape(self):
        rng = np.random.default_rng(0)
        sources = rng.normal(size=(2, 4000))
        mixture, rirs = render_convolutive_mixture(
            sources,
            ROOM,
            source_positions=[SOURCE, (2.0, 1.0, 1.5)],
            mic_positions=[MIC, (2.5, 2.5, 1.5)],
            sample_rate=FS,
            absorption=0.3,
        )
        assert mixture.shape == (2, 4000)
        assert len(rirs) == 2 and all(len(row) == 2 for row in rirs)
