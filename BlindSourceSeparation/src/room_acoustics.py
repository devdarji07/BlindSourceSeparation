"""
Room impulse response (RIR) generation via the image-source method for
shoebox (rectangular) rooms:

    J. B. Allen and D. A. Berkley, "Image method for efficiently
    simulating small-room acoustics," JASA 65(4), 1979.

Used to build realistic **convolutive** multi-microphone mixtures
(``x(t) = sum_tau h(tau) s(t-tau)``) as a contrast case to the
**instantaneous** mixing model (``X = A @ S``) the rest of this app is
built around -- see `src/mixer.py` and the Theory page.

Two simplifications, both documented so results aren't overclaimed:

- **Uniform absorption.** A single scalar absorption coefficient is
  applied to all six room surfaces, rather than per-wall coefficients.
  This keeps "absorption / RT60" a single configurable knob at the cost
  of not modeling asymmetric rooms (e.g. a carpeted floor vs. a glass
  window). RT60 <-> absorption conversion uses Sabine's formula, itself
  an approximation valid for moderately live, diffuse rooms -- treat
  RT60 inputs/outputs as approximate, not exact.
- **Linear-interpolation fractional delay.** Each image source's
  contribution is placed into the RIR at a fractional sample position
  via linear interpolation between its two nearest samples, rather than
  a full band-limited (sinc) fractional-delay filter. This is
  numerically cheap enough to stay interactive for a Streamlit app (it
  vectorizes into a handful of NumPy array ops instead of a per-image
  Python loop) at the cost of a small high-frequency smoothing error
  relative to a sinc-interpolated RIR.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve

SPEED_OF_SOUND_M_S = 343.0


class RoomAcousticsError(Exception):
    """Raised when a room/absorption/source/mic configuration is physically invalid or unworkable."""


def room_volume(room_dim: tuple[float, float, float]) -> float:
    lx, ly, lz = room_dim
    return lx * ly * lz


def room_surface_area(room_dim: tuple[float, float, float]) -> float:
    lx, ly, lz = room_dim
    return 2.0 * (lx * ly + ly * lz + lx * lz)


def rt60_to_absorption(rt60: float, room_dim: tuple[float, float, float]) -> float:
    """
    Sabine's formula, solved for a uniform absorption coefficient::

        RT60 = 0.161 * V / (S * alpha)  =>  alpha = 0.161 * V / (S * RT60)
    """
    if rt60 <= 0:
        raise RoomAcousticsError(f"rt60 must be positive, got {rt60}.")
    alpha = 0.161 * room_volume(room_dim) / (room_surface_area(room_dim) * rt60)
    if not (0.0 < alpha <= 1.0):
        raise RoomAcousticsError(
            f"RT60={rt60:.3f}s is not achievable for a room of size {room_dim} with "
            f"uniform absorption (Sabine implies alpha={alpha:.3f}, must be in (0, 1] -- "
            "try a longer RT60, a larger room, or a shorter target)."
        )
    return alpha


def absorption_to_rt60(absorption: float, room_dim: tuple[float, float, float]) -> float:
    """Sabine's formula: RT60 = 0.161 * V / (S * alpha)."""
    if not (0.0 < absorption <= 1.0):
        raise RoomAcousticsError(f"absorption must be in (0, 1], got {absorption}.")
    return 0.161 * room_volume(room_dim) / (room_surface_area(room_dim) * absorption)


@dataclass
class RIRResult:
    """A single source-to-mic room impulse response and the parameters that produced it."""

    rir: np.ndarray                  # (n_samples,) impulse response, h(t)
    sample_rate: int
    rt60_target: float | None         # requested RT60, if that's how absorption was specified
    rt60_nominal: float                # Sabine RT60 implied by `absorption` (== rt60_target when given)
    absorption: float                   # uniform absorption coefficient actually used
    reflection_coefficient: float        # beta = sqrt(1 - absorption)
    max_order: int                        # highest reflection order summed
    n_images: int                          # image sources actually summed (post-pruning)
    truncated: bool                         # True if max_order was capped before reaching amplitude_threshold


def _validate_point_in_room(point: tuple[float, float, float], room_dim: tuple[float, float, float], label: str) -> None:
    for coord, dim, axis in zip(point, room_dim, "xyz"):
        if not (0.0 < coord < dim):
            raise RoomAcousticsError(f"{label} {axis}={coord} is outside the room (0, {dim}).")


def generate_rir(
    room_dim: tuple[float, float, float],
    source_pos: tuple[float, float, float],
    mic_pos: tuple[float, float, float],
    sample_rate: int,
    absorption: float | None = None,
    rt60: float | None = None,
    duration_sec: float | None = None,
    amplitude_threshold: float = 1e-4,
    max_order_cap: int = 130,
) -> RIRResult:
    """
    Generate a room impulse response for one source-mic pair via the
    image-source method (Allen & Berkley, 1979).

    Exactly one of `absorption` or `rt60` must be given; the other is
    derived via Sabine's formula (see module docstring).

    Image sources are indexed by a periodic tile index ``n`` (one per
    axis, `Z`) and a mirror sign (one per axis, `{+1, -1}`), giving the
    standard 1-D family of image positions ``2*n*L + sign*s`` per axis
    (source position `s`, room length `L`); the 3-D image position is the
    per-axis combination, and its **reflection order** along an axis is
    the number of room-lengths the tiling has unfolded through --
    ``abs(floor(image_axis_position / L))`` -- which is what the uniform
    per-surface reflection coefficient `beta = sqrt(1 - absorption)` is
    raised to the power of (summed across axes) for that image's
    attenuation. Images are summed until their contribution drops below
    `amplitude_threshold` relative to the direct path, or until
    `max_order_cap` is hit first (whichever bounds the sum sooner) --
    the cap exists purely so a very live room (absorption near 0) can't
    blow up computation time; `RIRResult.truncated` reports if it did.
    """
    if (absorption is None) == (rt60 is None):
        raise RoomAcousticsError("Pass exactly one of `absorption` or `rt60`.")

    if absorption is None:
        absorption = rt60_to_absorption(rt60, room_dim)
    elif not (0.0 < absorption <= 1.0):
        raise RoomAcousticsError(f"absorption must be in (0, 1], got {absorption}.")

    beta = float(np.sqrt(1.0 - absorption))
    rt60_nominal = absorption_to_rt60(absorption, room_dim)

    _validate_point_in_room(source_pos, room_dim, "source")
    _validate_point_in_room(mic_pos, room_dim, "mic")

    if duration_sec is None:
        duration_sec = rt60_nominal * 1.3  # margin past the nominal -60dB point for a clean tail

    n_samples = max(int(np.ceil(duration_sec * sample_rate)), 1)
    rir = np.zeros(n_samples, dtype=np.float64)

    room_dim_arr = np.asarray(room_dim, dtype=np.float64)
    source_pos_arr = np.asarray(source_pos, dtype=np.float64)
    mic_pos_arr = np.asarray(mic_pos, dtype=np.float64)

    if beta < 1e-9:
        max_order = 0
    else:
        max_order = int(np.ceil(np.log(amplitude_threshold) / np.log(beta)))
    truncated = max_order > max_order_cap
    max_order = max(min(max_order, max_order_cap), 0)

    # An image's order along one axis alone can't exceed the total
    # budget, and grows as ~2*|n| with the tile index, so this bounds
    # the per-axis search range directly from max_order (tighter, and
    # cheaper, than bounding by physical travel distance).
    n_range = max_order // 2 + 1

    nx = np.arange(-n_range, n_range + 1)
    signs = np.array([1.0, -1.0])

    # Build every (axis tile index, axis mirror sign) combination as a
    # flat array per axis, then take the Cartesian product across the
    # three axes with broadcasting -- this replaces an 8*(2n+1)^3 Python
    # loop with a handful of vectorized NumPy operations.
    def _axis_images(n_vals: np.ndarray, length: float, source_coord: float) -> tuple[np.ndarray, np.ndarray]:
        n_grid, s_grid = np.meshgrid(n_vals, signs, indexing="ij")
        n_grid = n_grid.ravel()
        s_grid = s_grid.ravel()
        pos = 2.0 * n_grid * length + s_grid * source_coord
        order = np.abs(np.floor(pos / length)).astype(np.int64)
        return pos, order

    pos_x, order_x = _axis_images(nx, room_dim_arr[0], source_pos_arr[0])
    pos_y, order_y = _axis_images(nx, room_dim_arr[1], source_pos_arr[1])
    pos_z, order_z = _axis_images(nx, room_dim_arr[2], source_pos_arr[2])

    # Cartesian product across axes.
    PX, PY, PZ = np.meshgrid(pos_x, pos_y, pos_z, indexing="ij")
    OX, OY, OZ = np.meshgrid(order_x, order_y, order_z, indexing="ij")
    order = (OX + OY + OZ).ravel()
    keep = order <= max_order
    if not np.any(keep):
        return RIRResult(
            rir=rir, sample_rate=sample_rate, rt60_target=rt60, rt60_nominal=rt60_nominal,
            absorption=absorption, reflection_coefficient=beta, max_order=max_order,
            n_images=0, truncated=truncated,
        )

    images = np.stack([PX.ravel()[keep], PY.ravel()[keep], PZ.ravel()[keep]], axis=1)
    order = order[keep]

    distance = np.linalg.norm(images - mic_pos_arr[None, :], axis=1)
    direct_distance = max(float(np.linalg.norm(source_pos_arr - mic_pos_arr)), 1e-6)
    nonzero = distance > 1e-6
    images, order, distance = images[nonzero], order[nonzero], distance[nonzero]

    attenuation = (beta**order) / distance
    significant = attenuation >= amplitude_threshold / direct_distance
    order, distance, attenuation = order[significant], distance[significant], attenuation[significant]

    delay_samples = distance / SPEED_OF_SOUND_M_S * sample_rate
    in_range = delay_samples < (n_samples - 1)
    delay_samples, attenuation = delay_samples[in_range], attenuation[in_range]

    floor_idx = np.floor(delay_samples).astype(np.int64)
    frac = delay_samples - floor_idx

    np.add.at(rir, floor_idx, attenuation * (1.0 - frac))
    np.add.at(rir, floor_idx + 1, attenuation * frac)

    return RIRResult(
        rir=rir,
        sample_rate=sample_rate,
        rt60_target=rt60,
        rt60_nominal=rt60_nominal,
        absorption=absorption,
        reflection_coefficient=beta,
        max_order=max_order,
        n_images=int(delay_samples.size),
        truncated=truncated,
    )


def convolve_source_to_mic(source: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Apply a room impulse response to a source signal: ``x = h * s``
    (full linear convolution, truncated back to `len(source)` samples so
    every mic channel stays the same length as the inputs -- the tail
    beyond that point is reverberant energy past the analysis window,
    which a real recording would also just not capture within a clip of
    that length).

    Uses `scipy.signal.fftconvolve`: mathematically identical to direct
    (time-domain) convolution, just computed via the FFT for O(N log N)
    instead of O(N*M) -- RIRs here can run into the tens of thousands of
    samples (see `generate_rir`), where direct convolution against a
    multi-second source is the dominant cost of this whole module. This
    is standard-library signal processing, not part of the from-scratch
    ICA/room-acoustics math this project implements by hand.
    """
    convolved = fftconvolve(source, rir, mode="full")
    return convolved[: len(source)]


def render_convolutive_mixture(
    sources: np.ndarray,
    room_dim: tuple[float, float, float],
    source_positions: list[tuple[float, float, float]],
    mic_positions: list[tuple[float, float, float]],
    sample_rate: int,
    absorption: float | None = None,
    rt60: float | None = None,
    **rir_kwargs,
) -> tuple[np.ndarray, list[list[RIRResult]]]:
    """
    Build a convolutive multi-mic mixture: every mic sums every source
    convolved with that source-mic pair's RIR, ``x_m(t) = sum_n h_mn * s_n(t)``.

    sources         : (n_sources, n_samples)
    returns         : (n_mics, n_samples) mixture, and the (n_mics, n_sources)
                      grid of RIRResults used (for display/diagnostics).
    """
    if len(source_positions) != sources.shape[0]:
        raise RoomAcousticsError(
            f"Got {sources.shape[0]} sources but {len(source_positions)} source positions."
        )

    n_mics = len(mic_positions)
    n_sources, n_samples = sources.shape
    mixture = np.zeros((n_mics, n_samples), dtype=np.float64)
    rirs: list[list[RIRResult]] = []

    for m, mic_pos in enumerate(mic_positions):
        row: list[RIRResult] = []
        for s in range(n_sources):
            result = generate_rir(
                room_dim=room_dim,
                source_pos=source_positions[s],
                mic_pos=mic_pos,
                sample_rate=sample_rate,
                absorption=absorption,
                rt60=rt60,
                **rir_kwargs,
            )
            if result.truncated:
                warnings.warn(
                    f"RIR for source {s} -> mic {m} hit max_order_cap before reaching "
                    "amplitude_threshold; its tail is truncated (RT60 achieved may be "
                    "shorter than requested).",
                    stacklevel=2,
                )
            row.append(result)
            mixture[m] += convolve_source_to_mic(sources[s], result.rir)
        rirs.append(row)

    return mixture, rirs
