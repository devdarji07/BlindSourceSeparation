"""
Field-standard separation metrics: SI-SDR (from scratch) and BSS_Eval
SDR/SIR/SAR (via `mir_eval.separation`).

These replace the ad-hoc Pearson-correlation/SNR/MSE numbers in
`src/metrics.py` as the headline recovery-quality metrics. That module's
metrics are kept as a secondary panel in the UI, not removed, since they're
still a cheap sanity check -- but a single correlation number can't tell
you *why* a separation is bad, whereas SIR (how much of another source
leaked in) and SAR (how much non-source distortion the algorithm itself
introduced) are diagnostic.

ICA recovers sources only up to permutation, sign, and scale (see
`src/metrics.py` module docstring). Both metric families here resolve
that ambiguity, but differently:

- The **permutation** is resolved once, up front, via `match_signals`
  (Hungarian assignment on absolute Pearson correlation -- the same
  procedure the legacy metrics use), so SI-SDR and BSS_Eval always score
  the same source/estimate pairing.
- **Sign and scale** are then resolved independently by each metric's own
  formulation: SI-SDR's optimal projection coefficient `alpha` (which can
  be negative) absorbs both; BSS_Eval's time-invariant filter projection
  does the same internally. Neither needs the legacy module's separate
  least-squares scale/sign alignment step.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import mir_eval.separation as mir_separation
import numpy as np

from src.metrics import match_signals


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    r"""
    Scale-invariant signal-to-distortion ratio (SI-SDR), in dB.

    Le Roux, Wisdom, Erdogan, Hershey, "SDR - Half-baked or Well Done?",
    ICASSP 2019, eq. 5-7::

        alpha    = <s, s_hat> / <s, s>
        e_target = alpha * s
        e_res    = s_hat - e_target
        SI-SDR   = 10 * log10( ||e_target||^2 / ||e_res||^2 )

    where `s` is the reference and `s_hat` the estimate. `alpha` is the
    scalar that best explains `s_hat` as a copy of `s` in the
    least-squares sense; it absorbs any scale *and* sign difference
    between the two, so SI-SDR is invariant to both by construction --
    exactly the ambiguity ICA leaves unresolved. Both signals are
    zero-meaned first (standard convention; a DC offset is not part of
    the audio content). `eps` guards against division by zero for
    all-silence input; it also means SI-SDR is finite (not literally
    +inf) for perfect reconstruction.
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)

    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()

    ref_energy = np.dot(reference, reference) + eps
    alpha = np.dot(reference, estimate) / ref_energy

    e_target = alpha * reference
    e_res = estimate - e_target

    return float(10 * np.log10((np.dot(e_target, e_target) + eps) / (np.dot(e_res, e_res) + eps)))


@dataclass
class SeparationReport:
    """SI-SDR + BSS_Eval SDR/SIR/SAR for a set of recovered sources, permutation-aligned to reference."""

    permutation: np.ndarray      # indices into `estimated` rows that best match `reference` rows
    si_sdr_db: np.ndarray         # per component
    sdr_db: np.ndarray             # per component -- BSS_Eval SDR (overall distortion)
    sir_db: np.ndarray              # per component -- BSS_Eval SIR (interference from other sources)
    sar_db: np.ndarray               # per component -- BSS_Eval SAR (artifacts introduced by the algorithm)
    mean_si_sdr_db: float
    mean_sdr_db: float
    mean_sir_db: float
    mean_sar_db: float


def evaluate_separation(reference: np.ndarray, estimated: np.ndarray) -> SeparationReport:
    """
    Score recovered sources against ground truth with SI-SDR and BSS_Eval
    SDR/SIR/SAR.

    reference : (n_sources, n_samples) ground-truth signals
    estimated : (n_sources, n_samples) recovered signals, any order/sign/scale

    Permutation is resolved once (see module docstring) and applied to
    `estimated` before either metric is computed, so both describe the
    same pairing.
    """
    n = reference.shape[0]
    match = match_signals(reference, estimated)
    permuted = estimated[match.permutation]

    si_sdr_vals = np.array([si_sdr(reference[i], permuted[i]) for i in range(n)])

    with warnings.catch_warnings():
        # mir_eval 0.8 deprecates bss_eval_sources in favor of a
        # museval-based workflow but still ships and documents it; the
        # replacement is a separate package, not a drop-in mir_eval call,
        # so we keep using this until that migration is worth its own
        # phase.
        warnings.simplefilter("ignore", FutureWarning)
        sdr, sir, sar, _ = mir_separation.bss_eval_sources(reference, permuted, compute_permutation=False)

    return SeparationReport(
        permutation=match.permutation,
        si_sdr_db=si_sdr_vals,
        sdr_db=sdr,
        sir_db=sir,
        sar_db=sar,
        mean_si_sdr_db=float(np.mean(si_sdr_vals)),
        mean_sdr_db=float(np.mean(sdr)),
        mean_sir_db=float(np.mean(sir)),
        mean_sar_db=float(np.mean(sar)),
    )
