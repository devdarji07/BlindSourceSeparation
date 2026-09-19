"""
Evaluation metrics for comparing recovered signals against a reference
(ground-truth original sources, or another algorithm's recovered output).

ICA recovers sources only up to **permutation, sign, and scale** -- so
every metric here first finds the best-matching permutation (via
absolute Pearson correlation, solved as a linear assignment problem),
then the least-squares scale/sign that best aligns each matched pair,
before computing correlation / SNR / MSE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    # Avoids a circular import: separation_metrics.py imports match_signals
    # from this module, so it can't be imported here at runtime.
    from src.separation_metrics import SeparationReport


@dataclass
class MatchResult:
    """Best permutation + sign alignment between two sets of signals."""

    permutation: np.ndarray          # indices into `estimated` rows that best match `reference` rows
    signs: np.ndarray                 # +1/-1 per matched pair (sign of the raw correlation)
    correlation_matrix: np.ndarray     # abs(correlation), (n_reference, n_estimated)


def match_signals(reference: np.ndarray, estimated: np.ndarray) -> MatchResult:
    """
    Solve the linear assignment problem that maximizes total absolute
    correlation between rows of `reference` and rows of `estimated`.
    """
    n = reference.shape[0]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(reference[i], estimated[j])[0, 1]

    abs_corr = np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(-abs_corr)  # maximize -> minimize the negative
    permutation = col_ind[np.argsort(row_ind)]
    signs = np.array([1 if corr[i, permutation[i]] >= 0 else -1 for i in range(n)])

    return MatchResult(permutation=permutation, signs=signs, correlation_matrix=abs_corr)


def align_estimated(estimated: np.ndarray, match: MatchResult) -> np.ndarray:
    """Reorder + sign-flip estimated rows to align with the reference ordering."""
    return estimated[match.permutation] * match.signs[:, None]


def compute_correlation(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Pearson correlation between two 1-D signals."""
    return float(np.corrcoef(reference, estimated)[0, 1])


def compute_snr_db(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB: 10*log10(signal power / residual power)."""
    residual = reference - estimated
    signal_power = np.sum(reference**2)
    residual_power = np.sum(residual**2)
    if residual_power < 1e-20:
        return float("inf")
    return float(10 * np.log10(signal_power / residual_power))


def compute_mse(reference: np.ndarray, estimated: np.ndarray) -> float:
    """Mean squared error between two 1-D signals."""
    return float(np.mean((reference - estimated) ** 2))


@dataclass
class EvaluationReport:
    """Per-component and aggregate metrics comparing recovered vs. reference signals."""

    permutation: np.ndarray
    signs: np.ndarray
    correlation_matrix: np.ndarray   # full abs-correlation matrix used for matching
    correlations: np.ndarray          # per matched pair, after sign/scale alignment
    snr_db: np.ndarray                 # per matched pair
    mse: np.ndarray                     # per matched pair
    mean_abs_correlation: float
    mean_snr_db: float
    mean_mse: float
    recovery_percentage: float          # mean_abs_correlation expressed as a percentage


def evaluate(reference: np.ndarray, estimated: np.ndarray) -> EvaluationReport:
    """
    Full evaluation pipeline: best-match permutation/sign, best-fit
    least-squares scale per pair, then per-component correlation / SNR / MSE.
    """
    n = reference.shape[0]
    match = match_signals(reference, estimated)
    aligned = align_estimated(estimated, match)

    correlations = np.zeros(n)
    snrs = np.zeros(n)
    mses = np.zeros(n)

    for i in range(n):
        ref = reference[i]
        est = aligned[i]
        denom = np.dot(est, est)
        scale = np.dot(ref, est) / denom if denom > 1e-20 else 0.0
        est_scaled = scale * est

        correlations[i] = compute_correlation(ref, est_scaled)
        snrs[i] = compute_snr_db(ref, est_scaled)
        mses[i] = compute_mse(ref, est_scaled)

    finite_snrs = snrs[np.isfinite(snrs)]

    return EvaluationReport(
        permutation=match.permutation,
        signs=match.signs,
        correlation_matrix=match.correlation_matrix,
        correlations=correlations,
        snr_db=snrs,
        mse=mses,
        mean_abs_correlation=float(np.mean(np.abs(correlations))),
        mean_snr_db=float(np.mean(finite_snrs)) if finite_snrs.size else float("inf"),
        mean_mse=float(np.mean(mses)),
        recovery_percentage=float(np.mean(np.abs(correlations)) * 100),
    )


def build_metrics_report(
    component_names: list[str],
    my_eval: EvaluationReport,
    my_execution_time_sec: float,
    my_n_iterations: int,
    my_converged: bool,
    sk_eval: EvaluationReport | None = None,
    sk_execution_time_sec: float | None = None,
    sk_n_iterations: int | None = None,
    sk_converged: bool | None = None,
    my_sep: "SeparationReport | None" = None,
    sk_sep: "SeparationReport | None" = None,
) -> pd.DataFrame:
    """
    Build a flat, spreadsheet-friendly metrics report: one row per
    component per implementation, plus one aggregate "MEAN" row per
    implementation. Execution time / iteration / convergence / recovery-%
    columns are repeated on every row so each row is self-contained.

    `my_sep`/`sk_sep` are optional `SeparationReport`s (see
    `src/separation_metrics.py`) -- when given, the field-standard
    SI-SDR/SDR/SIR/SAR columns are included alongside the legacy
    correlation/SNR/MSE columns kept here for continuity.
    """

    def _implementation_rows(
        name: str,
        ev: EvaluationReport,
        exec_time: float,
        n_iter: int,
        converged: bool,
        sep: "SeparationReport | None",
    ) -> list[dict]:
        def _sep_fields(i: int | None) -> dict:
            if sep is None:
                return {"SI-SDR (dB)": None, "SDR (dB)": None, "SIR (dB)": None, "SAR (dB)": None}
            if i is None:
                return {
                    "SI-SDR (dB)": sep.mean_si_sdr_db,
                    "SDR (dB)": sep.mean_sdr_db,
                    "SIR (dB)": sep.mean_sir_db,
                    "SAR (dB)": sep.mean_sar_db,
                }
            return {
                "SI-SDR (dB)": sep.si_sdr_db[i],
                "SDR (dB)": sep.sdr_db[i],
                "SIR (dB)": sep.sir_db[i],
                "SAR (dB)": sep.sar_db[i],
            }

        rows = [
            {
                "Implementation": name,
                "Component": comp,
                **_sep_fields(i),
                "Correlation": ev.correlations[i],
                "SNR (dB)": ev.snr_db[i],
                "MSE": ev.mse[i],
                "Execution Time (ms)": exec_time * 1000,
                "Iterations": n_iter,
                "Converged": converged,
                "Mean |Correlation|": ev.mean_abs_correlation,
                "Mean SNR (dB)": ev.mean_snr_db,
                "Mean MSE": ev.mean_mse,
                "Recovery %": ev.recovery_percentage,
            }
            for i, comp in enumerate(component_names)
        ]
        rows.append(
            {
                "Implementation": name,
                "Component": "MEAN (aggregate)",
                **_sep_fields(None),
                "Correlation": ev.mean_abs_correlation,
                "SNR (dB)": ev.mean_snr_db,
                "MSE": ev.mean_mse,
                "Execution Time (ms)": exec_time * 1000,
                "Iterations": n_iter,
                "Converged": converged,
                "Mean |Correlation|": ev.mean_abs_correlation,
                "Mean SNR (dB)": ev.mean_snr_db,
                "Mean MSE": ev.mean_mse,
                "Recovery %": ev.recovery_percentage,
            }
        )
        return rows

    rows = _implementation_rows("My FastICA", my_eval, my_execution_time_sec, my_n_iterations, my_converged, my_sep)
    if sk_eval is not None:
        rows += _implementation_rows(
            "scikit-learn", sk_eval, sk_execution_time_sec, sk_n_iterations, sk_converged, sk_sep
        )

    return pd.DataFrame(rows)
