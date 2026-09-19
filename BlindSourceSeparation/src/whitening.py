"""
Whitening implemented from scratch (no sklearn PCA).

Whitening is the standard pre-processing step for ICA: it removes
correlations between mixed channels and rescales every direction to unit
variance. This turns the hard problem ("find any invertible un-mixing
matrix") into an easier one ("find the right *rotation*"), which is what
FastICA (Milestone 4) actually searches for.

Steps
-----
1. Mean subtraction   : center each channel to zero mean.
2. Covariance matrix  : Cov = (1/N) * Xc @ Xc.T
3. Eigen decomposition: Cov = E @ D @ E.T (D diagonal, E orthonormal columns)
4. Whitening transform: W = D^(-1/2) @ E.T,  Z = W @ Xc  =>  Cov(Z) ≈ I
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WhiteningResult:
    """Every intermediate artifact of the whitening pipeline, for display and reuse."""

    mean: np.ndarray                 # (n_channels,)
    centered: np.ndarray              # (n_channels, n_samples), X - mean
    covariance: np.ndarray             # (n_channels, n_channels), covariance of centered X
    eigenvalues: np.ndarray              # (n_channels,), descending
    eigenvectors: np.ndarray              # (n_channels, n_channels), columns = eigenvectors
    whitening_matrix: np.ndarray            # (n_channels, n_channels), Z = W @ Xc
    whitened: np.ndarray                     # (n_channels, n_samples), Z
    whitened_covariance: np.ndarray            # should be ~= identity


class WhiteningError(Exception):
    """Raised when whitening cannot proceed (e.g. a singular covariance matrix)."""


def whiten(X: np.ndarray, eps: float = 1e-10) -> WhiteningResult:
    """
    Whiten a stack of signals ``X`` (n_channels, n_samples), from scratch.

    Raises WhiteningError if the covariance matrix is (near) singular,
    which happens if two channels are linearly dependent (e.g. identical
    signals, or fewer independent sources than channels).
    """
    n_channels, n_samples = X.shape

    # Step 1: mean subtraction.
    mean = X.mean(axis=1, keepdims=True)
    centered = X - mean

    # Step 2: covariance matrix.
    covariance = (centered @ centered.T) / n_samples

    # Step 3: eigen decomposition (covariance is symmetric, so `eigh` is
    # both faster and numerically more stable than the general `eig`).
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]  # eigh returns ascending order; flip to descending
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    if np.any(eigenvalues < eps):
        raise WhiteningError(
            "Covariance matrix is (near) singular -- the mixed channels may be "
            "linearly dependent. Try uploading more varied source signals."
        )

    # Step 4: whitening transform. Rescaling each principal direction by
    # 1/sqrt(eigenvalue) gives every direction exactly unit variance.
    d_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    whitening_matrix = d_inv_sqrt @ eigenvectors.T

    whitened = whitening_matrix @ centered
    whitened_covariance = (whitened @ whitened.T) / n_samples

    return WhiteningResult(
        mean=mean.flatten(),
        centered=centered,
        covariance=covariance,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        whitening_matrix=whitening_matrix,
        whitened=whitened,
        whitened_covariance=whitened_covariance,
    )
