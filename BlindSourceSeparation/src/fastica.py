"""
FastICA implemented from scratch (no sklearn).

Implements the *symmetric* (parallel) fixed-point FastICA algorithm from:

    A. Hyvärinen and E. Oja, "Independent Component Analysis: Algorithms
    and Applications," Neural Networks, 13(4-5), 2000.
    A. Hyvärinen, "Fast and Robust Fixed-Point Algorithms for Independent
    Component Analysis," IEEE Trans. Neural Networks, 10(3), 1999.

Given whitened, zero-mean, unit-variance, uncorrelated data ``Z`` (see
`src/whitening.py`), FastICA searches for an **orthogonal** un-mixing
matrix ``W`` such that ``S_hat = W @ Z`` are maximally non-Gaussian (a
proxy for statistical independence, by the Central Limit Theorem: a
mixture of independent sources is *more* Gaussian than any of them
individually, so un-mixing = maximizing non-Gaussianity).

All components are estimated simultaneously (the "symmetric" variant),
with a symmetric orthogonal decorrelation step after every update so `W`
stays on the orthogonal manifold -- this is what makes the components
mutually uncorrelated at every iteration, not just at convergence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


def _logcosh(u: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """g(u) = tanh(alpha*u),  g'(u) = alpha*(1 - tanh(alpha*u)^2). Good general-purpose default."""
    tanh_au = np.tanh(alpha * u)
    return tanh_au, alpha * (1.0 - tanh_au**2)


def _exp(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """g(u) = u*exp(-u^2/2),  g'(u) = (1-u^2)*exp(-u^2/2). Robust to outliers."""
    exp_term = np.exp(-(u**2) / 2.0)
    return u * exp_term, (1.0 - u**2) * exp_term


def _cube(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """g(u) = u^3,  g'(u) = 3*u^2. Best for sub-Gaussian sources (e.g. uniform-like)."""
    return u**3, 3.0 * u**2


NONLINEARITIES: dict[str, Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
    "logcosh": _logcosh,
    "exp": _exp,
    "cube": _cube,
}


@dataclass
class FastICAResult:
    """Everything produced by a FastICA run, for display and downstream comparison."""

    unmixing_matrix: np.ndarray            # W (n_components, n_components), applied to whitened data
    sources: np.ndarray                     # S_hat = W @ Z (n_components, n_samples)
    n_iterations: int                        # iterations actually run
    converged: bool                           # whether tol was reached before max_iter
    convergence_history: list[float] = field(default_factory=list)  # delta per iteration
    nonlinearity: str = "logcosh"
    execution_time_sec: float = 0.0
    tol: float = 1e-6
    max_iter: int = 200


class FastICA:
    """
    From-scratch symmetric FastICA.

    Parameters
    ----------
    nonlinearity : one of "logcosh", "exp", "cube" -- the contrast function
        used as a proxy for non-Gaussianity.
    max_iter     : hard cap on fixed-point iterations.
    tol          : convergence tolerance on the per-iteration change in W.
    random_state : seed for the random weight initialization.
    """

    def __init__(
        self,
        nonlinearity: str = "logcosh",
        max_iter: int = 200,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        if nonlinearity not in NONLINEARITIES:
            raise ValueError(f"Unknown nonlinearity '{nonlinearity}'. Choose from {list(NONLINEARITIES)}.")
        self.nonlinearity = nonlinearity
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _initialize_weights(self, n_components: int, rng: np.random.Generator) -> np.ndarray:
        """Random Gaussian initial guess for W, immediately orthogonalized."""
        w_init = rng.normal(size=(n_components, n_components))
        return self._symmetric_decorrelation(w_init)

    @staticmethod
    def _symmetric_decorrelation(W: np.ndarray) -> np.ndarray:
        """
        Orthogonalize W symmetrically: W_orth = (W @ W.T)^(-1/2) @ W.

        This is the "symmetric" orthogonalization used by the parallel
        FastICA variant -- it decorrelates all rows of W simultaneously
        (no single row is treated as more important than another), which
        avoids the error accumulation of one-component-at-a-time
        (deflationary) orthogonalization.
        """
        WWt = W @ W.T
        eigenvalues, eigenvectors = np.linalg.eigh(WWt)
        eigenvalues = np.clip(eigenvalues, 1e-12, None)  # numerical safety
        inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        return inv_sqrt @ W

    def fit_transform(self, Z: np.ndarray) -> FastICAResult:
        """
        Run the fixed-point iteration on whitened data ``Z`` (n_components, n_samples).
        """
        start = time.perf_counter()

        n_components, n_samples = Z.shape
        rng = np.random.default_rng(self.random_state)
        g_func = NONLINEARITIES[self.nonlinearity]

        # --- Weight initialization ---
        W = self._initialize_weights(n_components, rng)

        history: list[float] = []
        converged = False
        n_iter_done = 0

        for iteration in range(1, self.max_iter + 1):
            # --- Fixed-point update ---
            WZ = W @ Z                       # (n_components, n_samples) projections
            gwz, g_prime_wz = g_func(WZ)      # nonlinearity + its derivative

            # Hyvärinen's fixed-point rule: E[Z * g(W.Z)] - E[g'(W.Z)] * W
            W_new = (gwz @ Z.T) / n_samples - np.diag(g_prime_wz.mean(axis=1)) @ W

            # --- Orthogonalization ---
            W_new = self._symmetric_decorrelation(W_new)

            # --- Convergence detection ---
            # Rows of W are unit-norm after decorrelation, so |dot(w_new_i, w_old_i)|
            # approaches 1 as the direction stops changing; this is the standard
            # FastICA convergence criterion (sign/permutation-invariant).
            delta = float(np.max(np.abs(np.abs(np.sum(W_new * W, axis=1)) - 1.0)))
            history.append(delta)

            W = W_new
            n_iter_done = iteration

            if delta < self.tol:
                converged = True
                break

        sources = W @ Z
        elapsed = time.perf_counter() - start

        return FastICAResult(
            unmixing_matrix=W,
            sources=sources,
            n_iterations=n_iter_done,
            converged=converged,
            convergence_history=history,
            nonlinearity=self.nonlinearity,
            execution_time_sec=elapsed,
            tol=self.tol,
            max_iter=self.max_iter,
        )


@dataclass
class SklearnFastICAResult:
    """Result of running scikit-learn's FastICA, for comparison against `FastICAResult`."""

    unmixing_matrix: np.ndarray
    sources: np.ndarray
    n_iterations: int
    converged: bool
    execution_time_sec: float


def run_sklearn_fastica(
    Z: np.ndarray,
    nonlinearity: str = "logcosh",
    max_iter: int = 200,
    tol: float = 1e-6,
    random_state: int | None = 0,
) -> SklearnFastICAResult:
    """
    Run scikit-learn's FastICA on already-whitened data, for a fair,
    apples-to-apples comparison against the from-scratch implementation
    above (Milestone 5).

    `whiten=False` because `Z` is already whitened by `src/whitening.py`
    -- both implementations then start from the exact same whitened data,
    so any difference is purely in the fixed-point solver itself.
    """
    from sklearn.decomposition import FastICA as SklearnFastICA

    start = time.perf_counter()
    model = SklearnFastICA(
        algorithm="parallel",
        whiten=False,
        fun=nonlinearity,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
    )
    sources = model.fit_transform(Z.T).T  # sklearn expects (n_samples, n_features)
    elapsed = time.perf_counter() - start

    n_iter = int(model.n_iter_)
    return SklearnFastICAResult(
        unmixing_matrix=model.components_,
        sources=sources,
        n_iterations=n_iter,
        converged=n_iter < max_iter,
        execution_time_sec=elapsed,
    )
