"""
Plotly-based visualizations for the BSS/ICA pipeline.

Kept in one module so every page of the app shares a single look
(dark theme, consistent margins/fonts) as more plots are added in
later milestones (spectrograms, covariance heatmaps, convergence, etc.).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from scipy import signal as scipy_signal

DEFAULT_COLORS = ["#5eead4", "#f472b6", "#60a5fa"]

_DARK_COLORWAY = [
    "#5eead4", "#f472b6", "#60a5fa", "#a78bfa", "#34d399",
    "#fbbf24", "#fb7185", "#38bdf8", "#c084fc", "#4ade80",
]
_DARK_AXIS_STYLE = dict(
    gridcolor="rgba(255,255,255,0.08)",
    linecolor="rgba(255,255,255,0.22)",
    zerolinecolor="rgba(255,255,255,0.14)",
)


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """
    Apply this app's dark-glassmorphism chart theme via explicit layout
    properties, rather than via `template="plotly_dark"`.

    Streamlit's `st.plotly_chart(..., theme=None)` -- needed so it doesn't
    override our colors with its own light theme -- strips the `template`
    key while serializing the figure for the frontend, which silently
    falls back to Plotly.js's light default. Explicit layout properties
    survive that path, so we set them directly instead.
    """
    fig.update_layout(
        paper_bgcolor="#10162b",
        plot_bgcolor="#10162b",
        font=dict(color="#eef1fb"),
        colorway=_DARK_COLORWAY,
    )
    fig.update_xaxes(**_DARK_AXIS_STYLE)
    fig.update_yaxes(**_DARK_AXIS_STYLE)
    fig.update_scenes(
        xaxis=_DARK_AXIS_STYLE,
        yaxis=_DARK_AXIS_STYLE,
        zaxis=_DARK_AXIS_STYLE,
        bgcolor="#10162b",
    )
    return fig


def plot_waveform(
    data: np.ndarray,
    sample_rate: int,
    title: str,
    color: str = DEFAULT_COLORS[0],
) -> go.Figure:
    """Return an interactive waveform plot (zoom/pan/hover) for one mono signal."""
    time_axis = np.arange(len(data)) / sample_rate
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=time_axis,
            y=data,
            mode="lines",
            line=dict(color=color, width=1),
            name=title,
            hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
    )
    return _apply_dark_theme(fig)


def plot_spectrogram(
    data: np.ndarray,
    sample_rate: int,
    title: str,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Return an interactive spectrogram (STFT magnitude, in dB) for one mono signal."""
    nperseg = min(1024, len(data))
    noverlap = nperseg // 2
    freqs, times, sxx = scipy_signal.spectrogram(
        data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap
    )
    sxx_db = 10 * np.log10(sxx + 1e-12)

    fig = go.Figure(
        data=go.Heatmap(
            z=sxx_db,
            x=times,
            y=freqs,
            colorscale=colorscale,
            colorbar=dict(title="dB"),
            hovertemplate="t=%{x:.2f}s<br>f=%{y:.0f}Hz<br>%{z:.1f}dB<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return _apply_dark_theme(fig)


def plot_matrix_heatmap(
    matrix: np.ndarray,
    title: str,
    row_labels: list[str] | None = None,
    col_labels: list[str] | None = None,
    colorscale: str = "RdBu",
    zmid: float | None = 0.0,
) -> go.Figure:
    """
    Return an interactive annotated heatmap for a small square/rectangular
    matrix (mixing matrix, covariance matrix, correlation matrix, ...).
    """
    n_rows, n_cols = matrix.shape
    row_labels = row_labels or [f"Row {i + 1}" for i in range(n_rows)]
    col_labels = col_labels or [f"Col {j + 1}" for j in range(n_cols)]
    text = np.round(matrix, 3).astype(str)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=col_labels,
            y=row_labels,
            colorscale=colorscale,
            zmid=zmid,
            text=text,
            texttemplate="%{text}",
            colorbar=dict(title="Value"),
        )
    )
    fig.update_layout(
        title=title,
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return _apply_dark_theme(fig)


def plot_scatter(
    signal_matrix: np.ndarray,
    title: str,
    color: str = DEFAULT_COLORS[0],
    max_points: int = 3000,
) -> go.Figure:
    """
    Return an interactive scatter plot of samples from a 2- or 3-channel
    signal matrix (n_channels, n_samples). Used to visually show
    correlation (a slanted cloud) vs. decorrelation (a circular/spherical
    cloud) before and after whitening.
    """
    n_channels, n_samples = signal_matrix.shape
    if n_channels not in (2, 3):
        raise ValueError("Scatter plot supports 2 or 3 channels only.")

    if n_samples > max_points:
        idx = np.linspace(0, n_samples - 1, max_points).astype(int)
    else:
        idx = np.arange(n_samples)

    if n_channels == 2:
        fig = go.Figure(
            data=go.Scattergl(
                x=signal_matrix[0, idx],
                y=signal_matrix[1, idx],
                mode="markers",
                marker=dict(size=3, color=color, opacity=0.5),
            )
        )
        fig.update_layout(xaxis_title="Channel 1", yaxis_title="Channel 2")
    else:
        fig = go.Figure(
            data=go.Scatter3d(
                x=signal_matrix[0, idx],
                y=signal_matrix[1, idx],
                z=signal_matrix[2, idx],
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.5),
            )
        )
        fig.update_layout(scene=dict(xaxis_title="Ch 1", yaxis_title="Ch 2", zaxis_title="Ch 3"))

    fig.update_layout(
        title=title,
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return _apply_dark_theme(fig)


def plot_line(
    x_values: np.ndarray,
    y_series: dict[str, np.ndarray],
    title: str,
    xaxis_title: str,
    yaxis_title: str,
) -> go.Figure:
    """
    Return an interactive multi-series line+marker plot -- used for
    metric-vs-parameter sweeps (e.g. SI-SDR/SIR vs. RT60 or mic spacing).

    y_series maps a trace name to a 1-D array the same length as `x_values`.
    """
    fig = go.Figure()
    for i, (name, y_values) in enumerate(y_series.items()):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines+markers",
                name=name,
                line=dict(color=_DARK_COLORWAY[i % len(_DARK_COLORWAY)]),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=360,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return _apply_dark_theme(fig)


def plot_convergence(
    history: list[float],
    tol: float,
    title: str = "FastICA Convergence",
) -> go.Figure:
    """
    Plot the per-iteration convergence delta (how much W changed) on a log
    scale, against the tolerance threshold that stops the fixed-point loop.
    """
    iterations = np.arange(1, len(history) + 1)
    safe_history = np.maximum(history, 1e-300)  # avoid log(0) for a perfect-convergence step

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=safe_history,
            mode="lines+markers",
            name="Δ (change in W)",
            line=dict(color=DEFAULT_COLORS[0]),
            hovertemplate="iteration=%{x}<br>Δ=%{y:.2e}<extra></extra>",
        )
    )
    fig.add_hline(
        y=tol,
        line_dash="dash",
        line_color="#EF553B",
        annotation_text=f"tolerance = {tol:g}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Δ (log scale)",
        yaxis_type="log",
        height=340,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return _apply_dark_theme(fig)
