"""Small formatting helpers shared across the Streamlit pages."""

from __future__ import annotations

from pathlib import Path


def load_css(path: str | Path) -> str:
    """Read a CSS file and return it wrapped in a <style> tag for st.markdown."""
    css_text = Path(path).read_text(encoding="utf-8")
    return f"<style>{css_text}</style>"


def format_duration(seconds: float) -> str:
    """Format seconds as '3.45s' or '1m 03.2s'."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m {sec:04.1f}s"


def format_hz(value: float) -> str:
    """Format a sample rate as '44.1 kHz' or '500 Hz'."""
    if value >= 1000:
        return f"{value / 1000:.1f} kHz"
    return f"{value:.0f} Hz"
