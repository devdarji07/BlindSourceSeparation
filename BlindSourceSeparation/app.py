"""
Blind Source Separation using ICA — Streamlit application entry point.

Run with:
    streamlit run app.py

This file wires up navigation and page rendering. Each page's real logic
lives in `src/`; `app.py` stays a thin orchestration layer so it keeps
reading like a table of contents as milestones are added.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))

from src.audio import compute_amplitude_stats, to_wav_bytes
from src.loader import AudioLoader, AudioLoadError
from src.fastica import NONLINEARITIES, FastICA, run_sklearn_fastica
from src.fdica import fd_ica
from src.metrics import build_metrics_report, evaluate
from src.mixer import MixingError, generate_mixing_matrix, mix_signals, normalize_rows_peak, stack_sources
from src.room_acoustics import RoomAcousticsError, render_convolutive_mixture, room_surface_area, room_volume
from src.separation_metrics import evaluate_separation
from src.utils import format_duration, format_hz, load_css
from src.visualization import (
    DEFAULT_COLORS,
    plot_convergence,
    plot_line,
    plot_matrix_heatmap,
    plot_scatter,
    plot_spectrogram,
    plot_waveform,
)
from src.whitening import WhiteningError, whiten

PAGES = [
    "Home",
    "Theory",
    "Upload Audio",
    "Mix Signals",
    "Whitening",
    "FastICA",
    "Comparison",
    "Metrics",
    "Visualizations",
    "Convolutive Mixing",
    "Failure Gallery",
    "Real Data",
    "Downloads",
    "About",
]

ASSETS_DIR = Path(__file__).parent / "assets"
DATA_DIR = Path(__file__).parent / "data"


def configure_page() -> None:
    st.set_page_config(
        page_title="Blind Source Separation — ICA",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(load_css(ASSETS_DIR / "style.css"), unsafe_allow_html=True)
    # Visual effects (waves, balloons, scroll reveals) — presentation only.
    fx_js = (ASSETS_DIR / "fx.js").read_text(encoding="utf-8")
    components.html(f"<script>{fx_js}</script>", height=0)


def init_session_state() -> None:
    # signals: list[AudioSignal] | None — populated by the Upload Audio page,
    # consumed by every page that comes after it in the pipeline.
    st.session_state.setdefault("signals", None)


def render_sidebar() -> str:
    st.sidebar.caption("Cocktail Party Problem · BSS")
    # keyed so Home's Theory / feature cards can navigate via session state
    page = st.sidebar.radio("Navigate", PAGES, key="nav_page")
    st.sidebar.divider()

    signals = st.session_state.get("signals")
    if signals:
        st.sidebar.success(f"{len(signals)} signal(s) loaded ✅")
        for sig in signals:
            st.sidebar.caption(f"• {sig.name} — {format_duration(sig.duration)}")
    else:
        st.sidebar.info("No audio loaded yet.")

    return page


# (page, glyph, one-line description, category tag)
FEATURES = [
    ("Mix Signals", "⨁", "Blend your sources into synthetic microphone channels with X = A·S.", "Signal Mixing"),
    ("Whitening", "◎", "Center, decorrelate and rescale the mixtures with PCA.", "Preprocessing"),
    ("FastICA", "∿", "Recover independent components with the from-scratch fixed-point algorithm.", "Source Separation"),
    ("Comparison", "⇄", "Compare your FastICA against scikit-learn and the original sources.", "Signal Analysis"),
    ("Metrics", "◈", "Correlation, SNR, MSE and convergence for every recovered component.", "Evaluation"),
    ("Visualizations", "▤", "Waveforms, spectrograms, covariance, mixing matrices and correlations.", "Signal Analysis"),
    ("Convolutive Mixing", "⌇", "Simulate a real room, then separate with frequency-domain ICA.", "Room Acoustics"),
    ("Failure Gallery", "△", "See exactly where and why ICA breaks down.", "Robustness"),
    ("Real Data", "◉", "Run the pipeline on bundled real speech recordings.", "Real Recordings"),
    ("Downloads", "⇩", "Export recovered audio and the metrics report.", "Export"),
]


def _goto(page: str) -> None:
    """Button callback: switch the sidebar page (presentation-level navigation)."""
    st.session_state["nav_page"] = page


def _render_hero() -> None:
    st.markdown(
        """
        <div class="dsp-hero">
            <h1 class="dsp-title">DIGITAL SIGNAL PROCESSING</h1>
            <h2 class="dsp-title-sub">COCKTAIL PARTY PROBLEM</h2>
            <div class="dsp-sub">Blind Source Separation<i>•</i>Signal Analysis<i>•</i>Independent Component Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feature_reveal() -> None:
    st.markdown(
        '<div class="dsp-locked"><span class="dot"></span>Signals locked · analysis modules unlocked</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dsp-label" style="margin-top:1.6rem">Modules</div>'
        '<div class="dsp-section-title">Analysis Toolkit</div>'
        '<div class="dsp-section-desc">Walk the pipeline from mixing to metrics, or jump straight to any module.</div>',
        unsafe_allow_html=True,
    )
    for row in range(0, len(FEATURES), 2):
        cols = st.columns(2, gap="large")
        for offset, col in enumerate(cols):
            if row + offset >= len(FEATURES):
                continue
            page, glyph, desc, tag = FEATURES[row + offset]
            slug = page.lower().replace(" ", "_")
            side = "featl" if offset == 0 else "featr"
            with col:
                with st.container(key=f"{side}_{slug}"):
                    st.markdown(
                        f"""
                        <div class="dsp-feat-top"><span class="dsp-icon">{glyph}</span><span class="dsp-arrow">→</span></div>
                        <div class="dsp-feat-title">{page}</div>
                        <p class="dsp-feat-desc">{desc}</p>
                        <div class="dsp-meta"><span>{tag}</span><span>{row + offset + 1:02d}</span></div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.button("Open module", key=f"go_{slug}", on_click=_goto, args=(page,))


def _render_pipeline_overview() -> None:
    """The original Home explanation, kept intact, below the fold."""
    with st.container(key="rv_home_about"):
        st.markdown(
            '<div class="dsp-label" style="margin-top:3.5rem">Overview</div>'
            '<div class="dsp-section-title">How the pipeline works</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "This app demonstrates **Blind Source Separation (BSS)** via "
            "**Independent Component Analysis (ICA)**: given only a mixture "
            "of independent signals, recover the originals without being told "
            "how they were mixed. The pipeline here uses a **synthetic, "
            "instantaneous mixture** — you upload the ground-truth sources, "
            "the app generates a random mixing matrix `A` and forms `X = A·S`, "
            "then FastICA (implemented from scratch, cross-checked against "
            "scikit-learn) recovers `S` from `X` alone."
        )
        st.info(
            "ℹ️ **This is not the real 'cocktail party problem.'** Real rooms "
            "mix sound *convolutively* — echoes and inter-mic delays mean each "
            "microphone hears a filtered, time-smeared sum of the sources, not "
            "an instant linear blend. The instantaneous model here is exact "
            "for domains like EEG/MEG, fetal ECG, and some telecom antenna "
            "arrays, and is a genuine building block toward convolutive "
            "separation — but it does not, by itself, separate voices "
            "recorded by real microphones in a real room. See **Theory** and "
            "the (upcoming) convolutive-mixing pages for why."
        )

        st.markdown(
            """
            <div class="nb-flow">
                <div class="nb-flow-step">🎤<br>Original<br>Signals</div>
                <div class="nb-flow-arrow">→</div>
                <div class="nb-flow-step">🎙️<br>Mixed<br>Signals</div>
                <div class="nb-flow-arrow">→</div>
                <div class="nb-flow-step">🧠<br>Recovered<br>Signals</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="nb-grid">
                <div class="nb-card yellow">
                    <h3>1️⃣ Original Signals</h3>
                    <p>Independent sources — e.g. separate speakers or instruments — each statistically unrelated to the others.</p>
                </div>
                <div class="nb-card blue">
                    <h3>2️⃣ Mixed Signals</h3>
                    <p>What each microphone actually records: a linear mixture <code>X = A·S</code> of every source at once.</p>
                </div>
                <div class="nb-card pink">
                    <h3>3️⃣ Recovered Signals</h3>
                    <p>Original sources reconstructed with FastICA, using only the mixed recordings — no knowledge of A or S.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_home() -> None:
    _render_hero()

    left, right = st.columns([1.7, 1], gap="large")
    with left:
        with st.container(key="upload_card"):
            st.markdown(
                """
                <div class="dsp-upload-head">
                    <div class="dsp-wave-icon"><b></b><b></b><b></b><b></b><b></b><b></b><b></b></div>
                    <div class="dsp-label">Audio Input</div>
                    <h3>Upload / Drop Audio</h3>
                    <p>2–3 WAV files · Drag &amp; drop supported</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_upload_audio(embedded=True)
    with right:
        with st.container(key="theory_card"):
            st.markdown(
                """
                <span class="dsp-icon">∫</span>
                <div class="dsp-label" style="margin-top:1.4rem">Theory</div>
                <div class="dsp-card-title">Learn the DSP behind the system</div>
                <p class="dsp-card-desc">Cocktail party problem, whitening, FastICA, convolutive mixing and frequency-domain ICA, explained from first principles.</p>
                """,
                unsafe_allow_html=True,
            )
            st.button("Explore →", key="go_theory", on_click=_goto, args=("Theory",))

    if st.session_state.get("signals"):
        _render_feature_reveal()
    else:
        st.markdown(
            '<div class="dsp-flow-note">Upload audio to unlock the analysis modules</div>',
            unsafe_allow_html=True,
        )

    _render_pipeline_overview()


THEORY_SECTIONS = [
    ("bss", "Cocktail Party & BSS"),
    ("clt", "Central Limit Theorem"),
    ("pca", "PCA"),
    ("white", "Whitening"),
    ("ica", "FastICA"),
    ("apps", "Applications"),
]


def _theory_section(key: str):
    """Keyed container for one Theory section (scroll target + reveal hook)."""
    number = [k for k, _ in THEORY_SECTIONS].index(key) + 1
    title = dict(THEORY_SECTIONS)[key]
    box = st.container(key=f"rv_theory_{key}")
    box.markdown(f'<div class="dsp-sec-num">{number:02d} / {len(THEORY_SECTIONS):02d} · {title.upper()}</div>', unsafe_allow_html=True)
    return box


def render_theory() -> None:
    st.markdown(
        """
        <div class="dsp-theory-hero">
            <div class="dsp-label">Theory</div>
            <h1>THE MATH BEHIND THE SEPARATION</h1>
            <p>Everything on this page is explained beginner-first: what the
            problem is, why it's hard, and the specific math trick this app
            uses to solve it. Every concept here is also implemented for real
            elsewhere in the app — this page is the 'why' behind those pages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    index_links = "".join(
        f'<a data-scroll="rv_theory_{k}"><em>{i + 1:02d}</em>{t}</a>'
        for i, (k, t) in enumerate(THEORY_SECTIONS)
    )
    st.markdown(f'<div class="dsp-index">{index_links}</div>', unsafe_allow_html=True)

    with _theory_section("bss"):
        st.subheader("The Cocktail Party Problem")
        st.write(
            "Picture a party: several people talking at once, and several "
            "microphones scattered around the room. Every microphone picks "
            "up a **blend** of everyone talking, just at different "
            "volumes depending on how close each speaker is. Yet your own "
            "ears can somehow focus on *one* voice and tune out the rest. "
            "**Blind Source Separation (BSS)** is the mathematical version "
            "of that trick: recover each original voice using *only* the "
            "microphone recordings."
        )
        st.write(
            "It's called **'blind'** because the algorithm is never told "
            "how the sources were mixed (no positions, no volumes, no "
            "delays) — it has to figure out the un-mixing purely from "
            "statistical properties of the recordings themselves."
        )
        st.latex(r"X = A \cdot S")
        st.write(
            "`S` = original independent sources (unknown), `A` = the "
            "mixing matrix (unknown), `X` = what the microphones actually "
            "record (the only thing we get to see). BSS solves for `S` "
            "given only `X`."
        )
        st.markdown(
            """
            <div class="nb-grid">
                <div class="nb-card white"><h3>Why not just divide by A?</h3><p>Because A is unknown! We never observe the true mixing matrix, only its effect on the microphones.</p></div>
                <div class="nb-card white"><h3>Why does this even have a solution?</h3><p>Because the sources are <b>statistically independent</b> — and that's a strong enough clue, as the next tab explains.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Instantaneous vs. convolutive mixing")
        st.write(
            "`X = A·S` is the **instantaneous** mixing model: every "
            "microphone's sample at time `t` is a fixed linear combination "
            "of every source's sample at that *same* `t`. That's exact for "
            "signals that reach every sensor with no delay and no echo — "
            "true of EEG/MEG electrodes, fetal ECG leads, and some "
            "closely-spaced telecom antenna arrays."
        )
        st.write(
            "It is **not** true of real microphones in a real room. Sound "
            "takes time to travel, so different mics hear each source "
            "delayed by different amounts, plus reflections off walls "
            "arriving even later. The correct model is a **convolutive** "
            "mixture, `x(t) = Σ_τ h(τ)·s(t−τ)`, where `h` is a whole room "
            "impulse response per source-mic pair, not a single number. "
            "Applying instantaneous ICA to a convolutive mixture is a "
            "model mismatch, and separation quality degrades — sometimes "
            "sharply — as reverberation and mic spacing increase. The "
            "**Convolutive Mixing** page demonstrates exactly this "
            "collapse, using a from-scratch room-impulse-response "
            "generator (image-source method)."
        )
        st.write(
            "**Frequency-domain ICA (FD-ICA)** is one way to recover from "
            "it: convolution in time is multiplication in frequency, so "
            "STFT each mic channel and, *within one narrow frequency "
            "bin*, the room's effect is approximately a single complex "
            "gain per source — locally instantaneous again. Running "
            "(complex-valued) ICA independently per bin, then resolving "
            "the per-bin permutation and scale ambiguity that creates, "
            "recovers a working un-mixing across the whole convolutive "
            "signal. Also on the **Convolutive Mixing** page, alongside "
            "an honest report of where it helps and where it doesn't."
        )

    with _theory_section("clt"):
        st.subheader("The Central Limit Theorem — Why ICA Works At All")
        st.write(
            "The **Central Limit Theorem (CLT)** says that when you add "
            "together many independent random variables, the result tends "
            "to look **more Gaussian (bell-curve-shaped)** than any of the "
            "original variables — no matter what shape they started as."
        )
        st.write(
            "Mixing is just weighted addition: each microphone signal is "
            "a weighted sum of the original sources. So, by the CLT, "
            "**every mixed signal is more Gaussian than any single original "
            "source.** This gives ICA its search strategy:"
        )
        st.markdown(
            "> To un-mix, search for the combination of microphone signals "
            "that is **least Gaussian** (most non-Gaussian). The least "
            "Gaussian directions correspond to the original, unmixed "
            "sources."
        )
        st.info(
            "ℹ️ This is exactly what `logcosh`, `exp`, and `cube` measure "
            "in the **FastICA** tab below — each is a different way of "
            "scoring 'how non-Gaussian is this signal?'"
        )

    with _theory_section("pca"):
        st.subheader("PCA — Principal Component Analysis")
        st.write(
            "PCA finds the directions in your data that carry the most "
            "**variance** (spread), ranks them, and lets you re-express the "
            "data along those directions instead of the original axes. "
            "Concretely, for a set of signals:"
        )
        st.markdown(
            "1. **Center** the data (subtract the mean).\n"
            "2. Compute the **covariance matrix** — how much every channel "
            "varies together with every other channel.\n"
            "3. **Eigen-decompose** the covariance matrix: the "
            "eigenvectors are the principal directions, the eigenvalues "
            "are how much variance lies along each one."
        )
        st.write(
            "ICA doesn't use PCA to reduce dimensions here — it uses "
            "exactly this machinery for a different purpose: **whitening**, "
            "covered next."
        )

    with _theory_section("white"):
        st.subheader("Whitening — PCA's Pre-processing Role in ICA")
        st.write(
            "Whitening rescales the PCA directions so **every direction "
            "has exactly unit variance**, and the channels become "
            "mutually **uncorrelated**. This matters because it simplifies "
            "the problem FastICA has to solve:"
        )
        st.markdown(
            "- **Without whitening**: FastICA would need to search over "
            "*all* invertible matrices — an enormous, poorly-behaved "
            "search space.\n"
            "- **With whitening**: the remaining unknown is just an "
            "**orthogonal rotation** — a much smaller, numerically "
            "well-behaved search space."
        )
        st.latex(r"W = D^{-1/2} E^T, \qquad Z = W \cdot X_{centered}")
        st.write(
            "`E` = eigenvectors, `D` = diagonal matrix of eigenvalues, `Z` "
            "= the whitened signal, ready for FastICA. See the **Whitening** "
            "page for the full worked example with your own audio."
        )

    with _theory_section("ica"):
        st.subheader("FastICA — Finding the Rotation")
        st.write(
            "Starting from whitened data `Z`, FastICA searches for an "
            "orthogonal matrix `W` such that `S_hat = W · Z` is **maximally "
            "non-Gaussian** — which, by the CLT argument above, means "
            "`S_hat` is as close as possible to the original independent "
            "sources."
        )
        st.markdown(
            "1. **Initialize** `W` randomly, then orthogonalize it.\n"
            "2. **Fixed-point update**: nudge each row of `W` in the "
            "direction that increases non-Gaussianity, measured by a "
            "nonlinearity `g` (`logcosh`, `exp`, or `cube`).\n"
            "3. **Re-orthogonalize** `W` so components stay uncorrelated.\n"
            "4. **Repeat** until `W` stops changing (convergence) or a "
            "maximum number of iterations is reached."
        )
        st.info(
            "ℹ️ ICA can only ever recover sources up to **scale, sign, and "
            "order** — never their exact original amplitude or which "
            "component 'comes first'. That's a fundamental property of "
            "the algorithm, not a bug. See the **FastICA** and "
            "**Comparison** pages."
        )
        st.warning(
            "⚠️ **Where this version applies, and where it doesn't.** The "
            "FastICA on this page assumes instantaneous mixing (see the "
            "**Cocktail Party & BSS** tab). It's a good match for EEG/MEG "
            "artifact removal, fetal ECG, and narrowband telecom arrays — "
            "and a poor match for real-room audio, where mixing is "
            "convolutive. Running it on convolutive audio anyway is "
            "exactly the failure case explored later in this project."
        )

    with _theory_section("apps"):
        st.subheader("Real-World Applications")
        st.markdown(
            """
            <div class="nb-grid">
                <div class="nb-card yellow"><h3>🎙️ Speech Enhancement</h3><p>Separating a speaker's voice from background noise or other talkers, e.g. in hearing aids and voice assistants.</p></div>
                <div class="nb-card blue"><h3>🧠 EEG / MEG (Neuroscience)</h3><p>Removing eye-blink and muscle artifacts from brain signal recordings so the underlying neural activity is cleaner.</p></div>
                <div class="nb-card pink"><h3>❤️ Biomedical Signals</h3><p>Separating a fetal heartbeat from the mother's in prenatal ECG monitoring.</p></div>
                <div class="nb-card green"><h3>🎵 Music Source Separation</h3><p>Splitting a mixed recording into vocals, drums, and instruments.</p></div>
                <div class="nb-card white"><h3>📡 Telecommunications</h3><p>Separating overlapping signals arriving at multiple antennas.</p></div>
                <div class="nb-card white"><h3>💰 Finance</h3><p>Finding independent driving factors behind correlated financial time series.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_upload_audio(embedded: bool = False) -> None:
    """Upload page. `embedded=True` renders just the uploader inside Home's card."""
    if not embedded:
        st.title("📤 Upload Audio")
        st.write(
            "Provide **2 or 3** independent sound sources (e.g. two people "
            "speaking, or speech + music) as WAV files. Every source is "
            "validated, resampled to a common rate if needed, trimmed to the "
            "shortest duration, and peak-normalized before being handed to the "
            "rest of the pipeline."
        )
    st.caption(
        "ℹ️ There's no 'record from microphone' option: the mixing model "
        "used here (`X = A·S`) assumes every channel is sampled from the "
        "same clock at the same instant, the way multiple inputs on one "
        "audio interface are. Recording sources one at a time through "
        "separate browser widgets — which is all a browser/cloud app can "
        "do — produces clips with arbitrary relative offsets, not "
        "synchronized microphone channels, so ICA's assumptions would be "
        "violated before the audio ever reached the pipeline."
    )

    named_files: list[tuple] = []

    uploaded_files = st.file_uploader(
        "Choose WAV files",
        type=["wav"],
        accept_multiple_files=True,
        help="Select between 2 and 3 .wav files.",
    )
    if uploaded_files:
        named_files = [(f, f.name) for f in uploaded_files]

    if embedded:
        # Home only: the uploader is empty whenever the page is revisited, so
        # audio loaded earlier stays active until the user actually clears the
        # uploader (main() resets this flag whenever Home is left).
        was_active = st.session_state.get("_home_upload_active", False)
        st.session_state["_home_upload_active"] = bool(named_files)
        if not named_files and not was_active and st.session_state.get("signals"):
            st.caption("Audio from this session is still loaded. Drop new files to replace it.")
            return

    if not named_files:
        st.session_state["signals"] = None
        st.caption("No audio provided yet.")
        return

    loader = AudioLoader()

    try:
        with st.spinner("Analyzing signals…"):
            loader.validate_count(len(named_files))
            raw_signals = [loader.load_file(f, name=name) for f, name in named_files]
            harmonized = loader.harmonize_batch(raw_signals)
    except AudioLoadError as exc:
        st.error(f"❌ {exc}")
        st.session_state["signals"] = None
        return

    st.session_state["signals"] = harmonized
    st.success(f"✅ Loaded and harmonized {len(harmonized)} signal(s).")

    common_sr = harmonized[0].sample_rate
    st.caption(
        f"Common sample rate after harmonization: **{format_hz(common_sr)}**  |  "
        f"Trimmed length: **{format_duration(harmonized[0].duration)}**"
    )

    # On Home the per-source detail folds away so the card stays compact.
    holder = st.expander("Loaded sources — stats, waveforms & playback") if embedded else st.container()
    with holder:
        for i, sig in enumerate(harmonized):
            raw = raw_signals[i]
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            with st.container(border=True):
                st.markdown(f"#### 🎵 Source {i + 1} — `{sig.name}`")

                meta_cols = st.columns(4)
                meta_cols[0].metric("Sample Rate", format_hz(sig.sample_rate))
                meta_cols[1].metric("Duration (used)", format_duration(sig.duration))
                meta_cols[2].metric("Original Duration", format_duration(raw.original_duration))
                meta_cols[3].metric("Samples", f"{sig.n_samples:,}")

                stats = compute_amplitude_stats(sig.data)
                stat_cols = st.columns(5)
                stat_cols[0].metric("Min", f"{stats.minimum:.3f}")
                stat_cols[1].metric("Max", f"{stats.maximum:.3f}")
                stat_cols[2].metric("Mean", f"{stats.mean:.4f}")
                stat_cols[3].metric("Std Dev", f"{stats.std:.3f}")
                stat_cols[4].metric("RMS", f"{stats.rms:.3f}")

                st.plotly_chart(
                    plot_waveform(sig.data, sig.sample_rate, f"Waveform — {sig.name}", color),
                    use_container_width=True, theme=None,
                )

                st.audio(to_wav_bytes(sig.data, sig.sample_rate), format="audio/wav")


def render_mix_signals() -> None:
    st.title("🎛️ Mix Signals")
    signals = st.session_state.get("signals")
    if not signals:
        st.warning("⚠️ Please upload 2–3 audio files on the **Upload Audio** page first.")
        return

    st.write(
        "A random **mixing matrix** simulates several microphones placed "
        "around a room, each picking up a different linear combination of "
        "every original source at once. This is the classic "
        "*instantaneous linear mixing* model used in blind source "
        "separation:"
    )
    st.latex(r"X = A \cdot S")
    st.write(
        "where `S` are the original sources, `A` is the mixing matrix, and "
        "`X` is what each microphone actually records."
    )

    n = len(signals)
    sr = signals[0].sample_rate
    mic_names = [f"Mic {i + 1}" for i in range(n)]
    source_col_names = [f"Source {i + 1}" for i in range(n)]

    if "mixing_matrix" not in st.session_state or st.session_state.get("mix_n_sources") != n:
        st.session_state["mixing_matrix"] = generate_mixing_matrix(n, np.random.default_rng())
        st.session_state["mix_n_sources"] = n

    if st.button("🎲 Regenerate Random Mixing Matrix"):
        st.session_state["mixing_matrix"] = generate_mixing_matrix(n, np.random.default_rng())

    mixing_matrix = st.session_state["mixing_matrix"]

    st.subheader("Mixing Matrix (A)")
    st.caption("Rows = microphones, columns = original sources.")
    st.plotly_chart(
        plot_matrix_heatmap(mixing_matrix, "Mixing Matrix A", mic_names, source_col_names),
        use_container_width=True, theme=None,
    )
    with st.expander("🔢 View raw matrix values"):
        st.dataframe(
            pd.DataFrame(mixing_matrix, index=mic_names, columns=source_col_names),
            use_container_width=True,
        )

    try:
        sources_stack = stack_sources(signals)
        mixed_stack = mix_signals(sources_stack, mixing_matrix)
    except MixingError as exc:
        st.error(f"❌ {exc}")
        return

    playback_mixed = normalize_rows_peak(mixed_stack)

    # Store the raw (un-normalized) mix for downstream ICA milestones; the
    # normalized copy below is only for waveform display / audio playback.
    st.session_state["mixed_signals"] = mixed_stack
    st.session_state["sample_rate"] = sr

    st.divider()
    st.subheader("Original → Mixed")

    wave_tab, spec_tab = st.tabs(["🌊 Waveforms", "🔥 Spectrograms"])

    with wave_tab:
        col_orig, col_mixed = st.columns(2)
        with col_orig:
            st.markdown("##### 🎤 Original Sources")
            for i, sig in enumerate(signals):
                st.plotly_chart(
                    plot_waveform(
                        sig.data, sr, f"Source {i + 1} — {sig.name}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                    ),
                    use_container_width=True, theme=None,
                )
                st.audio(to_wav_bytes(sig.data, sr))
        with col_mixed:
            st.markdown("##### 🎙️ Mixed (Microphone) Signals")
            for i in range(n):
                st.plotly_chart(
                    plot_waveform(playback_mixed[i], sr, f"{mic_names[i]} (mixed)", "#fbbf24"),
                    use_container_width=True, theme=None,
                )
                st.audio(to_wav_bytes(playback_mixed[i], sr))

    with spec_tab:
        col_orig, col_mixed = st.columns(2)
        with col_orig:
            st.markdown("##### 🎤 Original Sources")
            for i, sig in enumerate(signals):
                st.plotly_chart(
                    plot_spectrogram(sig.data, sr, f"Source {i + 1} — {sig.name}"),
                    use_container_width=True, theme=None,
                )
        with col_mixed:
            st.markdown("##### 🎙️ Mixed (Microphone) Signals")
            for i in range(n):
                st.plotly_chart(
                    plot_spectrogram(playback_mixed[i], sr, f"{mic_names[i]} (mixed)"),
                    use_container_width=True, theme=None,
                )


def render_whitening() -> None:
    st.title("⚪ Whitening")
    mixed = st.session_state.get("mixed_signals")
    if mixed is None:
        st.warning("⚠️ Please mix your signals on the **Mix Signals** page first.")
        return

    sr = st.session_state.get("sample_rate")
    n = mixed.shape[0]
    mic_names = [f"Mic {i + 1}" for i in range(n)]
    pc_names = [f"PC{i + 1}" for i in range(n)]

    st.write(
        "Before FastICA can search for independent components, the mixed "
        "signals are **whitened**: transformed to have zero mean, unit "
        "variance, and — crucially — **no correlation** between channels. "
        "This turns the hard problem *'find any invertible un-mixing "
        "matrix'* into the easier problem *'find the right rotation'*, "
        "which is exactly what FastICA (Milestone 4) solves."
    )

    try:
        result = whiten(mixed)
    except WhiteningError as exc:
        st.error(f"❌ {exc}")
        return

    st.session_state["whitening_result"] = result

    st.subheader("Step 1 — Mean Subtraction")
    with st.expander("ℹ️ What does this do?"):
        st.write(
            "Each microphone signal is shifted so its **average value "
            "becomes zero**. This removes any DC offset so the covariance "
            "matrix in the next step reflects only how the signals *vary* "
            "together, not their absolute levels."
        )
    st.dataframe(
        pd.DataFrame(
            {
                "Mic": mic_names,
                "Mean Before": mixed.mean(axis=1),
                "Mean After (≈0)": result.centered.mean(axis=1),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Step 2 — Covariance Matrix")
    with st.expander("ℹ️ What does this do?"):
        st.write(
            "The covariance matrix captures how much each microphone "
            "signal varies **together with** every other one. Large "
            "off-diagonal values mean the mixed channels are strongly "
            "correlated — exactly what mixing introduces, and exactly "
            "what whitening removes."
        )
    st.plotly_chart(
        plot_matrix_heatmap(result.covariance, "Covariance Matrix — Before Whitening", mic_names, mic_names),
        use_container_width=True, theme=None,
    )

    st.subheader("Step 3 — Eigen Decomposition")
    with st.expander("ℹ️ What does this do?"):
        st.write(
            "Any covariance matrix can be broken down into **eigenvectors** "
            "(directions of variance) and **eigenvalues** (how much "
            "variance lies along each direction). This is the mathematical "
            "core of PCA — it tells us exactly how to de-correlate and "
            "rescale the signals in the next step."
        )
    eig_col1, eig_col2 = st.columns(2)
    with eig_col1:
        st.markdown("**Eigenvalues** (variance along each principal direction)")
        st.dataframe(
            pd.DataFrame({"Component": pc_names, "Eigenvalue": result.eigenvalues}),
            use_container_width=True,
            hide_index=True,
        )
    with eig_col2:
        st.markdown("**Eigenvectors** (columns = principal directions)")
        st.dataframe(
            pd.DataFrame(result.eigenvectors, columns=pc_names, index=mic_names),
            use_container_width=True,
        )

    st.subheader("Step 4 — Whitening Transform")
    with st.expander("ℹ️ What does this do?"):
        st.write(
            "Each principal direction is rescaled by `1/√eigenvalue` so "
            "every direction ends up with **exactly unit variance**. The "
            "result, `Z = W · X_centered`, is the whitened signal: flat, "
            "uncorrelated, unit-variance — the ideal starting point for "
            "FastICA."
        )
    st.plotly_chart(
        plot_matrix_heatmap(
            result.whitened_covariance, "Covariance Matrix — After Whitening (≈ Identity)", mic_names, mic_names
        ),
        use_container_width=True, theme=None,
    )

    st.divider()
    st.subheader("Whitened Waveforms")
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.plotly_chart(
                plot_waveform(result.whitened[i], sr, f"Whitened {mic_names[i]}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                use_container_width=True, theme=None,
            )

    if n in (2, 3):
        st.divider()
        st.subheader("Visualizing Decorrelation")
        st.write(
            "Each point below is one instant in time, plotted using the "
            "amplitude of each channel as coordinates. **Mixed** signals "
            "form a slanted, stretched cloud (channels moving together). "
            "**Whitened** signals form a round, symmetric cloud — proof "
            "that the correlation between channels has been removed."
        )
        scatter_col1, scatter_col2 = st.columns(2)
        with scatter_col1:
            st.plotly_chart(
                plot_scatter(mixed, "Mixed Signals (correlated)", "#fbbf24"),
                use_container_width=True, theme=None,
            )
        with scatter_col2:
            st.plotly_chart(
                plot_scatter(result.whitened, "Whitened Signals (decorrelated)", DEFAULT_COLORS[0]),
                use_container_width=True, theme=None,
            )


NONLINEARITY_INFO = {
    "logcosh": {
        "label": "log-cosh (general purpose, recommended default)",
        "formula": r"g(u) = \tanh(u), \qquad g'(u) = 1 - \tanh^2(u)",
    },
    "exp": {
        "label": "exp (robust to outliers)",
        "formula": r"g(u) = u\,e^{-u^2/2}, \qquad g'(u) = (1-u^2)\,e^{-u^2/2}",
    },
    "cube": {
        "label": "cube (best for sub-Gaussian sources)",
        "formula": r"g(u) = u^3, \qquad g'(u) = 3u^2",
    },
}


def render_fastica() -> None:
    st.title("🧠 FastICA (From Scratch)")
    whitening_result = st.session_state.get("whitening_result")
    if whitening_result is None:
        st.warning("⚠️ Please complete the **Whitening** page first.")
        return

    sr = st.session_state.get("sample_rate")
    Z = whitening_result.whitened
    n = Z.shape[0]
    mic_names = [f"Mic {i + 1}" for i in range(n)]

    st.write(
        "FastICA searches for an **un-mixing matrix `W`** that makes the "
        "recovered signals `S_hat = W · Z` as *non-Gaussian* as possible. "
        "By the Central Limit Theorem, mixing independent sources makes "
        "them **more** Gaussian — so maximizing non-Gaussianity is "
        "equivalent to un-mixing them."
    )

    st.subheader("Configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        nonlinearity = st.selectbox(
            "Nonlinearity g(u)",
            options=list(NONLINEARITIES.keys()),
            format_func=lambda k: NONLINEARITY_INFO[k]["label"],
            index=0,
        )
    with col2:
        max_iter = st.number_input("Max iterations", min_value=10, max_value=2000, value=200, step=10)
    with col3:
        tol = st.select_slider(
            "Tolerance",
            options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            value=1e-6,
            format_func=lambda x: f"{x:g}",
        )

    st.latex(NONLINEARITY_INFO[nonlinearity]["formula"])

    with st.expander("ℹ️ How FastICA works, step by step"):
        st.markdown(
            "1. **Weight initialization** — start from a random matrix, "
            "immediately orthogonalized so its rows are orthonormal.\n"
            "2. **Fixed-point update** — for the chosen nonlinearity `g`, "
            "push each row of `W` toward "
            "`E[Z · g(W·Z)] - E[g'(W·Z)] · W`, the update that (locally) "
            "maximizes non-Gaussianity.\n"
            "3. **Orthogonalization** — symmetrically re-orthogonalize all "
            "rows of `W` together (`(W·Wᵀ)^(-1/2) · W`) so every recovered "
            "component stays uncorrelated with the others.\n"
            "4. **Convergence detection** — stop once the direction of "
            "every row stops changing (within `tol`), or after `max_iter` "
            "iterations."
        )

    if st.button("🚀 Run FastICA", type="primary"):
        model = FastICA(nonlinearity=nonlinearity, max_iter=int(max_iter), tol=tol, random_state=0)
        st.session_state["fastica_result"] = model.fit_transform(Z)

    result = st.session_state.get("fastica_result")
    if result is not None and result.unmixing_matrix.shape[0] != n:
        # Stale result from a previous upload with a different source count.
        result = None

    if result is None:
        st.info("👆 Configure the settings above and click **Run FastICA** to separate the sources.")
        return

    st.divider()
    st.subheader("Convergence")
    status_cols = st.columns(4)
    status_cols[0].metric("Converged", "✅ Yes" if result.converged else "❌ No")
    status_cols[1].metric("Iterations", result.n_iterations)
    status_cols[2].metric("Execution Time", f"{result.execution_time_sec * 1000:.1f} ms")
    status_cols[3].metric("Nonlinearity", result.nonlinearity)

    st.plotly_chart(plot_convergence(result.convergence_history, tol), use_container_width=True, theme=None)

    if not result.converged:
        st.warning(
            "⚠️ FastICA did not converge within the iteration limit. Try "
            "increasing **Max iterations**, relaxing **Tolerance**, or "
            "switching the **Nonlinearity**."
        )

    st.divider()
    st.subheader("Un-mixing Matrix (W)")
    st.caption("Applied to the *whitened* signals: `S_hat = W · Z`.")
    st.plotly_chart(
        plot_matrix_heatmap(result.unmixing_matrix, "Un-mixing Matrix W", mic_names, mic_names),
        use_container_width=True, theme=None,
    )

    st.divider()
    st.subheader("Recovered Independent Components")
    st.info(
        "ℹ️ ICA recovers sources only up to **scale, sign, and order** — "
        "the recovered components may be louder/quieter, phase-inverted, "
        "or in a different order than the originals. That's expected, and "
        "is resolved by correlation-based matching on the **Comparison** "
        "and **Metrics** pages."
    )
    playback_sources = normalize_rows_peak(result.sources)
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.plotly_chart(
                plot_waveform(
                    playback_sources[i], sr, f"Recovered Component {i + 1}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
                ),
                use_container_width=True, theme=None,
            )
            st.audio(to_wav_bytes(playback_sources[i], sr))


def render_comparison() -> None:
    st.title("🔎 Implementation Parity Check: Scratch FastICA vs. scikit-learn")

    signals = st.session_state.get("signals")
    whitening_result = st.session_state.get("whitening_result")
    my_result = st.session_state.get("fastica_result")

    if not signals or whitening_result is None:
        st.warning("⚠️ Please complete **Upload Audio**, **Mix Signals**, and **Whitening** first.")
        return
    if my_result is None or my_result.unmixing_matrix.shape[0] != len(signals):
        st.warning("⚠️ Please run **FastICA** first — this page reuses that run's settings for a fair comparison.")
        return

    st.write(
        "This page runs **scikit-learn's** `FastICA` on the exact same "
        "whitened data, using the same nonlinearity, tolerance, and "
        "iteration cap as your from-scratch run on the **FastICA** page."
    )
    st.info(
        "ℹ️ **This is a parity check, not validation.** scikit-learn's "
        "`FastICA` implements the *same* symmetric fixed-point algorithm "
        "from the *same* Hyvärinen & Oja papers as the from-scratch "
        "version on this page — it isn't an independent ground truth. "
        "Close agreement mainly shows the from-scratch math was "
        "transcribed correctly; it says nothing about whether FastICA "
        "itself is a good fit for the input — both implementations can "
        "agree perfectly and still be wrong, e.g. on convolutive mixtures "
        "or Gaussian sources (see the **Theory** page)."
    )

    Z = whitening_result.whitened
    sr = st.session_state.get("sample_rate")
    n = Z.shape[0]
    reference = stack_sources(signals)

    with st.spinner("Running scikit-learn FastICA..."):
        sk_result = run_sklearn_fastica(
            Z,
            nonlinearity=my_result.nonlinearity,
            max_iter=my_result.max_iter,
            tol=my_result.tol,
            random_state=0,
        )
    st.session_state["sklearn_fastica_result"] = sk_result

    st.subheader("Iterations to Convergence")
    st.caption(
        "Execution time is intentionally not headlined here: timing two "
        "short NumPy loops on a shared machine, with no warm-up, "
        "averaging, or isolation from other processes, is dominated by "
        "measurement noise — it is not a meaningful performance "
        "comparison. Iteration count is more informative since it "
        "reflects the actual convergence path of each solver."
    )
    time_cols = st.columns(2)
    time_cols[0].metric("My FastICA (from scratch)", f"{my_result.n_iterations} iterations")
    time_cols[1].metric("scikit-learn FastICA", f"{sk_result.n_iterations} iterations")
    with st.expander("⏱️ Raw execution time (noisy, shown for completeness)"):
        raw_time_cols = st.columns(2)
        raw_time_cols[0].metric("My FastICA (from scratch)", f"{my_result.execution_time_sec * 1000:.2f} ms")
        raw_time_cols[1].metric("scikit-learn FastICA", f"{sk_result.execution_time_sec * 1000:.2f} ms")

    st.subheader("Recovery Quality vs. Original Sources")
    st.caption("Both implementations are matched to the true sources by best-correlation permutation before scoring.")
    st.write(
        "**SI-SDR** (scale-invariant signal-to-distortion ratio) is the "
        "headline number: higher is better, and it's invariant to ICA's "
        "scale/sign ambiguity by construction. **SIR** and **SAR** (from "
        "`mir_eval`'s BSS_Eval) break a low SDR down into *why* it's low: "
        "a low **SIR** means another source is leaking into this "
        "component (a permutation/separation failure); a low **SAR** "
        "means the algorithm introduced its own distortion/artifacts even "
        "though the right source dominates. A single correlation number "
        "can't tell those two failure modes apart — that's the "
        "point of reporting them separately instead of one score."
    )
    my_sep = evaluate_separation(reference, my_result.sources)
    sk_sep = evaluate_separation(reference, sk_result.sources)

    summary_df = pd.DataFrame(
        {
            "Metric": ["Mean SI-SDR (dB)", "Mean SDR (dB)", "Mean SIR (dB)", "Mean SAR (dB)"],
            "My FastICA": [
                f"{my_sep.mean_si_sdr_db:.2f}",
                f"{my_sep.mean_sdr_db:.2f}",
                f"{my_sep.mean_sir_db:.2f}",
                f"{my_sep.mean_sar_db:.2f}",
            ],
            "scikit-learn": [
                f"{sk_sep.mean_si_sdr_db:.2f}",
                f"{sk_sep.mean_sdr_db:.2f}",
                f"{sk_sep.mean_sir_db:.2f}",
                f"{sk_sep.mean_sar_db:.2f}",
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with st.expander("🔬 Per-component breakdown (SI-SDR / SDR / SIR / SAR)"):
        per_component_sep_df = pd.DataFrame(
            {
                "Source": [f"Source {i + 1}" for i in range(n)],
                "My SI-SDR (dB)": my_sep.si_sdr_db,
                "My SDR (dB)": my_sep.sdr_db,
                "My SIR (dB)": my_sep.sir_db,
                "My SAR (dB)": my_sep.sar_db,
                "sklearn SI-SDR (dB)": sk_sep.si_sdr_db,
                "sklearn SDR (dB)": sk_sep.sdr_db,
                "sklearn SIR (dB)": sk_sep.sir_db,
                "sklearn SAR (dB)": sk_sep.sar_db,
            }
        )
        st.dataframe(per_component_sep_df, use_container_width=True, hide_index=True)

    my_eval = evaluate(reference, my_result.sources)
    sk_eval = evaluate(reference, sk_result.sources)

    with st.expander("📎 Legacy metrics (correlation / SNR / MSE)"):
        st.caption(
            "Kept for continuity with earlier milestones. SI-SDR/SDR/SIR/SAR "
            "above are the field-standard numbers and should be preferred."
        )
        legacy_summary_df = pd.DataFrame(
            {
                "Metric": ["Mean |Correlation|", "Mean SNR (dB)", "Mean MSE", "Recovery %"],
                "My FastICA": [
                    f"{my_eval.mean_abs_correlation:.4f}",
                    f"{my_eval.mean_snr_db:.2f}",
                    f"{my_eval.mean_mse:.6f}",
                    f"{my_eval.recovery_percentage:.2f}%",
                ],
                "scikit-learn": [
                    f"{sk_eval.mean_abs_correlation:.4f}",
                    f"{sk_eval.mean_snr_db:.2f}",
                    f"{sk_eval.mean_mse:.6f}",
                    f"{sk_eval.recovery_percentage:.2f}%",
                ],
            }
        )
        st.dataframe(legacy_summary_df, use_container_width=True, hide_index=True)

        per_component_df = pd.DataFrame(
            {
                "Source": [f"Source {i + 1}" for i in range(n)],
                "My |Correlation|": np.abs(my_eval.correlations),
                "My SNR (dB)": my_eval.snr_db,
                "My MSE": my_eval.mse,
                "sklearn |Correlation|": np.abs(sk_eval.correlations),
                "sklearn SNR (dB)": sk_eval.snr_db,
                "sklearn MSE": sk_eval.mse,
            }
        )
        st.dataframe(per_component_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Do Both Implementations Converge to the Same Solution?")
    st.write(
        "Beyond scoring each implementation against the ground truth "
        "separately, we can directly cross-correlate **my recovered "
        "components** against **scikit-learn's recovered components** "
        "(after best-matching order, sign, and scale). A value close to "
        "1.0 means the two solvers landed on the same fixed point — "
        "evidence the from-scratch fixed-point update, orthogonalization, "
        "and convergence check are implemented correctly, not evidence "
        "that fixed point is a good separation of the sources."
    )
    cross_eval = evaluate(my_result.sources, sk_result.sources)
    my_names = [f"My {i + 1}" for i in range(n)]
    sk_names = [f"sklearn {i + 1}" for i in range(n)]
    st.plotly_chart(
        plot_matrix_heatmap(
            cross_eval.correlation_matrix,
            "Cross-correlation: My Components vs. scikit-learn Components",
            my_names,
            sk_names,
            colorscale="Viridis",
            zmid=None,
        ),
        use_container_width=True, theme=None,
    )

    agreement = cross_eval.mean_abs_correlation
    if agreement > 0.95:
        st.success(f"✅ Both implementations converge to statistically equivalent solutions (mean cross-correlation = {agreement:.4f}) — consistent with a correct from-scratch transcription.")
    elif agreement > 0.8:
        st.info(f"ℹ️ Both implementations converge to similar solutions, with some deviation (mean cross-correlation = {agreement:.4f}).")
    else:
        st.warning(
            f"⚠️ The two solvers diverge noticeably (mean cross-correlation = {agreement:.4f}). "
            "Try more iterations or a tighter tolerance on the FastICA page."
        )

    st.divider()
    st.subheader("Recovered Waveforms Side-by-Side")
    my_playback = normalize_rows_peak(my_result.sources)
    sk_playback = normalize_rows_peak(sk_result.sources)
    for i in range(n):
        st.markdown(f"##### Component {i + 1}")
        col_mine, col_sklearn = st.columns(2)
        with col_mine:
            st.plotly_chart(
                plot_waveform(my_playback[i], sr, f"My FastICA — Component {i + 1}", DEFAULT_COLORS[0]),
                use_container_width=True, theme=None,
            )
            st.audio(to_wav_bytes(my_playback[i], sr))
        with col_sklearn:
            st.plotly_chart(
                plot_waveform(sk_playback[i], sr, f"scikit-learn — Component {i + 1}", DEFAULT_COLORS[1]),
                use_container_width=True, theme=None,
            )
            st.audio(to_wav_bytes(sk_playback[i], sr))


def _render_waveform_row(matrix: np.ndarray, sr: int, title_prefix: str) -> None:
    n = matrix.shape[0]
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.plotly_chart(
                plot_waveform(matrix[i], sr, f"{title_prefix} {i + 1}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                use_container_width=True, theme=None,
            )


def _render_spectrogram_row(matrix: np.ndarray, sr: int, title_prefix: str) -> None:
    n = matrix.shape[0]
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.plotly_chart(plot_spectrogram(matrix[i], sr, f"{title_prefix} {i + 1}"), use_container_width=True, theme=None)


def render_visualizations() -> None:
    st.title("📈 Interactive Visualizations")
    st.write(
        "A single gallery of every interactive plot generated across the "
        "pipeline, organized by chart type rather than by pipeline stage. "
        "Every chart here is a live Plotly figure on a dark background: "
        "**scroll/drag to zoom, hover for exact values, drag to pan.**"
    )

    signals = st.session_state.get("signals")
    if not signals:
        st.warning("⚠️ Please upload audio on the **Upload Audio** page first to populate this gallery.")
        return

    mixed = st.session_state.get("mixed_signals")
    mixing_matrix = st.session_state.get("mixing_matrix")
    whitening_result = st.session_state.get("whitening_result")
    my_result = st.session_state.get("fastica_result")
    sk_result = st.session_state.get("sklearn_fastica_result")
    sr = st.session_state.get("sample_rate")

    n = len(signals)
    mic_names = [f"Mic {i + 1}" for i in range(n)]
    reference = stack_sources(signals)

    tab_wave, tab_spec, tab_cov, tab_matrix, tab_corr = st.tabs(
        ["🌊 Waveforms", "🔥 Spectrograms", "🟩 Covariance", "🔀 Matrices", "🔗 Correlation"]
    )

    with tab_wave:
        st.subheader("🎤 Original Sources")
        _render_waveform_row(reference, sr, "Source")

        st.subheader("🎙️ Mixed Signals")
        if mixed is not None:
            _render_waveform_row(normalize_rows_peak(mixed), sr, "Mic")
        else:
            st.info("Visit **Mix Signals** to generate the mixed signals.")

        st.subheader("⚪ Whitened Signals")
        if whitening_result is not None:
            _render_waveform_row(whitening_result.whitened, sr, "Whitened")
        else:
            st.info("Visit **Whitening** to generate the whitened signals.")

        st.subheader("🧠 Recovered — My FastICA")
        if my_result is not None:
            _render_waveform_row(normalize_rows_peak(my_result.sources), sr, "Recovered")
        else:
            st.info("Visit **FastICA** to recover the independent components.")

        st.subheader("📚 Recovered — scikit-learn")
        if sk_result is not None:
            _render_waveform_row(normalize_rows_peak(sk_result.sources), sr, "sklearn")
        else:
            st.info("Visit **Comparison** to run scikit-learn's FastICA.")

    with tab_spec:
        st.subheader("🎤 Original Sources")
        _render_spectrogram_row(reference, sr, "Source")

        st.subheader("🎙️ Mixed Signals")
        if mixed is not None:
            _render_spectrogram_row(normalize_rows_peak(mixed), sr, "Mic")
        else:
            st.info("Visit **Mix Signals** to generate the mixed signals.")

        st.subheader("🧠 Recovered — My FastICA")
        if my_result is not None:
            _render_spectrogram_row(normalize_rows_peak(my_result.sources), sr, "Recovered")
        else:
            st.info("Visit **FastICA** to recover the independent components.")

    with tab_cov:
        if whitening_result is not None:
            cov_col1, cov_col2 = st.columns(2)
            with cov_col1:
                st.plotly_chart(
                    plot_matrix_heatmap(whitening_result.covariance, "Covariance — Before Whitening", mic_names, mic_names),
                    use_container_width=True, theme=None,
                )
            with cov_col2:
                st.plotly_chart(
                    plot_matrix_heatmap(
                        whitening_result.whitened_covariance, "Covariance — After Whitening (≈ Identity)", mic_names, mic_names
                    ),
                    use_container_width=True, theme=None,
                )
        else:
            st.info("Visit **Whitening** to compute the covariance matrices.")

    with tab_matrix:
        matrix_col1, matrix_col2, matrix_col3 = st.columns(3)
        with matrix_col1:
            if mixing_matrix is not None:
                st.plotly_chart(
                    plot_matrix_heatmap(mixing_matrix, "Mixing Matrix (A)", mic_names, [f"Source {i + 1}" for i in range(n)]),
                    use_container_width=True, theme=None,
                )
            else:
                st.info("Visit **Mix Signals** to generate the mixing matrix.")
        with matrix_col2:
            if my_result is not None:
                st.plotly_chart(
                    plot_matrix_heatmap(my_result.unmixing_matrix, "Recovered Matrix (My W)", mic_names, mic_names),
                    use_container_width=True, theme=None,
                )
            else:
                st.info("Visit **FastICA** to compute the un-mixing matrix.")
        with matrix_col3:
            if sk_result is not None:
                st.plotly_chart(
                    plot_matrix_heatmap(sk_result.unmixing_matrix, "Recovered Matrix (sklearn W)", mic_names, mic_names),
                    use_container_width=True, theme=None,
                )
            else:
                st.info("Visit **Comparison** to compute scikit-learn's un-mixing matrix.")

    with tab_corr:
        if my_result is not None:
            st.markdown("**My FastICA vs. Original Sources**")
            my_eval = evaluate(reference, my_result.sources)
            st.plotly_chart(
                plot_matrix_heatmap(
                    my_eval.correlation_matrix,
                    "|Correlation|: True Sources vs. My Recovered Components",
                    [f"Source {i + 1}" for i in range(n)],
                    [f"Recovered {i + 1}" for i in range(n)],
                    colorscale="Viridis",
                    zmid=None,
                ),
                use_container_width=True, theme=None,
            )
        else:
            st.info("Visit **FastICA** to compute this correlation heatmap.")

        if sk_result is not None:
            st.markdown("**scikit-learn vs. Original Sources**")
            sk_eval = evaluate(reference, sk_result.sources)
            st.plotly_chart(
                plot_matrix_heatmap(
                    sk_eval.correlation_matrix,
                    "|Correlation|: True Sources vs. scikit-learn Components",
                    [f"Source {i + 1}" for i in range(n)],
                    [f"sklearn {i + 1}" for i in range(n)],
                    colorscale="Viridis",
                    zmid=None,
                ),
                use_container_width=True, theme=None,
            )

            st.markdown("**My FastICA vs. scikit-learn**")
            cross_eval = evaluate(my_result.sources, sk_result.sources)
            st.plotly_chart(
                plot_matrix_heatmap(
                    cross_eval.correlation_matrix,
                    "|Correlation|: My Components vs. scikit-learn Components",
                    [f"My {i + 1}" for i in range(n)],
                    [f"sklearn {i + 1}" for i in range(n)],
                    colorscale="Viridis",
                    zmid=None,
                ),
                use_container_width=True, theme=None,
            )
        else:
            st.info("Visit **Comparison** to compute the scikit-learn correlation heatmaps.")


def render_metrics() -> None:
    st.title("📊 Evaluation Dashboard")

    signals = st.session_state.get("signals")
    my_result = st.session_state.get("fastica_result")
    sk_result = st.session_state.get("sklearn_fastica_result")

    if not signals or my_result is None:
        st.warning("⚠️ Please run **FastICA** first to populate this dashboard.")
        return
    if my_result.unmixing_matrix.shape[0] != len(signals):
        st.warning("⚠️ The stored FastICA result doesn't match the current audio — please re-run **FastICA**.")
        return

    reference = stack_sources(signals)
    n = len(signals)
    component_names = [f"Source {i + 1}" for i in range(n)]

    my_sep = evaluate_separation(reference, my_result.sources)
    sk_sep = evaluate_separation(reference, sk_result.sources) if sk_result is not None else None
    my_eval = evaluate(reference, my_result.sources)
    sk_eval = evaluate(reference, sk_result.sources) if sk_result is not None else None

    st.caption(
        "ℹ️ **SI-SDR** is the headline recovery-quality number (higher is "
        "better, invariant to ICA's scale/sign ambiguity). **SIR** "
        "diagnoses leakage from other sources; **SAR** diagnoses "
        "distortion the algorithm itself introduced — see the "
        "**Comparison** page for why both matter more than one "
        "correlation number."
    )

    st.subheader("🧠 My FastICA — Summary")
    cols_row1 = st.columns(4)
    cols_row1[0].metric("Mean SI-SDR (dB)", f"{my_sep.mean_si_sdr_db:.2f}")
    cols_row1[1].metric("Mean SDR (dB)", f"{my_sep.mean_sdr_db:.2f}")
    cols_row1[2].metric("Mean SIR (dB)", f"{my_sep.mean_sir_db:.2f}")
    cols_row1[3].metric("Mean SAR (dB)", f"{my_sep.mean_sar_db:.2f}")
    cols_row2 = st.columns(3)
    cols_row2[0].metric("Execution Time", f"{my_result.execution_time_sec * 1000:.2f} ms")
    cols_row2[1].metric("Iterations", my_result.n_iterations, "✅ Converged" if my_result.converged else "❌ Not converged")
    cols_row2[2].metric("Mean |Correlation| (legacy)", f"{my_eval.mean_abs_correlation:.4f}")

    st.markdown("**Per-Component Metrics**")
    my_component_df = pd.DataFrame(
        {
            "Component": component_names,
            "SI-SDR (dB)": my_sep.si_sdr_db,
            "SDR (dB)": my_sep.sdr_db,
            "SIR (dB)": my_sep.sir_db,
            "SAR (dB)": my_sep.sar_db,
        }
    )
    st.dataframe(my_component_df, use_container_width=True, hide_index=True)
    with st.expander("📎 Legacy per-component metrics (correlation / SNR / MSE)"):
        my_legacy_df = pd.DataFrame(
            {
                "Component": component_names,
                "Correlation": my_eval.correlations,
                "SNR (dB)": my_eval.snr_db,
                "MSE": my_eval.mse,
            }
        )
        st.dataframe(my_legacy_df, use_container_width=True, hide_index=True)

    st.markdown("**Convergence**")
    st.plotly_chart(
        plot_convergence(my_result.convergence_history, my_result.tol, "My FastICA Convergence"),
        use_container_width=True, theme=None,
    )

    st.divider()

    if sk_sep is not None:
        st.subheader("📚 scikit-learn — Summary")
        sk_cols_row1 = st.columns(4)
        sk_cols_row1[0].metric("Mean SI-SDR (dB)", f"{sk_sep.mean_si_sdr_db:.2f}")
        sk_cols_row1[1].metric("Mean SDR (dB)", f"{sk_sep.mean_sdr_db:.2f}")
        sk_cols_row1[2].metric("Mean SIR (dB)", f"{sk_sep.mean_sir_db:.2f}")
        sk_cols_row1[3].metric("Mean SAR (dB)", f"{sk_sep.mean_sar_db:.2f}")
        sk_cols_row2 = st.columns(3)
        sk_cols_row2[0].metric("Execution Time", f"{sk_result.execution_time_sec * 1000:.2f} ms")
        sk_cols_row2[1].metric(
            "Iterations", sk_result.n_iterations, "✅ Converged" if sk_result.converged else "❌ Not converged"
        )
        sk_cols_row2[2].metric("Mean |Correlation| (legacy)", f"{sk_eval.mean_abs_correlation:.4f}")

        st.markdown("**Per-Component Metrics**")
        sk_component_df = pd.DataFrame(
            {
                "Component": component_names,
                "SI-SDR (dB)": sk_sep.si_sdr_db,
                "SDR (dB)": sk_sep.sdr_db,
                "SIR (dB)": sk_sep.sir_db,
                "SAR (dB)": sk_sep.sar_db,
            }
        )
        st.dataframe(sk_component_df, use_container_width=True, hide_index=True)
        with st.expander("📎 Legacy per-component metrics (correlation / SNR / MSE)"):
            sk_legacy_df = pd.DataFrame(
                {
                    "Component": component_names,
                    "Correlation": sk_eval.correlations,
                    "SNR (dB)": sk_eval.snr_db,
                    "MSE": sk_eval.mse,
                }
            )
            st.dataframe(sk_legacy_df, use_container_width=True, hide_index=True)
        st.divider()
    else:
        st.info("Visit **Comparison** to also include scikit-learn's metrics in this dashboard and the CSV report.")
        st.divider()

    st.subheader("⬇️ Downloadable CSV Report")
    st.write(
        "One row per component per implementation, plus an aggregate "
        "'MEAN' row — every row also repeats the execution time, "
        "iteration count, and convergence status so it's readable "
        "standalone in Excel/Sheets. Includes both the headline "
        "SI-SDR/SDR/SIR/SAR columns and the legacy correlation/SNR/MSE "
        "columns."
    )
    report_df = build_metrics_report(
        component_names=component_names,
        my_eval=my_eval,
        my_execution_time_sec=my_result.execution_time_sec,
        my_n_iterations=my_result.n_iterations,
        my_converged=my_result.converged,
        sk_eval=sk_eval,
        sk_execution_time_sec=sk_result.execution_time_sec if sk_result is not None else None,
        sk_n_iterations=sk_result.n_iterations if sk_result is not None else None,
        sk_converged=sk_result.converged if sk_result is not None else None,
        my_sep=my_sep,
        sk_sep=sk_sep,
    )
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Metrics CSV",
        data=report_df.to_csv(index=False).encode("utf-8"),
        file_name="bss_ica_metrics_report.csv",
        mime="text/csv",
        type="primary",
    )


def _place_convolutive_geometry(
    room_dim: tuple[float, float, float], n: int, mic_spacing: float
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """
    Deterministic source/mic layout for the Convolutive Mixing page: sources
    spread along one wall, a linear mic array centered in the room. Keeps
    the UI to two tunable knobs (RT60, mic spacing) instead of 3*n position
    sliders.
    """
    room_x, room_y, room_z = room_dim
    z = min(1.5, room_z / 2)
    source_positions = [(1.0, room_y * (i + 1) / (n + 1), z) for i in range(n)]
    center_y = room_y / 2
    mic_positions = [
        (room_x / 2, center_y + (i - (n - 1) / 2) * mic_spacing, z) for i in range(n)
    ]
    return source_positions, mic_positions


@st.cache_data(show_spinner=False)
def _run_instantaneous_fastica(mixture: np.ndarray):
    """Whiten + run default from-scratch FastICA -- the exact pipeline used elsewhere in the app."""
    wh = whiten(mixture)
    result = FastICA(nonlinearity="logcosh", max_iter=200, tol=1e-6, random_state=0).fit_transform(wh.whitened)
    return result


@st.cache_data(show_spinner=False)
def _cached_convolutive_mixture(
    sources: np.ndarray,
    room_dim: tuple[float, float, float],
    source_positions: list[tuple[float, float, float]],
    mic_positions: list[tuple[float, float, float]],
    sample_rate: int,
    rt60: float,
):
    """Cached wrapper around `render_convolutive_mixture` -- RIR generation is the most
    expensive step on the Convolutive Mixing / Failure Gallery / Real Data pages, and every
    one of these arguments deterministically fixes the result, so repeat calls with the same
    room/geometry/RT60 (e.g. revisiting a page, or an unrelated widget triggering a rerun)
    return instantly instead of recomputing every image source and every RIR."""
    return render_convolutive_mixture(sources, room_dim, source_positions, mic_positions, sample_rate, rt60=rt60)


@st.cache_data(show_spinner=False)
def _cached_fd_ica(mixture: np.ndarray, n_fft: int, hop_length: int, random_state: int = 0):
    """Cached wrapper around `fd_ica` -- the per-bin complex FastICA + permutation alignment
    loop is the single slowest operation in the app; caching it avoids repeating that work
    for a mixture/config pair that's already been run."""
    return fd_ica(mixture, n_fft=n_fft, hop_length=hop_length, random_state=random_state)


def render_convolutive_mixing() -> None:
    st.title("🏠 Convolutive Mixing: Where Instantaneous ICA Breaks")

    signals = st.session_state.get("signals")
    if not signals:
        st.warning("⚠️ Please upload 2–3 audio files on the **Upload Audio** page first.")
        return

    st.write(
        "Everywhere else in this app, mixing is **instantaneous**: "
        "`X = A·S`. This page builds a physically realistic **convolutive** "
        "mixture instead — each microphone hears every source delayed and "
        "echoed by that source-mic pair's room impulse response (RIR), "
        "generated from scratch via the image-source method "
        "(Allen & Berkley, 1979): `x_mic(t) = Σ_source Σ_τ h(τ)·s_source(t−τ)`."
    )
    st.warning(
        "⚠️ The FastICA on this page is the *exact same instantaneous* "
        "solver used on the **FastICA** page — it is given a mixture its "
        "model doesn't describe, on purpose. Watch what happens to "
        "separation quality as RT60 (reverberation time) and mic spacing "
        "change."
    )

    n = len(signals)
    sr = signals[0].sample_rate
    reference = stack_sources(signals)

    st.subheader("Room & Array Configuration")
    room_cols = st.columns(3)
    room_x = room_cols[0].slider("Room length (m)", 3.0, 10.0, 6.0, 0.5)
    room_y = room_cols[1].slider("Room width (m)", 3.0, 10.0, 5.0, 0.5)
    room_z = room_cols[2].slider("Room height (m)", 2.5, 5.0, 3.0, 0.5)
    room_dim = (room_x, room_y, room_z)

    # Sabine's formula implies alpha = 0.161*V/(S*RT60); alpha <= 1 bounds
    # how short an RT60 this room size can physically produce.
    min_feasible_rt60 = 0.161 * room_volume(room_dim) / room_surface_area(room_dim)

    cfg_cols = st.columns(2)
    rt60 = cfg_cols[0].slider("RT60 (s)", 0.05, 0.8, 0.3, 0.05)
    mic_spacing = cfg_cols[1].slider("Mic spacing (m)", 0.02, 1.0, 0.2, 0.02)
    if rt60 < min_feasible_rt60:
        st.caption(
            f"⚠️ RT60={rt60:.2f}s isn't physically achievable at this room "
            f"size under Sabine's formula (needs ≥{min_feasible_rt60:.2f}s here) "
            "— Generate will error. Raise RT60 or shrink the room."
        )

    source_positions, mic_positions = _place_convolutive_geometry(room_dim, n, mic_spacing)
    with st.expander("📍 Source/mic layout used"):
        st.caption(
            "Sources spread along one wall (x=1m), 1.5m high. Mics form a "
            "linear array centered in the room, also 1.5m high, spaced by "
            "the slider above."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Point": [f"Source {i + 1}" for i in range(n)] + [f"Mic {i + 1}" for i in range(n)],
                    "x (m)": [p[0] for p in source_positions] + [p[0] for p in mic_positions],
                    "y (m)": [p[1] for p in source_positions] + [p[1] for p in mic_positions],
                    "z (m)": [p[2] for p in source_positions] + [p[2] for p in mic_positions],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("🔊 Generate Convolutive Mixture", type="primary"):
        try:
            with st.spinner("Generating room impulse responses and convolving..."):
                mixture, rirs = _cached_convolutive_mixture(
                    reference, room_dim, source_positions, mic_positions, sr, rt60
                )
        except RoomAcousticsError as exc:
            st.error(f"❌ {exc}")
            return
        st.session_state["convolutive_mixture"] = mixture
        st.session_state["convolutive_rirs"] = rirs
        st.session_state["convolutive_config"] = (room_dim, rt60, mic_spacing, n)

    mixture = st.session_state.get("convolutive_mixture")
    stored_config = st.session_state.get("convolutive_config")
    if mixture is None or stored_config is None or stored_config[3] != n:
        st.info("👆 Configure the room above and click **Generate Convolutive Mixture**.")
        return
    stale = stored_config != (room_dim, rt60, mic_spacing, n)
    if stale:
        st.caption("ℹ️ Showing the last *generated* mixture — sliders have changed since; click Generate again to update it.")

    rirs = st.session_state["convolutive_rirs"]
    mic_names = [f"Mic {i + 1}" for i in range(n)]

    st.divider()
    st.subheader("Room Impulse Response (Source 1 → Mic 1)")
    example_rir = rirs[0][0]
    st.plotly_chart(
        plot_waveform(example_rir.rir, sr, "RIR: Source 1 → Mic 1", DEFAULT_COLORS[0]),
        use_container_width=True, theme=None,
    )
    st.caption(
        f"Nominal RT60 (Sabine): {example_rir.rt60_nominal:.2f}s · "
        f"absorption: {example_rir.absorption:.3f} · "
        f"{example_rir.n_images:,} image sources summed up to reflection order {example_rir.max_order}"
        + (" · ⚠️ truncated before reaching the amplitude threshold (RT60 achieved may fall short of target)" if example_rir.truncated else "")
    )

    st.subheader("Convolutive Mixture — Waveforms")
    playback_mixed = normalize_rows_peak(mixture)
    cols = st.columns(n)
    for i in range(n):
        with cols[i]:
            st.plotly_chart(
                plot_waveform(playback_mixed[i], sr, f"{mic_names[i]} (convolutive)", "#fbbf24"),
                use_container_width=True, theme=None,
            )
            st.audio(to_wav_bytes(playback_mixed[i], sr))

    st.divider()
    st.subheader("Running Instantaneous FastICA on the Convolutive Mixture")
    try:
        conv_result = _run_instantaneous_fastica(mixture)
    except WhiteningError as exc:
        st.error(f"❌ Whitening failed on this mixture: {exc}")
        return
    conv_sep = evaluate_separation(reference, conv_result.sources)

    st.write(
        "For comparison, here's the same solver on a well-conditioned "
        "**instantaneous** mixture of the same sources (freshly generated, "
        "not tuned for this comparison)."
    )
    baseline_matrix = generate_mixing_matrix(n, np.random.default_rng(0))
    baseline_mixed = mix_signals(reference, baseline_matrix)
    baseline_result = _run_instantaneous_fastica(baseline_mixed)
    baseline_sep = evaluate_separation(reference, baseline_result.sources)

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "SI-SDR: instantaneous", f"{baseline_sep.mean_si_sdr_db:.1f} dB",
    )
    metric_cols[1].metric(
        "SI-SDR: convolutive", f"{conv_sep.mean_si_sdr_db:.1f} dB",
        f"{conv_sep.mean_si_sdr_db - baseline_sep.mean_si_sdr_db:.1f} dB", delta_color="inverse",
    )
    metric_cols[2].metric("SIR: instantaneous", f"{baseline_sep.mean_sir_db:.1f} dB")
    metric_cols[3].metric(
        "SIR: convolutive", f"{conv_sep.mean_sir_db:.1f} dB",
        f"{conv_sep.mean_sir_db - baseline_sep.mean_sir_db:.1f} dB", delta_color="inverse",
    )
    st.caption(
        "The instantaneous-mixture numbers are the ceiling this solver can "
        "reach when its model assumption actually holds; the gap below "
        "that ceiling on the convolutive mixture is the cost of the model "
        "mismatch, not a bug in the solver."
    )

    st.divider()
    st.subheader("🎯 Frequency-Domain ICA: The Actual Fix")
    st.write(
        "Convolution in time is multiplication in frequency. STFT each "
        "mic channel, and within one narrow frequency bin the room's "
        "effect is *approximately* a single complex gain per source — "
        "locally instantaneous again. So: whiten and run FastICA "
        "**independently per bin**, using a complex-valued formulation, "
        "then solve the two problems that creates — per-bin permutation "
        "and per-bin scale — before inverting back to a waveform. "
        "See the **Theory** page and `src/fdica.py` for the full "
        "algorithm (Bingham & Hyvärinen 2000 complex FastICA, "
        "envelope/DOA-based permutation alignment, minimal-distortion "
        "projection back)."
    )
    fd_cols = st.columns(2)
    n_fft = fd_cols[0].select_slider("STFT window (samples)", options=[512, 1024, 2048, 4096], value=1024)
    st.caption(
        "ℹ️ Bigger windows better match the room's reflection tail (more "
        "accurate 'locally instantaneous' assumption) but leave fewer "
        "time frames per bin for FastICA to estimate statistics from — "
        "for this page's ~3s clips, testing found **1024 is usually the "
        "sweet spot**; larger windows measurably hurt on short clips."
    )

    if st.button("🧠 Run FD-ICA", type="primary"):
        with st.spinner("Running per-bin complex FastICA + permutation alignment..."):
            fdica_result = _cached_fd_ica(mixture, n_fft, n_fft // 4, random_state=0)
        st.session_state["fdica_result"] = fdica_result
        st.session_state["fdica_n_fft"] = n_fft

    fdica_result = st.session_state.get("fdica_result")
    if fdica_result is not None and fdica_result.recovered.shape[0] == n:
        fdica_sep = evaluate_separation(reference, fdica_result.recovered)
        st.markdown(f"**Results** (`n_fft={st.session_state.get('fdica_n_fft')}`, {fdica_result.execution_time_sec:.1f}s)")
        fd_metric_cols = st.columns(4)
        fd_metric_cols[0].metric("SI-SDR", f"{fdica_sep.mean_si_sdr_db:.1f} dB", f"{fdica_sep.mean_si_sdr_db - conv_sep.mean_si_sdr_db:+.1f} dB vs. instantaneous ICA")
        fd_metric_cols[1].metric("SDR", f"{fdica_sep.mean_sdr_db:.1f} dB", f"{fdica_sep.mean_sdr_db - conv_sep.mean_sdr_db:+.1f} dB vs. instantaneous ICA")
        fd_metric_cols[2].metric("SIR", f"{fdica_sep.mean_sir_db:.1f} dB", f"{fdica_sep.mean_sir_db - conv_sep.mean_sir_db:+.1f} dB vs. instantaneous ICA")
        fd_metric_cols[3].metric("SAR", f"{fdica_sep.mean_sar_db:.1f} dB", f"{fdica_sep.mean_sar_db - conv_sep.mean_sar_db:+.1f} dB vs. instantaneous ICA")
        st.info(
            "ℹ️ **Read SDR/SIR and SI-SDR as telling different parts of "
            "the story, not disagreeing.** In this implementation's own "
            "testing, FD-ICA reliably beats instantaneous ICA on "
            "BSS_Eval **SDR and SIR** — genuinely less cross-source "
            "interference. **SI-SDR is noisier, and sometimes worse**: "
            "it fits a single scalar between reference and estimate, "
            "with no tolerance for the mild linear coloration the "
            "STFT/ISTFT round trip and per-bin estimation noise "
            "introduce; BSS_Eval's SDR fits a short linear *filter* "
            "instead, which absorbs that coloration. Low SAR alongside "
            "high SIR is the signature of exactly this: real separation, "
            "real reconstruction artifacts."
        )

        st.markdown("**Recovered Waveforms**")
        fd_playback = normalize_rows_peak(fdica_result.recovered)
        cols = st.columns(n)
        for i in range(n):
            with cols[i]:
                st.plotly_chart(
                    plot_waveform(fd_playback[i], sr, f"FD-ICA Component {i + 1}", DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                    use_container_width=True, theme=None,
                )
                st.audio(to_wav_bytes(fd_playback[i], sr))

        st.markdown("**Permutation Alignment, Before vs. After**")
        st.write(
            "Each line is one separated component's average energy "
            "across frequency. Raw per-bin FastICA output has no "
            "guaranteed labeling from one bin to the next, so before "
            "alignment the lines cross and jump between sources at "
            "essentially every bin. After alignment they should read as "
            "coherent per-source spectral shapes — this is the single "
            "most interesting picture this project produces."
        )
        freq_axis = np.arange(fdica_result.n_freq_bins)
        raw_mean = fdica_result.raw_envelopes.mean(axis=2)       # (n_freq_bins, n)
        aligned_mean = fdica_result.aligned_envelopes.mean(axis=2)
        align_cols = st.columns(2)
        with align_cols[0]:
            st.plotly_chart(
                plot_line(
                    freq_axis,
                    {f"Component {k + 1}": raw_mean[:, k] for k in range(n)},
                    "Before Alignment (raw per-bin order)",
                    "Frequency bin",
                    "Mean |amplitude|",
                ),
                use_container_width=True, theme=None,
            )
        with align_cols[1]:
            st.plotly_chart(
                plot_line(
                    freq_axis,
                    {f"Component {k + 1}": aligned_mean[:, k] for k in range(n)},
                    "After Alignment (consistent labeling)",
                    "Frequency bin",
                    "Mean |amplitude|",
                ),
                use_container_width=True, theme=None,
            )
    else:
        st.info("👆 Click **Run FD-ICA** to separate the convolutive mixture in the frequency domain.")

    st.divider()
    st.subheader("Separation Quality vs. Reverberation and Mic Spacing")
    st.write(
        "The single run above is one point on a curve. This sweep "
        "re-generates the convolutive mixture and re-runs instantaneous "
        "FastICA at several RT60 values (mic spacing fixed at the slider "
        "value above), and separately at several mic spacings (RT60 fixed "
        "at the slider value above) — the degradation is the headline "
        "result of this page, not the single-point numbers."
    )

    sweep_cols = st.columns(2)
    with sweep_cols[0]:
        if st.button("📉 Run RT60 sweep"):
            rt60_values = np.linspace(0.05, 0.8, 6)
            si_sdrs, sirs = [], []
            with st.spinner("Sweeping RT60..."):
                for rt in rt60_values:
                    try:
                        mix, _ = _cached_convolutive_mixture(
                            reference, room_dim, source_positions, mic_positions, sr, float(rt)
                        )
                        res = _run_instantaneous_fastica(mix)
                        sep = evaluate_separation(reference, res.sources)
                    except (RoomAcousticsError, WhiteningError):
                        si_sdrs.append(np.nan)
                        sirs.append(np.nan)
                        continue
                    si_sdrs.append(sep.mean_si_sdr_db)
                    sirs.append(sep.mean_sir_db)
            st.session_state["convolutive_rt60_sweep"] = (rt60_values, si_sdrs, sirs)

        sweep = st.session_state.get("convolutive_rt60_sweep")
        if sweep is not None:
            rt60_values, si_sdrs, sirs = sweep
            st.plotly_chart(
                plot_line(
                    rt60_values,
                    {"SI-SDR (dB)": np.array(si_sdrs), "SIR (dB)": np.array(sirs)},
                    "Separation Quality vs. RT60",
                    "RT60 (s)",
                    "dB",
                ),
                use_container_width=True, theme=None,
            )

    with sweep_cols[1]:
        if st.button("📉 Run mic-spacing sweep"):
            spacing_values = np.linspace(0.02, 1.0, 6)
            si_sdrs, sirs = [], []
            with st.spinner("Sweeping mic spacing..."):
                for spacing in spacing_values:
                    src_pos, mic_pos = _place_convolutive_geometry(room_dim, n, float(spacing))
                    try:
                        mix, _ = _cached_convolutive_mixture(
                            reference, room_dim, src_pos, mic_pos, sr, rt60
                        )
                        res = _run_instantaneous_fastica(mix)
                        sep = evaluate_separation(reference, res.sources)
                    except (RoomAcousticsError, WhiteningError):
                        si_sdrs.append(np.nan)
                        sirs.append(np.nan)
                        continue
                    si_sdrs.append(sep.mean_si_sdr_db)
                    sirs.append(sep.mean_sir_db)
            st.session_state["convolutive_spacing_sweep"] = (spacing_values, si_sdrs, sirs)

        sweep = st.session_state.get("convolutive_spacing_sweep")
        if sweep is not None:
            spacing_values, si_sdrs, sirs = sweep
            st.plotly_chart(
                plot_line(
                    spacing_values,
                    {"SI-SDR (dB)": np.array(si_sdrs), "SIR (dB)": np.array(sirs)},
                    "Separation Quality vs. Mic Spacing",
                    "Mic spacing (m)",
                    "dB",
                ),
                use_container_width=True, theme=None,
            )


def render_failure_gallery() -> None:
    st.title("💥 Failure Gallery")
    st.write(
        "Every page so far shows this pipeline working, at least under "
        "some conditions. This page is the opposite: five ways to break "
        "it on purpose, each tied to a specific assumption FastICA "
        "actually depends on — not a bug, a real model boundary. Every "
        "demo here generates its own synthetic sources (no upload "
        "needed) so the failure is reproducible and isolated to the one "
        "assumption being tested."
    )

    tab_gauss, tab_under, tab_noise, tab_cond, tab_conv = st.tabs(
        ["🎲 Gaussian Sources", "📉 Underdetermined", "📡 Sensor Noise", "🔀 Ill-Conditioned", "🏠 Convolutive"]
    )

    with tab_gauss:
        st.subheader("Two Gaussian Sources")
        st.write(
            "FastICA's entire search strategy is *maximize non-Gaussianity* "
            "(see **Theory → Central Limit Theorem**). If the sources "
            "themselves are Gaussian, there's no non-Gaussian direction to "
            "find: any orthogonal rotation of independent Gaussian "
            "variables is itself a set of independent Gaussian variables, "
            "so the mixture and the true sources are statistically "
            "indistinguishable by rotation. At most one Gaussian source "
            "can ever be identified — with two, the problem is genuinely "
            "unsolvable, not just hard."
        )
        if st.button("🎲 Run Gaussian-source demo"):
            n, T = 2, 8000
            rng = np.random.default_rng(0)
            gaussian_sources = rng.normal(size=(n, T))
            laplace_sources = rng.laplace(size=(n, T))  # non-Gaussian control
            A = generate_mixing_matrix(n, np.random.default_rng(1))

            gauss_seps = []
            laplace_seps = []
            for seed in range(5):
                X_g = mix_signals(gaussian_sources, A)
                res_g = FastICA(random_state=seed).fit_transform(whiten(X_g).whitened)
                gauss_seps.append(evaluate_separation(gaussian_sources, res_g.sources).mean_si_sdr_db)

                X_l = mix_signals(laplace_sources, A)
                res_l = FastICA(random_state=seed).fit_transform(whiten(X_l).whitened)
                laplace_seps.append(evaluate_separation(laplace_sources, res_l.sources).mean_si_sdr_db)

            st.session_state["failure_gauss"] = (gauss_seps, laplace_seps)

        result = st.session_state.get("failure_gauss")
        if result is not None:
            gauss_seps, laplace_seps = result
            cols = st.columns(2)
            cols[0].metric("Gaussian sources — SI-SDR", f"{np.mean(gauss_seps):.1f} dB", f"± {np.std(gauss_seps):.1f} dB across 5 random inits", delta_color="off")
            cols[1].metric("Non-Gaussian (Laplace) control — SI-SDR", f"{np.mean(laplace_seps):.1f} dB", f"± {np.std(laplace_seps):.1f} dB across 5 random inits", delta_color="off")
            st.caption(
                "Same mixing matrix, same solver, only the source "
                "distribution changes — and the Gaussian case recovers "
                "meaningfully worse. Both converge *consistently* across "
                "random initializations (low std either way): with a "
                "finite sample, Gaussian noise is never *exactly* "
                "isotropic, so FastICA still finds a repeatable direction "
                "of maximum sample non-Gaussianity — it's just fitting "
                "sampling noise in that direction rather than real source "
                "structure, which is exactly what population-level "
                "non-identifiability predicts: a well-defined but "
                "spurious answer, not an unstable one."
            )

    with tab_under:
        st.subheader("Underdetermined: 3 Sources, 2 Microphones")
        st.write(
            "Every other page assumes as many microphones as sources "
            "(square, invertible `A`). With more sources than sensors, no "
            "linear un-mixing matrix exists — 2 microphones give only 2 "
            "linear measurements per instant, which can't determine 3 "
            "independent unknowns. FastICA still runs (it just treats the "
            "mic count as the component count), but 2 outputs can't "
            "losslessly represent 3 independent sources — at least one "
            "'recovered' component is unavoidably still a mixture of more "
            "than one true source."
        )
        if st.button("📉 Run underdetermined demo"):
            n_sources, n_mics, T = 3, 2, 8000
            rng = np.random.default_rng(0)
            sources = rng.laplace(size=(n_sources, T))
            A = rng.uniform(0.3, 1.0, size=(n_mics, n_sources)) * rng.choice([-1.0, 1.0], size=(n_mics, n_sources))
            X = A @ sources  # bypasses mix_signals' square-matrix requirement on purpose

            wh = whiten(X)
            result = FastICA(random_state=0).fit_transform(wh.whitened)

            corr = np.zeros((n_mics, n_sources))
            for i in range(n_mics):
                for j in range(n_sources):
                    corr[i, j] = abs(np.corrcoef(result.sources[i], sources[j])[0, 1])
            st.session_state["failure_underdetermined"] = corr

        corr = st.session_state.get("failure_underdetermined")
        if corr is not None:
            st.plotly_chart(
                plot_matrix_heatmap(
                    corr, "|Correlation|: Recovered Components vs. All 3 True Sources",
                    [f"Recovered {i + 1}" for i in range(corr.shape[0])],
                    [f"True Source {j + 1}" for j in range(corr.shape[1])],
                    colorscale="Viridis", zmid=None,
                ),
                use_container_width=True, theme=None,
            )
            st.caption(
                "With a correct (square) problem, this heatmap is close to "
                "a permutation matrix — every recovered component lights "
                "up for exactly one true source, with the rest near zero. "
                "Here it can't be: there are only 2 outputs to explain 3 "
                "independent things, so **at least one** recovered "
                "component is unavoidably a blend of more than one true "
                "source's correlation (look for the row without one "
                "dominant, near-1.0 entry) — which one is a coincidence of "
                "this particular random mixing, but that at least one must "
                "blend is not."
            )

    with tab_noise:
        st.subheader("Additive Sensor Noise")
        st.write(
            "FastICA's model is noiseless: `X = A·S`, full stop — no `+N` "
            "term. Real microphones always have some noise floor. This "
            "sweep adds i.i.d. Gaussian noise directly to an otherwise "
            "clean instantaneous mixture at increasing levels and tracks "
            "recovery quality against the resulting SNR."
        )
        if st.button("📡 Run noise-degradation sweep"):
            n, T = 2, 8000
            rng = np.random.default_rng(0)
            sources = rng.laplace(size=(n, T))
            A = generate_mixing_matrix(n, np.random.default_rng(1))
            X_clean = mix_signals(sources, A)
            clean_power = float(np.mean(X_clean**2))

            snr_db_values = np.linspace(30, -10, 9)
            si_sdrs = []
            with st.spinner("Sweeping sensor noise..."):
                for snr_db in snr_db_values:
                    noise_power = clean_power / (10 ** (snr_db / 10))
                    noisy = X_clean + rng.normal(scale=np.sqrt(noise_power), size=X_clean.shape)
                    try:
                        res = FastICA(random_state=0).fit_transform(whiten(noisy).whitened)
                        sep = evaluate_separation(sources, res.sources)
                        si_sdrs.append(sep.mean_si_sdr_db)
                    except WhiteningError:
                        si_sdrs.append(np.nan)
            st.session_state["failure_noise"] = (snr_db_values, si_sdrs)

        result = st.session_state.get("failure_noise")
        if result is not None:
            snr_db_values, si_sdrs = result
            st.plotly_chart(
                plot_line(snr_db_values, {"SI-SDR (dB)": np.array(si_sdrs)}, "Recovery Quality vs. Sensor SNR", "Sensor SNR (dB)", "SI-SDR (dB)"),
                use_container_width=True, theme=None,
            )
            st.caption(
                "Degradation here is expected to be **graceful**, not a "
                "cliff: FastICA has no explicit noise model, but whitening "
                "and the fixed-point update are continuous functions of "
                "the data, so small noise perturbs the solution smoothly. "
                "Contrast this shape with the Gaussian-source and "
                "underdetermined cases above, which fail outright "
                "regardless of noise."
            )

    with tab_cond:
        st.subheader("Ill-Conditioned Mixing Matrix")
        st.write(
            "`src/mixer.py` normally rejects near-singular mixing matrices "
            "(see `generate_mixing_matrix`'s condition-number check) "
            "because as two microphones' linear combinations become "
            "nearly identical, the information needed to separate them "
            "is genuinely disappearing, not just getting harder to "
            "estimate. This demo builds matrices with deliberately "
            "controlled condition number to find exactly where that "
            "breaks down — the answer (see the caption below the plot) "
            "isn't the smooth degradation you might expect."
        )
        if st.button("🔀 Run ill-conditioning sweep"):
            n, T = 2, 8000
            rng = np.random.default_rng(0)
            sources = rng.laplace(size=(n, T))

            condition_numbers = np.logspace(0, 15, 10)
            si_sdrs = []
            with st.spinner("Sweeping condition number..."):
                for cond in condition_numbers:
                    U, _ = np.linalg.qr(rng.normal(size=(n, n)))
                    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
                    singular_values = np.array([cond**0.5, cond**-0.5])
                    A = U @ np.diag(singular_values) @ V.T
                    X = A @ sources
                    try:
                        res = FastICA(random_state=0).fit_transform(whiten(X).whitened)
                        sep = evaluate_separation(sources, res.sources)
                        si_sdrs.append(sep.mean_si_sdr_db)
                    except WhiteningError:
                        si_sdrs.append(np.nan)
            st.session_state["failure_cond"] = (condition_numbers, si_sdrs)

        result = st.session_state.get("failure_cond")
        if result is not None:
            condition_numbers, si_sdrs = result
            fig = plot_line(condition_numbers, {"SI-SDR (dB)": np.array(si_sdrs)}, "Recovery Quality vs. Mixing-Matrix Condition Number", "Condition number", "SI-SDR (dB)")
            fig.update_xaxes(type="log")
            st.plotly_chart(fig, use_container_width=True, theme=None)
            st.caption(
                "ℹ️ **This curve is flat, then a cliff — not a gradual "
                "slope, and that's the real finding.** Whitening operates "
                "on the *mixed signal's* covariance, not on `A` directly, "
                "so it absorbs an ill-conditioned `A` almost completely: "
                "quality barely moves from condition number 1 up to "
                "roughly 1e6–1e7. Past that, the mixed channels become "
                "numerically indistinguishable in float64 — you'll likely "
                "see a `WhiteningError` (near-singular covariance, caught "
                "as a gap in this plot) right around 1e9–1e10, and "
                "'successful' runs beyond that are really just float64 "
                "precision exhausted, not a real separation. This is a "
                "property of the *linear algebra and floating-point "
                "precision*, not of FastICA specifically."
            )

    with tab_conv:
        st.subheader("Convolutive Mixing Under the Instantaneous Model")
        st.write(
            "The full interactive version of this failure — with your own "
            "room dimensions, RT60, and mic spacing, plus the FD-ICA fix "
            "— lives on the **Convolutive Mixing** page. This is the "
            "same failure in miniature: one fixed room, one fixed RT60, "
            "just enough to place it in this gallery alongside the other "
            "four."
        )
        if st.button("🏠 Run convolutive-mixing demo"):
            n, T = 2, 3 * 16000
            sr = 16000
            rng = np.random.default_rng(0)
            sources = rng.laplace(size=(n, T))
            sources = np.clip(sources / np.max(np.abs(sources)), -1, 1)

            room_dim = (6.0, 5.0, 3.0)
            source_positions = [(1.0, 1.67, 1.5), (1.0, 3.33, 1.5)]
            mic_positions = [(3.0, 2.4, 1.5), (3.0, 2.6, 1.5)]
            try:
                mixture, _ = _cached_convolutive_mixture(sources, room_dim, source_positions, mic_positions, sr, 0.3)
                inst_result = FastICA(random_state=0).fit_transform(whiten(mixture).whitened)
                inst_sep = evaluate_separation(sources, inst_result.sources)
                baseline_A = generate_mixing_matrix(n, np.random.default_rng(1))
                baseline_result = FastICA(random_state=0).fit_transform(whiten(mix_signals(sources, baseline_A)).whitened)
                baseline_sep = evaluate_separation(sources, baseline_result.sources)
                st.session_state["failure_conv"] = (baseline_sep.mean_si_sdr_db, inst_sep.mean_si_sdr_db)
            except (RoomAcousticsError, WhiteningError) as exc:
                st.error(f"❌ {exc}")

        result = st.session_state.get("failure_conv")
        if result is not None:
            baseline_sdr, conv_sdr = result
            cols = st.columns(2)
            cols[0].metric("SI-SDR: instantaneous mixture (ceiling)", f"{baseline_sdr:.1f} dB")
            cols[1].metric("SI-SDR: convolutive mixture, RT60=0.3s", f"{conv_sdr:.1f} dB", f"{conv_sdr - baseline_sdr:+.1f} dB", delta_color="inverse")
            st.caption("See the **Convolutive Mixing** page for the full RT60/mic-spacing sweep, the room impulse responses, and the FD-ICA recovery attempt.")


@st.cache_data
def _load_real_speech_sources():
    loader = AudioLoader()
    with open(DATA_DIR / "real_speech_1.wav", "rb") as f1, open(DATA_DIR / "real_speech_2.wav", "rb") as f2:
        raw = [
            loader.load_file(f1, "real_speech_1.wav"),
            loader.load_file(f2, "real_speech_2.wav"),
        ]
    return loader.harmonize_batch(raw)


def render_real_data() -> None:
    st.title("🎙️ Real Data: A Non-Synthesized Experiment")
    st.write(
        "Every other page in this app runs on either uploaded audio or "
        "signals this project generated itself (sine waves, Laplace "
        "noise, AM-modulated harmonics). This page runs the exact same "
        "pipeline on two recordings this project didn't create: real "
        "human speech, publicly and permissively licensed, bundled with "
        "the app so this experiment is reproducible without any upload."
    )

    with st.expander("📜 Data provenance and license"):
        st.markdown(
            "Both clips are public-domain readings from **LibriVox** "
            "(volunteer-read public-domain audiobooks), sourced via the "
            "Internet Archive:\n\n"
            "- *\"Because I could not stop for Death\"* (Emily Dickinson) — "
            "[Short Poetry Collection 001](https://archive.org/details/short_poetry_001_librivox), reader initials AC\n"
            "- *\"A Dead Boche\"* (Robert Graves) — "
            "[Short Poetry Collection 001](https://archive.org/details/short_poetry_001_librivox), reader initials SM\n\n"
            "Both are in the **public domain** (LibriVox's entire catalog "
            "is public-domain text read into public-domain recordings). "
            "Bundled here as `data/real_speech_1.wav` / `real_speech_2.wav` "
            "— the first 12 seconds of each original recording (0.3s "
            "lead-in trimmed), downsampled/converted to WAV, nothing else "
            "altered."
        )

    signals = _load_real_speech_sources()
    reference = stack_sources(signals)
    sr = signals[0].sample_rate
    n = len(signals)

    st.subheader("The Two Real Sources")
    cols = st.columns(2)
    titles = ["\"Because I could not stop for Death\" — Emily Dickinson", "\"A Dead Boche\" — Robert Graves"]
    for i in range(n):
        with cols[i]:
            st.plotly_chart(
                plot_waveform(signals[i].data, sr, titles[i], DEFAULT_COLORS[i % len(DEFAULT_COLORS)]),
                use_container_width=True, theme=None,
            )
            st.audio(to_wav_bytes(signals[i].data, sr))

    st.divider()
    st.subheader("1️⃣ Instantaneous Mixing — the Ceiling Case")
    st.write("Same pipeline as the **Mix Signals → FastICA** pages, run once here for a self-contained baseline.")
    if st.button("▶️ Run instantaneous-mixing experiment"):
        A = generate_mixing_matrix(n, np.random.default_rng(0))
        mixed = mix_signals(reference, A)
        result = _run_instantaneous_fastica(mixed)
        sep = evaluate_separation(reference, result.sources)
        st.session_state["realdata_instantaneous"] = sep

    sep = st.session_state.get("realdata_instantaneous")
    if sep is not None:
        cols = st.columns(4)
        cols[0].metric("SI-SDR", f"{sep.mean_si_sdr_db:.1f} dB")
        cols[1].metric("SDR", f"{sep.mean_sdr_db:.1f} dB")
        cols[2].metric("SIR", f"{sep.mean_sir_db:.1f} dB")
        cols[3].metric("SAR", f"{sep.mean_sar_db:.1f} dB")

    st.divider()
    st.subheader("2️⃣ Convolutive Mixing — Instantaneous ICA vs. FD-ICA")
    room_dim = (6.0, 5.0, 3.0)
    min_feasible_rt60 = 0.161 * room_volume(room_dim) / room_surface_area(room_dim)
    rt60 = st.slider("RT60 (s)", 0.15, 0.6, 0.2, 0.05)
    if rt60 < min_feasible_rt60:
        st.caption(f"⚠️ RT60={rt60:.2f}s isn't achievable at this fixed room size (needs ≥{min_feasible_rt60:.2f}s).")

    if st.button("🧠 Run convolutive experiment (instantaneous ICA + FD-ICA)"):
        source_positions = [(1.0, 1.67, 1.5), (1.0, 3.33, 1.5)]
        mic_positions = [(3.0, 2.4, 1.5), (3.0, 2.6, 1.5)]
        try:
            with st.spinner("Generating convolutive mixture, running instantaneous ICA and FD-ICA..."):
                mixture, _ = _cached_convolutive_mixture(reference, room_dim, source_positions, mic_positions, sr, rt60)
                inst_result = _run_instantaneous_fastica(mixture)
                inst_sep = evaluate_separation(reference, inst_result.sources)
                fdica_result = _cached_fd_ica(mixture, 1024, 256, random_state=0)
                fdica_sep = evaluate_separation(reference, fdica_result.recovered)
            st.session_state["realdata_conv"] = (inst_sep, fdica_sep)
        except (RoomAcousticsError, WhiteningError) as exc:
            st.error(f"❌ {exc}")

    result = st.session_state.get("realdata_conv")
    if result is not None:
        inst_sep, fdica_sep = result
        metric_cols = st.columns(4)
        metric_cols[0].metric("SI-SDR: instantaneous ICA", f"{inst_sep.mean_si_sdr_db:.1f} dB")
        metric_cols[1].metric(
            "SI-SDR: FD-ICA", f"{fdica_sep.mean_si_sdr_db:.1f} dB",
            f"{fdica_sep.mean_si_sdr_db - inst_sep.mean_si_sdr_db:+.1f} dB",
        )
        metric_cols[2].metric("SIR: instantaneous ICA", f"{inst_sep.mean_sir_db:.1f} dB")
        metric_cols[3].metric(
            "SIR: FD-ICA", f"{fdica_sep.mean_sir_db:.1f} dB",
            f"{fdica_sep.mean_sir_db - inst_sep.mean_sir_db:+.1f} dB",
        )
        st.warning(
            "⚠️ **Honest finding, not a flattering one.** On the synthetic "
            "AM-modulated harmonic signals used to develop and test FD-ICA "
            "(**Convolutive Mixing** page), it substantially beat "
            "instantaneous ICA on SDR/SIR. On this real, unscripted "
            "speech, testing this exact configuration across several RT60 "
            "values found FD-ICA's advantage much smaller and "
            "**inconsistent** — sometimes a couple of dB better on "
            "SDR/SIR, sometimes no better or slightly worse, and SI-SDR "
            "did not improve in any configuration tested. The most likely "
            "reason: the synthetic test signal was deliberately built "
            "with strong, clean amplitude-envelope correlation across "
            "frequency (the cue permutation alignment leans on); real "
            "speech has that structure too, but weaker and noisier, over "
            "only 12 seconds of audio per source. This is the gap between "
            "a clean benchmark and a real recording, not a bug — see "
            "`src/fdica.py` and the **Convolutive Mixing** page for the "
            "algorithm and its synthetic-data results."
        )

    st.divider()
    st.subheader("What This Experiment Shows")
    st.markdown(
        "- **Instantaneous ICA works on real speech** as well as it does "
        "on synthetic sources — the math doesn't care what generated the "
        "signal, only that it's non-Gaussian and independent, and real "
        "speech clearly qualifies.\n"
        "- **Convolutive mixing breaks instantaneous ICA on real speech "
        "too**, exactly as the synthetic Convolutive Mixing demo predicts.\n"
        "- **FD-ICA's benefit, measured honestly on real data, is real "
        "but modest and inconsistent** — a meaningfully weaker result "
        "than the synthetic benchmark suggested, most likely because of "
        "how much the permutation-alignment cue's reliability depends on "
        "the specific signal's cross-frequency structure and how much "
        "data is available to estimate it from."
    )


def render_downloads() -> None:
    st.title("⬇️ Downloads")
    st.write(
        "Every artifact produced by the pipeline, in one place: separated "
        "audio, mixed audio, key interactive plots, and the full CSV "
        "metrics report."
    )

    signals = st.session_state.get("signals")
    if not signals:
        st.warning("⚠️ Please upload audio first — there's nothing to download yet.")
        return

    n = len(signals)
    sr = st.session_state.get("sample_rate") or signals[0].sample_rate
    mixed = st.session_state.get("mixed_signals")
    mixing_matrix = st.session_state.get("mixing_matrix")
    my_result = st.session_state.get("fastica_result")
    sk_result = st.session_state.get("sklearn_fastica_result")
    mic_names = [f"Mic {i + 1}" for i in range(n)]
    source_names = [f"Source {i + 1}" for i in range(n)]

    st.subheader("🎤 Original Audio")
    cols = st.columns(n)
    for i, sig in enumerate(signals):
        with cols[i]:
            st.download_button(
                f"Source {i + 1}",
                data=to_wav_bytes(sig.data, sig.sample_rate),
                file_name=f"source_{i + 1}.wav",
                mime="audio/wav",
                key=f"dl_source_{i}",
            )

    st.subheader("🎙️ Mixed Audio")
    if mixed is not None:
        mixed_playback = normalize_rows_peak(mixed)
        cols = st.columns(n)
        for i in range(n):
            with cols[i]:
                st.download_button(
                    f"Mic {i + 1} (mixed)",
                    data=to_wav_bytes(mixed_playback[i], sr),
                    file_name=f"mixed_mic_{i + 1}.wav",
                    mime="audio/wav",
                    key=f"dl_mixed_{i}",
                )
    else:
        st.info("Visit **Mix Signals** to generate mixed audio.")

    st.subheader("🧠 Separated (Recovered) Audio")
    if my_result is not None:
        st.markdown("**My FastICA**")
        my_playback = normalize_rows_peak(my_result.sources)
        cols = st.columns(n)
        for i in range(n):
            with cols[i]:
                st.download_button(
                    f"Component {i + 1}",
                    data=to_wav_bytes(my_playback[i], sr),
                    file_name=f"recovered_my_fastica_{i + 1}.wav",
                    mime="audio/wav",
                    key=f"dl_my_{i}",
                )
    else:
        st.info("Visit **FastICA** to generate separated audio.")

    if sk_result is not None:
        st.markdown("**scikit-learn FastICA**")
        sk_playback = normalize_rows_peak(sk_result.sources)
        cols = st.columns(n)
        for i in range(n):
            with cols[i]:
                st.download_button(
                    f"Component {i + 1}",
                    data=to_wav_bytes(sk_playback[i], sr),
                    file_name=f"recovered_sklearn_{i + 1}.wav",
                    mime="audio/wav",
                    key=f"dl_sk_{i}",
                )

    st.divider()
    st.subheader("📈 Interactive Plots (HTML)")
    st.caption("HTML files — open in any browser, fully interactive (zoom/hover/pan). Requires an internet connection to load the Plotly.js library from CDN.")
    plot_cols = st.columns(3)
    with plot_cols[0]:
        if mixing_matrix is not None:
            fig = plot_matrix_heatmap(mixing_matrix, "Mixing Matrix A", mic_names, source_names)
            st.download_button(
                "Mixing Matrix",
                data=fig.to_html(include_plotlyjs="cdn"),
                file_name="mixing_matrix.html",
                mime="text/html",
                key="dl_plot_mixing",
            )
        else:
            st.caption("Visit **Mix Signals** first.")
    with plot_cols[1]:
        if my_result is not None:
            fig = plot_convergence(my_result.convergence_history, my_result.tol, "My FastICA Convergence")
            st.download_button(
                "Convergence Plot",
                data=fig.to_html(include_plotlyjs="cdn"),
                file_name="convergence.html",
                mime="text/html",
                key="dl_plot_convergence",
            )
        else:
            st.caption("Visit **FastICA** first.")
    with plot_cols[2]:
        if my_result is not None:
            reference = stack_sources(signals)
            my_eval = evaluate(reference, my_result.sources)
            fig = plot_matrix_heatmap(
                my_eval.correlation_matrix,
                "Correlation: Truth vs. Recovered",
                source_names,
                [f"Recovered {i + 1}" for i in range(n)],
                colorscale="Viridis",
                zmid=None,
            )
            st.download_button(
                "Correlation Heatmap",
                data=fig.to_html(include_plotlyjs="cdn"),
                file_name="correlation_heatmap.html",
                mime="text/html",
                key="dl_plot_corr",
            )
        else:
            st.caption("Visit **FastICA** first.")

    st.divider()
    st.subheader("📄 CSV Metrics Report")
    if my_result is not None:
        reference = stack_sources(signals)
        my_eval = evaluate(reference, my_result.sources)
        sk_eval = evaluate(reference, sk_result.sources) if sk_result is not None else None
        my_sep = evaluate_separation(reference, my_result.sources)
        sk_sep = evaluate_separation(reference, sk_result.sources) if sk_result is not None else None
        report_df = build_metrics_report(
            component_names=source_names,
            my_eval=my_eval,
            my_execution_time_sec=my_result.execution_time_sec,
            my_n_iterations=my_result.n_iterations,
            my_converged=my_result.converged,
            sk_eval=sk_eval,
            sk_execution_time_sec=sk_result.execution_time_sec if sk_result is not None else None,
            sk_n_iterations=sk_result.n_iterations if sk_result is not None else None,
            sk_converged=sk_result.converged if sk_result is not None else None,
            my_sep=my_sep,
            sk_sep=sk_sep,
        )
        st.download_button(
            "📥 Download Metrics CSV",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="bss_ica_metrics_report.csv",
            mime="text/csv",
            type="primary",
            key="dl_csv_downloads_page",
        )
    else:
        st.info("Visit **FastICA** to generate the metrics report.")


def render_about() -> None:
    st.title("ℹ️ About This Project")
    st.write(
        "A final-year DSP mini-project demonstrating **Blind Source "
        "Separation** via **Independent Component Analysis (ICA)** — a "
        "from-scratch FastICA implementation, validated against "
        "scikit-learn, wrapped in an interactive Streamlit app."
    )

    st.subheader("🛠️ Tech Stack")
    st.markdown(
        """
        <div>
            <span class="nb-badge yellow">Python 3</span>
            <span class="nb-badge blue">NumPy</span>
            <span class="nb-badge pink">SciPy</span>
            <span class="nb-badge green">Matplotlib</span>
            <span class="nb-badge yellow">Plotly</span>
            <span class="nb-badge blue">SoundFile</span>
            <span class="nb-badge pink">Librosa</span>
            <span class="nb-badge green">scikit-learn</span>
            <span class="nb-badge yellow">Streamlit</span>
            <span class="nb-badge blue">Pandas</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("🧩 What Was Built From Scratch")
    st.markdown(
        "- **Whitening** (mean subtraction → covariance → eigendecomposition → whitening transform) — `src/whitening.py`\n"
        "- **FastICA** (symmetric fixed-point algorithm, 3 nonlinearities, orthogonalization, convergence detection) — `src/fastica.py`\n"
        "- **Evaluation metrics** (permutation/sign matching, correlation, SNR, MSE) — `src/metrics.py`\n\n"
        "scikit-learn is used **only** for the Milestone 5 comparison — it never powers the app's actual separation pipeline."
    )

    st.subheader("📖 References")
    st.markdown(
        "1. A. Hyvärinen, \"Fast and Robust Fixed-Point Algorithms for "
        "Independent Component Analysis,\" *IEEE Transactions on Neural "
        "Networks*, vol. 10, no. 3, pp. 626–634, 1999.\n"
        "2. A. Hyvärinen and E. Oja, \"Independent Component Analysis: "
        "Algorithms and Applications,\" *Neural Networks*, vol. 13, "
        "no. 4-5, pp. 411–430, 2000.\n"
        "3. A. Hyvärinen, J. Karhunen, and E. Oja, *Independent Component "
        "Analysis*, Wiley-Interscience, 2001.\n"
        "4. scikit-learn documentation: `sklearn.decomposition.FastICA`"
    )

    st.subheader("📂 Project Structure")
    st.code(
        "BlindSourceSeparation/\n"
        "├── app.py                 # Streamlit entry point / page router\n"
        "├── requirements.txt\n"
        "├── assets/                # Custom CSS (Neo-Brutalism theme)\n"
        "├── data/, outputs/\n"
        "└── src/\n"
        "    ├── loader.py          # Audio loading, validation, harmonization\n"
        "    ├── audio.py           # Amplitude stats + in-memory WAV encoding\n"
        "    ├── mixer.py           # Random mixing matrix + mixing\n"
        "    ├── whitening.py       # PCA/whitening from scratch\n"
        "    ├── fastica.py         # FastICA from scratch + sklearn wrapper\n"
        "    ├── metrics.py         # Correlation / SNR / MSE / CSV report\n"
        "    ├── visualization.py   # Plotly charts (waveform, spectrogram, matrix, ...)\n"
        "    └── utils.py           # Formatting + CSS loading helpers",
        language="text",
    )


def main() -> None:
    configure_page()
    init_session_state()
    page = render_sidebar()
    if page != "Home":
        st.session_state["_home_upload_active"] = False

    if page == "Home":
        render_home()
    elif page == "Upload Audio":
        render_upload_audio()
    elif page == "Mix Signals":
        render_mix_signals()
    elif page == "Whitening":
        render_whitening()
    elif page == "FastICA":
        render_fastica()
    elif page == "Comparison":
        render_comparison()
    elif page == "Visualizations":
        render_visualizations()
    elif page == "Metrics":
        render_metrics()
    elif page == "Theory":
        render_theory()
    elif page == "Convolutive Mixing":
        render_convolutive_mixing()
    elif page == "Failure Gallery":
        render_failure_gallery()
    elif page == "Real Data":
        render_real_data()
    elif page == "Downloads":
        render_downloads()
    elif page == "About":
        render_about()


if __name__ == "__main__":
    main()
