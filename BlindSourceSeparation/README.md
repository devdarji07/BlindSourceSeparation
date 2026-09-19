# Blind Source Separation using ICA

> 🚧 **Work in progress.** This project is being built milestone by milestone.
> Current status: **Phase 6 — Deployment prep** complete (pinned deps, caching); live Streamlit Community Cloud deployment is a pending manual step (see below).

## What is this?

An interactive Streamlit app that demonstrates **Blind Source Separation (BSS)**
using **Independent Component Analysis (ICA)**: given only a mixture of
independent signals, recover the originals without being told how they were
mixed. FastICA (the symmetric fixed-point algorithm) and PCA/whitening are
implemented from scratch in NumPy; scikit-learn is used only as an
implementation parity check, never to power the actual separation.

**What it currently does, precisely:** you upload 2–3 WAV files as ground-truth
sources, the app generates a random **instantaneous** mixing matrix `A` and
forms `X = A·S`, then recovers `S` from `X` alone. This is *not* the real
"cocktail party problem" — real microphones in a real room mix sound
**convolutively** (each mic hears a delayed, echo-smeared sum of the sources,
not an instantaneous linear blend), and even a few milliseconds of inter-mic
delay breaks the instantaneous model this pipeline currently uses. Instantaneous
ICA is, however, the *right* model for domains like EEG/MEG artifact removal,
fetal ECG extraction, and some telecom antenna arrays. See the in-app
**Theory** page for the full explanation, and the Roadmap below for how this
project extends to the convolutive (real-room) case.

There is no "record from microphone" input: recording sources one at a time
through separate browser widgets produces clips with arbitrary relative
offsets, not sample-synchronized microphone channels, so it would silently
violate the instantaneous mixing model. WAV upload only.

## Quick start

```bash
cd BlindSourceSeparation
pip install -r requirements.txt
streamlit run app.py
```

## Running tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Folder structure

```
BlindSourceSeparation/
├── app.py                 # Streamlit entry point / page router
├── requirements.txt / requirements-dev.txt
├── conftest.py             # pytest import path setup
├── assets/                # style.css (Neo-Brutalism theme)             ✅ Milestone 8
├── data/                  # real_speech_*.wav (Phase 5, public domain -- see in-app provenance)
├── outputs/                # Generated audio/plots/reports (git-ignored in spirit)
├── tests/                   # pytest unit tests (one file per src/ math module)
└── src/
    ├── loader.py           # Audio loading, validation, harmonization  ✅ Milestone 1
    ├── audio.py             # Amplitude stats + in-memory WAV encoding ✅ Milestone 1
    ├── visualization.py      # Plotly plots (waveform, spectrogram, matrix, sweeps) ✅ Milestone 1-2, Phase 2
    ├── utils.py               # Small formatting helpers               ✅ Milestone 1
    ├── mixer.py                 # Random mixing matrix + mixing          ✅ Milestone 2
    ├── whitening.py               # PCA/whitening from scratch           ✅ Milestone 3
    ├── fastica.py                   # FastICA from scratch + sklearn wrapper ✅ Milestone 4, 5
    ├── metrics.py                     # Legacy correlation/SNR/MSE + CSV report ✅ Milestone 5, 7
    ├── separation_metrics.py           # SI-SDR (from scratch) + BSS_Eval SDR/SIR/SAR ✅ Phase 1
    ├── room_acoustics.py                # Image-source RIR generator + convolutive mixing ✅ Phase 2
    ├── stft.py                            # From-scratch STFT/ISTFT (overlap-add)  ✅ Phase 3
    └── fdica.py                            # Frequency-domain ICA (complex FastICA, ✅ Phase 3
                                             # permutation alignment, projection back)
```

## Milestone 1 — Audio Loading

- Upload 2 or 3 WAV files via the **Upload Audio** page.
- Validation: file count (2–3), non-empty, minimum duration.
- Automatic harmonization: resample to a common (lowest) sample rate,
  trim every signal to the shortest duration, peak-normalize to [-1, 1].
- Per-signal display: waveform (interactive Plotly), duration, sample
  rate, and amplitude statistics (min/max/mean/std/RMS).
- In-browser playback for every loaded signal.

## Milestone 2 — Signal Mixing

- Random, well-conditioned mixing matrix `A` generated on the **Mix Signals**
  page, with a button to regenerate it on demand.
- Instantaneous linear mixing model applied: `X = A · S`.
- Mixing matrix displayed as an interactive, annotated heatmap (and as a
  raw value table).
- Side-by-side **Original vs. Mixed** comparison with tabs for waveforms
  and spectrograms, plus in-browser playback for every original and mixed
  channel.

## Milestone 3 — Whitening (from scratch)

- Implemented without sklearn's PCA: mean subtraction → covariance matrix
  → eigen decomposition (`np.linalg.eigh`) → whitening transform
  (`W = D^-1/2 · Eᵀ`).
- **Whitening** page walks through all four steps with a plain-language
  explanation under each one.
- Covariance matrix visualized as an interactive heatmap **before**
  whitening (correlated) and **after** whitening (≈ identity matrix).
- Eigenvalues/eigenvectors shown as tables; whitened waveforms plotted
  per channel.
- Bonus scatter plots (2D/3D) show the mixed signals as a correlated,
  slanted cloud and the whitened signals as a decorrelated, round cloud.

## Milestone 4 — FastICA (from scratch)

- Implemented without sklearn's FastICA: the *symmetric* fixed-point
  algorithm (Hyvärinen & Oja) with random weight initialization, three
  selectable nonlinearities (`logcosh`, `exp`, `cube`), symmetric
  orthogonalization after every update, and convergence detection with
  configurable tolerance / max iterations.
- **FastICA** page lets you pick the nonlinearity, tolerance, and
  iteration cap, then run separation on the whitened signals from
  Milestone 3.
- Convergence visualized as an interactive log-scale plot of the
  per-iteration change in `W` against the tolerance threshold.
- Un-mixing matrix shown as a heatmap; recovered independent components
  plotted and playable, with a clear note on ICA's inherent scale/sign/
  order ambiguity.
- Verified on synthetic 3-source mixtures (sine, square wave, uniform
  noise): recovers each source with >0.999 correlation and a correct,
  non-degenerate permutation.

## Milestone 5 — scikit-learn Comparison

- **`src/metrics.py`**: permutation/sign matching (linear assignment on
  absolute correlation), least-squares scale alignment, then per-component
  **correlation**, **SNR (dB)**, and **MSE** — since ICA recovers sources
  only up to scale/sign/order.
- `run_sklearn_fastica` in `src/fastica.py` runs scikit-learn's `FastICA`
  on the *same* whitened data (`whiten=False`) with the *same*
  nonlinearity/tolerance/iterations as the from-scratch run, for a fair
  comparison.
- **Comparison** page shows execution time & iteration count side by
  side, a recovery-quality table (both implementations vs. ground truth),
  a cross-correlation heatmap between the two implementations' recovered
  components, and a plain-language verdict on whether they agree.
- Verified on a synthetic 3-source mixture: both implementations recover
  the true sources with >0.9999 correlation, and agree with each other
  at >0.9999 cross-correlation.

## Milestone 6 — Interactive Visualizations

- **Visualizations** page: a gallery consolidating every interactive
  Plotly chart from the pipeline, organized by chart type (not by
  pipeline stage) across 5 tabs — Waveforms, Spectrograms, Covariance,
  Matrices, Correlation.
- Waveforms & spectrograms for every stage: original, mixed, whitened,
  recovered (mine), recovered (scikit-learn).
- Covariance matrices before/after whitening, the mixing matrix, both
  implementations' recovered (un-mixing) matrices, and correlation
  heatmaps (mine vs. truth, sklearn vs. truth, mine vs. sklearn) — all
  in one place.
- Every chart uses Plotly's dark theme with native zoom, hover, and pan;
  pages gracefully prompt the user to an earlier step if a given stage
  hasn't been run yet.

## Milestone 7 — Evaluation Dashboard

- **Metrics** page: a single dashboard consolidating recovery quality
  (correlation, SNR, MSE, recovery %), execution time, iteration count,
  convergence status, and the convergence graph for **My FastICA**, with
  an equivalent section for **scikit-learn** when the Comparison page has
  been run.
- Per-component metric tables for each implementation.
- `build_metrics_report()` in `src/metrics.py` assembles a flat,
  spreadsheet-friendly report (per-component rows + an aggregate "MEAN"
  row per implementation, with execution time/iterations/convergence
  repeated on every row) — downloadable as CSV via a button on the page.
- Degrades gracefully: works with just "My FastICA" if Comparison hasn't
  been run yet, and still produces a valid CSV in that case.

## Milestone 8 — Neo-Brutalism UI

- Custom **Neo-Brutalism** theme (`assets/style.css`, ~250 lines): thick
  black borders, bold offset drop-shadows, bright flat colors (yellow /
  blue / pink / green), rounded corners, heavy typography, and hover/press
  animations — applied app-wide via Streamlit's stable `data-testid` hooks
  (buttons, cards, metrics, tabs, expanders, file uploader, dataframes,
  sidebar), so no page needed per-widget styling.
- **Home** page redesigned as a hero section (gradient banner, badge
  pills) with an animated "Original → Mixed → Recovered" flow diagram and
  a 3-card explainer grid.
- **Theory** page: full beginner-friendly write-up across 6 tabs —
  Cocktail Party & BSS, Central Limit Theorem, PCA, Whitening, FastICA,
  and real-world Applications.
- **Downloads** page: every original/mixed/recovered audio channel (both
  implementations), 3 key interactive plots as self-contained HTML, and
  the CSV metrics report — all as one-click downloads.
- **About** page: tech stack, what was built from scratch vs. scikit-learn,
  references (Hyvärinen's FastICA papers, the ICA textbook), and the
  project structure.
- **Bug found & fixed during visual verification**: `st.plotly_chart`'s
  default `theme="streamlit"` silently overrides a figure's own template
  with Streamlit's light theme; fixed by passing `theme=None` everywhere.
  That in turn revealed Streamlit strips the `template` key even when
  `theme=None`, so `src/visualization.py` now applies the dark palette via
  explicit `paper_bgcolor`/`plot_bgcolor`/`colorway`/axis-color properties
  instead of `template="plotly_dark"`.
- Verified visually end-to-end with a Playwright-driven headless Chromium
  session (all 11 pages, real audio uploaded and run through the full
  pipeline) — confirmed dark charts, styled buttons/cards/metrics, and
  zero console errors.

## Phase 0 — Honest Framing

- Removed the "record from microphone" input mode: sequential browser
  recordings aren't sample-synchronized, so they violated the instantaneous
  mixing model the pipeline assumes. WAV upload only, with an in-app note
  explaining why.
- Rewrote Home and Theory copy to stop claiming this solves "the cocktail
  party problem" — it demonstrates instantaneous BSS on a synthetic mixture,
  which is a real and useful thing, just not that.
- Reframed the scikit-learn comparison as an **implementation parity check**:
  both implementations run the same published algorithm, so agreement mostly
  validates the from-scratch transcription, not separation quality against
  some independent authority. De-emphasized the execution-time metric (two
  short NumPy loops on a shared machine is noise, not benchmarking).
- Added explicit convolutive-vs-instantaneous mixing explanations to the
  Theory page, naming where instantaneous ICA is and isn't the right model.

## Phase 1 — Proper Separation Metrics

- Added **SI-SDR** (from scratch, `src/separation_metrics.py`, tested against
  hand-derived analytic cases) and **BSS_Eval SDR/SIR/SAR** (`mir_eval`) as
  the headline recovery-quality numbers on the Comparison and Metrics pages,
  replacing ad-hoc correlation/SNR/MSE (kept as a secondary "legacy metrics"
  panel, and in the CSV export).
- Permutation ambiguity is resolved once via Hungarian assignment on absolute
  correlation and reused by both metric families; sign/scale ambiguity is
  resolved independently by each metric's own formulation (SI-SDR's optimal
  projection coefficient, BSS_Eval's linear-filter projection).
- UI explains why SIR (interference from other sources) and SAR (artifacts
  the algorithm introduced) are more diagnostic than a single correlation or
  SDR number: they separate *why* a separation is bad, not just *how* bad.

## Phase 2 — Convolutive Mixing & the Failure It Predicts

- Added an image-source-method room impulse response generator from scratch
  (`src/room_acoustics.py`, Allen & Berkley 1979; vectorized NumPy; uniform
  per-surface absorption via Sabine's formula for RT60↔absorption), tested
  against direct-path physics and measured RT60 (Schroeder backward
  integration) within statistical tolerance of the Sabine target.
- Added a **Convolutive Mixing** page: builds a real multi-mic convolutive
  mixture from the uploaded sources, runs the *same* instantaneous FastICA
  used everywhere else in the app on it (on purpose), and shows the result
  side by side with a well-conditioned instantaneous mixture of the same
  sources — the gap between them is the cost of the model mismatch.
- Added RT60 and mic-spacing degradation sweeps on that page: SI-SDR/SIR
  plotted against reverberation time and mic spacing, making the collapse a
  first-class, reproducible result rather than a one-off number. Even mild
  reverberation collapses instantaneous ICA's separation quality by tens of
  dB, consistent with the Theory page's claim that inter-mic delay alone
  breaks the instantaneous model.
- RIR→mic convolution uses `scipy.signal.fftconvolve` (not part of the
  from-scratch scope; ~400x faster than direct convolution at the RIR
  lengths involved, needed to keep the sweeps interactive).

## Phase 3 — Frequency-Domain ICA

- Added STFT/ISTFT from scratch (`src/stft.py`, overlap-add, tested for
  perfect reconstruction to machine precision) and complex-valued
  whitening + FastICA per frequency bin (`src/fdica.py`, Bingham &
  Hyvärinen 2000). Testing found their suggested concave nonlinearities
  (`sqrt`, `log`) actually reward the *equal-mixture* direction over the
  true sources for strongly super-Gaussian circular data — every random
  init converged to the same wrong ~1/√2-correlation fixed point. Switched
  to a kurtosis-type nonlinearity (`g(y)=y`), documented in the module.
- Solved the per-bin **permutation** ambiguity by combining envelope
  correlation (Murata/Sawada) with a direction-of-arrival-style cue (the
  estimated per-bin mixing vector's phase/scale-normalized direction),
  refined iteratively against a global consensus. Plain sequential
  bin-to-bin propagation was tried first and measured at only 55-70%
  correct labeling on realistic synthetic mixtures; the combined,
  globally-refined version reaches ~85-90% on the same data.
- Solved the per-bin **scale** ambiguity via the minimal distortion
  principle (projection back onto a fixed reference microphone).
- Added a before/after permutation-alignment visualization to the
  **Convolutive Mixing** page, alongside a head-to-head FD-ICA vs.
  instantaneous-ICA comparison. Result, reported honestly: FD-ICA
  consistently and substantially improves BSS_Eval **SDR and SIR**
  (genuinely less interference) but **SI-SDR is noisier and sometimes
  worse** — traced to SI-SDR's rigid single-scalar alignment being
  unforgiving of the mild linear coloration the STFT/ISTFT round trip
  introduces, where BSS_Eval's linear-filter projection absorbs it. Both
  numbers are shown, not just the flattering one.

## Phase 4 — Failure Gallery

- Added a **Failure Gallery** page: five self-contained demonstrations
  (synthetic sources, no upload needed) of specific ICA assumptions
  breaking — two Gaussian sources (non-identifiability), an
  underdetermined 3-sources/2-mics mixture, additive sensor noise,
  an ill-conditioned mixing matrix, and convolutive mixing under the
  instantaneous model.
- The ill-conditioning demo's first version (condition numbers up to
  1e4) showed *no* degradation at all — whitening operates on the mixed
  signal's covariance, not on `A` directly, and absorbs an ill-conditioned
  `A` almost completely. Sweeping up to 1e15 found the real shape: flat
  until roughly 1e6-1e7, then a cliff (a caught `WhiteningError`, then
  float64-precision-exhausted noise beyond it) — not the gradual slope
  originally assumed. Copy was rewritten to describe what's actually
  observed instead of the original guess.
- The Gaussian-sources demo's first draft claimed the failure would show
  up as *run-to-run instability* across random initializations; tested
  and found both the Gaussian and non-Gaussian cases converge
  consistently (low variance across seeds) — a finite sample of Gaussian
  noise is never exactly isotropic, so FastICA still finds a repeatable,
  if spurious, direction. Copy corrected to match.

## Phase 5 — Real Data

- Added a **Real Data** page: two public-domain LibriVox poem readings
  (different narrators, sourced from archive.org), trimmed to 12s each
  and bundled as `data/real_speech_*.wav`, run through the exact same
  instantaneous-mixing, convolutive-mixing, and FD-ICA pipeline as every
  synthetic demo elsewhere in the app. Provenance and license documented
  in-app.
- Instantaneous ICA works on real speech about as well as on synthetic
  sources (~50dB SI-SDR) — the algorithm only needs non-Gaussian,
  independent signals, and real speech qualifies easily.
- Convolutive mixing breaks it the same way the synthetic Convolutive
  Mixing demo predicts.
- **FD-ICA's real-data result is honestly weaker than its synthetic
  benchmark.** On real speech, tested across several RT60 values, its
  advantage over instantaneous ICA is small and inconsistent (occasionally
  a couple of dB better on SDR/SIR, sometimes not better at all, SI-SDR
  not reliably improved) — a real gap from the synthetic AM-modulated
  harmonic signal used to develop it, which was (unintentionally) an
  easier case: strong, clean cross-frequency envelope correlation, which
  is exactly the cue permutation alignment depends on. Reported as found,
  not smoothed over.

## Phase 6 — Deployment (local readiness done; live deploy pending)

- `requirements.txt` / `requirements-dev.txt` pinned to exact versions
  (`==`, not `>=`) this project was built and tested against, for
  reproducible installs.
- Added `st.cache_data` wrappers around `render_convolutive_mixture` and
  `fd_ica` — the two slowest operations in the app (RIR generation and
  per-bin complex FastICA) — plus the shared instantaneous-FastICA helper,
  keyed on their actual arguments. Verified via a headless-browser run:
  an identical repeat call drops from ~32s to effectively instant.
  Every call site across the Convolutive Mixing, Failure Gallery, and
  Real Data pages goes through the cached wrappers.
- Checked repo/asset size for bloat: `data/` (the two Phase 5 real-audio
  clips) is 1.6MB, `.git` is 1.8MB — no cleanup needed.
- **Not done yet, by choice**: pushing to GitHub and connecting Streamlit
  Community Cloud. That needs your own GitHub and Streamlit accounts, so
  it wasn't done automatically. To deploy when you're ready:
  1. Create a GitHub repo and push this one to it:
     `git remote add origin <your-repo-url>`, then `git push -u origin master`.
  2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
     GitHub, click **New app**, and point it at your repo,
     `BlindSourceSeparation` as the app directory, `app.py` as the entry
     point, and `requirements.txt` (already pinned) for dependencies.
  3. If Streamlit Cloud's default Python version doesn't match this
     project's (developed on 3.13), add a `runtime.txt` with
     `python-3.13` (or whatever version Streamlit Cloud offers that's
     closest) at the repo root.
  4. Once live, put the URL at the top of this README.

## Roadmap

This is a from-scratch DSP/ML project built and documented in phases, each
adding real capability rather than polish:

1. ~~**Honest framing**~~ — done (above).
2. ~~**Proper separation metrics**~~ — done (above).
3. ~~**Convolutive mixing**~~ — done (above).
4. ~~**Frequency-domain ICA**~~ — done (above).
5. ~~**Failure gallery**~~ — done (above).
6. ~~**Real data**~~ — done (above).
7. **Deployment** — pinned dependencies and caching done (above); the
   live Streamlit Community Cloud deployment is a manual step (see
   above) pending your GitHub/Streamlit accounts.

Installation details, screenshots, and a final report will be filled in as
later phases land.
