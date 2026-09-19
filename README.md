# Blind Source Separation with Independent Component Analysis

An interactive Streamlit web application that demonstrates Blind Source Separation (BSS) using Independent Component Analysis (ICA). The core FastICA algorithm is implemented entirely from scratch in NumPy. scikit-learn is used only as a parity check.

## Overview

Given only mixtures of independent signals, the app recovers the originals without knowing how they were mixed. It covers what ICA can do, where it fails, and how frequency-domain methods extend it to real rooms.

## Features

- From-scratch FastICA (symmetric fixed-point, three nonlinearities: logcosh, exp, cube)
- From-scratch PCA whitening
- From-scratch STFT/ISTFT (overlap-add)
- From-scratch SI-SDR metric
- Convolutive room mixing via image-source method (Allen and Berkley)
- Frequency-domain ICA with permutation alignment
- Failure Gallery demonstrating five failure modes
- Real speech separation using LibriVox clips
- scikit-learn parity check (cross-correlation > 0.9999)
- Neo-Brutalism CSS theme
- 14 interactive Streamlit pages

## Project Structure
BlindSourceSeparation/
├── app.py # Streamlit entry point
├── requirements.txt # Pinned dependencies
├── src/
│ ├── loader.py # Audio I/O and validation
│ ├── audio.py # Resampling, normalization
│ ├── utils.py # General utilities
│ ├── mixer.py # Random mixing matrix generation
│ ├── whitening.py # PCA whitening (from scratch)
│ ├── fastica.py # FastICA algorithm (from scratch)
│ ├── metrics.py # Correlation, SNR, MSE, CSV export
│ ├── separation_metrics.py # SI-SDR, BSS_Eval SDR/SIR/SAR
│ ├── room_acoustics.py # Image-source RIR, convolutive mixing
│ ├── stft.py # STFT/ISTFT (from scratch)
│ ├── fdica.py # Frequency-domain ICA
│ └── visualization.py # Plotly charts
├── pages/
│ ├── Home.py
│ ├── Upload.py
│ ├── Mix.py
│ ├── Whitening.py
│ ├── FastICA.py
│ ├── Comparison.py
│ ├── Visualizations.py
│ ├── Metrics.py
│ ├── Theory.py
│ ├── Downloads.py
│ ├── About.py
│ ├── Convolutive_Mixing.py
│ ├── Failure_Gallery.py
│ └── Real_Data.py
└── tests/
└── ... # pytest: STFT, room acoustics, metrics, FD-ICA


## Installation

```bash
git clone https://github.com/<your-username>/BlindSourceSeparation.git
cd BlindSourceSeparation
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Pipeline

1. Load: Upload 2-3 WAV files. They are validated, resampled, trimmed, and peak-normalized.
2. Mix: A random well-conditioned matrix produces X = A * S.
3. Whiten: PCA whitening (mean, covariance, eigendecomposition, D^(-1/2) * E^T).
4. FastICA: Symmetric fixed-point iteration with convergence checks.
5. Evaluate: Compare against ground truth and scikit-learn.

## Key Results

| Experiment                        | Result                        |
|-----------------------------------|-------------------------------|
| Synthetic sources (instantaneous) | Correlation > 0.999           |
| From-scratch vs. scikit-learn     | Cross-correlation > 0.9999    |
| Real speech (instantaneous ICA)   | ~50 dB SI-SDR                 |
| Convolutive mixing                | Tens of dB quality drop       |
| FD-ICA permutation alignment      | ~85-90% correct labeling      |

## Failure Modes

The Failure Gallery demonstrates five cases where ICA breaks down:

1. Gaussian sources -- all rotations look identical, no unique solution exists.
2. Underdetermined mixtures -- more sources than microphones, mixing matrix not invertible.
3. Sensor noise -- additive noise degrades separation proportionally.
4. Ill-conditioned mixing -- cliff failure at condition number ~1e6-1e7.
5. Convolutive mixing -- delays and reflections violate the instantaneous model.

## Tech Stack

numpy, scipy, matplotlib, plotly, soundfile, librosa, scikit-learn (parity only), streamlit, pandas, mir_eval

## Limitations

- Instantaneous ICA is not a true cocktail-party solution. Real rooms mix convolutively.
- FD-ICA on real speech is only modestly better than instantaneous ICA.
- ICA recovers sources only up to scale, sign, and order.
- Real audio data is limited to two public-domain LibriVox clips (12 seconds each).

## Tests

```bash
pytest tests/
```

Covers STFT, room acoustics, separation metrics, and FD-ICA.

## Team

- Harshil Nagori (24070123046)
- Dev Darji (24070123033)
- Krishna Chabbaria (24070123149)
- Joy Thakkar (24070123051)

B.Tech ENTC, Symbiosis Institute of Technology, Pune
Under Dr. Mrinal Bachute | Digital Signal Processing Mini Project

## License

Academic project. Not licensed for commercial use.
