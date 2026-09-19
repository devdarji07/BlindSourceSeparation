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
