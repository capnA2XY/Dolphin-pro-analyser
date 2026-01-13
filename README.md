# 🐬 DOLPHIN PRO ANALYZER v4.0 CAI

**Deep Learning-Based Behavioral Classification of Indo-Pacific Bottlenose Dolphin (*Tursiops aduncus*) Acoustic Signals**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxxx-blue)](https://doi.org/)

---

## 📖 Overview

DOLPHIN PRO ANALYZER is a comprehensive computational pipeline for automated behavioral state classification of dolphin acoustic signals. The system integrates:

- **LSTM Autoencoders** for latent space representation learning
- **Harmonic-Percussive Source Separation (HPSS)** for acoustic feature extractionPermission is hereby granted, free of charge, to any person obtaining a copy
6		-
of this software and associated documentation files (the "Software"), to deal
7		-
in the Software without restriction, including without limitation the rights
8		-
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
9		-
copies of the Software, and to permit persons to whom the Software is
10		-
furnished to do so, subject to the following conditions:
11		-
12		-
The above copyright notice and this permission notice shall be included in all
13		-

- **Recurrence Quantification Analysis (RQA)** for temporal dynamics characterization
- **Markov Chain Modeling** for behavioral transition analysis
- **Social Network Analysis** for group-level behavioral patterns

### Associated Publication

> Aradi, A. (2026). Deep learning-based behavioral state classification of Indo-Pacific bottlenose dolphin (*Tursiops aduncus*) acoustic signals using LSTM autoencoders. *Marine Mammal Science*. (submitted)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| 🧠 **8 Behavioral States** | Automatic clustering with interpretable labels |
| 📊 **30 Visualizations** | Comprehensive analytical output suite |
| 🔬 **Enhanced ICI Analysis** | Bimodal detection, terminal buzz identification |
| 📈 **Bout Analysis** | Survival modeling, fragmentation metrics |
| 🌐 **Social Dynamics** | Turn-taking, contagion cascades, dominance hierarchies |
| ⚡ **GPU Accelerated** | PyTorch-based LSTM training |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/capnA2XY/Dolphin-pro-analyser.git
cd Dolphin-pro-analyser

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
librosa>=0.8.0
torch>=1.9.0
networkx>=2.6.0
plotly>=5.0.0
imageio>=2.9.0
```

---

## 💻 Usage

### Basic Usage

```python
from dolphin_pro_analyzer_v40_cai import DolphinProAnalyzer

# Initialize analyzer with audio file
analyzer = DolphinProAnalyzer("your_recording.wav")

# Run full analysis pipeline
analyzer.run()
```

### Command Line

```bash
# Place your .wav file in the directory and run
python dolphin_pro_analyzer_v40_cai.py
```

### Example Script

```python
"""example_usage.py - Demonstration of DOLPHIN PRO ANALYZER"""

import os
from dolphin_pro_analyzer_v40_cai import DolphinProAnalyzer

# Find first .wav file in current directory
wav_files = [f for f in os.listdir('.') if f.endswith('.wav')]

if wav_files:
    analyzer = DolphinProAnalyzer(wav_files[0])
    analyzer.run()
    print(f"Analysis complete! Check output PNG/GIF/HTML files.")
else:
    print("No .wav file found. Place a recording in this directory.")
```

---

## 🎯 Behavioral States

The analyzer automatically classifies acoustic signals into 8 behavioral states:

| State | Label | Acoustic Signature | Description |
|-------|-------|-------------------|-------------|
| 0 | **SILENCE** | Low all channels | Baseline/resting period |
| 1 | **SCAN** | High click, low whistle | Active echolocation |
| 2 | **WHISTLE** | High whistle, low burst | Contact calls |
| 3 | **Soft BURST** | Moderate burst | Social contact signals |
| 4 | **SCAN+WHISTLE** | Mixed click/whistle | Combined scanning |
| 5 | **BURST+SCAN** | High burst + click | Approach behavior |
| 6 | **PLAY** | Mixed high activity | Social play |
| 7 | **INTENSE PLAY** | Maximum all channels | High-arousal interaction |

---

## 📊 Output Visualizations

The pipeline generates **30 distinct visualizations** organized into categories:

### Main Paper Figures (5)

| Figure | Filename | Description |
|--------|----------|-------------|
| 1 | `dolphin_streamgraph.png` | Temporal dynamics of acoustic channels |
| 2 | `dolphin_recurrence_plot.png` | RQA recurrence visualization |
| 3 | `dolphin_ici_analysis.png` | Inter-click interval analysis |
| 4 | `dolphin_bout_analysis.png` | Comprehensive bout analysis |
| 5 | `dolphin_social_network.png` | Social network structure |

### Supplementary Figures (S1-S25)

<details>
<summary><b>Click to expand full visualization list</b></summary>

| Fig | Filename | Category | Description |
|-----|----------|----------|-------------|
| S1 | `dolphin_dashboard_full.png` | Overview | Integrated analysis dashboard |
| S2 | `dolphin_mandala.png` | Cyclicity | Circular behavioral pattern |
| S3 | `dolphin_helix_3d.png` | Evolution | 3D latent space trajectory |
| S4 | `dolphin_vector_field.png` | Dynamics | Behavioral attractor field |
| S5 | `dolphin_burst_deep_dive.png` | Micro-structure | Burst pulse sub-classification |
| S6 | `dolphin_sankey.png` | Transitions | State transition flow diagram |
| S7 | `dolphin_horizon_chart.png` | Multi-channel | Horizon chart analysis |
| S8 | `dolphin_chord_diagram.png` | Network | Transition chord diagram |
| S9 | `dolphin_spectrogram_overlay.png` | Validation | Spectrogram with states |
| S10 | `dolphin_polar_histogram.png` | Temporal | Windrose distribution |
| S11 | `dolphin_phase_portrait.png` | Dynamics | Phase space velocity field |
| S12 | `dolphin_entropy_plot.png` | Complexity | Shannon entropy dynamics |
| S13 | `dolphin_voronoi_map.png` | Territories | Behavioral territory mapping |
| S14 | `dolphin_ridge_plot.png` | Fingerprints | Acoustic feature distributions |
| S15 | `dolphin_markov_analysis.png` | Stationary | Markov chain properties |
| S16 | `dolphin_turn_taking.png` | Social | Vocal turn-taking dynamics |
| S17 | `dolphin_contagion_cascade.png` | Social | Acoustic behavior spreading |
| S18 | `dolphin_dynamic_network.png` | Social | Network evolution over time |
| S19 | `dolphin_information_flow.png` | Social | Transfer entropy analysis |
| S20 | `dolphin_repertoire_similarity.png` | Culture | Vocal repertoire clustering |
| S21 | `dolphin_reciprocity_dominance.png` | Social | Dominance hierarchy |
| S22 | `dolphin_summary_report.png` | Summary | Statistical summary report |
| S23 | `dolphin_helix_3d_anim.gif` | Animation | Animated behavioral evolution |
| S24 | `dolphin_flow_field.gif` | Animation | Animated attractor dynamics |
| S25 | `dolphin_sequence_sunburst.html` | Interactive | Behavioral sequence hierarchy |

</details>

---

## 🔬 Methodology

### Signal Processing Pipeline

```
Raw Audio (.wav, 96 kHz)
    │
    ▼
┌─────────────────────────────────┐
│  Harmonic-Percussive Separation │
│  (HPSS via librosa)             │
└─────────────────────────────────┘
    │
    ├──► Harmonic Channel (Whistles)
    ├──► Percussive Channel (Burst Pulses)  
    └──► Onset Strength (Clicks)
           │
           ▼
    ┌─────────────────────────────────┐
    │  Feature Vector Extraction      │
    │  (50-sample windows, 80% overlap)│
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  LSTM Autoencoder               │
    │  Encoder: 128→64→16 latent dims │
    │  Decoder: 16→64→128→3 output    │
    └─────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────┐
    │  K-means Clustering (K=8)       │
    │  Silhouette optimization        │
    └─────────────────────────────────┘
           │
           ▼
    8 Behavioral States + 30 Visualizations
```

### LSTM Architecture

```
Input (seq_len, 3) ──► LSTM(128) ──► LSTM(64) ──► Latent(16)
                                                      │
Output (seq_len, 3) ◄── LSTM(128) ◄── LSTM(64) ◄─────┘
```

### Analysis Methods

| Method | Purpose | Key Metrics |
|--------|---------|-------------|
| **RQA** | Temporal structure | RR, DET, LAM, L_max |
| **Shannon Entropy** | Behavioral complexity | H, H_norm |
| **ICI Analysis** | Click characterization | Bimodality, terminal buzz % |
| **Bout Analysis** | Episode duration | T_50, fragmentation index |
| **Markov Chain** | Transition probabilities | Stationary distribution |
| **Transfer Entropy** | Information flow | Directed coupling |

---

## 📁 Repository Structure

```
Dolphin-pro-analyser/
│
├── dolphin_pro_analyzer_v40_cai.py   # Main analyzer (4658 lines)
├── example_usage.py                   # Usage demonstration
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── README.md                          # This file
│
├── paper/
│   ├── dolphin_mms_wiley_template.tex # Manuscript (Wiley format)
│   └── dolphin_mms_paper.pdf          # Compiled PDF
│
└── figures/                           # Output visualizations
    ├── dolphin_streamgraph.png
    ├── dolphin_recurrence_plot.png
    ├── dolphin_ici_analysis.png
    ├── dolphin_bout_analysis.png
    ├── dolphin_social_network.png
    └── ... (25 supplementary figures)
```

---

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@article{aradi2025dolphin,
  title={Deep learning-based behavioral state classification of Indo-Pacific 
         bottlenose dolphin (\textit{Tursiops aduncus}) acoustic signals 
         using LSTM autoencoders},
  author={Aradi, Attila},
  journal={Marine Mammal Science},
  year={2025},
  volume={00},
  pages={1--15},
  doi={10.xxxx/xxxxx}
}
```

---

## 🤝 Acknowledgments

- * Access to dolphin facility and acoustic recordings
- **ION-Technik Kft., Tarnaszentmária, Hungary** - Computational resources
- **University of Miskolc** - Institutional support

---

## 📄 License


Copyright (c) 2025 Attila Aradi



## 📧 Contact

**Attila Aradi**  
ION-Technik Kft.

ION Alkalmazott Kutatási NonProfit Kft.

University of Miskolc, Hungary  

📧 attila.aradi@gmail.com

---

<p align="center">
  <i>Developed with 🐬 for marine mammal science</i>
</p>
