# 🐬 Dolphin Pro Analyzer - Ultimate Edition v4.3

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Integrated Bioacoustic Analysis Tool for Dolphin Communication Studies**

A comprehensive Python toolkit for analyzing dolphin acoustic communication using LSTM autoencoders, dynamical systems theory, and advanced visualization techniques. Designed for research on *Tursiops* species (bottlenose dolphins).

---

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Behavioral States](#-behavioral-states)
- [Output Files](#-output-files)
- [Methods](#-methods)
- [Requirements](#-requirements)
- [Citation](#-citation)
- [License](#-license)

---

## ✨ Features

### Deep Learning Analysis
- **LSTM Autoencoder** for unsupervised behavioral state discovery
- Latent space embedding and clustering
- Explainable AI (XAI) feature importance analysis

### Acoustic Feature Extraction
- Spectral centroid, bandwidth, rolloff, and flatness
- Zero-crossing rate and RMS energy
- Mel-frequency cepstral coefficients (MFCCs)
- Click detection and Inter-Click Interval (ICI) analysis

### Dynamical Systems Analysis
- **Recurrence Quantification Analysis (RQA)** - determinism, laminarity, recurrence rate
- **Markov chain modeling** - state transition probabilities
- **Phase portrait reconstruction** - attractor dynamics
- **Temporal irreversibility** - time asymmetry detection
- **Self-organized criticality** - avalanche analysis

### Advanced Analytics
- Topological Data Analysis (TDA) with persistent homology
- Vocal interaction timing (Floor Transfer Offset analysis)
- Whistle contour extraction and clustering
- Soundscape analysis with Lombard effect detection
- Cognitive kinematics in latent space
- Motif/syntax analysis with statistical significance testing

### Visualization Suite (37 outputs)
- Interactive dashboards and summary reports
- Sankey diagrams for state transitions
- Chord diagrams for behavioral connections
- Streamgraphs, horizon charts, and ridge plots
- Phase portraits and recurrence plots
- Polar histograms and Voronoi tessellations
- Animated 3D helix trajectories

---

## 🔧 Installation

### Prerequisites

```bash
# Create a virtual environment (recommended)
python -m venv dolphin_env
source dolphin_env/bin/activate  # Linux/Mac
# or
dolphin_env\Scripts\activate  # Windows
```

### Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm
pip install torch torchvision torchaudio
pip install librosa soundfile
pip install networkx plotly imageio
```

### Optional Dependencies

```bash
# For topological data analysis
pip install ripser persim

# For GPU acceleration (CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Basic Usage

1. Place your `.wav` audio file in the same directory as the script
2. Run the analyzer:

```bash
python cai_nsnp_0145.py
```

The analyzer will automatically:
- Detect and load the audio file
- Trim the first 3 minutes and last 2 minutes (configurable)
- Extract acoustic features
- Train the LSTM autoencoder
- Perform clustering and state identification
- Generate all visualizations and CSV exports

### Input Requirements

- **Audio format:** WAV file (mono or stereo)
- **Sample rate:** Any (automatically resampled to 22050 Hz)
- **Minimum duration:** ~10 minutes recommended
- **Content:** Dolphin vocalizations (clicks, whistles, burst-pulses)

### Configuration

Key parameters can be modified in the `DolphinProVisualizer` class:

```python
self.window_size = 40      # Temporal window for sequences
self.hidden_dim = 32       # LSTM hidden dimension
self.batch_size = 64       # Training batch size
```

---

## 🎯 Behavioral States

The analyzer identifies 8 distinct behavioral states based on acoustic signatures:

| State | Description | Acoustic Characteristics |
|-------|-------------|-------------------------|
| **BASELINE** | Low activity baseline | Minimal acoustic output |
| **SOFT BURST** | Moderate burst activity | Medium-energy burst pulses |
| **INTENSE BURST** | High burst energy episodes | High-amplitude burst sequences |
| **CONTACT CALL** | Whistle-dominant communication | Signature whistles, frequency modulation |
| **SOCIAL PLAY** | Mixed high-activity interaction | Variable clicks and whistles |
| **SCAN BURST** | Scanning with burst pulses | Regular click trains |
| **VIGILANT REST** | Alert resting state | Sparse, low-energy signals |
| **DEEP REST** | Low activity rest | Minimal vocalization |

---

## 📁 Output Files

### Visualizations (PNG/GIF)

| # | Filename | Description |
|---|----------|-------------|
| 1 | `fig01_dashboard.png` | Main analysis dashboard |
| 2 | `fig02_mandala.png` | Circular state visualization |
| 3 | `fig03_burst_analysis.png` | Burst-pulse substructure |
| 4 | `fig04_streamgraph.png` | Temporal state proportions |
| 5 | `fig05_helix.png` | 3D chronological helix |
| 6 | `fig06_helix_animation.gif` | Animated helix trajectory |
| 7 | `fig07_vector_field.png` | Latent space dynamics |
| 8 | `fig08_sankey.png` | State transition flows |
| 9 | `fig09_recurrence.png` | Recurrence plot |
| 10 | `fig10_horizon.png` | Horizon chart |
| 11 | `fig11_chord.png` | State connectivity diagram |
| 12 | `fig12_spectrogram.png` | Spectrogram with state overlay |
| 13 | `fig13_polar.png` | Polar histogram |
| 14 | `fig14_phase_portrait.png` | Phase space reconstruction |
| 15 | `fig15_entropy.png` | Shannon entropy analysis |
| 16 | `fig16_voronoi.png` | Voronoi tessellation map |
| 17 | `fig17_ridge.png` | Ridge plot distributions |
| 18 | `fig18_sunburst.png` | Hierarchical state sequences |
| 19 | `fig19_ici_analysis.png` | Inter-click interval analysis |
| 20 | `fig20_bout_analysis.png` | Behavioral bout statistics |
| 21 | `fig21_markov.png` | Markov transition analysis |
| 22 | `fig22_vocal_interaction.png` | Vocal timing patterns |
| 23 | `fig23_whistle_catalog.png` | Whistle type classification |
| 24 | `fig24_soundscape.png` | Acoustic environment analysis |
| 25 | `fig25_kinematics.png` | Cognitive kinematics |
| 26 | `fig26_motif_analysis.png` | Behavioral motif patterns |
| 27 | `fig27_xai_analysis.png` | Feature importance (XAI) |
| 28 | `fig28_irreversibility.png` | Temporal irreversibility |
| 29 | `fig29_criticality.png` | Self-organized criticality |
| 30 | `fig30_topology.png` | Topological data analysis |
| 31 | `fig31_summary_report.png` | Comprehensive summary |

### CSV Data Files (31 files)

#### Core Statistics
- `paper_statistics_summary.csv` - Master summary with all metrics
- `paper_audio_stats.csv` - Recording and audio parameters
- `paper_behavioral_states.csv` - 8 behavioral state characteristics

#### Click & ICI Analysis
- `paper_ici_statistics.csv` - Inter-click interval statistics

#### Bout Analysis
- `paper_bout_statistics.csv` - Behavioral bout metrics
- `paper_bout_by_state.csv` - Bout statistics per state

#### Dynamics & Transitions
- `paper_rqa_statistics.csv` - Recurrence quantification metrics
- `paper_entropy_statistics.csv` - Shannon entropy values
- `paper_markov_transitions.csv` - 8×8 state transition matrix
- `paper_markov_statistics.csv` - Markov chain properties
- `paper_transition_flows.csv` - Sankey-style transition data

#### Vocal Interaction
- `paper_vocal_interaction.csv` - FTO, gaps, burstiness metrics
- `paper_event_transitions.csv` - 3×3 event type matrix

#### Whistle Analysis
- `paper_whistle_catalog.csv` - Whistle contour type summary
- `paper_whistle_centroids.csv` - Whistle type centroid shapes

#### Soundscape
- `paper_soundscape.csv` - Lombard effect and acoustic environment

#### Kinematics
- `paper_kinematics.csv` - Cognitive kinematics summary
- `paper_kinematics_by_state.csv` - Velocity by behavioral state

#### Motif/Syntax Analysis
- `paper_motif_summary.csv` - Behavioral motif statistics
- `paper_motif_details.csv` - All motif patterns with Z-scores

#### Explainable AI
- `paper_xai_dominant_features.csv` - Dominant feature per state
- `paper_xai_latent_pca.csv` - Latent space PCA variance
- `paper_xai_cluster_importance.csv` - Feature deviation matrix

#### Advanced Dynamics
- `paper_irreversibility.csv` - Temporal irreversibility metrics
- `paper_criticality.csv` - Self-organized criticality metrics
- `paper_topology.csv` - Topological data analysis (TDA)

#### Latent Space
- `paper_latent_space.csv` - Latent space dimension statistics
- `paper_state_centroids.csv` - State centroids in latent space
- `paper_phase_portrait.csv` - Phase space trajectory statistics

#### Temporal & Feature Distributions
- `paper_feature_distributions.csv` - Feature statistics per state
- `paper_temporal_proportions.csv` - State proportions over time (60s bins)

---

## 🔬 Methods

### Feature Extraction

The analyzer extracts the following acoustic features using `librosa`:

- **Spectral features:** centroid, bandwidth, rolloff, flatness, contrast
- **Temporal features:** zero-crossing rate, RMS energy
- **Cepstral features:** 13 MFCCs
- **Rhythm features:** tempo, onset strength

### LSTM Autoencoder Architecture

```
Input → LSTM Encoder → Latent Space (32-dim) → LSTM Decoder → Reconstruction
```

The model learns compressed representations of acoustic sequences, enabling:
- Dimensionality reduction
- Behavioral state clustering (K-means, k=8)
- Anomaly detection via reconstruction error

### Recurrence Quantification Analysis

Based on [Marwan et al. (2007)](https://doi.org/10.1016/j.physrep.2006.11.001):
- **Recurrence Rate (RR):** Density of recurrence points
- **Determinism (DET):** Predictability of the system
- **Laminarity (LAM):** Presence of laminar states
- **Entropy (ENTR):** Complexity of diagonal structures

### Topological Data Analysis

Uses persistent homology to identify:
- **Betti-0:** Connected components
- **Betti-1:** Loops/cycles in the data
- **Betti-2:** Voids/cavities

---

## 📦 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-learn>=0.24.0
torch>=1.9.0
librosa>=0.9.0
tqdm>=4.62.0
networkx>=2.6.0 (optional)
plotly>=5.3.0 (optional)
imageio>=2.9.0 (optional)
```

---

## 📖 Citation

If you use this tool in your research, please cite:

```bibtex
@software{dolphin_pro_analyzer,
  title = {Dolphin Pro Analyzer: Integrated Bioacoustic Analysis Tool},
  version = {4.3},
  year = {2025},
  url = {https://github.com/yourusername/dolphin-pro-analyzer}
}
```

### Key References

- Au, W.W.L. (1993). *The Sonar of Dolphins*. Springer-Verlag.
- Janik, V.M. (2013). Cognitive skills in bottlenose dolphin communication. *Trends in Cognitive Sciences*, 17(4), 157-159.
- Marwan, N., Romano, M.C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports*, 438(5-6), 237-329.
- Shannon, C.E. (1948). A mathematical theory of communication. *Bell System Technical Journal*, 27, 379-423.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

<p align="center">
  <i>Advancing our understanding of dolphin communication through computational bioacoustics</i>
</p>
