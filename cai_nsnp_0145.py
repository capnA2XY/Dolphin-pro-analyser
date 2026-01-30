#!/usr/bin/env python3
"""
DOLPHIN PRO ANALYZER - ULTIMATE EDITION v4.3
=============================================
Integrated Bioacoustic Analysis Tool for Tursiops aduncus
Uses LSTM Autoencoders, Dynamical Systems Theory, and Advanced Visualization.



BEHAVIORAL STATE NOMENCLATURE:
- BASELINE: Low activity baseline state
- SOFT BURST: Moderate burst activity
- INTENSE BURST: High burst energy episodes
- CONTACT CALL: Whistle-dominant communication
- SOCIAL PLAY: Mixed high-activity social interaction
- SCAN BURST: Scanning with burst pulses
- VIGILANT REST: Alert resting state
- DEEP REST: Low activity rest


Output Files:
Visualizations (37 PNG/GIF):
  1-7:   Dashboard, Mandala, Burst Analysis, Streamgraph, Helix, Helix Animation, Vector Field
  8-14:  Sankey, Recurrence, Horizon, Chord, Spectrogram, Polar, Phase Portrait
  15-21: Entropy, Voronoi, Ridge, Sunburst, ICI Analysis, Bout Analysis, Markov
  22-28: Vocal Interaction, Whistle Catalog, Soundscape, Kinematics
  29-31: Motif Analysis, XAI Feature Importance, Temporal Irreversibility
  32-34: Criticality Analysis, Topological Data Analysis
  35-37: Summary Report + animations

CSV Files for Paper (31 files):
  # Core Statistics
  paper_statistics_summary.csv    - Master summary with all metrics
  paper_audio_stats.csv           - Recording & audio parameters
  paper_behavioral_states.csv     - 8 behavioral state characteristics
  
  # Click & ICI Analysis  
  paper_ici_statistics.csv        - Inter-click interval analysis
  
  # Bout Analysis
  paper_bout_statistics.csv       - Behavioral bout analysis
  paper_bout_by_state.csv         - Bout statistics per state
  
  # Dynamics & Transitions
  paper_rqa_statistics.csv        - Recurrence quantification
  paper_entropy_statistics.csv    - Shannon entropy metrics
  paper_markov_transitions.csv    - 8x8 state transition matrix
  paper_markov_statistics.csv     - Markov chain metrics
  paper_transition_flows.csv      - Sankey-style transition flows
  
  # Vocal Interaction
  paper_vocal_interaction.csv     - FTO, gaps, burstiness, etc.
  paper_event_transitions.csv     - 3x3 event type matrix
  
  # Whistle Analysis
  paper_whistle_catalog.csv       - Whistle contour type summary
  paper_whistle_centroids.csv     - Whistle type centroid shapes
  
  # Soundscape
  paper_soundscape.csv            - Lombard effect & acoustic environment
  
  # Kinematics
  paper_kinematics.csv            - Cognitive kinematics summary
  paper_kinematics_by_state.csv   - Velocity by behavioral state
  
  # Motif/Syntax Analysis
  paper_motif_summary.csv         - Behavioral motif statistics
  paper_motif_details.csv         - All motif patterns with Z-scores
  
  # Explainable AI
  paper_xai_dominant_features.csv - Dominant feature per state
  paper_xai_latent_pca.csv        - Latent space PCA variance
  paper_xai_cluster_importance.csv- Full feature deviation matrix
  
  # Advanced Dynamics
  paper_irreversibility.csv       - Temporal irreversibility metrics
  paper_criticality.csv           - Self-organized criticality metrics
  paper_topology.csv              - Topological data analysis (TDA)
  
  # Latent Space
  paper_latent_space.csv          - Latent space dimension stats
  paper_state_centroids.csv       - State centroids in latent space
  paper_phase_portrait.csv        - Phase space trajectory stats
  
  # Temporal & Feature Distributions
  paper_feature_distributions.csv - Feature stats per state
  paper_temporal_proportions.csv  - State proportions over time (60s bins)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.collections import LineCollection
import seaborn as sns
import librosa
import librosa.display
import pandas as pd
import os
import glob
import sys
import shutil
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import Voronoi
from scipy import signal

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    print("Warning: networkx not found.")
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    print("Warning: plotly not found.")
    HAS_PLOTLY = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    print("Warning: imageio not found.")
    HAS_IMAGEIO = False


# =============================================================================
# 1. LSTM AUTOENCODER MODEL
# =============================================================================
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden_n, _) = self.encoder(x)
        latent = hidden_n[-1]
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded_seq, _ = self.decoder(decoder_input)
        reconstructed = self.output_layer(decoded_seq)
        return reconstructed, latent


# =============================================================================
# 2. MAIN ANALYZER CLASS
# =============================================================================
class DolphinProVisualizer:
    def __init__(self):
        self.window_size = 40
        self.hidden_dim = 32
        self.batch_size = 64
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"{'='*60}")
        print("DOLPHIN PRO ANALYZER - ULTIMATE EDITION v4.3")
        print("*** REAL DATA ONLY - No synthetic data ***")
        print(f"{'='*60}")
        print(f"Device: {self.device}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def find_and_load_audio(self):
        """
        Find and load audio file. REQUIRES real .wav file - no synthetic data.
        """
        wav_files = glob.glob("*.wav")
        if not wav_files:
            print("\n" + "="*60)
            print("ERROR: No .wav file found in current directory!")
            print("="*60)
            print("This analyzer requires REAL acoustic data.")
            print("Please provide a .wav file and run again.")
            print("="*60)
            raise FileNotFoundError("No .wav audio file found. Real data required.")
        
        self.file_path = wav_files[0]

        print(f"\n1. File selected: {self.file_path}")

        total_duration = librosa.get_duration(path=self.file_path)
        trim_start = 3 * 60
        trim_end = 2 * 60

        if total_duration < (trim_start + trim_end + 60):
            print("   ! File too short, analyzing without trimming.")
            offset = 0
            duration = None
            self.start_offset_sec = 0
        else:
            offset = trim_start
            duration = total_duration - trim_start - trim_end
            self.start_offset_sec = trim_start
            print(f"   -> TRIMMING: Skipping first {trim_start}s and last {trim_end}s.")

        print("   -> Loading audio into memory...")
        self.y, self.sr = librosa.load(self.file_path, sr=None, offset=offset, duration=duration)
        print(f"   -> Done! SR: {self.sr} Hz, Samples: {len(self.y)}")

    # =========================================================================
    # FEATURE EXTRACTION
    # =========================================================================
    def extract_features(self):
        print("\n2. Feature Extraction (HPSS + Chunk Processing)...")

        hop = 1024
        chunk_duration = 30
        chunk_samples = int(chunk_duration * self.sr)
        total_samples = len(self.y)

        rms_harm_list = []
        rms_perc_list = []
        onset_list = []

        with tqdm(total=total_samples, desc="   HPSS+RMS", unit="sample") as pbar:
            for i in range(0, total_samples, chunk_samples):
                y_chunk = self.y[i: i + chunk_samples]
                if len(y_chunk) < hop * 2:
                    break

                y_h, y_p = librosa.effects.hpss(y_chunk, margin=3.0)

                rh = librosa.feature.rms(y=y_h, hop_length=hop, center=False)[0]
                rp = librosa.feature.rms(y=y_p, hop_length=hop, center=False)[0]
                on = librosa.onset.onset_strength(y=y_chunk, sr=self.sr, hop_length=hop, center=False)

                min_L = min(len(rh), len(rp), len(on))
                rms_harm_list.append(rh[:min_L])
                rms_perc_list.append(rp[:min_L])
                onset_list.append(on[:min_L])
                pbar.update(len(y_chunk))

        print("   -> Concatenating...")
        rms_harm = np.concatenate(rms_harm_list)
        rms_perc = np.concatenate(rms_perc_list)
        onset_env = np.concatenate(onset_list)

        self.times = np.arange(len(rms_harm)) * hop / self.sr + self.start_offset_sec

        raw = np.column_stack((rms_harm, rms_perc, onset_env))
        raw_log = np.log1p(raw * 100)

        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(raw_log)

        self.feat_df = pd.DataFrame(self.features, columns=['Whistle', 'Burst', 'Click'])
        self.quantiles = self.feat_df.quantile([0.5, 0.60, 0.70, 0.80])

        print(f"   -> Data points: {len(self.features)}")

    # =========================================================================
    # MODEL TRAINING
    # =========================================================================
    def train_model(self):
        print("\n3. Deep Learning Training (LSTM)...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        X = []
        for i in tqdm(range(len(self.features) - self.window_size), desc="   Sequencing"):
            X.append(self.features[i:(i + self.window_size)])

        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        dataset = TensorDataset(X_tensor)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        inference_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model = LSTMAutoencoder(3, self.hidden_dim, self.window_size).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=0.0015)
        criterion = nn.MSELoss()

        self.model.train()
        epochs = 12

        epoch_bar = tqdm(range(epochs), desc="   Epochs", position=0)
        for epoch in epoch_bar:
            batch_losses = []
            for batch in train_loader:
                x = batch[0].to(self.device)
                optimizer.zero_grad()
                rec, _ = self.model(x)
                loss = criterion(rec, x)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            epoch_bar.set_postfix({"Loss": f"{sum(batch_losses) / len(batch_losses):.5f}"})

        print("\n   -> Extracting Latent Space...")
        self.model.eval()
        latent_vectors = []
        with torch.no_grad():
            for batch in tqdm(inference_loader, desc="   Inferring"):
                x = batch[0].to(self.device)
                _, latent = self.model(x)
                latent_vectors.append(latent.cpu().numpy())

        self.latent_space = np.concatenate(latent_vectors, axis=0)
        self.valid_times = self.times[self.window_size:]

    # =========================================================================
    # CLUSTERING - UPDATED WITH NEW STATE NAMES
    # =========================================================================
    def analyze_clusters(self):
        print("\n4. Clustering and Labeling...")
        self.n_clusters = 8
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')

        self.clusters = kmeans.fit_predict(self.latent_space)
        self.existing_clusters = sorted(list(set(self.clusters)))

        df = pd.DataFrame(self.features[self.window_size:], columns=['Whistle', 'Burst', 'Click'])
        df['Cluster'] = self.clusters
        stats = df.groupby('Cluster').mean()

        self.labels = {}
        self.all_colors = sns.color_palette("bright", self.n_clusters)

        print("\n   --- BEHAVIORAL STATE CLASSIFICATION ---")
        print("   (Based on acoustic feature profiles)")
        
        # Collect stats for all clusters to determine relative rankings
        cluster_stats = {}
        for i in self.existing_clusters:
            row = stats.loc[i]
            cluster_stats[i] = {
                'whistle': row['Whistle'],
                'burst': row['Burst'],
                'click': row['Click']
            }
        
        # Find min/max for relative comparison
        all_whistles = [s['whistle'] for s in cluster_stats.values()]
        all_bursts = [s['burst'] for s in cluster_stats.values()]
        
        # Sort clusters by burst (for INTENSE BURST detection)
        sorted_by_burst = sorted(cluster_stats.items(), key=lambda x: x[1]['burst'], reverse=True)
        # Sort by whistle (for SOCIAL PLAY and DEEP REST detection)
        sorted_by_whistle = sorted(cluster_stats.items(), key=lambda x: x[1]['whistle'], reverse=True)
        
        # Identify key states
        intense_burst_id = sorted_by_burst[0][0]  # Highest burst
        social_play_id = sorted_by_whistle[0][0]  # Highest whistle
        deep_rest_id = sorted_by_whistle[-1][0]   # Lowest whistle
        
        # Track assigned labels to avoid duplicates
        assigned_labels = {}
        
        for i in self.existing_clusters:
            w = cluster_stats[i]['whistle']
            b = cluster_stats[i]['burst']
            c = cluster_stats[i]['click']
            
            # Classification based on relative rankings
            if i == intense_burst_id and b > np.percentile(all_bursts, 70):
                label = "INTENSE BURST"
            elif i == social_play_id and w > np.percentile(all_whistles, 70):
                label = "SOCIAL PLAY"
            elif i == deep_rest_id and w < np.percentile(all_whistles, 30):
                label = "DEEP REST"
            elif w > np.percentile(all_whistles, 60) and b < np.percentile(all_bursts, 50):
                label = "CONTACT CALL"
            elif b > np.percentile(all_bursts, 40) and b < np.percentile(all_bursts, 70):
                # Moderate burst - distinguish SOFT vs SCAN
                if w > np.median(all_whistles):
                    label = "SCAN BURST"
                else:
                    label = "SOFT BURST"
            elif w < np.percentile(all_whistles, 40) and b < np.percentile(all_bursts, 40):
                label = "BASELINE"
            elif w > np.percentile(all_whistles, 30) and w < np.percentile(all_whistles, 50):
                label = "VIGILANT REST"
            else:
                # Fallback assignment
                if b > np.median(all_bursts):
                    label = "SOFT BURST"
                elif w > np.median(all_whistles):
                    label = "CONTACT CALL"
                else:
                    label = "BASELINE"
            
            # Handle duplicates by adding distinguishing suffix
            base_label = label
            counter = 1
            while label in assigned_labels.values():
                if base_label in ["BASELINE", "SOFT BURST", "VIGILANT REST"]:
                    # These can have variants
                    label = f"{base_label} ({counter})"
                    counter += 1
                else:
                    # Force to a different category
                    if w > np.median(all_whistles):
                        label = "CONTACT CALL" if "CONTACT CALL" not in assigned_labels.values() else f"VIGILANT REST ({counter})"
                    else:
                        label = "BASELINE" if "BASELINE" not in assigned_labels.values() else f"SOFT BURST ({counter})"
                    counter += 1
            
            assigned_labels[i] = label
            final_name = f"{label} [{i}]"
            self.labels[i] = final_name
            print(f"   [{i}] {final_name:<30} (W:{w:.3f} B:{b:.3f} C:{c:.3f})")

    # =========================================================================
    # BURST ANALYSIS (PRIORITY LOGIC)
    # =========================================================================
    def analyze_burst_substructure(self):
        print("\n5. Burst Deep Dive (Priority Logic)...")

        burst_indices = []
        for i, label in enumerate(self.clusters):
            label_text = self.labels[self.clusters[i]]
            if "BURST" in label_text or "PLAY" in label_text or "INTENSE" in label_text:
                burst_indices.append(i)

        if len(burst_indices) < 50:
            print("   ! Not enough burst data.")
            return

        burst_features_raw = self.features[self.window_size:][burst_indices]
        burst_times = self.valid_times[burst_indices]

        local_scaler = MinMaxScaler()
        burst_features_norm = local_scaler.fit_transform(burst_features_raw)

        local_pca = PCA(n_components=2)
        burst_2d = local_pca.fit_transform(burst_features_norm)

        n_sub = 4
        sub_kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init='auto')
        sub_clusters = sub_kmeans.fit_predict(burst_features_norm)

        df_sub = pd.DataFrame(burst_features_norm, columns=['Whistle', 'Burst', 'Click'])
        df_sub['SubID'] = sub_clusters
        stats = df_sub.groupby('SubID').mean()

        sub_labels = {}
        print("   -> Sub-type profiles:")

        for i in range(n_sub):
            row = stats.loc[i]
            w, b, c = row['Whistle'], row['Burst'], row['Click']

            if b > 0.25:
                if w > 0.4:
                    main_type = "INTENSE SOCIAL (Burst+Whistle)"
                else:
                    main_type = "PHYSICAL CONTACT (Burst)"
            elif w > 0.5:
                if c > 0.6:
                    main_type = "COORDINATED SCAN (+Click)"
                else:
                    main_type = "VOCAL EXCHANGE (Whistle)"
            elif c > 0.5:
                main_type = "ECHOLOCATION SCAN"
            else:
                main_type = "TRANSITION STATE"

            sub_labels[i] = main_type
            print(f"      [{i}] {main_type:<30} (Norm -> B:{b:.2f} W:{w:.2f} C:{c:.2f})")

        df_micro = pd.DataFrame({
            'Time': burst_times,
            'Level2_Label': [sub_labels[c] for c in sub_clusters],
            'Level2_ClusterID': sub_clusters,
            'Whistle_Norm': burst_features_norm[:, 0],
            'Burst_Norm': burst_features_norm[:, 1],
            'Click_Norm': burst_features_norm[:, 2]
        })
        df_micro.to_csv("dolphin_micro_burst_analysis.csv", index=False)

        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2)
        ax_map = fig.add_subplot(gs[0, 0])

        sub_colors = sns.color_palette("bright", n_sub)
        for i in range(n_sub):
            mask = (sub_clusters == i)
            ax_map.scatter(burst_2d[mask, 0], burst_2d[mask, 1],
                           label=sub_labels[i], color=sub_colors[i], alpha=0.7, s=30)
        ax_map.set_title("Burst/Play Micro-Structure", fontsize=14)
        ax_map.legend()
        ax_map.grid(True, alpha=0.2)

        ax_stat = fig.add_subplot(gs[0, 1])
        stats.plot(kind='bar', ax=ax_stat, colormap='viridis', alpha=0.8)
        ax_stat.set_xticklabels([sub_labels[i] for i in range(n_sub)], rotation=30, ha='right')
        ax_stat.set_title("Acoustic Profile (Relative Strength)", fontsize=14)
        ax_stat.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('dolphin_burst_deep_dive.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_burst_deep_dive.png")

    # =========================================================================
    # DATA EXPORT
    # =========================================================================
    def export_language_data(self):
        print("\n6. Exporting Linguistic Data...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        df_lang = pd.DataFrame({
            'Time': self.valid_times,
            'Cluster_ID': self.clusters,
            'Label': [self.labels[c] for c in self.clusters],
            'Latent_X': latent_2d[:, 0],
            'Latent_Y': latent_2d[:, 1],
            'Whistle': self.features[self.window_size:, 0],
            'Burst': self.features[self.window_size:, 1],
            'Click': self.features[self.window_size:, 2]
        })
        df_lang.to_csv("dolphin_language_data.csv", index=False)
        print("   -> SAVED: dolphin_language_data.csv")

    # =========================================================================
    # VISUALIZATION 7: DASHBOARD
    # =========================================================================
    def generate_dashboard(self):
        print("\n7. Generating Dashboard...")
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 2)

        ax_map = fig.add_subplot(gs[0:2, 0])
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        for i in self.existing_clusters:
            mask = (self.clusters == i)
            ax_map.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                           color=self.all_colors[i], label=self.labels[i], alpha=0.7, s=15, edgecolors='none')
        ax_map.set_title("Deep Learning Manifold (PCA)", fontsize=14)
        ax_map.legend(loc='lower right', fontsize='x-small')

        ax_net = fig.add_subplot(gs[0:2, 1])
        if HAS_NETWORKX:
            max_idx = max(self.existing_clusters) + 1
            trans_mat = np.zeros((max_idx, max_idx))
            for k in range(len(self.clusters) - 1):
                trans_mat[self.clusters[k], self.clusters[k + 1]] += 1

            G = nx.DiGraph()
            for i in self.existing_clusters:
                freq = np.sum(self.clusters == i)
                G.add_node(self.labels[i], color=self.all_colors[i], size=freq)

            threshold = np.mean(trans_mat) * 0.05
            for i in self.existing_clusters:
                for j in self.existing_clusters:
                    if i != j and trans_mat[i, j] > threshold:
                        weight = trans_mat[i, j]
                        width = np.log1p(weight) * 0.3
                        G.add_edge(self.labels[i], self.labels[j], weight=width)

            pos = nx.circular_layout(G) if len(self.existing_clusters) > 1 else nx.spring_layout(G)
            node_colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
            node_sizes = [np.log(nx.get_node_attributes(G, 'size')[n] + 1) * 100 for n in G.nodes()]
            edge_widths = [d['weight'] for u, v, d in G.edges(data=True)]

            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, ax=ax_net, alpha=0.9)
            nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold', ax=ax_net)
            nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', arrowstyle='->', arrowsize=10,
                                   connectionstyle="arc3,rad=0.1", ax=ax_net)
        ax_net.set_title("Behavioral Transition Network", fontsize=14)
        ax_net.axis('off')

        ax_time = fig.add_subplot(gs[2, :])
        step = max(1, len(self.clusters) // 4000)
        cluster_downsampled = self.clusters[::step]
        times_downsampled = self.valid_times[::step]

        for i in range(len(cluster_downsampled)):
            c = cluster_downsampled[i]
            t = times_downsampled[i]
            ax_time.vlines(t, 0, 1, colors=self.all_colors[c], linewidth=1)

        ax_time.set_title("Behavioral Timeline", fontsize=14)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_yticks([])
        ax_time.set_xlim(times_downsampled[0], times_downsampled[-1])

        plt.tight_layout()
        plt.savefig('dolphin_dashboard_full.png', dpi=300)
        plt.close()

        df_export = pd.DataFrame(
            {'Time': self.valid_times, 'Cluster': self.clusters, 'Label': [self.labels[c] for c in self.clusters]})
        df_export.to_csv('dolphin_full_analysis.csv', index=False)
        print("   -> SAVED: dolphin_dashboard_full.png")

    # =========================================================================
    # VISUALIZATION 8-14: MANDALA, STREAMGRAPH, HELIX, etc.
    # =========================================================================
    def generate_mandala(self):
        print("\n8. Generating Mandala...")
        downsample_rate = max(1, len(self.valid_times) // 2000)
        times_ds = self.valid_times[::downsample_rate]
        clusters_ds = self.clusters[::downsample_rate]
        features_ds = self.features[self.window_size:][::downsample_rate]

        angles = np.linspace(0, 2 * np.pi, len(times_ds))
        energies = np.sum(features_ds, axis=1)
        energies = pd.Series(energies).rolling(window=10, center=True).mean().fillna(0).values
        radii = energies + 1.0

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='polar')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        change_points = np.where(clusters_ds[:-1] != clusters_ds[1:])[0]
        change_points = np.concatenate(([0], change_points + 1, [len(clusters_ds)]))

        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            c_id = clusters_ds[start]
            ax.fill_between(angles[start:end], 1.0, radii[start:end], color=self.all_colors[c_id], alpha=0.9)

        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        plt.tight_layout()
        plt.savefig('dolphin_mandala.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_mandala.png")

    def generate_streamgraph(self):
        print("\n9. Generating Streamgraph...")
        window_sec = 10
        dt = self.valid_times[1] - self.valid_times[0]
        smooth_window = int(window_sec / dt)

        df_stream = pd.DataFrame(self.features[self.window_size:], columns=['Whistle', 'Burst', 'Click'])
        df_smooth = df_stream.rolling(window=smooth_window, center=True).mean().fillna(0)
        time_min = (self.valid_times - self.valid_times[0]) / 60

        fig, ax = plt.subplots(figsize=(20, 8))
        pal = ["#1f77b4", "#d62728", "#2ca02c"]
        ax.stackplot(time_min, df_smooth['Whistle'], df_smooth['Burst'], df_smooth['Click'],
                     labels=['Whistle (Contact Call)', 'Burst (Social)', 'Click (Echolocation)'], colors=pal, alpha=0.85)

        ax.set_title("Behavioral Streamgraph - Acoustic Channel Activity", fontsize=16)
        ax.set_xlabel("Time (min)")
        ax.set_xlim(0, time_min[-1])
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('dolphin_streamgraph.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_streamgraph.png")

    def generate_chrono_helix(self):
        print("\n10. Generating 3D Helix...")
        from mpl_toolkits.mplot3d import Axes3D

        step = max(1, len(self.valid_times) // 3000)
        t = self.valid_times[::step]
        z = (t - t[0]) / (t[-1] - t[0]) * 100

        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        x = latent_2d[::step, 0]
        y = latent_2d[::step, 1]
        clusters_ds = self.clusters[::step]
        colors = [self.all_colors[c] for c in clusters_ds]
        features_ds = self.features[self.window_size:][::step]
        energy = np.sum(features_ds, axis=1)
        sizes = 5 + (energy / energy.max()) * 80

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color='gray', alpha=0.3, linewidth=0.5)
        ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.8, edgecolors='none')
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.set_zlabel('TIME')
        ax.set_title('Behavioral Trajectory in Latent Space', fontsize=14)
        plt.savefig('dolphin_helix_3d.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_helix_3d.png")

    def generate_helix_animation(self):
        print("\n11. Generating Helix Animation...")
        if not HAS_IMAGEIO:
            print("   ! imageio not available, skipping animation.")
            return

        from mpl_toolkits.mplot3d import Axes3D

        step = max(1, len(self.valid_times) // 2500)
        t = self.valid_times[::step]
        z = (t - t[0]) / (t[-1] - t[0]) * 100
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        x = latent_2d[::step, 0]
        y = latent_2d[::step, 1]
        colors = [self.all_colors[c] for c in self.clusters[::step]]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, color='gray', alpha=0.3)
        ax.scatter(x, y, z, c=colors, s=10, alpha=0.8, edgecolors='none')
        ax.axis('off')

        frame_dir = "helix_frames"
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

        frames = []
        for i, angle in enumerate(range(0, 360, 5)):
            ax.view_init(elev=25, azim=angle)
            fname = os.path.join(frame_dir, f"f_{i:03d}.png")
            plt.savefig(fname, transparent=True)
            frames.append(fname)
        plt.close()

        images = [imageio.imread(f) for f in frames]
        imageio.mimsave("dolphin_helix_3d_anim.gif", images, duration=0.05)
        shutil.rmtree(frame_dir)
        print("   -> SAVED: dolphin_helix_3d_anim.gif")

    def generate_vector_field(self):
        print("\n12. Generating Vector Field...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        x, y = latent_2d[:, 0], latent_2d[:, 1]

        grid_size = 20
        x_min, x_max = x.min() - 0.5, x.max() + 0.5
        y_min, y_max = y.min() - 0.5, y.max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        u = np.zeros_like(xx)
        v = np.zeros_like(yy)
        count = np.zeros_like(xx)

        for i in range(len(x) - 1):
            xi = int((x[i] - x_min) / (x_max - x_min) * (grid_size - 1))
            yi = int((y[i] - y_min) / (y_max - y_min) * (grid_size - 1))
            if 0 <= xi < grid_size and 0 <= yi < grid_size:
                u[yi, xi] += x[i + 1] - x[i]
                v[yi, xi] += y[i + 1] - y[i]
                count[yi, xi] += 1

        mask = count > 0
        u[mask] /= count[mask]
        v[mask] /= count[mask]
        u = gaussian_filter(u, sigma=1)
        v = gaussian_filter(v, sigma=1)

        fig, ax = plt.subplots(figsize=(14, 10))
        sns.kdeplot(x=x, y=y, fill=True, cmap="Greys", levels=15, ax=ax, alpha=0.3)
        speed = np.sqrt(u ** 2 + v ** 2)
        ax.streamplot(xx, yy, u, v, color=speed, cmap='autumn', linewidth=1.5)
        ax.set_title("Behavioral Vector Field - State Transition Dynamics")
        plt.tight_layout()
        plt.savefig('dolphin_vector_field.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_vector_field.png")

    def generate_sankey_diagram(self):
        print("\n14. Generating Sankey...")
        max_idx = max(self.existing_clusters) + 1
        trans_mat = np.zeros((max_idx, max_idx))
        for k in range(len(self.clusters) - 1):
            trans_mat[self.clusters[k], self.clusters[k + 1]] += 1

        if HAS_PLOTLY:
            sources, targets, values = [], [], []
            thresh = np.percentile(trans_mat[trans_mat > 0], 30) if np.any(trans_mat > 0) else 0
            for i in self.existing_clusters:
                for j in self.existing_clusters:
                    if trans_mat[i, j] > thresh:
                        sources.append(i)
                        targets.append(j + len(self.existing_clusters))
                        values.append(trans_mat[i, j])

            labels = [self.labels[i] for i in self.existing_clusters] * 2
            colors = [f'rgba({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)},0.8)' for c in
                      [self.all_colors[i] for i in self.existing_clusters]] * 2

            fig = go.Figure(data=[go.Sankey(
                node=dict(label=labels, color=colors),
                link=dict(source=sources, target=targets, value=values)
            )])
            fig.write_html("dolphin_sankey.html")
            print("   -> SAVED: dolphin_sankey.html")

        fig, ax = plt.subplots(figsize=(12, 8))
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_existing = trans_mat[np.ix_(self.existing_clusters, self.existing_clusters)]
        row_sums_existing = trans_existing.sum(axis=1, keepdims=True)
        row_sums_existing[row_sums_existing == 0] = 1
        
        labels_short = [self.labels[i].split('[')[0].strip()[:15] for i in self.existing_clusters]
        sns.heatmap(trans_existing / row_sums_existing, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                   xticklabels=labels_short, yticklabels=labels_short)
        ax.set_title("State Transition Probabilities")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('dolphin_sankey.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_sankey.png")

    # =========================================================================
    # BURST ANALYSIS (PRIORITY LOGIC)
    # =========================================================================
    def analyze_burst_substructure(self):
        print("\n5. Burst Deep Dive (Priority Logic)...")

        burst_indices = []
        for i, label in enumerate(self.clusters):
            label_text = self.labels[self.clusters[i]]
            if "BURST" in label_text or "PLAY" in label_text or "INTENSE" in label_text:
                burst_indices.append(i)

        if len(burst_indices) < 50:
            print("   ! Not enough burst data.")
            return

        burst_features_raw = self.features[self.window_size:][burst_indices]
        burst_times = self.valid_times[burst_indices]

        local_scaler = MinMaxScaler()
        burst_features_norm = local_scaler.fit_transform(burst_features_raw)

        local_pca = PCA(n_components=2)
        burst_2d = local_pca.fit_transform(burst_features_norm)

        n_sub = 4
        sub_kmeans = KMeans(n_clusters=n_sub, random_state=42, n_init='auto')
        sub_clusters = sub_kmeans.fit_predict(burst_features_norm)

        df_sub = pd.DataFrame(burst_features_norm, columns=['Whistle', 'Burst', 'Click'])
        df_sub['SubID'] = sub_clusters
        stats = df_sub.groupby('SubID').mean()

        sub_labels = {}
        print("   -> Sub-type profiles:")

        for i in range(n_sub):
            row = stats.loc[i]
            w, b, c = row['Whistle'], row['Burst'], row['Click']

            if b > 0.25:
                if w > 0.4:
                    main_type = "INTENSE SOCIAL (Burst+Whistle)"
                else:
                    main_type = "PHYSICAL CONTACT (Burst)"
            elif w > 0.5:
                if c > 0.6:
                    main_type = "COORDINATED SCAN (+Click)"
                else:
                    main_type = "VOCAL EXCHANGE (Whistle)"
            elif c > 0.5:
                main_type = "ECHOLOCATION SCAN"
            else:
                main_type = "TRANSITION STATE"

            sub_labels[i] = main_type
            print(f"      [{i}] {main_type:<30} (Norm -> B:{b:.2f} W:{w:.2f} C:{c:.2f})")

        df_micro = pd.DataFrame({
            'Time': burst_times,
            'Level2_Label': [sub_labels[c] for c in sub_clusters],
            'Level2_ClusterID': sub_clusters,
            'Whistle_Norm': burst_features_norm[:, 0],
            'Burst_Norm': burst_features_norm[:, 1],
            'Click_Norm': burst_features_norm[:, 2]
        })
        df_micro.to_csv("dolphin_micro_burst_analysis.csv", index=False)

        fig = plt.figure(figsize=(14, 7))
        gs = fig.add_gridspec(1, 2)
        ax_map = fig.add_subplot(gs[0, 0])

        sub_colors = sns.color_palette("bright", n_sub)
        for i in range(n_sub):
            mask = (sub_clusters == i)
            ax_map.scatter(burst_2d[mask, 0], burst_2d[mask, 1],
                           label=sub_labels[i], color=sub_colors[i], alpha=0.7, s=30)
        ax_map.set_title("Burst/Play Micro-Structure", fontsize=14)
        ax_map.legend()
        ax_map.grid(True, alpha=0.2)

        ax_stat = fig.add_subplot(gs[0, 1])
        stats.plot(kind='bar', ax=ax_stat, colormap='viridis', alpha=0.8)
        ax_stat.set_xticklabels([sub_labels[i] for i in range(n_sub)], rotation=30, ha='right')
        ax_stat.set_title("Acoustic Profile (Relative Strength)", fontsize=14)
        ax_stat.set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig('dolphin_burst_deep_dive.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_burst_deep_dive.png")

    # =========================================================================
    # DATA EXPORT
    # =========================================================================
    def export_language_data(self):
        print("\n6. Exporting Linguistic Data...")
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        df_lang = pd.DataFrame({
            'Time': self.valid_times,
            'Cluster_ID': self.clusters,
            'Label': [self.labels[c] for c in self.clusters],
            'Latent_X': latent_2d[:, 0],
            'Latent_Y': latent_2d[:, 1],
            'Whistle': self.features[self.window_size:, 0],
            'Burst': self.features[self.window_size:, 1],
            'Click': self.features[self.window_size:, 2]
        })
        df_lang.to_csv("dolphin_language_data.csv", index=False)
        print("   -> SAVED: dolphin_language_data.csv")

    # =========================================================================
    # VISUALIZATION 15-24: RECURRENCE, HORIZON, CHORD, etc.
    # =========================================================================
    def generate_recurrence_plot(self):
        print("\n15. Generating Recurrence Plot...")
        step = max(1, len(self.latent_space) // 1000)
        latent_ds = self.latent_space[::step]

        print(f"   -> Distance Matrix Calculation ({len(latent_ds)}x{len(latent_ds)})...")
        dists = squareform(pdist(latent_ds, metric='euclidean'))
        thresh = np.percentile(dists, 15)
        rec = (dists < thresh).astype(int)

        rr = np.sum(rec) / (rec.shape[0] * rec.shape[1]) * 100
        print(f"   -> Recurrence Rate (RR): {rr:.2f}%")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(rec, cmap='binary', origin='lower', aspect='auto')
        axes[0].set_title(f"Recurrence Plot (RR={rr:.1f}%)", fontsize=14)
        axes[0].set_xlabel("Time index i")
        axes[0].set_ylabel("Time index j")

        im = axes[1].imshow(dists, cmap='magma_r', origin='lower', aspect='auto')
        axes[1].set_title("Distance Matrix", fontsize=14)
        axes[1].set_xlabel("Time index i")
        axes[1].set_ylabel("Time index j")
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig('dolphin_recurrence_plot.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_recurrence_plot.png")

    def generate_horizon_chart(self):
        print("\n16. Generating Enhanced Horizon Chart...")
        
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1
        window = max(1, int(5 / dt))
        
        df = pd.DataFrame(self.features[self.window_size:], columns=['Whistle', 'Burst', 'Click'])
        df_smooth = df.rolling(window, center=True).mean().fillna(0)
        t = (self.valid_times - self.valid_times[0]) / 60
        
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(5, 3, height_ratios=[2, 2, 2, 1.5, 1.5], 
                              width_ratios=[3, 1, 1], hspace=0.3, wspace=0.25)
        
        channel_colors = {
            'Whistle': ['#c6dbef', '#6baed6', '#2171b5', '#08306b'],
            'Burst': ['#fcbba1', '#fb6a4a', '#cb181d', '#67000d'],
            'Click': ['#c7e9c0', '#74c476', '#238b45', '#00441b']
        }
        channel_base = {'Whistle': '#2171b5', 'Burst': '#cb181d', 'Click': '#238b45'}
        chans = ['Whistle', 'Burst', 'Click']
        
        n_bands = 4
        
        for idx, chan in enumerate(chans):
            ax = fig.add_subplot(gs[idx, 0])
            
            d = df_smooth[chan].values
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            
            colors = channel_colors[chan]
            for b in range(n_bands):
                band_height = 1 / n_bands
                d_band = np.clip(d_norm - b * band_height, 0, band_height)
                ax.fill_between(t, 0, d_band * n_bands, color=colors[b], alpha=0.85)
            
            ax.set_ylabel(f'{chan}\nIntensity', fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_xlim(t[0], t[-1])
            
            if idx < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (min)', fontsize=10)
        
        # Distribution histograms
        for idx, chan in enumerate(chans):
            ax = fig.add_subplot(gs[idx, 1])
            d = df_smooth[chan].values
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            ax.hist(d_norm, bins=30, orientation='horizontal', color=channel_base[chan], alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Count', fontsize=9)
            ax.set_title('Distribution', fontsize=9)
        
        # Correlation panel
        ax_corr = fig.add_subplot(gs[3, :2])
        corr_window = int(60 / dt)
        
        corr_wb, corr_wc, corr_bc = [], [], []
        corr_times = []
        
        step = max(1, corr_window // 4)
        for i in range(0, len(df_smooth) - corr_window, step):
            chunk = df_smooth.iloc[i:i+corr_window]
            corr_wb.append(chunk['Whistle'].corr(chunk['Burst']))
            corr_wc.append(chunk['Whistle'].corr(chunk['Click']))
            corr_bc.append(chunk['Burst'].corr(chunk['Click']))
            corr_times.append(t[i + corr_window//2])
        
        ax_corr.plot(corr_times, corr_wb, color='#9467bd', lw=1.5, label='Whistle-Burst', alpha=0.8)
        ax_corr.plot(corr_times, corr_wc, color='#17becf', lw=1.5, label='Whistle-Click', alpha=0.8)
        ax_corr.plot(corr_times, corr_bc, color='#bcbd22', lw=1.5, label='Burst-Click', alpha=0.8)
        ax_corr.axhline(0, color='gray', linestyle='-', lw=1, alpha=0.5)
        ax_corr.set_ylabel('Correlation (r)', fontsize=10)
        ax_corr.set_xlabel('Time (min)', fontsize=10)
        ax_corr.set_ylim(-1, 1)
        ax_corr.legend(loc='upper right', fontsize=8, ncol=3)
        ax_corr.set_title('Cross-Channel Correlation (60s rolling window)', fontsize=11, fontweight='bold')
        ax_corr.grid(True, alpha=0.3)
        
        # Activity index
        ax_act = fig.add_subplot(gs[4, :2])
        activity_index = (df_smooth['Whistle'] + df_smooth['Burst'] + df_smooth['Click']) / 3
        activity_norm = (activity_index - activity_index.min()) / (activity_index.max() - activity_index.min() + 1e-8)
        
        ax_act.fill_between(t, 0, activity_norm, color='#756bb1', alpha=0.4, label='Activity Index')
        ax_act.set_ylabel('Activity\nIndex', fontsize=10)
        ax_act.set_xlabel('Time (min)', fontsize=10)
        ax_act.set_ylim(0, 1)
        ax_act.set_xlim(t[0], t[-1])
        ax_act.set_title('Combined Acoustic Activity', fontsize=11, fontweight='bold')
        
        plt.suptitle('Enhanced Horizon Chart: Multi-Channel Acoustic Intensity Analysis',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_horizon_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_horizon_chart.png")

    def generate_chord_diagram(self):
        print("\n17. Generating Chord Diagram...")

        n = len(self.existing_clusters)
        mat = np.zeros((n, n))
        c2i = {c: i for i, c in enumerate(self.existing_clusters)}
        for k in range(len(self.clusters) - 1):
            mat[c2i[self.clusters[k]], c2i[self.clusters[k + 1]]] += 1
        mat = mat / mat.sum() * 100

        fig, ax = plt.subplots(figsize=(10, 10))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        width = 2 * np.pi / n * 0.8

        for i, cid in enumerate(self.existing_clusters):
            theta = angles[i]
            ax.bar(theta, 1, width=width, bottom=10, color=self.all_colors[cid], alpha=0.8)
            label_short = self.labels[cid].split('[')[0].strip()[:12]
            ax.text(theta, 11.5, label_short, rotation=np.degrees(theta), ha='center', va='center', fontsize=8)

        thresh = np.percentile(mat[mat > 0], 60) if np.any(mat > 0) else 0
        for i in range(n):
            for j in range(n):
                if mat[i, j] > thresh and i != j:
                    x1, y1 = 10 * np.cos(angles[i]), 10 * np.sin(angles[i])
                    x2, y2 = 10 * np.cos(angles[j]), 10 * np.sin(angles[j])
                    verts = [(x1, y1), (0, 0), (x2, y2)]
                    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                    path = Path(verts, codes)
                    ax.add_patch(
                        PathPatch(path, facecolor='none', edgecolor=self.all_colors[self.existing_clusters[i]],
                                  alpha=0.3))

        ax.set_xlim(-12, 12)
        ax.set_ylim(-12, 12)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('dolphin_chord_diagram.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_chord_diagram.png")

    def generate_spectrogram_overlay(self):
        print("\n18. Generating Spectrogram Overlay...")
        dur = 120
        start = len(self.y) // 2 - int(dur / 2 * self.sr)
        if start < 0:
            start = 0
        end = start + int(dur * self.sr)
        y_seg = self.y[start:end]

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_seg)), ref=np.max)
        times_spec = np.linspace(0, dur, D.shape[1])

        fig, ax = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log', ax=ax[0])

        real_start = start / self.sr + self.start_offset_sec
        real_end = end / self.sr + self.start_offset_sec
        mask = (self.valid_times >= real_start) & (self.valid_times <= real_end)
        seg_times = self.valid_times[mask] - real_start
        seg_clust = self.clusters[mask]

        for i in range(len(seg_times) - 1):
            ax[1].axvspan(seg_times[i], seg_times[i + 1], color=self.all_colors[seg_clust[i]], alpha=0.8)
        ax[1].set_yticks([])
        plt.tight_layout()
        plt.savefig('dolphin_spectrogram_overlay.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_spectrogram_overlay.png")

    def generate_polar_histogram(self):
        print("\n19. Generating Polar Histogram...")
        n_sect = 24
        dur = self.valid_times[-1] - self.valid_times[0]
        sect_dur = dur / n_sect
        counts = {i: np.zeros(n_sect) for i in self.existing_clusters}

        for t, c in zip(self.valid_times, self.clusters):
            idx = min(int((t - self.valid_times[0]) / sect_dur), n_sect - 1)
            counts[c][idx] += 1

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        angles = np.linspace(0, 2 * np.pi, n_sect, endpoint=False)
        width = 2 * np.pi / n_sect * 0.9
        bottom = np.zeros(n_sect)

        for cid in self.existing_clusters:
            vals = counts[cid]
            vals = vals / (np.sum(list(counts.values()), axis=0) + 1e-8)
            ax.bar(angles, vals, width=width, bottom=bottom, color=self.all_colors[cid], alpha=0.8)
            bottom += vals

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticklabels([])
        plt.savefig('dolphin_polar_histogram.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_polar_histogram.png")

    def generate_phase_portrait(self):
        print("\n20. Generating Phase Portrait...")
        pca = PCA(n_components=2)
        l2d = pca.fit_transform(self.latent_space)
        x, y = l2d[:, 0], l2d[:, 1]
        dx = gaussian_filter1d(np.gradient(x), 5)
        dy = gaussian_filter1d(np.gradient(y), 5)

        fig, ax = plt.subplots(figsize=(10, 8))
        st = max(1, len(x) // 500)
        ax.quiver(x[::st], y[::st], dx[::st], dy[::st], np.sqrt(dx ** 2 + dy ** 2)[::st], cmap='hot', alpha=0.7)
        ax.set_title("Phase Portrait")
        plt.savefig('dolphin_phase_portrait.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_phase_portrait.png")

    def generate_entropy_plot(self):
        print("\n21. Generating Entropy Analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1
        window_sec = 60
        ws = max(10, int(window_sec / dt))
        n_states = len(self.existing_clusters)

        ents = []
        times_ent = []
        step = max(1, ws // 4)

        for i in range(0, len(self.clusters) - ws, step):
            chunk = self.clusters[i:i + ws]
            counts = np.bincount(chunk.astype(int), minlength=max(self.existing_clusters) + 1)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            ents.append(h)
            times_ent.append(self.valid_times[i + ws // 2])

        if not ents:
            print("   ! Not enough data for entropy calculation")
            plt.close()
            return

        ents = np.array(ents)
        times_ent = np.array(times_ent)
        ents_smooth = gaussian_filter1d(ents, sigma=3)
        tm = (times_ent - self.valid_times[0]) / 60

        max_ent = np.log2(n_states)
        mean_ent = np.mean(ents)

        ax1 = axes[0, 0]
        ax1.fill_between(tm, 0, ents_smooth, color='#6a0dad', alpha=0.3)
        ax1.plot(tm, ents_smooth, color='#6a0dad', lw=2, label='Shannon Entropy')
        ax1.axhline(max_ent, color='red', linestyle='--', lw=1.5, label=f'Max H = {max_ent:.2f} bits')
        ax1.axhline(mean_ent, color='orange', linestyle=':', lw=1.5, label=f'Mean H = {mean_ent:.2f} bits')
        ax1.set_xlabel('Time (min)', fontsize=11)
        ax1.set_ylabel('Entropy (bits)', fontsize=11)
        ax1.set_title('Behavioral Complexity (Rolling Entropy)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        ax2.hist(ents, bins=40, color='#6a0dad', edgecolor='black', alpha=0.7, density=True)
        ax2.axvline(mean_ent, color='red', linestyle='--', lw=2, label=f'Mean: {mean_ent:.2f}')
        ax2.set_xlabel('Entropy (bits)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Entropy Distribution', fontsize=12, fontweight='bold')
        ax2.legend()

        ax3 = axes[1, 0]
        switch_rate = []
        switch_times = []
        for i in range(0, len(self.clusters) - ws, step):
            chunk = self.clusters[i:i + ws]
            switches = np.sum(np.diff(chunk.astype(int)) != 0)
            rate = switches / ws * 100
            switch_rate.append(rate)
            switch_times.append(self.valid_times[i + ws // 2])

        switch_rate = gaussian_filter1d(switch_rate, sigma=3)
        st = (np.array(switch_times) - self.valid_times[0]) / 60

        ax3.fill_between(st, 0, switch_rate, color='#e74c3c', alpha=0.3)
        ax3.plot(st, switch_rate, color='#c0392b', lw=2)
        ax3.set_xlabel('Time (min)', fontsize=11)
        ax3.set_ylabel('Switching Rate (%)', fontsize=11)
        ax3.set_title('Behavioral Switching Frequency', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        activity_vals = []
        entropy_vals = []
        for i in range(0, len(self.clusters) - ws, step):
            chunk = self.clusters[i:i + ws]
            counts = np.bincount(chunk.astype(int), minlength=max(self.existing_clusters) + 1)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            entropy_vals.append(h)
            feat_chunk = self.features[self.window_size + i:self.window_size + i + ws]
            if len(feat_chunk) > 0:
                activity = np.mean(feat_chunk)
            else:
                activity = 0
            activity_vals.append(activity)

        colors = np.linspace(0, 1, len(entropy_vals))
        scatter = ax4.scatter(activity_vals, entropy_vals, c=colors, cmap='viridis', s=30, alpha=0.6)
        ax4.set_xlabel('Mean Acoustic Activity', fontsize=11)
        ax4.set_ylabel('Entropy (bits)', fontsize=11)
        ax4.set_title('Complexity vs Activity Phase Space', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Time (relative)')

        plt.suptitle('Cognitive Complexity Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_entropy_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   -> SAVED: dolphin_entropy_plot.png")

    def generate_voronoi_map(self):
        print("\n22. Generating Voronoi Map...")
        pca = PCA(n_components=2)
        l2d = pca.fit_transform(self.latent_space)
        centroids = []
        c_ids = []
        for i in self.existing_clusters:
            mask = self.clusters == i
            if np.sum(mask) > 0:
                centroids.append(np.mean(l2d[mask], axis=0))
                c_ids.append(i)

        if len(centroids) < 3:
            print("   ! Not enough clusters for Voronoi")
            return

        centroids = np.array(centroids)

        fig, ax = plt.subplots(figsize=(12, 10))

        x_min, x_max = l2d[:, 0].min() - 1, l2d[:, 0].max() + 1
        y_min, y_max = l2d[:, 1].min() - 1, l2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        dists = np.linalg.norm(np.c_[xx.ravel(), yy.ravel()][:, np.newaxis] - centroids, axis=2)
        regions = np.argmin(dists, axis=1).reshape(xx.shape)

        cmap = matplotlib.colors.ListedColormap([self.all_colors[i] for i in c_ids])
        ax.imshow(regions, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap=cmap, alpha=0.3, aspect='auto')
        ax.scatter(l2d[:, 0], l2d[:, 1], c='k', s=1, alpha=0.1)

        for i, cent in enumerate(centroids):
            ax.scatter(cent[0], cent[1], s=200, c=[self.all_colors[c_ids[i]]], marker='o', edgecolors='black', lw=2)
            label_short = self.labels[c_ids[i]].split('[')[0].strip()[:12]
            ax.text(cent[0], cent[1] + 0.3, label_short, ha='center', fontsize=8, fontweight='bold')

        ax.set_title("Behavioral Territories (Voronoi)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        plt.tight_layout()
        plt.savefig('dolphin_voronoi_map.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_voronoi_map.png")

    def generate_ridge_plot(self):
        print("\n23. Generating Ridge Plot...")
        recs = []
        step = max(1, len(self.clusters) // 5000)
        for i in range(0, len(self.clusters), step):
            cid = self.clusters[i]
            if cid not in self.existing_clusters:
                continue
            row = self.features[self.window_size:][i]
            recs.append({'Val': row[0], 'Feat': 'Whistle', 'Clust': self.labels[cid]})
            recs.append({'Val': row[1], 'Feat': 'Burst', 'Clust': self.labels[cid]})
            recs.append({'Val': row[2], 'Feat': 'Click', 'Clust': self.labels[cid]})

        if not recs:
            print("   ! No data for ridge plot")
            return

        df = pd.DataFrame(recs)

        try:
            g = sns.FacetGrid(df, row="Clust", hue="Feat", aspect=5, height=1.2, palette="bright")
            g.map(sns.kdeplot, "Val", fill=True, alpha=0.6)
            g.map(plt.axhline, y=0, lw=2)
            g.set(yticks=[])
            g.despine(bottom=True, left=True)
            plt.savefig('dolphin_ridge_plot.png', dpi=300)
            plt.close()
            print("   -> SAVED: dolphin_ridge_plot.png")
        except Exception as e:
            print(f"   ! Ridge plot error: {e}")

    def generate_sunburst_sequence(self):
        print("\n24. Generating Sunburst...")
        if not HAS_PLOTLY:
            print("   ! plotly not available, skipping.")
            return

        seq = [self.clusters[0]]
        for c in self.clusters[1:]:
            if c != seq[-1]:
                seq.append(c)

        if len(seq) < 4:
            print("   ! Not enough transitions for sunburst")
            return

        grams = []
        for i in range(len(seq) - 2):
            grams.append([self.labels[seq[i]], self.labels[seq[i + 1]], self.labels[seq[i + 2]]])

        df = pd.DataFrame(grams, columns=['S1', 'S2', 'S3'])
        df['C'] = 1

        try:
            fig = px.sunburst(df, path=['S1', 'S2', 'S3'], values='C')
            fig.write_html("dolphin_sequence_sunburst.html")
            print("   -> SAVED: dolphin_sequence_sunburst.html")
        except Exception as e:
            print(f"   ! Sunburst error: {e}")

    # =========================================================================
    # VISUALIZATION 25: ICI ANALYSIS
    # =========================================================================
    def generate_ici_analysis(self):
        print("\n25. Generating Enhanced ICI Analysis...")

        clicks = self.features[self.window_size:, 2]
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1

        threshold_high = np.percentile(clicks, 85)
        threshold_low = np.percentile(clicks, 70)
        threshold_high = max(threshold_high, 0.15)
        threshold_low = max(threshold_low, 0.10)
        
        click_indices = []
        in_click = False
        for i in range(len(clicks)):
            if not in_click and clicks[i] >= threshold_high:
                click_indices.append(i)
                in_click = True
            elif in_click and clicks[i] < threshold_low:
                in_click = False
        
        click_indices = np.array(click_indices)
        
        if len(click_indices) < 100:
            threshold_high = np.percentile(clicks, 75)
            threshold_low = np.percentile(clicks, 60)
            threshold_high = max(threshold_high, 0.10)
            threshold_low = max(threshold_low, 0.05)
            
            click_indices = []
            in_click = False
            for i in range(len(clicks)):
                if not in_click and clicks[i] >= threshold_high:
                    click_indices.append(i)
                    in_click = True
                elif in_click and clicks[i] < threshold_low:
                    in_click = False
            click_indices = np.array(click_indices)
        
        if len(click_indices) < 30:
            print("   ! Not enough click events detected - skipping ICI analysis")
            return

        icis = np.diff(click_indices)
        click_times = self.valid_times[click_indices[1:]]
        ici_ms = icis * dt * 1000

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # Panel 1: ICI Histogram
        ax1 = fig.add_subplot(gs[0, 0:2])
        ax1.hist(ici_ms, bins=60, color='#3498db', edgecolor='white', alpha=0.7, density=True)
        ax1.axvline(np.median(ici_ms), color='orange', linestyle=':', lw=2, label=f'Median: {np.median(ici_ms):.1f} ms')
        ax1.set_xlabel('Inter-Click Interval (ms)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('ICI Distribution', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xlim(0, np.percentile(ici_ms, 98))
        ax1.grid(True, alpha=0.3)

        # Panel 2: ICI Categories
        ax2 = fig.add_subplot(gs[0, 2:4])
        short_thresh = 50
        long_thresh = 200
        
        short_ici = ici_ms[ici_ms < short_thresh]
        medium_ici = ici_ms[(ici_ms >= short_thresh) & (ici_ms < long_thresh)]
        long_ici = ici_ms[ici_ms >= long_thresh]
        
        categories = ['Terminal Buzz\n(<50ms)', 'Regular Echo\n(50-200ms)', 'Social/Pause\n(>200ms)']
        counts = [len(short_ici), len(medium_ici), len(long_ici)]
        percentages = [c/len(ici_ms)*100 for c in counts]
        colors_cat = ['#e74c3c', '#3498db', '#2ecc71']
        
        bars = ax2.bar(categories, counts, color=colors_cat, edgecolor='black', alpha=0.8)
        for bar, pct in zip(bars, percentages):
            ax2.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('ICI Functional Categories', fontsize=12, fontweight='bold')

        # Panel 3: ICI over time
        ax3 = fig.add_subplot(gs[1, :3])
        scatter = ax3.scatter(click_times, ici_ms, c=np.arange(len(ici_ms)), cmap='viridis', s=8, alpha=0.5)
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('ICI (ms)', fontsize=11)
        ax3.set_title(f'ICI Temporal Evolution (n={len(ici_ms)} intervals)', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, np.percentile(ici_ms, 95))
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, shrink=0.6, label='Sequence Order')

        # Panel 4: Click rate
        ax4 = fig.add_subplot(gs[1, 3])
        window_sec = 10
        rate_times = []
        click_rates = []
        time_start = self.valid_times[0]
        time_end = self.valid_times[-1]
        
        for t_center in np.arange(time_start + window_sec/2, time_end - window_sec/2, window_sec/4):
            mask = (click_times >= t_center - window_sec/2) & (click_times < t_center + window_sec/2)
            rate = mask.sum() / window_sec
            rate_times.append(t_center)
            click_rates.append(rate)
        
        ax4.fill_between(rate_times, 0, click_rates, color='#9b59b6', alpha=0.5)
        ax4.plot(rate_times, click_rates, color='#8e44ad', lw=2)
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Click Rate (Hz)', fontsize=10)
        ax4.set_title('Click Rate', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Panel 5: ICI Return Map
        ax5 = fig.add_subplot(gs[2, 0:2])
        if len(icis) > 1:
            ici_n = ici_ms[:-1]
            ici_n1 = ici_ms[1:]
            ax5.scatter(ici_n, ici_n1, c='cyan', s=3, alpha=0.3)
            max_val = np.percentile(ici_ms, 95)
            ax5.plot([0, max_val], [0, max_val], 'r--', lw=2, label='No Change')
            ax5.set_xlabel('ICI(n) (ms)', fontsize=11)
            ax5.set_ylabel('ICI(n+1) (ms)', fontsize=11)
            ax5.set_title('ICI Return Map', fontsize=12, fontweight='bold')
            ax5.legend(loc='upper right', fontsize=8)
            ax5.set_xlim(0, max_val)
            ax5.set_ylim(0, max_val)

        # Panel 6: Click train analysis
        train_threshold = 150
        in_train = ici_ms < train_threshold
        train_lengths = []
        i = 0
        while i < len(in_train):
            if in_train[i]:
                start = i
                while i < len(in_train) and in_train[i]:
                    i += 1
                if i - start >= 3:
                    train_lengths.append(i - start)
            else:
                i += 1

        ax7 = fig.add_subplot(gs[3, 0:2])
        if train_lengths:
            ax7.hist(train_lengths, bins=30, color='#1abc9c', edgecolor='black', alpha=0.7)
            ax7.axvline(np.median(train_lengths), color='red', linestyle='--', lw=2,
                       label=f'Median: {np.median(train_lengths):.0f} clicks')
            ax7.set_xlabel('Click Train Length (# clicks)', fontsize=11)
            ax7.set_ylabel('Frequency', fontsize=11)
            ax7.set_title(f'Click Train Length Distribution (n={len(train_lengths)} trains)', fontsize=12, fontweight='bold')
            ax7.legend(fontsize=9)
            ax7.grid(True, alpha=0.3)

        # Panel 9: Statistics
        ax9 = fig.add_subplot(gs[3, 3])
        ax9.axis('off')
        stats_text = f"""
ICI STATISTICS
══════════════════════
Total Clicks:    {len(click_indices)}
Total Intervals: {len(ici_ms)}

ICI (ms):
  Mean:          {np.mean(ici_ms):.1f}
  Median:        {np.median(ici_ms):.1f}
  Std:           {np.std(ici_ms):.1f}

Categories:
  Terminal Buzz: {percentages[0]:.1f}%
  Regular Echo:  {percentages[1]:.1f}%
  Social/Pause:  {percentages[2]:.1f}%

Click Trains:
  Count:         {len(train_lengths)}
  Mean Length:   {np.mean(train_lengths) if train_lengths else 0:.1f}
"""
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=9, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Enhanced Inter-Click Interval (ICI) Analysis', fontsize=14, fontweight='bold', y=0.98)
        plt.savefig('dolphin_ici_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_ici_analysis.png")

    # =========================================================================
    # VISUALIZATION 26: BOUT ANALYSIS
    # =========================================================================
    def generate_bout_analysis(self):
        print("\n26. Generating Enhanced Bout Analysis...")

        clusters = self.clusters
        times = self.valid_times
        features = self.features[self.window_size:]
        dt = times[1] - times[0] if len(times) > 1 else 0.1

        min_bout_duration = 3
        bouts = []
        current_state = clusters[0]
        bout_start = 0

        for i in range(1, len(clusters)):
            if clusters[i] != current_state:
                duration = i - bout_start
                if duration >= min_bout_duration:
                    bout_features = features[bout_start:i]
                    bouts.append({
                        'state': current_state,
                        'start_idx': bout_start,
                        'end_idx': i,
                        'duration': duration,
                        'duration_sec': duration * dt,
                        'start_time': times[bout_start],
                        'total_energy': np.sum(bout_features) if len(bout_features) > 0 else 0,
                    })
                current_state = clusters[i]
                bout_start = i
        
        duration = len(clusters) - bout_start
        if duration >= min_bout_duration:
            bout_features = features[bout_start:]
            bouts.append({
                'state': current_state,
                'duration': duration,
                'duration_sec': duration * dt,
                'start_time': times[bout_start],
                'total_energy': np.sum(bout_features) if len(bout_features) > 0 else 0,
            })

        if len(bouts) < 5:
            min_bout_duration = 1
            bouts = []
            current_state = clusters[0]
            bout_start = 0
            
            for i in range(1, len(clusters)):
                if clusters[i] != current_state:
                    duration = i - bout_start
                    if duration >= min_bout_duration:
                        bout_features = features[bout_start:i]
                        bouts.append({
                            'state': current_state,
                            'duration': duration,
                            'duration_sec': duration * dt,
                            'start_time': times[bout_start],
                            'total_energy': np.sum(bout_features) if len(bout_features) > 0 else 0,
                        })
                    current_state = clusters[i]
                    bout_start = i
        
        if len(bouts) < 3:
            print("   ! Not enough bouts detected - skipping bout analysis")
            return

        bout_df = pd.DataFrame(bouts)
        
        ibis = []
        for i in range(1, len(bouts)):
            ibi = bouts[i]['start_time'] - (bouts[i-1]['start_time'] + bouts[i-1]['duration_sec'])
            if ibi > 0:
                ibis.append(ibi)

        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)

        # Panel 1: Duration distribution
        ax1 = fig.add_subplot(gs[0, 0:2])
        durations_sec = bout_df['duration_sec'].values
        ax1.hist(durations_sec, bins=40, color='#3498db', edgecolor='white', alpha=0.7, density=True)
        ax1.axvline(np.median(durations_sec), color='green', linestyle='--', lw=2, label=f'Median: {np.median(durations_sec):.2f}s')
        ax1.axvline(np.mean(durations_sec), color='orange', linestyle=':', lw=2, label=f'Mean: {np.mean(durations_sec):.2f}s')
        ax1.set_xlabel('Bout Duration (seconds)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Bout Duration Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Survival curve
        ax2 = fig.add_subplot(gs[0, 2:4])
        sorted_durations = np.sort(durations_sec)
        survival_prob = 1 - np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        ax2.step(sorted_durations, survival_prob, where='post', color='#e74c3c', lw=2)
        ax2.fill_between(sorted_durations, 0, survival_prob, step='post', alpha=0.3, color='#e74c3c')
        ax2.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax2.set_xlabel('Bout Duration (seconds)', fontsize=11)
        ax2.set_ylabel('Survival Probability', fontsize=11)
        ax2.set_title('Bout Survival Curve', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Mean duration by state
        ax3 = fig.add_subplot(gs[1, 0:2])
        state_stats = bout_df.groupby('state').agg({'duration_sec': ['mean', 'std', 'count']}).reset_index()
        state_stats.columns = ['state', 'mean', 'std', 'count']
        state_stats = state_stats.sort_values('mean', ascending=True)
        
        y_pos = np.arange(len(state_stats))
        colors_bar = [self.all_colors[int(s)] for s in state_stats['state']]
        labels_bar = [self.labels[int(s)].split('[')[0].strip()[:15] for s in state_stats['state']]
        
        ax3.barh(y_pos, state_stats['mean'], xerr=state_stats['std'], color=colors_bar, edgecolor='black', alpha=0.8, capsize=3)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels_bar, fontsize=9)
        ax3.set_xlabel('Mean Bout Duration (seconds)', fontsize=11)
        ax3.set_title('Bout Duration by Behavioral State', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # Panel 4: IBI distribution
        ax4 = fig.add_subplot(gs[1, 2:4])
        if len(ibis) > 5:
            ax4.hist(ibis, bins=40, color='#9b59b6', edgecolor='white', alpha=0.7)
            ax4.axvline(np.median(ibis), color='red', linestyle='--', lw=2, label=f'Median IBI: {np.median(ibis):.2f}s')
            ax4.set_xlabel('Inter-Bout Interval (seconds)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Inter-Bout Interval Distribution', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)

        # Panel 5: Statistics
        ax9 = fig.add_subplot(gs[3, 3])
        ax9.axis('off')
        
        total_bouts = len(bouts)
        total_time_sec = times[-1] - times[0]
        mean_bouts_per_min = total_bouts / (total_time_sec / 60)
        
        stats_text = f"""
BOUT STATISTICS
══════════════════════
Total Bouts:      {total_bouts}
Recording Time:   {total_time_sec:.1f} s
Bouts/min:        {mean_bouts_per_min:.2f}

DURATION (sec):
  Mean:           {np.mean(durations_sec):.2f}
  Median:         {np.median(durations_sec):.2f}
  Std:            {np.std(durations_sec):.2f}
  Min:            {np.min(durations_sec):.2f}
  Max:            {np.max(durations_sec):.2f}

INTER-BOUT (sec):
  Mean IBI:       {np.mean(ibis) if ibis else 0:.2f}
  Median IBI:     {np.median(ibis) if ibis else 0:.2f}
"""
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes, fontsize=8, verticalalignment='top', fontfamily='monospace')

        plt.suptitle('Enhanced Behavioral Bout Analysis', fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('dolphin_bout_analysis.png', dpi=200, facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_bout_analysis.png")

    # =========================================================================
    # VISUALIZATION 27: MARKOV ANALYSIS
    # =========================================================================
    def generate_markov_analysis(self):
        print("\n27. Generating Markov Analysis...")

        n_states = max(self.existing_clusters) + 1
        transitions = np.zeros((n_states, n_states))
        for i in range(len(self.clusters) - 1):
            transitions[self.clusters[i], self.clusters[i + 1]] += 1

        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = transitions / row_sums

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        ax1 = axes[0, 0]
        P_existing = P[np.ix_(self.existing_clusters, self.existing_clusters)]
        im = ax1.imshow(P_existing, cmap='Blues', vmin=0, vmax=1)
        ax1.set_xticks(range(len(self.existing_clusters)))
        ax1.set_yticks(range(len(self.existing_clusters)))
        labels_short = [self.labels[i].split('[')[0].strip()[:12] for i in self.existing_clusters]
        ax1.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(labels_short, fontsize=8)
        ax1.set_title('Transition Matrix P', fontweight='bold')
        plt.colorbar(im, ax=ax1, shrink=0.8)

        ax2 = axes[0, 1]
        pi = np.ones(n_states) / n_states
        for _ in range(100):
            pi = pi @ P
        empirical = np.bincount(self.clusters, minlength=n_states) / len(self.clusters)

        x_pos = np.arange(len(self.existing_clusters))
        width = 0.35
        pi_existing = [pi[i] for i in self.existing_clusters]
        emp_existing = [empirical[i] for i in self.existing_clusters]

        ax2.bar(x_pos - width / 2, emp_existing, width, label='Empirical', color='steelblue', alpha=0.8)
        ax2.bar(x_pos + width / 2, pi_existing, width, label='Stationary π', color='coral', alpha=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Probability')
        ax2.set_title('Stationary vs Empirical Distribution', fontweight='bold')
        ax2.legend()

        ax3 = axes[1, 0]
        eigenvalues = np.linalg.eigvals(P)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        ax3.bar(range(1, min(len(eigenvalues_sorted), 10) + 1), eigenvalues_sorted[:10], color='purple', alpha=0.7)
        ax3.axhline(1, color='red', linestyle='--', label='λ=1')
        ax3.set_xlabel('Eigenvalue Index')
        ax3.set_ylabel('|λ|')
        ax3.set_title('Eigenvalue Spectrum', fontweight='bold')
        ax3.legend()

        ax4 = axes[1, 1]
        self_trans = [P[i, i] for i in self.existing_clusters]
        colors_bar = [self.all_colors[i] for i in self.existing_clusters]
        ax4.bar(x_pos, self_trans, color=colors_bar, edgecolor='black', alpha=0.8)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
        ax4.set_ylabel('P(stay)')
        ax4.set_title('State Persistence (Self-transition)', fontweight='bold')
        ax4.set_ylim(0, 1)

        plt.suptitle('Markov Chain Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_markov_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   -> SAVED: dolphin_markov_analysis.png")

    # =========================================================================
    # VISUALIZATION 28: SOCIAL NETWORK
    # =========================================================================
    def generate_vocal_interaction_analysis(self):
        """
        Generate REAL vocal interaction analysis based on actual acoustic data.
        
        This analyzes acoustic EVENT patterns (not individual dolphin identification,
        which is impossible with single-hydrophone recording).
        
        Computes:
        - Vocal event detection (whistles, bursts, click trains)
        - Inter-vocalization intervals (IVI)
        - Call-response patterns
        - Event transition network
        - Turn-taking proxy metrics
        """
        print("\n28. Generating Vocal Interaction Analysis (REAL DATA)...")
        
        features = self.features[self.window_size:]
        times = self.valid_times
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        # =====================================================================
        # 1. DETECT VOCAL EVENTS
        # =====================================================================
        print("   -> Detecting vocal events...")
        
        events = []
        event_colors = {'WHISTLE': '#2ecc71', 'BURST': '#e74c3c', 'CLICK_TRAIN': '#3498db'}
        
        # Adaptive thresholds
        w_high = np.percentile(features[:, 0], 70)
        w_low = np.percentile(features[:, 0], 55)
        b_high = np.percentile(features[:, 1], 65)
        b_low = np.percentile(features[:, 1], 50)
        c_high = np.percentile(features[:, 2], 75)
        c_low = np.percentile(features[:, 2], 60)
        
        min_duration = 3  # samples
        
        for channel_idx, (name, high, low) in enumerate([
            ('WHISTLE', w_high, w_low),
            ('BURST', b_high, b_low),
            ('CLICK_TRAIN', c_high, c_low)
        ]):
            signal = features[:, channel_idx]
            in_event = False
            event_start = 0
            
            for i in range(len(signal)):
                if not in_event and signal[i] >= high:
                    in_event = True
                    event_start = i
                elif in_event and signal[i] < low:
                    in_event = False
                    duration = i - event_start
                    if duration >= min_duration:
                        events.append({
                            'type': name,
                            'start_idx': event_start,
                            'end_idx': i,
                            'start_time': times[event_start],
                            'end_time': times[i],
                            'duration': duration * dt,
                            'peak_energy': np.max(signal[event_start:i]),
                            'mean_energy': np.mean(signal[event_start:i])
                        })
            
            if in_event:
                duration = len(signal) - event_start
                if duration >= min_duration:
                    events.append({
                        'type': name,
                        'start_idx': event_start,
                        'end_idx': len(signal) - 1,
                        'start_time': times[event_start],
                        'end_time': times[-1],
                        'duration': duration * dt,
                        'peak_energy': np.max(signal[event_start:]),
                        'mean_energy': np.mean(signal[event_start:])
                    })
        
        # Sort by time
        events = sorted(events, key=lambda x: x['start_time'])
        
        print(f"      Total events: {len(events)}")
        for etype in ['WHISTLE', 'BURST', 'CLICK_TRAIN']:
            count = sum(1 for e in events if e['type'] == etype)
            print(f"        {etype}: {count}")
        
        if len(events) < 10:
            print("   ! Not enough events for interaction analysis")
            return
        
        # =====================================================================
        # 2. COMPUTE FLOOR TRANSFER OFFSETS (FTO)
        # =====================================================================
        print("   -> Computing inter-event timing...")
        
        ftos = []
        for i in range(len(events) - 1):
            fto = events[i + 1]['start_time'] - events[i]['end_time']
            ftos.append(fto)
        ftos = np.array(ftos)
        
        gaps = ftos[ftos > 0.05]
        overlaps = ftos[ftos < -0.05]
        near_zero = ftos[(ftos >= -0.05) & (ftos <= 0.05)]
        
        gap_ratio = len(gaps) / len(ftos) if len(ftos) > 0 else 0
        overlap_ratio = len(overlaps) / len(ftos) if len(ftos) > 0 else 0
        
        print(f"      Mean FTO: {np.mean(ftos)*1000:.1f} ms")
        print(f"      Gap ratio: {gap_ratio*100:.1f}%")
        print(f"      Overlap ratio: {overlap_ratio*100:.1f}%")
        
        # =====================================================================
        # 3. EVENT TRANSITION NETWORK
        # =====================================================================
        print("   -> Building event transition network...")
        
        event_types = ['WHISTLE', 'BURST', 'CLICK_TRAIN']
        n_types = len(event_types)
        type_to_idx = {t: i for i, t in enumerate(event_types)}
        
        transitions = np.zeros((n_types, n_types))
        for i in range(len(events) - 1):
            from_type = events[i]['type']
            to_type = events[i + 1]['type']
            transitions[type_to_idx[from_type], type_to_idx[to_type]] += 1
        
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = transitions / row_sums
        
        # Transition entropy
        trans_entropies = []
        for i in range(n_types):
            probs = trans_prob[i]
            probs = probs[probs > 0]
            if len(probs) > 0:
                trans_entropies.append(-np.sum(probs * np.log2(probs)))
        mean_trans_entropy = np.mean(trans_entropies) if trans_entropies else 0
        
        print(f"      Transition entropy: {mean_trans_entropy:.3f} bits")
        
        # =====================================================================
        # 4. CALL-RESPONSE DETECTION
        # =====================================================================
        print("   -> Detecting call-response patterns...")
        
        max_response_time = 2.0  # seconds
        call_responses = []
        
        for i in range(len(events) - 1):
            current = events[i]
            for j in range(i + 1, len(events)):
                next_event = events[j]
                latency = next_event['start_time'] - current['end_time']
                if latency > max_response_time:
                    break
                if latency > 0:
                    call_responses.append({
                        'call_type': current['type'],
                        'response_type': next_event['type'],
                        'latency': latency
                    })
                    break
        
        response_prob = len(call_responses) / (len(events) - 1) if len(events) > 1 else 0
        mean_latency = np.mean([cr['latency'] for cr in call_responses]) if call_responses else 0
        
        print(f"      Call-response pairs: {len(call_responses)}")
        print(f"      Response probability: {response_prob*100:.1f}%")
        print(f"      Mean latency: {mean_latency*1000:.1f} ms")
        
        # =====================================================================
        # 5. BURSTINESS INDEX
        # =====================================================================
        window_size = 10.0  # seconds
        n_windows = max(1, int((times[-1] - times[0]) / window_size))
        events_per_window = []
        
        for w in range(n_windows):
            t_start = times[0] + w * window_size
            t_end = t_start + window_size
            count = sum(1 for e in events if t_start <= e['start_time'] < t_end)
            events_per_window.append(count)
        
        burstiness = np.std(events_per_window) / np.mean(events_per_window) if np.mean(events_per_window) > 0 else 0
        print(f"      Burstiness index: {burstiness:.3f}")
        
        # =====================================================================
        # 6. CREATE COMPREHENSIVE FIGURE
        # =====================================================================
        print("   -> Creating visualization...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)
        
        # Panel 1: Event Timeline (first 2 minutes)
        ax1 = fig.add_subplot(gs[0, :3])
        max_time = min(120, times[-1] - times[0])
        for event in events:
            t_rel = event['start_time'] - times[0]
            if t_rel <= max_time:
                ax1.barh(event['type'], event['duration'], 
                        left=t_rel, color=event_colors[event['type']], 
                        alpha=0.7, height=0.6)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_title('Vocal Event Timeline (First 2 Minutes)', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, max_time)
        
        # Panel 2: Event Counts
        ax2 = fig.add_subplot(gs[0, 3])
        type_counts = {t: sum(1 for e in events if e['type'] == t) for t in event_types}
        bars = ax2.bar(event_types, [type_counts[t] for t in event_types],
                      color=[event_colors[t] for t in event_types], edgecolor='black')
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Event Counts', fontsize=12, fontweight='bold')
        for bar, t in zip(bars, event_types):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(type_counts[t]), ha='center', fontsize=10)
        
        # Panel 3: FTO Distribution
        ax3 = fig.add_subplot(gs[1, 0:2])
        ftos_ms = ftos * 1000
        ftos_clipped = np.clip(ftos_ms, -500, 2000)
        ax3.hist(ftos_clipped, bins=50, color='#9b59b6', edgecolor='white', alpha=0.7)
        ax3.axvline(0, color='red', linestyle='--', lw=2, label='Simultaneous')
        ax3.axvline(np.median(ftos_ms), color='orange', linestyle=':',
                   lw=2, label=f'Median: {np.median(ftos_ms):.0f}ms')
        ax3.set_xlabel('Floor Transfer Offset (ms)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title('Inter-Event Timing Distribution', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=9)
        ax3.set_xlim(-500, 2000)
        
        # Panel 4: Gap/Overlap Pie
        ax4 = fig.add_subplot(gs[1, 2])
        sizes = [len(gaps), len(near_zero), len(overlaps)]
        labels = [f'Gaps\n({len(gaps)})', f'Near-zero\n({len(near_zero)})', f'Overlaps\n({len(overlaps)})']
        pie_colors = ['#27ae60', '#f39c12', '#c0392b']
        ax4.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', textprops={'fontsize': 9})
        ax4.set_title('Timing Categories', fontsize=12, fontweight='bold')
        
        # Panel 5: Burstiness
        ax5 = fig.add_subplot(gs[1, 3])
        ax5.bar(range(len(events_per_window)), events_per_window, color='#8e44ad', alpha=0.7)
        ax5.axhline(np.mean(events_per_window), color='red', linestyle='--',
                   label=f'Mean: {np.mean(events_per_window):.1f}')
        ax5.set_xlabel('Time Window (10s)', fontsize=10)
        ax5.set_ylabel('Events', fontsize=10)
        ax5.set_title(f'Burstiness (CV={burstiness:.2f})', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=8)
        
        # Panel 6: Transition Matrix
        ax6 = fig.add_subplot(gs[2, 0:2])
        im = ax6.imshow(trans_prob, cmap='YlOrRd', vmin=0, vmax=1)
        ax6.set_xticks(range(n_types))
        ax6.set_yticks(range(n_types))
        ax6.set_xticklabels(event_types, fontsize=10)
        ax6.set_yticklabels(event_types, fontsize=10)
        for i in range(n_types):
            for j in range(n_types):
                ax6.text(j, i, f'{trans_prob[i,j]:.2f}', ha='center', va='center',
                        color='white' if trans_prob[i,j] > 0.5 else 'black', fontsize=11)
        ax6.set_title('Event Transition Probabilities', fontsize=12, fontweight='bold')
        ax6.set_xlabel('To Event', fontsize=11)
        ax6.set_ylabel('From Event', fontsize=11)
        plt.colorbar(im, ax=ax6, shrink=0.8)
        
        # Panel 7: Transition Network
        ax7 = fig.add_subplot(gs[2, 2:4])
        if HAS_NETWORKX:
            G = nx.DiGraph()
            for t in event_types:
                G.add_node(t, size=type_counts[t])
            for i, from_t in enumerate(event_types):
                for j, to_t in enumerate(event_types):
                    if trans_prob[i, j] > 0.1:
                        G.add_edge(from_t, to_t, weight=trans_prob[i, j])
            
            pos = nx.circular_layout(G)
            node_sizes = [type_counts[t] * 3 for t in G.nodes()]
            node_colors = [event_colors[t] for t in G.nodes()]
            
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9, ax=ax7)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax7)
            
            for (u, v, d) in G.edges(data=True):
                ax7.annotate('', xy=pos[v], xytext=pos[u],
                           arrowprops=dict(arrowstyle='-|>', color='gray',
                                         lw=d['weight']*5, alpha=0.6,
                                         connectionstyle='arc3,rad=0.2'))
        ax7.set_title('Event Transition Network', fontsize=12, fontweight='bold')
        ax7.axis('off')
        
        # Panel 8: Response Latency
        ax8 = fig.add_subplot(gs[3, 0:2])
        if call_responses:
            latencies_ms = [cr['latency'] * 1000 for cr in call_responses]
            ax8.hist(latencies_ms, bins=30, color='#1abc9c', edgecolor='white', alpha=0.7)
            ax8.axvline(mean_latency*1000, color='red', linestyle='--', lw=2,
                       label=f'Mean: {mean_latency*1000:.0f}ms')
            ax8.set_xlabel('Response Latency (ms)', fontsize=11)
            ax8.set_ylabel('Frequency', fontsize=11)
            ax8.set_title(f'Call-Response Latencies (n={len(call_responses)})', fontsize=12, fontweight='bold')
            ax8.legend(fontsize=9)
        
        # Panel 9: Call-Response Type Matrix
        ax9 = fig.add_subplot(gs[3, 2])
        if call_responses:
            cr_matrix = np.zeros((n_types, n_types))
            for cr in call_responses:
                cr_matrix[type_to_idx[cr['call_type']], type_to_idx[cr['response_type']]] += 1
            im = ax9.imshow(cr_matrix, cmap='Blues')
            ax9.set_xticks(range(n_types))
            ax9.set_yticks(range(n_types))
            ax9.set_xticklabels(event_types, rotation=45, ha='right', fontsize=9)
            ax9.set_yticklabels(event_types, fontsize=9)
            for i in range(n_types):
                for j in range(n_types):
                    ax9.text(j, i, f'{cr_matrix[i,j]:.0f}', ha='center', va='center', fontsize=10)
            ax9.set_title('Call→Response Types', fontsize=11, fontweight='bold')
        
        # Panel 10: Summary Statistics
        ax10 = fig.add_subplot(gs[3, 3])
        ax10.axis('off')
        
        stats_text = f"""
VOCAL INTERACTION SUMMARY
══════════════════════════════
Total Events:     {len(events)}
  Whistles:       {type_counts['WHISTLE']}
  Bursts:         {type_counts['BURST']}
  Click Trains:   {type_counts['CLICK_TRAIN']}

TIMING METRICS
══════════════════════════════
Mean FTO:         {np.mean(ftos)*1000:.1f} ms
Median FTO:       {np.median(ftos)*1000:.1f} ms
Gap Ratio:        {gap_ratio*100:.1f}%
Overlap Ratio:    {overlap_ratio*100:.1f}%
Burstiness:       {burstiness:.3f}

CALL-RESPONSE
══════════════════════════════
Pairs Detected:   {len(call_responses)}
Response Prob:    {response_prob*100:.1f}%
Mean Latency:     {mean_latency*1000:.1f} ms

NETWORK
══════════════════════════════
Trans. Entropy:   {mean_trans_entropy:.3f} bits
"""
        ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('VOCAL INTERACTION ANALYSIS\n(Acoustic Event Patterns - Real Data)',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_vocal_interaction.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_vocal_interaction.png")
        
        # Store results for CSV export
        self.vocal_interaction_results = {
            'events': events,
            'n_events': len(events),
            'type_counts': type_counts,
            'ftos': ftos,
            'mean_fto': np.mean(ftos),
            'gap_ratio': gap_ratio,
            'overlap_ratio': overlap_ratio,
            'burstiness': burstiness,
            'trans_prob': trans_prob,
            'trans_entropy': mean_trans_entropy,
            'call_responses': call_responses,
            'response_prob': response_prob,
            'mean_latency': mean_latency
        }

    # =========================================================================
    # VISUALIZATION 29: WHISTLE CONTOUR CATALOG (SIGNATURE IDENTIFICATION)
    # =========================================================================
    def generate_whistle_catalog(self):
        """
        Extract and cluster whistle contours as signature whistle candidates.
        Uses PYIN pitch tracking on real audio data.
        """
        print("\n29. Generating Whistle Contour Catalog (Signature Proxy)...")
        
        # 1. Whistle detection from features
        whistle_energy = self.features[self.window_size:, 0]
        times = self.valid_times
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        # Threshold
        thresh = np.percentile(whistle_energy, 85)
        
        # Event detection
        contours = []
        in_whistle = False
        start = 0
        
        # Pitch tracking parameters (dolphin whistle range)
        fmin = 2000   # Hz 
        fmax = 20000  # Hz
        
        print("   -> Extracting pitch contours from real audio...")
        
        for i in range(len(whistle_energy)):
            if not in_whistle and whistle_energy[i] > thresh:
                start = i
                in_whistle = True
            elif in_whistle and whistle_energy[i] < thresh * 0.8:
                if i - start > 10:  # Minimum length
                    t_start = times[start]
                    t_end = times[i]
                    
                    # Audio slice indices
                    samp_start = int((t_start - self.start_offset_sec) * self.sr)
                    samp_end = int((t_end - self.start_offset_sec) * self.sr)
                    
                    if 0 <= samp_start < len(self.y) and samp_end <= len(self.y) and samp_end - samp_start > 512:
                        y_slice = self.y[samp_start:samp_end]
                        
                        try:
                            # PYIN pitch tracking (robust algorithm)
                            f0, voiced_flag, _ = librosa.pyin(y_slice, fmin=fmin, fmax=fmax, 
                                                               sr=self.sr, frame_length=1024)
                            
                            # Only voiced segments
                            f0_clean = f0[voiced_flag]
                            
                            if len(f0_clean) > 5:
                                # Normalization (shape matters, not absolute pitch)
                                # 1. Time scaling to fixed length (50 points)
                                from scipy.interpolate import interp1d
                                x_old = np.linspace(0, 1, len(f0_clean))
                                x_new = np.linspace(0, 1, 50)
                                f = interp1d(x_old, f0_clean, kind='linear')
                                contour_norm = f(x_new)
                                
                                # 2. Frequency normalization (z-score)
                                contour_norm = (contour_norm - np.mean(contour_norm)) / (np.std(contour_norm) + 1e-6)
                                
                                contours.append(contour_norm)
                        except Exception as e:
                            pass  # Skip problematic segments
                            
                in_whistle = False
                
        if len(contours) < 10:
            print(f"   ! Only {len(contours)} whistle contours found - insufficient for clustering.")
            return
            
        contours = np.array(contours)
        print(f"   -> Found {len(contours)} robust whistle contours from real audio.")
        
        # 2. Clustering (shape-based grouping)
        # Try to find 4 main types (as proxy for 4 dolphins - but NOT claiming individual ID)
        n_types = min(4, len(contours) // 5)  # At least 5 contours per cluster
        if n_types < 2:
            n_types = 2
            
        kmeans = KMeans(n_clusters=n_types, n_init=10)
        labels = kmeans.fit_predict(contours)
        
        # 3. Visualization
        fig, axes = plt.subplots(1, n_types, figsize=(5*n_types, 5), sharey=True)
        
        if n_types == 1:
            axes = [axes]
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:n_types]
        
        for i in range(n_types):
            ax = axes[i]
            cluster_contours = contours[labels == i]
            count = len(cluster_contours)
            percentage = count / len(contours) * 100
            
            # Plot all contours faintly
            for c in cluster_contours:
                ax.plot(c, color=colors[i], alpha=0.1)
                
            # Plot centroid (mean) boldly
            centroid = np.mean(cluster_contours, axis=0)
            ax.plot(centroid, color='black', linewidth=3, linestyle='--')
            
            ax.set_title(f"TYPE {i+1}\n(n={count}, {percentage:.1f}%)", fontweight='bold', fontsize=12)
            ax.set_xticks([])
            if i == 0:
                ax.set_ylabel("Normalized Pitch (z-score)", fontsize=10)
            
            # Shape description (simple heuristics)
            slope = np.polyfit(np.arange(50), centroid, 1)[0]
            curvature = np.std(np.diff(centroid))
            
            if curvature > 0.5:
                shape = "Complex/Modulated"
            elif slope > 0.05:
                shape = "Upsweep"
            elif slope < -0.05:
                shape = "Downsweep"
            else:
                shape = "Flat/Constant"
            
            ax.text(25, ax.get_ylim()[0] + 0.5, shape, ha='center', fontsize=10, style='italic', 
                    bbox=dict(facecolor='white', alpha=0.8))
            
        plt.suptitle("Whistle Contour Catalog (Signature Candidates)\nClustered by shape - NOT individual identification", 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_whistle_catalog.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   -> SAVED: dolphin_whistle_catalog.png")
        
        # Store results
        self.whistle_catalog_results = {
            'n_contours': len(contours),
            'n_types': n_types,
            'type_counts': {i: int(np.sum(labels == i)) for i in range(n_types)},
            'centroids': kmeans.cluster_centers_
        }

    # =========================================================================
    # VISUALIZATION 30: SOUNDSCAPE & LOMBARD EFFECT ANALYSIS
    # =========================================================================
    def generate_soundscape_analysis(self):
        """
        Analyze acoustic environment and test for Lombard effect.
        All data from real audio recording.
        """
        print("\n30. Generating Soundscape & Lombard Effect Analysis...")
        
        # 1. Long-Term Spectral Average (LTSA)
        n_fft = 4096
        hop_length = 2048
        
        print("   -> Computing STFT (real audio)...")
        S = np.abs(librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        
        # Time and frequency axes
        times_ltsa = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length) + self.start_offset_sec
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        
        # 2. Noise floor estimation
        noise_profile = np.percentile(S_db, 10, axis=1)
        
        # 3. Lombard effect analysis
        # Low freq noise (<2kHz) vs High freq signal (>5kHz)
        low_freq_idx = np.where(freqs < 2000)[0]
        high_freq_idx = np.where(freqs > 5000)[0]
        
        if len(low_freq_idx) == 0 or len(high_freq_idx) == 0:
            print("   ! Frequency range issue, skipping Lombard analysis")
            return
            
        noise_energy = np.mean(S[low_freq_idx, :], axis=0)
        signal_energy = np.mean(S[high_freq_idx, :], axis=0)
        
        # Smoothing
        noise_smooth = gaussian_filter1d(noise_energy, sigma=50)
        signal_smooth = gaussian_filter1d(signal_energy, sigma=50)
        
        # Correlation
        corr = np.corrcoef(noise_smooth, signal_smooth)[0, 1]
        
        print(f"   -> Noise-Signal correlation: r={corr:.3f}")
        
        # --- VISUALIZATION ---
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2)
        
        # A. LTSA
        ax1 = fig.add_subplot(gs[0, :])
        try:
            # librosa.display already imported at top of file
            img = librosa.display.specshow(S_db, sr=self.sr, hop_length=hop_length, 
                                           x_axis='time', y_axis='linear', ax=ax1, cmap='magma')
            plt.colorbar(img, ax=ax1, format="%+2.0f dB")
        except Exception as e:
            # Fallback if librosa.display fails
            extent = [times_ltsa[0], times_ltsa[-1], freqs[0], freqs[-1]]
            ax1.imshow(S_db, aspect='auto', origin='lower', extent=extent, cmap='magma')
            
        ax1.set_title("Long-Term Spectral Average (LTSA) - Acoustic Soundscape", fontsize=14, fontweight='bold')
        ax1.set_ylim(0, min(40000, self.sr/2))
        ax1.set_ylabel("Frequency (Hz)")
        ax1.set_xlabel("Time (s)")
        
        # B. Noise vs Signal timeseries
        ax2 = fig.add_subplot(gs[1, :])
        # Normalize for display
        n_norm = (noise_smooth - noise_smooth.min()) / (noise_smooth.max() - noise_smooth.min() + 1e-6)
        s_norm = (signal_smooth - signal_smooth.min()) / (signal_smooth.max() - signal_smooth.min() + 1e-6)
        
        t_axis = np.linspace(self.valid_times[0], self.valid_times[-1], len(n_norm))
        
        ax2.plot(t_axis, n_norm, color='gray', label='Background Noise (<2kHz)', alpha=0.7)
        ax2.plot(t_axis, s_norm, color='#2ecc71', label='Dolphin Signal (>5kHz)', alpha=0.8)
        ax2.fill_between(t_axis, 0, s_norm, color='#2ecc71', alpha=0.2)
        
        ax2.set_title("Signal vs. Noise Levels (Lombard Effect Check)", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Normalized Energy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # C. Scatter Plot (Correlation)
        ax3 = fig.add_subplot(gs[2, 0])
        # Downsample for scatter plot
        step = max(1, len(noise_smooth) // 500)
        sns.regplot(x=noise_smooth[::step], y=signal_smooth[::step], ax=ax3, 
                    scatter_kws={'alpha':0.3, 'color':'purple', 's':10}, 
                    line_kws={'color':'red'})
        ax3.set_xlabel("Background Noise Energy")
        ax3.set_ylabel("Dolphin Signal Energy")
        ax3.set_title(f"Adaptive Response Correlation (r={corr:.3f})", fontsize=12, fontweight='bold')
        
        # Interpretation
        if corr > 0.2:
            verdict = "POSSIBLE LOMBARD EFFECT\n(Louder noise → Louder dolphins)"
            bg_col = '#e8f8f5'
        elif corr < -0.2:
            verdict = "NOISE AVOIDANCE\n(Louder noise → Quieter dolphins)"
            bg_col = '#fdedec'
        else:
            verdict = "INDEPENDENT ACTIVITY\n(No clear adaptation observed)"
            bg_col = '#f8f9fa'
            
        ax3.text(0.05, 0.9, verdict, transform=ax3.transAxes, fontsize=10, fontweight='bold',
                 bbox=dict(facecolor=bg_col, alpha=1.0))
        
        # D. Spectral Partitioning
        ax4 = fig.add_subplot(gs[2, 1])
        mean_spec = np.mean(S, axis=1)
        mean_spec_db = librosa.amplitude_to_db(mean_spec, ref=np.max)
        
        ax4.plot(freqs, mean_spec_db, color='black', linewidth=0.5)
        ax4.fill_between(freqs, -80, mean_spec_db, where=(freqs < 2000), 
                         color='gray', alpha=0.5, label='Noise Niche (<2kHz)')
        ax4.fill_between(freqs, -80, mean_spec_db, where=((freqs > 4000) & (freqs < 20000)), 
                         color='#2ecc71', alpha=0.5, label='Whistle Niche (4-20kHz)')
        ax4.fill_between(freqs, -80, mean_spec_db, where=(freqs > 20000), 
                         color='#3498db', alpha=0.5, label='Click Niche (>20kHz)')
        
        ax4.set_xlim(0, min(40000, self.sr/2))
        ax4.set_ylim(-80, 0)
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Power (dB)")
        ax4.set_title("Acoustic Niche Partitioning", fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('dolphin_soundscape_analysis.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_soundscape_analysis.png")
        
        # Store results
        self.soundscape_results = {
            'lombard_correlation': corr,
            'noise_floor_db': np.mean(noise_profile),
            'signal_mean_db': np.mean(S_db[high_freq_idx, :]),
        }

    # =========================================================================
    # VISUALIZATION 31: COGNITIVE KINEMATICS
    # =========================================================================
    def generate_kinematic_analysis(self):
        """
        Analyze behavioral dynamics through latent space kinematics.
        Computes velocity and acceleration of state transitions.
        All data derived from real acoustic features via LSTM latent space.
        """
        print("\n31. Generating Cognitive Kinematics Analysis...")
        
        # Latent Space Trajectory (Z)
        Z = self.latent_space
        times = self.valid_times
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        # 1. Velocity - Rate of state change
        # v(t) = || z(t+1) - z(t) || / dt
        diffs = np.diff(Z, axis=0)
        velocities = np.linalg.norm(diffs, axis=1) / dt
        
        # 2. Acceleration - Rate of velocity change (Force/Surprise)
        # a(t) = (v(t+1) - v(t)) / dt
        accels = np.diff(velocities) / dt
        
        # Time axes alignment
        t_v = times[:-1]
        t_a = times[:-2]
        
        # Smoothing for visualization
        v_smooth = gaussian_filter1d(velocities, sigma=5)
        a_smooth = gaussian_filter1d(accels, sigma=5)
        
        # --- VISUALIZATION ---
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
        
        # PANEL A: Behavioral Velocity
        axes[0].fill_between(t_v, 0, v_smooth, color='#2980b9', alpha=0.3)
        axes[0].plot(t_v, v_smooth, color='#2980b9', lw=1.5)
        axes[0].set_ylabel("Velocity\n(State Change Rate)", fontsize=11)
        axes[0].set_xlabel("Time (s)")
        axes[0].set_title("Cognitive Kinematics: The Speed of Behavioral Change", fontsize=14, fontweight='bold')
        
        # Mark peak transitions (Rapid Transition Events)
        peaks, _ = signal.find_peaks(v_smooth, height=np.mean(v_smooth) + 2*np.std(v_smooth))
        if len(peaks) > 0:
            axes[0].plot(t_v[peaks], v_smooth[peaks], 'rx', markersize=8, label=f'Rapid Transitions (n={len(peaks)})')
            axes[0].legend(loc='upper right')
        
        # PANEL B: Behavioral Acceleration (Force/Surprise)
        axes[1].plot(t_a, a_smooth, color='#e74c3c', lw=1)
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.5)
        axes[1].fill_between(t_a, 0, a_smooth, where=(a_smooth > 0), color='#e74c3c', alpha=0.2)
        axes[1].fill_between(t_a, 0, a_smooth, where=(a_smooth < 0), color='#3498db', alpha=0.2)
        axes[1].set_ylabel("Acceleration\n(Force/Surprise)", fontsize=11)
        axes[1].set_xlabel("Time (s)")
        axes[1].set_title("Behavioral Forces: Acceleration in Latent Space (Red=Speed up, Blue=Slow down)", 
                          fontsize=12, fontweight='bold')
        
        # Statistics text
        inertia = np.corrcoef(v_smooth[:-1], v_smooth[1:])[0, 1] if len(v_smooth) > 2 else 0
        stats_txt = f"""KINEMATIC STATISTICS
Mean Velocity: {np.mean(velocities):.2f} units/s
Max Surge: {np.max(velocities):.2f} units/s
Std Velocity: {np.std(velocities):.2f} units/s
Inertia (Autocorr): {inertia:.3f}
Rapid Events: {len(peaks)}"""
        axes[1].text(0.02, 0.95, stats_txt, transform=axes[1].transAxes, 
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'), 
                     fontsize=9, fontfamily='monospace', verticalalignment='top')
        
        # PANEL C: Mean Velocity by Behavioral State
        # Show which states have fastest transitions
        df_kin = pd.DataFrame({'Velocity': velocities, 'Cluster': self.clusters[:-1]})
        mean_vel_by_cluster = df_kin.groupby('Cluster')['Velocity'].mean()
        std_vel_by_cluster = df_kin.groupby('Cluster')['Velocity'].std()
        
        existing_in_kin = [c for c in self.existing_clusters if c in mean_vel_by_cluster.index]
        colors = [self.all_colors[i] for i in existing_in_kin]
        labels = [self.labels[i].split('[')[0].strip()[:15] for i in existing_in_kin]
        means = [mean_vel_by_cluster[i] for i in existing_in_kin]
        stds = [std_vel_by_cluster[i] for i in existing_in_kin]
        
        bars = axes[2].bar(range(len(labels)), means, yerr=stds, color=colors, 
                           edgecolor='black', alpha=0.8, capsize=3)
        axes[2].set_xticks(range(len(labels)))
        axes[2].set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        axes[2].set_ylabel("Mean Velocity (units/s)")
        axes[2].set_title("Kinematic Profile by Behavioral State (Error bars = SD)", fontsize=12, fontweight='bold')
        
        # Highlight fastest and slowest states
        if len(means) > 0:
            max_idx = np.argmax(means)
            min_idx = np.argmin(means)
            axes[2].annotate('FASTEST', (max_idx, means[max_idx]), 
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='red')
            axes[2].annotate('SLOWEST', (min_idx, means[min_idx]), 
                            textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, color='blue')
        
        plt.tight_layout()
        plt.savefig('dolphin_kinematics.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_kinematics.png")
        
        # Store results
        self.kinematic_results = {
            'mean_velocity': float(np.mean(velocities)),
            'max_velocity': float(np.max(velocities)),
            'std_velocity': float(np.std(velocities)),
            'inertia': float(inertia),
            'n_rapid_events': int(len(peaks)),
            'velocity_by_state': {self.labels[i]: float(mean_vel_by_cluster[i]) 
                                  for i in existing_in_kin}
        }

    # =========================================================================
    # VISUALIZATION 32: BEHAVIORAL MOTIF ANALYSIS (SYNTAX SEARCH)
    # =========================================================================
    def generate_motif_analysis(self):
        """
        Search for recurring behavioral motifs (trigrams) in state sequences.
        Uses null model permutation testing for statistical significance.
        """
        print("\n32. Generating Behavioral Motif Analysis (Syntax Search)...")
        
        # 1. Compress sequence (RLE) - we care about transitions, not duration
        sequence_compressed = [self.clusters[0]]
        for c in self.clusters[1:]:
            if c != sequence_compressed[-1]:
                sequence_compressed.append(c)
        
        print(f"   -> Compressed sequence: {len(self.clusters)} -> {len(sequence_compressed)} states")
        
        # 2. Find 3-grams (trigrams)
        motif_len = 3
        motifs = {}
        
        for i in range(len(sequence_compressed) - motif_len + 1):
            pattern = tuple(sequence_compressed[i : i + motif_len])
            motifs[pattern] = motifs.get(pattern, 0) + 1
        
        # 3. NULL MODEL (Permutation test)
        print("   -> Computing Z-scores against null model (50 permutations)...")
        n_shuffles = 50
        random_motif_counts = {p: [] for p in motifs.keys()}
        
        # Use numpy random with seed for reproducibility
        rng = np.random.RandomState(42)
        
        for _ in range(n_shuffles):
            shuffled_seq = rng.permutation(sequence_compressed)
            temp_counts = {}
            for i in range(len(shuffled_seq) - motif_len + 1):
                pattern = tuple(shuffled_seq[i : i + motif_len])
                temp_counts[pattern] = temp_counts.get(pattern, 0) + 1
            
            for p in motifs.keys():
                random_motif_counts[p].append(temp_counts.get(p, 0))
                
        # 4. Z-Score calculation
        z_scores = []
        labels = []
        counts = []
        motif_data = []  # For CSV export
        
        for p, count in motifs.items():
            rand_vals = random_motif_counts[p]
            mean_rand = np.mean(rand_vals)
            std_rand = np.std(rand_vals)
            
            if std_rand > 0 and count > 10:  # Only frequent motifs
                z = (count - mean_rand) / std_rand
                
                # Store all for CSV
                l1 = self.labels[p[0]].split('[')[0].strip()[:8]
                l2 = self.labels[p[1]].split('[')[0].strip()[:8]
                l3 = self.labels[p[2]].split('[')[0].strip()[:8]
                label = f"{l1}→{l2}→{l3}"
                
                motif_data.append({
                    'Pattern': label,
                    'Count': count,
                    'Expected': mean_rand,
                    'Z_Score': z,
                    'Significant': abs(z) > 1.96
                })
                
                # Only significant for plot
                if abs(z) > 2.0:
                    z_scores.append(z)
                    counts.append(count)
                    labels.append(label)

        # 5. Visualization
        if not z_scores:
            print("   ! No significant motifs found (|z| > 2).")
            self.motif_results = {'n_motifs': 0, 'motif_data': motif_data}
            return

        # Sort by Z-score
        sorted_indices = np.argsort(z_scores)
        z_scores = np.array(z_scores)[sorted_indices]
        labels = np.array(labels)[sorted_indices]
        
        # Limit to top/bottom 20 for visibility
        if len(z_scores) > 40:
            top_idx = np.argsort(z_scores)[-20:]
            bottom_idx = np.argsort(z_scores)[:20]
            keep_idx = np.concatenate([bottom_idx, top_idx])
            z_scores = z_scores[keep_idx]
            labels = labels[keep_idx]
            sorted_indices = np.argsort(z_scores)
            z_scores = z_scores[sorted_indices]
            labels = labels[sorted_indices]
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(labels) * 0.3)))
        
        colors = ['#e74c3c' if z < 0 else '#3498db' for z in z_scores]
        bars = ax.barh(range(len(labels)), z_scores, color=colors, edgecolor='black', alpha=0.8)
        
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(1.96, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(-1.96, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel("Z-Score (Deviation from Random)", fontsize=12)
        ax.set_title(f"Behavioral Motif Analysis: Syntax Discovery (n={len(z_scores)} significant)", 
                     fontsize=14, fontweight='bold')
        
        # Legend box
        txt = "Blue (Z>2): MOTIFS\nPreferred sequences\n\nRed (Z<-2): ANTI-MOTIFS\nAvoided sequences"
        ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('dolphin_motif_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> SAVED: dolphin_motif_analysis.png ({len(z_scores)} significant motifs)")
        
        # Store results
        n_motifs = sum(1 for z in z_scores if z > 1.96)
        n_antimotifs = sum(1 for z in z_scores if z < -1.96)
        self.motif_results = {
            'n_significant': len(z_scores),
            'n_motifs': n_motifs,
            'n_antimotifs': n_antimotifs,
            'top_motif': labels[-1] if len(labels) > 0 else None,
            'top_motif_z': float(z_scores[-1]) if len(z_scores) > 0 else 0,
            'top_antimotif': labels[0] if len(labels) > 0 else None,
            'top_antimotif_z': float(z_scores[0]) if len(z_scores) > 0 else 0,
            'motif_data': motif_data
        }

    # =========================================================================
    # VISUALIZATION 33: EXPLAINABLE AI (XAI) - FEATURE IMPORTANCE
    # =========================================================================
    def generate_xai_analysis(self):
        """
        Analyze which acoustic features drive each behavioral state.
        Provides interpretability for the LSTM autoencoder clustering.
        """
        print("\n33. Generating Explainable AI (XAI) Feature Importance...")
        
        # 1. Feature contribution to Clusters
        df_feats = pd.DataFrame(self.features[self.window_size:], columns=['Whistle', 'Burst', 'Click'])
        df_feats['Cluster'] = self.clusters
        
        # Deviation from global mean
        global_mean = df_feats[['Whistle', 'Burst', 'Click']].mean()
        cluster_importance = df_feats.groupby('Cluster')[['Whistle', 'Burst', 'Click']].mean() - global_mean
        
        # 2. Latent Space Correlation
        pca = PCA(n_components=min(10, self.latent_space.shape[1]))
        latent_pca = pca.fit_transform(self.latent_space)
        
        input_data = self.features[self.window_size:]
        n_inputs = input_data.shape[1]
        n_latent = latent_pca.shape[1]
        
        corr_mat = np.zeros((n_inputs, n_latent))
        for i in range(n_inputs):
            for j in range(n_latent):
                corr_mat[i, j] = np.corrcoef(input_data[:, i], latent_pca[:, j])[0, 1]
        
        # 3. Visualization
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # A) Cluster Drivers Heatmap
        # Filter to existing clusters only
        cluster_importance_filtered = cluster_importance.loc[self.existing_clusters]
        
        sns.heatmap(cluster_importance_filtered, cmap='coolwarm', center=0, annot=True, fmt='.3f', 
                    ax=axes[0], cbar_kws={'label': 'Deviation from Mean'})
        
        ytick_labels = [self.labels[i].split('[')[0].strip()[:15] for i in cluster_importance_filtered.index]
        axes[0].set_yticklabels(ytick_labels, rotation=0)
        axes[0].set_title("XAI: Which features drive each behavior?", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Acoustic Feature")
        axes[0].set_ylabel("Behavioral State")
        
        # B) Input-Latent Correlation
        sns.heatmap(corr_mat, cmap='PiYG', center=0, annot=True, fmt='.2f', ax=axes[1],
                    xticklabels=[f'PC{i+1}' for i in range(n_latent)],
                    yticklabels=['Whistle', 'Burst', 'Click'])
        axes[1].set_title("Neural Network Decoding: Input-Latent Correlation", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("Latent Principal Component")
        axes[1].set_ylabel("Input Feature")
        
        plt.suptitle("Explainable AI Analysis: Understanding the LSTM Autoencoder", 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_xai_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   -> SAVED: dolphin_xai_analysis.png")
        
        # Store results
        # Find dominant feature for each cluster
        dominant_features = {}
        for idx in cluster_importance_filtered.index:
            row = cluster_importance_filtered.loc[idx]
            dominant = row.abs().idxmax()
            dominant_features[self.labels[idx].split('[')[0].strip()] = {
                'dominant_feature': dominant,
                'deviation': float(row[dominant])
            }
        
        self.xai_results = {
            'cluster_importance': cluster_importance_filtered.to_dict(),
            'input_latent_correlation': corr_mat.tolist(),
            'dominant_features': dominant_features,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }

    # =========================================================================
    # VISUALIZATION 34: TEMPORAL IRREVERSIBILITY (TIME ARROW)
    # =========================================================================
    def generate_irreversibility_analysis(self):
        """
        Test for temporal irreversibility in behavioral dynamics.
        Non-zero asymmetry indicates non-equilibrium/goal-directed behavior.
        """
        print("\n34. Generating Temporal Irreversibility (Time Arrow)...")
        
        # Project latent space to 1D (first PC)
        pca = PCA(n_components=1)
        ts = pca.fit_transform(self.latent_space).flatten()
        
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1
        
        # Time asymmetry statistic for different lags
        max_tau = min(100, len(ts) // 10)
        taus = np.arange(1, max_tau)
        asymmetry = []
        
        for tau in taus:
            diff = ts[:-tau] - ts[tau:]
            # Third moment (asymmetry) normalized by second moment
            num = np.mean(diff**3)
            den = np.mean(diff**2)**1.5
            val = num / (den + 1e-9)
            asymmetry.append(val)
            
        asymmetry = np.array(asymmetry)
        asym_smooth = gaussian_filter1d(asymmetry, sigma=2)
        
        # Time axis in seconds
        time_scales = taus * dt
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # A) Asymmetry vs Time Scale
        axes[0].plot(time_scales, asym_smooth, color='#d35400', lw=2)
        axes[0].axhline(0, color='black', linestyle='--', label='Reversible (Equilibrium)')
        axes[0].fill_between(time_scales, 0, asym_smooth, 
                             where=(asym_smooth > 0), color='#e74c3c', alpha=0.3, label='Forward bias')
        axes[0].fill_between(time_scales, 0, asym_smooth, 
                             where=(asym_smooth < 0), color='#3498db', alpha=0.3, label='Backward bias')
        
        axes[0].set_xlabel("Time Scale (seconds)", fontsize=12)
        axes[0].set_ylabel("Time Asymmetry Statistic (A)", fontsize=12)
        axes[0].set_title("Temporal Irreversibility: The 'Arrow of Time'", fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Interpretation
        mean_asym = np.mean(np.abs(asym_smooth))
        max_asym = np.max(np.abs(asym_smooth))
        
        if mean_asym > 0.1:
            verdict = "HIGH IRREVERSIBILITY\n(Non-equilibrium / Goal-directed)"
            verdict_color = '#e74c3c'
        elif mean_asym > 0.05:
            verdict = "MODERATE IRREVERSIBILITY\n(Partially structured)"
            verdict_color = '#f39c12'
        else:
            verdict = "LOW IRREVERSIBILITY\n(Near-equilibrium / Stochastic)"
            verdict_color = '#27ae60'
            
        axes[0].text(0.98, 0.95, verdict, transform=axes[0].transAxes, ha='right', va='top',
                     fontsize=11, fontweight='bold',
                     bbox=dict(facecolor=verdict_color, alpha=0.2, edgecolor=verdict_color))
        
        # B) Asymmetry Distribution
        axes[1].hist(asymmetry, bins=30, color='#9b59b6', edgecolor='white', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Reversible')
        axes[1].axvline(np.mean(asymmetry), color='orange', linestyle='-', lw=2, 
                        label=f'Mean: {np.mean(asymmetry):.3f}')
        axes[1].set_xlabel("Asymmetry Value", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Distribution of Time Asymmetry", fontsize=14, fontweight='bold')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('dolphin_irreversibility.png', dpi=300)
        plt.close()
        print(f"   -> SAVED: dolphin_irreversibility.png (Mean |A| = {mean_asym:.3f})")
        
        # Store results
        self.irreversibility_results = {
            'mean_asymmetry': float(np.mean(asymmetry)),
            'mean_abs_asymmetry': float(mean_asym),
            'max_abs_asymmetry': float(max_asym),
            'std_asymmetry': float(np.std(asymmetry)),
            'interpretation': verdict.split('\n')[0],
            'asymmetry_curve': asym_smooth.tolist(),
            'time_scales': time_scales.tolist()
        }

    # =========================================================================
    # VISUALIZATION 35: CRITICALITY & AVALANCHE ANALYSIS (SELF-ORGANIZED CRITICALITY)
    # =========================================================================
    def generate_criticality_analysis(self):
        """
        Analyze behavioral avalanches for signatures of self-organized criticality.
        Power-law distributions indicate optimal information processing capacity.
        Based on: Beggs & Plenz (2003), Cocchi et al. (2017)
        """
        print("\n35. Generating Criticality Analysis (Self-Organized Criticality)...")
        
        # 1. Compute global energy (sum of all acoustic channels)
        global_energy = np.sum(self.features[self.window_size:], axis=1)
        
        # 2. Detect avalanches (periods above threshold)
        threshold = np.mean(global_energy) + 0.5 * np.std(global_energy)
        is_active = global_energy > threshold
        
        avalanche_sizes = []
        avalanche_durations = []
        
        current_size = 0
        current_dur = 0
        
        for active, energy in zip(is_active, global_energy):
            if active:
                current_size += energy
                current_dur += 1
            else:
                if current_dur > 0:
                    avalanche_sizes.append(current_size)
                    avalanche_durations.append(current_dur)
                    current_size = 0
                    current_dur = 0
        
        # Handle last avalanche if still active
        if current_dur > 0:
            avalanche_sizes.append(current_size)
            avalanche_durations.append(current_dur)
        
        if len(avalanche_sizes) < 10:
            print("   ! Not enough avalanches for analysis.")
            self.criticality_results = {'n_avalanches': len(avalanche_sizes), 'slope': None}
            return

        # 3. Log-Log Histogram (Power Law search)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # A) Avalanche Size Distribution
        sizes = np.array(avalanche_sizes)
        
        # Logarithmic binning
        bins = np.logspace(np.log10(max(sizes.min(), 0.01)), np.log10(sizes.max()), 25)
        hist, edges = np.histogram(sizes, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        
        mask = hist > 0
        x = centers[mask]
        y = hist[mask]
        
        axes[0].loglog(x, y, 'o', color='#e74c3c', markersize=8, alpha=0.7, label='Observed Avalanches')
        
        # Power Law Fit
        slope = None
        if len(x) > 4:
            log_x = np.log10(x)
            log_y = np.log10(y)
            z = np.polyfit(log_x, log_y, 1)
            slope = z[0]
            p = np.poly1d(z)
            axes[0].loglog(x, 10**p(log_x), 'k--', lw=2, label=f'Power Law Fit (α={slope:.2f})')
            
            # Interpretation
            if -2.0 < slope < -1.0:
                status = "CRITICAL STATE\n(Optimal Info Processing)"
                bg_color = '#d5f5e3'
            elif slope < -2.0:
                status = "SUBCRITICAL\n(Over-damped)"
                bg_color = '#fadbd8'
            else:
                status = "SUPERCRITICAL\n(Under-damped)"
                bg_color = '#fdebd0'
                
            axes[0].text(0.05, 0.15, status, transform=axes[0].transAxes, fontsize=11, fontweight='bold',
                         bbox=dict(facecolor=bg_color, alpha=0.9, edgecolor='gray'))
        
        axes[0].set_xlabel('Avalanche Size (Energy)', fontsize=12)
        axes[0].set_ylabel('Probability P(S)', fontsize=12)
        axes[0].set_title('Avalanche Size Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, which="both", ls="--", alpha=0.3)
        
        # B) Scaling Relation (Duration vs Size)
        durs = np.array(avalanche_durations)
        axes[1].loglog(durs, sizes, 'o', color='#8e44ad', alpha=0.4, markersize=5)
        
        gamma = None
        if len(durs) > 4:
            # Filter to avoid log(0)
            valid = (durs > 0) & (sizes > 0)
            if np.sum(valid) > 4:
                z2 = np.polyfit(np.log10(durs[valid]), np.log10(sizes[valid]), 1)
                gamma = z2[0]
                x_fit = np.logspace(np.log10(durs[valid].min()), np.log10(durs[valid].max()), 50)
                y_fit = 10**np.polyval(z2, np.log10(x_fit))
                axes[1].loglog(x_fit, y_fit, 'k-', lw=2, label=f'Scaling: S ~ T^{gamma:.2f}')
        
        axes[1].set_xlabel('Avalanche Duration (time steps)', fontsize=12)
        axes[1].set_ylabel('Avalanche Size (Energy)', fontsize=12)
        axes[1].set_title('Universal Scaling Relation', fontsize=14, fontweight='bold')
        if gamma is not None:
            axes[1].legend()
        axes[1].grid(True, which="both", ls="--", alpha=0.3)
        
        plt.suptitle("Self-Organized Criticality: Evidence for Collective Intelligence", 
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_criticality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        slope_str = f"{slope:.2f}" if slope is not None else "N/A"
        gamma_str = f"{gamma:.2f}" if gamma is not None else "N/A"
        print(f"   -> SAVED: dolphin_criticality.png (α={slope_str}, γ={gamma_str})")
        
        # Store results
        self.criticality_results = {
            'n_avalanches': len(avalanche_sizes),
            'mean_avalanche_size': float(np.mean(sizes)),
            'max_avalanche_size': float(np.max(sizes)),
            'mean_duration': float(np.mean(durs)),
            'power_law_slope': float(slope) if slope else None,
            'scaling_exponent': float(gamma) if gamma else None,
            'is_critical': bool(-2.0 < slope < -1.0) if slope else False,
            'threshold_used': float(threshold)
        }

    # =========================================================================
    # VISUALIZATION 36: TOPOLOGICAL DATA ANALYSIS (PERSISTENT HOMOLOGY)
    # =========================================================================
    def generate_topological_analysis(self):
        """
        Use Topological Data Analysis (TDA) to find structural features in latent space.
        Persistent homology reveals loops, voids, and higher-dimensional structure.
        Based on: Carlsson (2009), Giusti et al. (2015)
        """
        print("\n36. Generating Topological Data Analysis (Persistent Homology)...")
        
        try:
            from ripser import ripser
            from persim import plot_diagrams
        except ImportError:
            print("   ! ripser/persim not installed. Skipping TDA.")
            print("   ! Install with: pip install ripser persim")
            self.tda_results = {'available': False}
            return
        
        # 1. Landmark sampling (uniform spacing to reduce computation)
        n_landmarks = min(800, len(self.latent_space))
        indices = np.linspace(0, len(self.latent_space)-1, n_landmarks).astype(int)
        point_cloud = self.latent_space[indices]
        
        # 2. Compute Persistent Homology
        print("   -> Computing Vietoris-Rips filtration...")
        result = ripser(point_cloud, maxdim=2)
        diagrams = result['dgms']
        
        # 3. Visualization
        fig = plt.figure(figsize=(16, 7))
        gs = fig.add_gridspec(1, 2)
        
        # A) Persistence Diagram
        ax1 = fig.add_subplot(gs[0, 0])
        plot_diagrams(diagrams, show=False, ax=ax1)
        ax1.set_title("Persistence Diagram", fontsize=14, fontweight='bold')
        
        # Legend explanation
        txt = "H0 (Red): Connected components\nH1 (Blue): Loops/Cycles\nH2 (Green): Voids/Cavities\n\nFar from diagonal = Persistent"
        ax1.text(0.55, 0.15, txt, transform=ax1.transAxes, fontsize=9, 
                 bbox=dict(facecolor='white', alpha=0.9))
        
        # B) Betti Curves
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Calculate Betti curves
        if len(diagrams[1]) > 0:
            max_death_1 = np.max(diagrams[1][np.isfinite(diagrams[1][:, 1]), 1])
            thresholds = np.linspace(0, max_death_1, 100)
            betti_1 = []
            for t in thresholds:
                count = np.sum((diagrams[1][:, 0] <= t) & (diagrams[1][:, 1] > t))
                betti_1.append(count)
            betti_1 = np.array(betti_1)
        else:
            thresholds = np.linspace(0, 1, 100)
            betti_1 = np.zeros(100)
        
        if len(diagrams) > 2 and len(diagrams[2]) > 0:
            finite_deaths = diagrams[2][np.isfinite(diagrams[2][:, 1]), 1]
            if len(finite_deaths) > 0:
                max_death_2 = np.max(finite_deaths)
                thresh_2 = np.linspace(0, max_death_2, 100)
                betti_2 = []
                for t in thresh_2:
                    count = np.sum((diagrams[2][:, 0] <= t) & (diagrams[2][:, 1] > t))
                    betti_2.append(count)
                betti_2 = np.array(betti_2)
            else:
                betti_2 = np.zeros(100)
        else:
            betti_2 = np.zeros(100)
            
        ax2.plot(thresholds, betti_1, color='#3498db', lw=3, label='H1 (Loops)')
        ax2.plot(thresholds[:len(betti_2)], betti_2, color='#2ecc71', lw=3, label='H2 (Voids)')
        
        ax2.set_xlabel("Filtration Scale", fontsize=12)
        ax2.set_ylabel("Betti Number", fontsize=12)
        ax2.set_title("Betti Curves: Topological Complexity", fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Interpretation
        max_b1 = int(np.max(betti_1))
        max_b2 = int(np.max(betti_2))
        
        if max_b1 > 5:
            conclusion = "COMPLEX TOPOLOGY\n(Semantic Structure Detected)"
            title_color = '#8e44ad'
        elif max_b1 > 2:
            conclusion = "MODERATE TOPOLOGY\n(Some Cyclic Structure)"
            title_color = '#2980b9'
        else:
            conclusion = "SIMPLE TOPOLOGY\n(Minimal Structure)"
            title_color = '#7f8c8d'
            
        plt.suptitle(f"Topological Data Analysis: {conclusion}", 
                     fontsize=14, fontweight='bold', color=title_color, y=1.02)
        
        plt.tight_layout()
        plt.savefig('dolphin_topology.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   -> SAVED: dolphin_topology.png (Max H1={max_b1}, Max H2={max_b2})")
        
        # Calculate persistence statistics
        h1_persistence = diagrams[1][:, 1] - diagrams[1][:, 0] if len(diagrams[1]) > 0 else np.array([0])
        h1_persistence = h1_persistence[np.isfinite(h1_persistence)]
        
        # Store results
        self.tda_results = {
            'available': True,
            'n_landmarks': n_landmarks,
            'max_betti_1': max_b1,
            'max_betti_2': max_b2,
            'n_h1_features': len(diagrams[1]),
            'n_h2_features': len(diagrams[2]) if len(diagrams) > 2 else 0,
            'mean_h1_persistence': float(np.mean(h1_persistence)) if len(h1_persistence) > 0 else 0,
            'max_h1_persistence': float(np.max(h1_persistence)) if len(h1_persistence) > 0 else 0,
            'interpretation': conclusion.split('\n')[0]
        }

    # =========================================================================
    # VISUALIZATION 37: SUMMARY REPORT
    # =========================================================================
    def generate_summary_report(self):
        print("\n37. Generating Summary Report...")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # Timeline
        ax1 = fig.add_subplot(gs[0, :])
        step = max(1, len(self.clusters) // 3000)
        for i in range(0, len(self.clusters), step):
            ax1.axvspan(self.valid_times[i], self.valid_times[min(i + step, len(self.valid_times) - 1)],
                        color=self.all_colors[self.clusters[i]], alpha=0.8)
        ax1.set_xlim(self.valid_times[0], self.valid_times[-1])
        ax1.set_ylabel('State')
        ax1.set_yticks([])
        ax1.set_title('Behavioral Timeline', fontweight='bold', fontsize=12)

        # State distribution pie
        ax2 = fig.add_subplot(gs[1, 0])
        counts = np.bincount(self.clusters, minlength=max(self.existing_clusters) + 1)
        counts_filtered = [counts[i] for i in self.existing_clusters]
        colors_pie = [self.all_colors[i] for i in self.existing_clusters]
        ax2.pie(counts_filtered, colors=colors_pie, autopct='%1.0f%%', textprops={'fontsize': 8})
        ax2.set_title('State Distribution', fontweight='bold', fontsize=10)

        # Channel energies
        ax3 = fig.add_subplot(gs[1, 1])
        channel_means = self.features[self.window_size:].mean(axis=0)
        ax3.bar(['Whistle', 'Burst', 'Click'], channel_means, color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='black')
        ax3.set_ylabel('Mean Energy')
        ax3.set_title('Channel Activity', fontweight='bold', fontsize=10)

        # Mini latent space
        ax4 = fig.add_subplot(gs[1, 2])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.latent_space)
        ax4.scatter(coords[::10, 0], coords[::10, 1], c=self.clusters[::10], cmap='Set1', s=5, alpha=0.5)
        ax4.set_title('Latent Space', fontweight='bold', fontsize=10)

        # Mini transition matrix
        ax5 = fig.add_subplot(gs[1, 3])
        n_states = max(self.existing_clusters) + 1
        trans = np.zeros((n_states, n_states))
        for i in range(len(self.clusters) - 1):
            trans[self.clusters[i], self.clusters[i + 1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans / row_sums
        trans_existing = trans_prob[np.ix_(self.existing_clusters, self.existing_clusters)]
        ax5.imshow(trans_existing, cmap='Blues')
        ax5.set_title('Transitions', fontweight='bold', fontsize=10)
        ax5.set_xticks([])
        ax5.set_yticks([])

        # Acoustic channels over time
        ax6 = fig.add_subplot(gs[2, :2])
        t_min = (self.valid_times - self.valid_times[0]) / 60
        ax6.fill_between(t_min, self.features[self.window_size:, 0], alpha=0.5, color='#2ecc71', label='Whistle')
        ax6.fill_between(t_min, self.features[self.window_size:, 1], alpha=0.5, color='#e74c3c', label='Burst')
        ax6.fill_between(t_min, self.features[self.window_size:, 2], alpha=0.5, color='#3498db', label='Click')
        ax6.set_xlabel('Time (min)')
        ax6.set_ylabel('Energy')
        ax6.set_title('Acoustic Channels', fontweight='bold', fontsize=10)
        ax6.legend(loc='upper right', fontsize=8)

        # Complexity
        ax7 = fig.add_subplot(gs[2, 2:])
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1
        ws = max(10, int(60 / dt))
        ents = []
        times_e = []
        for i in range(0, len(self.clusters) - ws, ws // 4):
            chunk = self.clusters[i:i + ws]
            cnts = np.bincount(chunk.astype(int), minlength=max(self.existing_clusters) + 1)
            probs = cnts / cnts.sum()
            probs = probs[probs > 0]
            ents.append(-np.sum(probs * np.log2(probs)))
            times_e.append(self.valid_times[i + ws // 2])
        if ents:
            te = (np.array(times_e) - self.valid_times[0]) / 60
            ax7.fill_between(te, gaussian_filter1d(ents, 3), alpha=0.5, color='purple')
            ax7.set_xlabel('Time (min)')
            ax7.set_ylabel('Entropy')
            ax7.set_title('Behavioral Complexity', fontweight='bold', fontsize=10)

        # Statistics
        ax8 = fig.add_subplot(gs[3, 0:2])
        ax8.axis('off')
        total_duration = self.valid_times[-1] - self.valid_times[0]
        n_transitions = np.sum(np.diff(self.clusters) != 0)
        dominant_state = self.labels[np.argmax(counts)]

        stats_text = f"""
╔══════════════════════════════════════════════════════════╗
║                  SUMMARY STATISTICS                      ║
╠══════════════════════════════════════════════════════════╣
║  Total Duration:       {total_duration:.1f} seconds ({total_duration / 60:.1f} min)
║  Data Points:          {len(self.valid_times)} samples
║  Behavioral States:    {len(self.existing_clusters)} clusters
║  
║  Whistle Mean:         {channel_means[0]:.3f}
║  Burst Mean:           {channel_means[1]:.3f}
║  Click Mean:           {channel_means[2]:.3f}
║  
║  Transition Count:     {n_transitions}
║  Switching Rate:       {n_transitions / len(self.clusters) * 100:.2f}%
║  Dominant State:       {dominant_state[:30]}
╚══════════════════════════════════════════════════════════╝
        """
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace')

        # Legend
        ax9 = fig.add_subplot(gs[3, 2:])
        ax9.axis('off')
        patches = [mpatches.Patch(color=self.all_colors[i], label=self.labels[i].split('[')[0].strip()[:20])
                   for i in self.existing_clusters]
        ax9.legend(handles=patches, loc='center', ncol=2, fontsize=9, title='Behavioral States')

        plt.suptitle('DOLPHIN BIOACOUSTIC ANALYSIS - SUMMARY REPORT', fontsize=16, fontweight='bold', y=0.98)
        plt.savefig('dolphin_summary_report.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_summary_report.png")

    # =========================================================================
    # EXPORT PAPER STATISTICS TO CSV
    # =========================================================================
    def export_paper_statistics(self):
        print("\n" + "="*60)
        print("EXPORTING PAPER STATISTICS TO CSV")
        print("="*60)
        
        features = self.features[self.window_size:]
        times = self.valid_times
        clusters = self.clusters
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        n_transitions = np.sum(np.diff(clusters) != 0)  # Total state transitions
        
        # 1. BEHAVIORAL STATE STATISTICS
        print("\n   -> Calculating behavioral state statistics...")
        state_stats = []
        for state_id in self.existing_clusters:
            mask = clusters == state_id
            state_features = features[mask]
            state_stats.append({
                'State_ID': state_id,
                'Label': self.labels[state_id],
                'Count': np.sum(mask),
                'Proportion_Percent': np.sum(mask) / len(clusters) * 100,
                'Mean_Whistle': np.mean(state_features[:, 0]) if len(state_features) > 0 else 0,
                'Mean_Burst': np.mean(state_features[:, 1]) if len(state_features) > 0 else 0,
                'Mean_Click': np.mean(state_features[:, 2]) if len(state_features) > 0 else 0,
                'Std_Whistle': np.std(state_features[:, 0]) if len(state_features) > 0 else 0,
                'Std_Burst': np.std(state_features[:, 1]) if len(state_features) > 0 else 0,
                'Std_Click': np.std(state_features[:, 2]) if len(state_features) > 0 else 0,
            })
        df_states = pd.DataFrame(state_stats)
        df_states.to_csv('paper_behavioral_states.csv', index=False)
        print("      SAVED: paper_behavioral_states.csv")
        
        # 2. ICI ANALYSIS
        print("   -> Calculating ICI statistics...")
        clicks = features[:, 2]
        threshold_high = max(np.percentile(clicks, 85), 0.15)
        threshold_low = max(np.percentile(clicks, 70), 0.10)
        
        click_indices = []
        in_click = False
        for i in range(len(clicks)):
            if not in_click and clicks[i] >= threshold_high:
                click_indices.append(i)
                in_click = True
            elif in_click and clicks[i] < threshold_low:
                in_click = False
        click_indices = np.array(click_indices)
        
        if len(click_indices) < 100:
            threshold_high = max(np.percentile(clicks, 75), 0.10)
            threshold_low = max(np.percentile(clicks, 60), 0.05)
            click_indices = []
            in_click = False
            for i in range(len(clicks)):
                if not in_click and clicks[i] >= threshold_high:
                    click_indices.append(i)
                    in_click = True
                elif in_click and clicks[i] < threshold_low:
                    in_click = False
            click_indices = np.array(click_indices)
        
        ici_stats = {}
        if len(click_indices) > 30:
            icis = np.diff(click_indices)
            ici_ms = icis * dt * 1000
            click_times = times[click_indices[1:]]
            
            short_thresh = 50
            terminal_buzz_count = np.sum(ici_ms < short_thresh)
            terminal_buzz_percent = terminal_buzz_count / len(ici_ms) * 100
            
            train_threshold = 150
            in_train = ici_ms < train_threshold
            train_lengths = []
            i = 0
            while i < len(in_train):
                if in_train[i]:
                    start = i
                    while i < len(in_train) and in_train[i]:
                        i += 1
                    if i - start >= 3:
                        train_lengths.append(i - start)
                else:
                    i += 1
            
            ici_stats = {
                'Total_Clicks': len(click_indices),
                'Total_Intervals': len(ici_ms),
                'ICI_Mean_ms': np.mean(ici_ms),
                'ICI_Median_ms': np.median(ici_ms),
                'ICI_Std_ms': np.std(ici_ms),
                'ICI_Min_ms': np.min(ici_ms),
                'ICI_Max_ms': np.max(ici_ms),
                'Terminal_Buzz_Percent': terminal_buzz_percent,
                'Regular_Echo_Percent': np.sum((ici_ms >= 50) & (ici_ms < 200)) / len(ici_ms) * 100,
                'Click_Train_Count': len(train_lengths),
                'Click_Train_Mean_Length': np.mean(train_lengths) if train_lengths else 0,
            }
        
        df_ici = pd.DataFrame([ici_stats])
        df_ici.to_csv('paper_ici_statistics.csv', index=False)
        print("      SAVED: paper_ici_statistics.csv")
        
        # 3. BOUT ANALYSIS
        print("   -> Calculating bout statistics...")
        min_bout_duration = 3
        bouts = []
        current_state = clusters[0]
        bout_start = 0
        
        for i in range(1, len(clusters)):
            if clusters[i] != current_state:
                duration = i - bout_start
                if duration >= min_bout_duration:
                    bouts.append({'state': current_state, 'duration_sec': duration * dt, 'start_time': times[bout_start]})
                current_state = clusters[i]
                bout_start = i
        
        if len(bouts) < 5:
            min_bout_duration = 1
            bouts = []
            current_state = clusters[0]
            bout_start = 0
            for i in range(1, len(clusters)):
                if clusters[i] != current_state:
                    duration = i - bout_start
                    if duration >= min_bout_duration:
                        bouts.append({'state': current_state, 'duration_sec': duration * dt, 'start_time': times[bout_start]})
                    current_state = clusters[i]
                    bout_start = i
        
        bout_stats = {}
        if len(bouts) > 5:
            bout_df = pd.DataFrame(bouts)
            durations = bout_df['duration_sec'].values
            total_time = times[-1] - times[0]
            
            bout_stats = {
                'Total_Bouts': len(bouts),
                'Recording_Duration_s': total_time,
                'Bouts_Per_Minute': len(bouts) / (total_time / 60),
                'Duration_Mean_s': np.mean(durations),
                'Duration_Median_s': np.median(durations),
                'Duration_Std_s': np.std(durations),
                'Duration_Geometric_Mean_s': np.exp(np.mean(np.log(durations[durations > 0]))),
            }
        
        df_bout = pd.DataFrame([bout_stats])
        df_bout.to_csv('paper_bout_statistics.csv', index=False)
        print("      SAVED: paper_bout_statistics.csv")
        
        # 4. RQA STATISTICS
        print("   -> Calculating RQA statistics...")
        step = max(1, len(self.latent_space) // 1000)
        latent_ds = self.latent_space[::step]
        dists = squareform(pdist(latent_ds, metric='euclidean'))
        thresh = np.percentile(dists, 15)
        rec = (dists < thresh).astype(int)
        n = rec.shape[0]
        
        rr = np.sum(rec) / (n * n) * 100
        
        min_line = 2
        diag_points = 0
        total_rec = np.sum(rec)
        all_diag_lengths = []
        
        for k in range(-n+1, n):
            diag = np.diag(rec, k)
            current_length = 0
            for val in diag:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= min_line:
                        all_diag_lengths.append(current_length)
                        diag_points += current_length
                    current_length = 0
            if current_length >= min_line:
                all_diag_lengths.append(current_length)
                diag_points += current_length
        
        det = diag_points / total_rec * 100 if total_rec > 0 else 0
        
        vert_points = 0
        for col in range(n):
            column = rec[:, col]
            current_length = 0
            for val in column:
                if val == 1:
                    current_length += 1
                else:
                    if current_length >= min_line:
                        vert_points += current_length
                    current_length = 0
            if current_length >= min_line:
                vert_points += current_length
        
        lam = vert_points / total_rec * 100 if total_rec > 0 else 0
        l_mean = np.mean(all_diag_lengths) if all_diag_lengths else 0
        
        if all_diag_lengths:
            hist, _ = np.histogram(all_diag_lengths, bins=20)
            hist = hist[hist > 0]
            probs = hist / hist.sum()
            entr_diag = -np.sum(probs * np.log2(probs))
        else:
            entr_diag = 0
        
        rqa_stats = {
            'Recurrence_Rate_Percent': rr,
            'Determinism_Percent': det,
            'Laminarity_Percent': lam,
            'Mean_Diagonal_Length_samples': l_mean,
            'Diagonal_Entropy_bits': entr_diag,
            'DET_RR_Ratio': det / rr if rr > 0 else 0,
        }
        
        df_rqa = pd.DataFrame([rqa_stats])
        df_rqa.to_csv('paper_rqa_statistics.csv', index=False)
        print("      SAVED: paper_rqa_statistics.csv")
        
        # 5. ENTROPY
        print("   -> Calculating entropy statistics...")
        window_sec = 60
        ws = max(10, int(window_sec / dt))
        n_states = len(self.existing_clusters)
        max_entropy = np.log2(n_states)
        
        ents = []
        for i in range(0, len(clusters) - ws, ws // 4):
            chunk = clusters[i:i + ws]
            counts = np.bincount(chunk.astype(int), minlength=max(self.existing_clusters) + 1)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            ents.append(h)
        
        entropy_stats = {
            'Max_Possible_Entropy_bits': max_entropy,
            'Num_States': n_states,
            'Entropy_Mean_bits': np.mean(ents),
            'Normalized_Entropy_Mean': np.mean(ents) / max_entropy,
        }
        
        df_entropy = pd.DataFrame([entropy_stats])
        df_entropy.to_csv('paper_entropy_statistics.csv', index=False)
        print("      SAVED: paper_entropy_statistics.csv")
        
        # 6. MARKOV TRANSITIONS
        print("   -> Calculating Markov transitions...")
        n_states_full = max(self.existing_clusters) + 1
        transitions = np.zeros((n_states_full, n_states_full))
        for i in range(len(clusters) - 1):
            transitions[clusters[i], clusters[i + 1]] += 1
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = transitions / row_sums
        trans_existing = trans_prob[np.ix_(self.existing_clusters, self.existing_clusters)]
        
        labels = [self.labels[i].split('[')[0].strip()[:15] for i in self.existing_clusters]
        df_trans = pd.DataFrame(trans_existing, index=labels, columns=labels)
        df_trans.to_csv('paper_markov_transitions.csv')
        print("      SAVED: paper_markov_transitions.csv")
        
        self_trans = {f'Self_Transition_{self.labels[i].split("[")[0].strip()[:12]}': trans_prob[i, i] 
                     for i in self.existing_clusters}
        df_markov_stats = pd.DataFrame([{'Num_Transitions': int(np.sum(transitions)), **self_trans}])
        df_markov_stats.to_csv('paper_markov_statistics.csv', index=False)
        print("      SAVED: paper_markov_statistics.csv")
        
        # 7. VOCAL INTERACTION STATISTICS (REAL DATA)
        print("   -> Calculating vocal interaction statistics...")
        if hasattr(self, 'vocal_interaction_results') and self.vocal_interaction_results:
            vir = self.vocal_interaction_results
            
            # Calculate additional metrics from stored FTOs
            ftos = vir['ftos']
            gaps = ftos[ftos > 0.05]
            overlaps = ftos[ftos < -0.05]
            
            vocal_stats = {
                'Total_Vocal_Events': vir['n_events'],
                'Whistle_Events': vir['type_counts']['WHISTLE'],
                'Burst_Events': vir['type_counts']['BURST'],
                'Click_Train_Events': vir['type_counts']['CLICK_TRAIN'],
                'Mean_FTO_ms': vir['mean_fto'] * 1000,
                'Median_FTO_ms': float(np.median(ftos) * 1000),
                'Std_FTO_ms': float(np.std(ftos) * 1000),
                'N_Gaps': int(len(gaps)),
                'N_Overlaps': int(len(overlaps)),
                'Gap_Ratio': vir['gap_ratio'],
                'Overlap_Ratio': vir['overlap_ratio'],
                'Mean_Gap_Duration_ms': float(np.mean(gaps) * 1000) if len(gaps) > 0 else 0,
                'Mean_Overlap_Duration_ms': float(np.mean(np.abs(overlaps)) * 1000) if len(overlaps) > 0 else 0,
                'Burstiness_Index': vir['burstiness'],
                'Transition_Entropy_bits': vir['trans_entropy'],
                'Call_Response_Pairs': len(vir['call_responses']),
                'Response_Probability': vir['response_prob'],
                'Mean_Response_Latency_ms': vir['mean_latency'] * 1000,
            }
            
            df_vocal = pd.DataFrame([vocal_stats])
            df_vocal.to_csv('paper_vocal_interaction.csv', index=False)
            print("      SAVED: paper_vocal_interaction.csv")
            
            # Export transition matrix
            event_types = ['WHISTLE', 'BURST', 'CLICK_TRAIN']
            df_event_trans = pd.DataFrame(
                vir['trans_prob'],
                index=event_types,
                columns=event_types
            )
            df_event_trans.to_csv('paper_event_transitions.csv')
            print("      SAVED: paper_event_transitions.csv")
        else:
            print("      ! Vocal interaction results not available")
        
        # 8. WHISTLE CONTOUR CATALOG
        print("   -> Exporting whistle contour statistics...")
        if hasattr(self, 'whistle_catalog_results') and self.whistle_catalog_results:
            wcr = self.whistle_catalog_results
            
            # Summary statistics
            whistle_stats = {
                'Total_Contours_Detected': wcr['n_contours'],
                'Num_Whistle_Types': wcr['n_types'],
            }
            
            # Add type-specific counts
            for type_id, count in wcr['type_counts'].items():
                whistle_stats[f'Type_{type_id+1}_Count'] = count
                whistle_stats[f'Type_{type_id+1}_Percent'] = count / wcr['n_contours'] * 100
            
            df_whistle = pd.DataFrame([whistle_stats])
            df_whistle.to_csv('paper_whistle_catalog.csv', index=False)
            print("      SAVED: paper_whistle_catalog.csv")
            
            # Export centroids (average contour shapes)
            centroids_df = pd.DataFrame(
                wcr['centroids'],
                index=[f'Type_{i+1}' for i in range(wcr['n_types'])],
                columns=[f'Point_{i+1}' for i in range(50)]
            )
            centroids_df.to_csv('paper_whistle_centroids.csv')
            print("      SAVED: paper_whistle_centroids.csv")
        else:
            print("      ! Whistle catalog results not available")
        
        # 9. SOUNDSCAPE & LOMBARD EFFECT
        print("   -> Exporting soundscape statistics...")
        if hasattr(self, 'soundscape_results') and self.soundscape_results:
            ssr = self.soundscape_results
            
            soundscape_stats = {
                'Lombard_Correlation_r': ssr['lombard_correlation'],
                'Noise_Floor_Mean_dB': ssr['noise_floor_db'],
                'Signal_Mean_dB': ssr['signal_mean_db'],
                'SNR_Estimate_dB': ssr['signal_mean_db'] - ssr['noise_floor_db'],
            }
            
            # Interpretation
            if ssr['lombard_correlation'] > 0.2:
                soundscape_stats['Lombard_Interpretation'] = 'Positive_Lombard_Effect'
            elif ssr['lombard_correlation'] < -0.2:
                soundscape_stats['Lombard_Interpretation'] = 'Noise_Avoidance'
            else:
                soundscape_stats['Lombard_Interpretation'] = 'Independent_Activity'
            
            df_soundscape = pd.DataFrame([soundscape_stats])
            df_soundscape.to_csv('paper_soundscape.csv', index=False)
            print("      SAVED: paper_soundscape.csv")
        else:
            print("      ! Soundscape results not available")
        
        # 10. COGNITIVE KINEMATICS
        print("   -> Exporting kinematic statistics...")
        if hasattr(self, 'kinematic_results') and self.kinematic_results:
            kr = self.kinematic_results
            
            kinematic_stats = {
                'Mean_Velocity_units_per_s': kr['mean_velocity'],
                'Max_Velocity_units_per_s': kr['max_velocity'],
                'Std_Velocity_units_per_s': kr['std_velocity'],
                'Behavioral_Inertia_autocorr': kr['inertia'],
                'Rapid_Transition_Events': kr['n_rapid_events'],
            }
            
            df_kinematic = pd.DataFrame([kinematic_stats])
            df_kinematic.to_csv('paper_kinematics.csv', index=False)
            print("      SAVED: paper_kinematics.csv")
            
            # Export velocity by state
            if kr['velocity_by_state']:
                vel_by_state = []
                for label, vel in kr['velocity_by_state'].items():
                    vel_by_state.append({
                        'State': label.split('[')[0].strip(),
                        'Mean_Velocity': vel
                    })
                df_vel_state = pd.DataFrame(vel_by_state)
                df_vel_state = df_vel_state.sort_values('Mean_Velocity', ascending=False)
                df_vel_state.to_csv('paper_kinematics_by_state.csv', index=False)
                print("      SAVED: paper_kinematics_by_state.csv")
        else:
            print("      ! Kinematic results not available")
        
        # 11. MOTIF ANALYSIS
        print("   -> Exporting motif analysis statistics...")
        if hasattr(self, 'motif_results') and self.motif_results:
            mr = self.motif_results
            
            motif_summary = {
                'N_Significant_Patterns': mr.get('n_significant', 0),
                'N_Motifs_Preferred': mr.get('n_motifs', 0),
                'N_Antimotifs_Avoided': mr.get('n_antimotifs', 0),
                'Top_Motif': mr.get('top_motif', 'N/A'),
                'Top_Motif_Z_Score': mr.get('top_motif_z', 0),
                'Top_Antimotif': mr.get('top_antimotif', 'N/A'),
                'Top_Antimotif_Z_Score': mr.get('top_antimotif_z', 0),
            }
            
            df_motif_summary = pd.DataFrame([motif_summary])
            df_motif_summary.to_csv('paper_motif_summary.csv', index=False)
            print("      SAVED: paper_motif_summary.csv")
            
            # Export all motif data
            if mr.get('motif_data'):
                df_motifs = pd.DataFrame(mr['motif_data'])
                df_motifs = df_motifs.sort_values('Z_Score', ascending=False)
                df_motifs.to_csv('paper_motif_details.csv', index=False)
                print("      SAVED: paper_motif_details.csv")
        else:
            print("      ! Motif results not available")
        
        # 12. XAI ANALYSIS
        print("   -> Exporting XAI feature importance...")
        if hasattr(self, 'xai_results') and self.xai_results:
            xr = self.xai_results
            
            # Export dominant features per state
            if xr.get('dominant_features'):
                dom_feat_list = []
                for state, data in xr['dominant_features'].items():
                    dom_feat_list.append({
                        'State': state,
                        'Dominant_Feature': data['dominant_feature'],
                        'Deviation_From_Mean': data['deviation']
                    })
                df_dominant = pd.DataFrame(dom_feat_list)
                df_dominant.to_csv('paper_xai_dominant_features.csv', index=False)
                print("      SAVED: paper_xai_dominant_features.csv")
            
            # Export PCA explained variance
            if xr.get('explained_variance_ratio'):
                pca_data = {
                    f'PC{i+1}_Variance_Ratio': v 
                    for i, v in enumerate(xr['explained_variance_ratio'])
                }
                pca_data['Total_Variance_Explained'] = sum(xr['explained_variance_ratio'])
                df_pca = pd.DataFrame([pca_data])
                df_pca.to_csv('paper_xai_latent_pca.csv', index=False)
                print("      SAVED: paper_xai_latent_pca.csv")
        else:
            print("      ! XAI results not available")
        
        # 13. TEMPORAL IRREVERSIBILITY
        print("   -> Exporting irreversibility statistics...")
        if hasattr(self, 'irreversibility_results') and self.irreversibility_results:
            ir = self.irreversibility_results
            
            irrev_stats = {
                'Mean_Asymmetry': ir['mean_asymmetry'],
                'Mean_Abs_Asymmetry': ir['mean_abs_asymmetry'],
                'Max_Abs_Asymmetry': ir['max_abs_asymmetry'],
                'Std_Asymmetry': ir['std_asymmetry'],
                'Interpretation': ir['interpretation'],
            }
            
            df_irrev = pd.DataFrame([irrev_stats])
            df_irrev.to_csv('paper_irreversibility.csv', index=False)
            print("      SAVED: paper_irreversibility.csv")
        else:
            print("      ! Irreversibility results not available")
        
        # 14. CRITICALITY ANALYSIS
        print("   -> Exporting criticality statistics...")
        if hasattr(self, 'criticality_results') and self.criticality_results:
            cr = self.criticality_results
            
            crit_stats = {
                'N_Avalanches': cr['n_avalanches'],
                'Mean_Avalanche_Size': cr.get('mean_avalanche_size', 'N/A'),
                'Max_Avalanche_Size': cr.get('max_avalanche_size', 'N/A'),
                'Mean_Duration_steps': cr.get('mean_duration', 'N/A'),
                'Power_Law_Slope_Alpha': cr.get('power_law_slope', 'N/A'),
                'Scaling_Exponent_Gamma': cr.get('scaling_exponent', 'N/A'),
                'Is_Critical': cr.get('is_critical', False),
                'Threshold_Used': cr.get('threshold_used', 'N/A'),
            }
            
            df_crit = pd.DataFrame([crit_stats])
            df_crit.to_csv('paper_criticality.csv', index=False)
            print("      SAVED: paper_criticality.csv")
        else:
            print("      ! Criticality results not available")
        
        # 15. TOPOLOGICAL DATA ANALYSIS
        print("   -> Exporting TDA statistics...")
        if hasattr(self, 'tda_results') and self.tda_results.get('available', False):
            tr = self.tda_results
            
            tda_stats = {
                'N_Landmarks': tr['n_landmarks'],
                'Max_Betti_1_Loops': tr['max_betti_1'],
                'Max_Betti_2_Voids': tr['max_betti_2'],
                'N_H1_Features': tr['n_h1_features'],
                'N_H2_Features': tr['n_h2_features'],
                'Mean_H1_Persistence': tr['mean_h1_persistence'],
                'Max_H1_Persistence': tr['max_h1_persistence'],
                'Interpretation': tr['interpretation'],
            }
            
            df_tda = pd.DataFrame([tda_stats])
            df_tda.to_csv('paper_topology.csv', index=False)
            print("      SAVED: paper_topology.csv")
        else:
            print("      ! TDA results not available (ripser not installed?)")
        
        # 16. AUDIO & RECORDING STATISTICS
        print("   -> Exporting audio statistics...")
        audio_stats = {
            'Recording_Duration_s': times[-1] - times[0],
            'Recording_Duration_min': (times[-1] - times[0]) / 60,
            'Sample_Rate_Hz': self.sr,
            'Total_Audio_Samples': len(self.audio) if hasattr(self, 'audio') else 'N/A',
            'Feature_Vectors': len(features),
            'Time_Resolution_ms': dt * 1000,
            'Frequency_Resolution_Hz': self.sr / 2048,  # n_fft default
        }
        df_audio = pd.DataFrame([audio_stats])
        df_audio.to_csv('paper_audio_stats.csv', index=False)
        print("      SAVED: paper_audio_stats.csv")
        
        # 17. LATENT SPACE STATISTICS
        print("   -> Exporting latent space statistics...")
        latent_stats = {
            'Latent_Dimensions': self.latent_space.shape[1],
            'N_Samples': self.latent_space.shape[0],
            'Mean_L2_Norm': float(np.mean(np.linalg.norm(self.latent_space, axis=1))),
            'Std_L2_Norm': float(np.std(np.linalg.norm(self.latent_space, axis=1))),
            'Max_L2_Norm': float(np.max(np.linalg.norm(self.latent_space, axis=1))),
        }
        
        # Add per-dimension statistics
        for dim in range(min(10, self.latent_space.shape[1])):
            latent_stats[f'Dim_{dim+1}_Mean'] = float(np.mean(self.latent_space[:, dim]))
            latent_stats[f'Dim_{dim+1}_Std'] = float(np.std(self.latent_space[:, dim]))
        
        df_latent = pd.DataFrame([latent_stats])
        df_latent.to_csv('paper_latent_space.csv', index=False)
        print("      SAVED: paper_latent_space.csv")
        
        # 18. STATE CENTROIDS (for Voronoi)
        print("   -> Exporting state centroids...")
        centroids_data = []
        for state_id in self.existing_clusters:
            mask = clusters == state_id
            if np.sum(mask) > 0:
                centroid = np.mean(self.latent_space[mask], axis=0)
                row = {
                    'State_ID': state_id,
                    'State_Label': self.labels[state_id].split('[')[0].strip(),
                    'N_Samples': int(np.sum(mask)),
                }
                for dim in range(min(10, len(centroid))):
                    row[f'Centroid_Dim_{dim+1}'] = float(centroid[dim])
                centroids_data.append(row)
        
        df_centroids = pd.DataFrame(centroids_data)
        df_centroids.to_csv('paper_state_centroids.csv', index=False)
        print("      SAVED: paper_state_centroids.csv")
        
        # 19. FEATURE DISTRIBUTIONS (for Ridge plot)
        print("   -> Exporting feature distributions per state...")
        feature_dist_data = []
        feature_names = ['Whistle', 'Burst', 'Click']
        
        for state_id in self.existing_clusters:
            mask = clusters == state_id
            if np.sum(mask) > 0:
                state_features = features[mask]
                for feat_idx, feat_name in enumerate(feature_names):
                    feature_dist_data.append({
                        'State_ID': state_id,
                        'State_Label': self.labels[state_id].split('[')[0].strip(),
                        'Feature': feat_name,
                        'Mean': float(np.mean(state_features[:, feat_idx])),
                        'Median': float(np.median(state_features[:, feat_idx])),
                        'Std': float(np.std(state_features[:, feat_idx])),
                        'Min': float(np.min(state_features[:, feat_idx])),
                        'Max': float(np.max(state_features[:, feat_idx])),
                        'Q25': float(np.percentile(state_features[:, feat_idx], 25)),
                        'Q75': float(np.percentile(state_features[:, feat_idx], 75)),
                    })
        
        df_feat_dist = pd.DataFrame(feature_dist_data)
        df_feat_dist.to_csv('paper_feature_distributions.csv', index=False)
        print("      SAVED: paper_feature_distributions.csv")
        
        # 20. TEMPORAL PROPORTIONS (for Streamgraph)
        print("   -> Exporting temporal state proportions...")
        # Divide recording into 60-second bins
        bin_duration = 60  # seconds
        n_bins = int((times[-1] - times[0]) / bin_duration) + 1
        
        temporal_data = []
        for bin_idx in range(n_bins):
            bin_start = times[0] + bin_idx * bin_duration
            bin_end = bin_start + bin_duration
            mask = (times >= bin_start) & (times < bin_end)
            
            if np.sum(mask) > 0:
                bin_clusters = clusters[mask]
                row = {
                    'Bin_Index': bin_idx,
                    'Time_Start_s': bin_start,
                    'Time_End_s': bin_end,
                    'N_Samples': int(np.sum(mask)),
                }
                for state_id in self.existing_clusters:
                    state_label = self.labels[state_id].split('[')[0].strip()
                    state_count = np.sum(bin_clusters == state_id)
                    row[f'{state_label}_Count'] = int(state_count)
                    row[f'{state_label}_Percent'] = float(state_count / np.sum(mask) * 100)
                temporal_data.append(row)
        
        df_temporal = pd.DataFrame(temporal_data)
        df_temporal.to_csv('paper_temporal_proportions.csv', index=False)
        print("      SAVED: paper_temporal_proportions.csv")
        
        # 21. TRANSITION FLOWS (for Sankey)
        print("   -> Exporting transition flows...")
        transition_flows = []
        for i in self.existing_clusters:
            for j in self.existing_clusters:
                if i != j:  # Skip self-transitions for Sankey
                    count = np.sum((clusters[:-1] == i) & (clusters[1:] == j))
                    if count > 0:
                        transition_flows.append({
                            'From_State_ID': i,
                            'From_State': self.labels[i].split('[')[0].strip(),
                            'To_State_ID': j,
                            'To_State': self.labels[j].split('[')[0].strip(),
                            'Count': int(count),
                            'Proportion': float(count / n_transitions * 100),
                        })
        
        df_flows = pd.DataFrame(transition_flows)
        df_flows = df_flows.sort_values('Count', ascending=False)
        df_flows.to_csv('paper_transition_flows.csv', index=False)
        print("      SAVED: paper_transition_flows.csv")
        
        # 22. PHASE PORTRAIT STATISTICS
        print("   -> Exporting phase portrait statistics...")
        # Use first 2 PCA components
        pca = PCA(n_components=min(3, self.latent_space.shape[1]))
        latent_pca = pca.fit_transform(self.latent_space)
        
        # Calculate trajectory statistics
        velocities = np.sqrt(np.sum(np.diff(latent_pca, axis=0)**2, axis=1))
        
        phase_stats = {
            'PC1_Explained_Variance': float(pca.explained_variance_ratio_[0]),
            'PC2_Explained_Variance': float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0,
            'Total_Variance_Explained': float(sum(pca.explained_variance_ratio_)),
            'Trajectory_Length': float(np.sum(velocities)),
            'Mean_Velocity': float(np.mean(velocities)),
            'Max_Velocity': float(np.max(velocities)),
            'Trajectory_Complexity': float(np.std(velocities) / np.mean(velocities)) if np.mean(velocities) > 0 else 0,
        }
        
        df_phase = pd.DataFrame([phase_stats])
        df_phase.to_csv('paper_phase_portrait.csv', index=False)
        print("      SAVED: paper_phase_portrait.csv")
        
        # 23. XAI CLUSTER IMPORTANCE (full matrix)
        print("   -> Exporting XAI cluster importance matrix...")
        xai_matrix = []
        global_mean = np.mean(features, axis=0)
        
        for state_id in self.existing_clusters:
            mask = clusters == state_id
            if np.sum(mask) > 0:
                state_mean = np.mean(features[mask], axis=0)
                deviation = state_mean - global_mean
                xai_matrix.append({
                    'State_ID': state_id,
                    'State_Label': self.labels[state_id].split('[')[0].strip(),
                    'Whistle_Deviation': float(deviation[0]),
                    'Burst_Deviation': float(deviation[1]),
                    'Click_Deviation': float(deviation[2]),
                    'Whistle_Mean': float(state_mean[0]),
                    'Burst_Mean': float(state_mean[1]),
                    'Click_Mean': float(state_mean[2]),
                })
        
        df_xai_matrix = pd.DataFrame(xai_matrix)
        df_xai_matrix.to_csv('paper_xai_cluster_importance.csv', index=False)
        print("      SAVED: paper_xai_cluster_importance.csv")
        
        # 24. BOUT DETAILS BY STATE
        print("   -> Exporting bout details by state...")
        bout_by_state = []
        for state_id in self.existing_clusters:
            state_bouts = [b for b in bouts if b['state'] == state_id]
            if state_bouts:
                durations = [b['duration_sec'] for b in state_bouts]
                bout_by_state.append({
                    'State_ID': state_id,
                    'State_Label': self.labels[state_id].split('[')[0].strip(),
                    'N_Bouts': len(state_bouts),
                    'Duration_Mean_s': float(np.mean(durations)),
                    'Duration_Median_s': float(np.median(durations)),
                    'Duration_Std_s': float(np.std(durations)),
                    'Duration_Min_s': float(np.min(durations)),
                    'Duration_Max_s': float(np.max(durations)),
                    'Total_Time_s': float(np.sum(durations)),
                })
        
        df_bout_state = pd.DataFrame(bout_by_state)
        df_bout_state = df_bout_state.sort_values('Total_Time_s', ascending=False)
        df_bout_state.to_csv('paper_bout_by_state.csv', index=False)
        print("      SAVED: paper_bout_by_state.csv")
        
        # 25. MASTER SUMMARY
        print("   -> Creating master summary...")
        total_duration = times[-1] - times[0]
        
        master_summary = {
            'Recording_Duration_s': total_duration,
            'Recording_Duration_min': total_duration / 60,
            'Feature_Samples': len(features),
            'Num_Behavioral_States': len(self.existing_clusters),
            'Total_State_Transitions': n_transitions,
            'RQA_Recurrence_Rate': rqa_stats['Recurrence_Rate_Percent'],
            'RQA_Determinism': rqa_stats['Determinism_Percent'],
            'RQA_Laminarity': rqa_stats['Laminarity_Percent'],
            'Shannon_Entropy_Mean_bits': entropy_stats['Entropy_Mean_bits'],
            'Shannon_Entropy_Normalized': entropy_stats['Normalized_Entropy_Mean'],
            'ICI_Total_Clicks': ici_stats.get('Total_Clicks', 'N/A'),
            'ICI_Mean_ms': ici_stats.get('ICI_Mean_ms', 'N/A'),
            'ICI_Median_ms': ici_stats.get('ICI_Median_ms', 'N/A'),
            'ICI_Terminal_Buzz_Percent': ici_stats.get('Terminal_Buzz_Percent', 'N/A'),
            'Bout_Total_Count': bout_stats.get('Total_Bouts', 'N/A'),
            'Bout_Duration_Mean_s': bout_stats.get('Duration_Mean_s', 'N/A'),
            'Bout_Bouts_Per_Minute': bout_stats.get('Bouts_Per_Minute', 'N/A'),
        }
        
        # Add vocal interaction to master summary if available
        if hasattr(self, 'vocal_interaction_results') and self.vocal_interaction_results:
            vir = self.vocal_interaction_results
            master_summary.update({
                'Vocal_Total_Events': vir['n_events'],
                'Vocal_Mean_FTO_ms': vir['mean_fto'] * 1000,
                'Vocal_Gap_Ratio': vir['gap_ratio'],
                'Vocal_Burstiness': vir['burstiness'],
                'Vocal_Response_Prob': vir['response_prob'],
            })
        
        # Add whistle catalog results if available
        if hasattr(self, 'whistle_catalog_results') and self.whistle_catalog_results:
            wcr = self.whistle_catalog_results
            master_summary.update({
                'Whistle_Contours_Detected': wcr['n_contours'],
                'Whistle_Types_Clustered': wcr['n_types'],
            })
        
        # Add soundscape results if available
        if hasattr(self, 'soundscape_results') and self.soundscape_results:
            ssr = self.soundscape_results
            master_summary.update({
                'Lombard_Correlation': ssr['lombard_correlation'],
                'Noise_Floor_dB': ssr['noise_floor_db'],
            })
        
        # Add kinematic results if available
        if hasattr(self, 'kinematic_results') and self.kinematic_results:
            kr = self.kinematic_results
            master_summary.update({
                'Kinematic_Mean_Velocity': kr['mean_velocity'],
                'Kinematic_Max_Velocity': kr['max_velocity'],
                'Kinematic_Inertia': kr['inertia'],
                'Kinematic_Rapid_Events': kr['n_rapid_events'],
            })
        
        # Add motif results if available
        if hasattr(self, 'motif_results') and self.motif_results:
            mr = self.motif_results
            master_summary.update({
                'Motif_N_Significant': mr.get('n_significant', 0),
                'Motif_N_Preferred': mr.get('n_motifs', 0),
                'Motif_N_Avoided': mr.get('n_antimotifs', 0),
                'Motif_Top_Z': mr.get('top_motif_z', 0),
            })
        
        # Add irreversibility results if available
        if hasattr(self, 'irreversibility_results') and self.irreversibility_results:
            ir = self.irreversibility_results
            master_summary.update({
                'Irreversibility_Mean_Abs': ir['mean_abs_asymmetry'],
                'Irreversibility_Max_Abs': ir['max_abs_asymmetry'],
            })
        
        # Add criticality results if available
        if hasattr(self, 'criticality_results') and self.criticality_results:
            cr = self.criticality_results
            master_summary.update({
                'Criticality_N_Avalanches': cr['n_avalanches'],
                'Criticality_Power_Law_Slope': cr.get('power_law_slope', 'N/A'),
                'Criticality_Is_Critical': cr.get('is_critical', False),
            })
        
        # Add TDA results if available
        if hasattr(self, 'tda_results') and self.tda_results.get('available', False):
            tr = self.tda_results
            master_summary.update({
                'TDA_Max_Betti_1': tr['max_betti_1'],
                'TDA_Max_Betti_2': tr['max_betti_2'],
            })
        
        df_master = pd.DataFrame([master_summary])
        df_master.to_csv('paper_statistics_summary.csv', index=False)
        print("      SAVED: paper_statistics_summary.csv")
        
        print("\n" + "="*60)
        print("CSV FILES GENERATED FOR PAPER:")
        print("-"*40)
        csv_files = [
            # Core statistics
            "paper_statistics_summary.csv",
            "paper_audio_stats.csv",
            "paper_behavioral_states.csv",
            # ICI & Click analysis
            "paper_ici_statistics.csv",
            # Bout analysis
            "paper_bout_statistics.csv",
            "paper_bout_by_state.csv",
            # Dynamics
            "paper_rqa_statistics.csv",
            "paper_entropy_statistics.csv",
            "paper_markov_transitions.csv",
            "paper_markov_statistics.csv",
            "paper_transition_flows.csv",
            # Vocal interaction
            "paper_vocal_interaction.csv",
            "paper_event_transitions.csv",
            # Whistle analysis
            "paper_whistle_catalog.csv",
            "paper_whistle_centroids.csv",
            # Soundscape
            "paper_soundscape.csv",
            # Kinematics
            "paper_kinematics.csv",
            "paper_kinematics_by_state.csv",
            # Motif/Syntax
            "paper_motif_summary.csv",
            "paper_motif_details.csv",
            # XAI
            "paper_xai_dominant_features.csv",
            "paper_xai_latent_pca.csv",
            "paper_xai_cluster_importance.csv",
            # Irreversibility
            "paper_irreversibility.csv",
            # Criticality
            "paper_criticality.csv",
            # Topology
            "paper_topology.csv",
            # Latent space
            "paper_latent_space.csv",
            "paper_state_centroids.csv",
            "paper_phase_portrait.csv",
            # Feature distributions
            "paper_feature_distributions.csv",
            "paper_temporal_proportions.csv",
        ]
        for f in csv_files:
            print(f"  - {f}")
        print(f"\nTOTAL: {len(csv_files)} CSV files")
        print("="*60)

    # =========================================================================
    # MAIN RUNNER
    # =========================================================================
    def run(self):
        try:
            self.find_and_load_audio()
            self.extract_features()
            self.train_model()
            self.analyze_clusters()

            self.generate_dashboard()
            self.generate_mandala()
            self.export_language_data()
            self.analyze_burst_substructure()
            self.generate_streamgraph()
            self.generate_chrono_helix()
            self.generate_helix_animation()
            self.generate_vector_field()
            self.generate_sankey_diagram()
            self.generate_recurrence_plot()
            self.generate_horizon_chart()
            self.generate_chord_diagram()
            self.generate_spectrogram_overlay()
            self.generate_polar_histogram()
            self.generate_phase_portrait()
            self.generate_entropy_plot()
            self.generate_voronoi_map()
            self.generate_ridge_plot()
            self.generate_sunburst_sequence()
            self.generate_ici_analysis()
            self.generate_bout_analysis()
            self.generate_markov_analysis()
            self.generate_vocal_interaction_analysis()
            self.generate_whistle_catalog()
            self.generate_soundscape_analysis()
            self.generate_kinematic_analysis()
            self.generate_motif_analysis()
            self.generate_xai_analysis()
            self.generate_irreversibility_analysis()
            self.generate_criticality_analysis()
            self.generate_topological_analysis()
            self.generate_summary_report()
            self.export_paper_statistics()

            print(f"\n{'=' * 60}")
            print("ANALYSIS COMPLETE!")
            print(f"{'=' * 60}")

        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    viz = DolphinProVisualizer()
    viz.run()
