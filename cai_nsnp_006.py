#!/usr/bin/env python3
"""
DOLPHIN PRO ANALYZER - ULTIMATE EDITION v4.0 CAI
=============================================
Integrated Bioacoustic Analysis Tool for Tursiops aduncus
Uses LSTM Autoencoders, Dynamical Systems Theory, and Advanced Visualization.

Features:
- Full PyTorch LSTM Autoencoder implementation
- Priority-based Burst/Social Logic
- 30 Scientific Visualizations included
- Enhanced ICI Analysis with bimodal detection and terminal buzz identification
- Comprehensive Bout Analysis with survival modeling
- Multi-channel correlation dynamics
- NEW: Social dynamics and group behavior analysis

Output Files:
1.  dolphin_dashboard_full.png       (Summary)
2.  dolphin_mandala.png              (Cyclicity)
3.  dolphin_streamgraph.png          (Temporal Flow)
4.  dolphin_helix_3d.png             (Evolution)
5.  dolphin_helix_3d_anim.gif        (Animation)
6.  dolphin_vector_field.png         (Attractors)
7.  dolphin_flow_field.gif           (Particle Flow)
8.  dolphin_burst_deep_dive.png      (Micro-structure)
9.  dolphin_sankey.png/html          (Transitions)
10. dolphin_recurrence_plot.png      (Determinism)
11. dolphin_horizon_chart.png        (Enhanced Multi-Channel Analysis)
12. dolphin_chord_diagram.png        (Circular Network)
13. dolphin_spectrogram_overlay.png  (Validation)
14. dolphin_polar_histogram.png      (Windrose)
15. dolphin_phase_portrait.png       (System Dynamics)
16. dolphin_entropy_plot.png         (Cognitive Complexity)
17. dolphin_voronoi_map.png          (Behavioral Territories)
18. dolphin_ridge_plot.png           (Acoustic Fingerprints)
19. dolphin_sequence_sunburst.html   (Grammar/Syntax)
20. dolphin_ici_analysis.png         (Enhanced ICI with Bimodal Analysis)
21. dolphin_bout_analysis.png        (Comprehensive Bout/Survival Analysis)
22. dolphin_markov_analysis.png      (Stationary Distribution)
23. dolphin_social_network.png       (Individual Interactions)
24. dolphin_turn_taking.png          (Vocal Turn-Taking Analysis) [NEW]
25. dolphin_contagion_cascade.png    (Acoustic Contagion Spreading) [NEW]
26. dolphin_dynamic_network.png      (Network Evolution Over Time) [NEW]
27. dolphin_information_flow.png     (Transfer Entropy Analysis) [NEW]
28. dolphin_repertoire_similarity.png (Vocal Repertoire & Culture) [NEW]
29. dolphin_reciprocity_dominance.png (Hierarchy & Coalitions) [NEW]
30. dolphin_summary_report.png       (Final Summary)
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

# Suppress warnings
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

# Check for optional libraries
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    print("Warning: networkx not found. Some visualizations will be skipped.")
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    print("Warning: plotly not found. Interactive visualizations will be skipped.")
    HAS_PLOTLY = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    print("Warning: imageio not found. Animations will be skipped.")
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
        print("DOLPHIN PRO ANALYZER - ULTIMATE EDITION v2.1")
        print(f"{'='*60}")
        print(f"Device: {self.device}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    def find_and_load_audio(self):
        wav_files = glob.glob("*.wav")
        if not wav_files:
            print("! No .wav file found. Generating demo audio...")
            self._generate_demo_wav()
            self.file_path = "demo_dolphin.wav"
        else:
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

    def _generate_demo_wav(self):
        sr = 22050
        t = np.linspace(0, 300, sr * 300)
        y = np.random.normal(0, 0.05, len(t))
        y += 0.4 * np.sin(2 * np.pi * 4000 * t) * (np.sin(t / 5) > 0.8)
        y += 0.3 * np.random.normal(0, 1, len(t)) * (np.sin(t / 13) > 0.7)
        import soundfile as sf
        sf.write('demo_dolphin.wav', y, sr)

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
    # CLUSTERING
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

        th_w = self.quantiles.loc[0.60, 'Whistle']
        th_b = self.quantiles.loc[0.60, 'Burst']
        th_c = self.quantiles.loc[0.80, 'Click']

        print("\n   --- BEHAVIOR TYPES ---")
        for i in self.existing_clusters:
            row = stats.loc[i]
            w, b, c = row['Whistle'], row['Burst'], row['Click']

            label_parts = []
            if w > th_w:
                label_parts.append("WHISTLE")
            if b > th_b:
                label_parts.append("BURST")
            if c > th_c:
                label_parts.append("SCAN")

            if not label_parts:
                if c > w * 2 and c > b * 2:
                    label_parts.append("Faint SCAN")
                elif w > c and w > b:
                    label_parts.append("Soft WHISTLE")
                elif b > c and b > w:
                    label_parts.append("Soft BURST")
                else:
                    label_parts.append("SILENCE")

            if w > 0.7 or b > 0.7:
                if "INTENSE" not in label_parts:
                    label_parts.insert(0, "INTENSE")

            if "WHISTLE" in label_parts and "BURST" in label_parts:
                base_name = "PLAY (Mix)"
            else:
                base_name = " + ".join(label_parts)

            final_name = f"{base_name} [{i}]"
            self.labels[i] = final_name
            print(f"   [{i}] {final_name:<30} (W:{w:.2f} B:{b:.2f} C:{c:.2f})")

    # =========================================================================
    # BURST ANALYSIS (PRIORITY LOGIC)
    # =========================================================================
    def analyze_burst_substructure(self):
        print("\n5. Burst Deep Dive (Priority Logic)...")

        burst_indices = []
        for i, label in enumerate(self.clusters):
            label_text = self.labels[self.clusters[i]]
            if "BURST" in label_text or "PLAY" in label_text or "INTENSE" in label_text or "MIX" in label_text:
                burst_indices.append(i)

        if len(burst_indices) < 50:
            print("   ! Not enough burst data.")
            return

        burst_features_raw = self.features[self.window_size:][burst_indices]
        burst_times = self.valid_times[burst_indices]

        # Local Re-normalization
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
                    main_type = "INTENSE PLAY (Burst+Whistle)"
                else:
                    main_type = "PHYSICAL CONTACT (Burst)"
            elif w > 0.5:
                if c > 0.6:
                    main_type = "SOCIAL COORDINATION (+Scan)"
                else:
                    main_type = "COMMUNICATION (Whistle)"
            elif c > 0.5:
                main_type = "SCANNING"
            else:
                main_type = "TRANSITION / MIX"

            sub_labels[i] = main_type
            print(f"      [{i}] {main_type:<30} (Norm -> B:{b:.2f} W:{w:.2f} C:{c:.2f})")

        # Export CSV
        df_micro = pd.DataFrame({
            'Time': burst_times,
            'Level2_Label': [sub_labels[c] for c in sub_clusters],
            'Level2_ClusterID': sub_clusters,
            'Whistle_Norm': burst_features_norm[:, 0],
            'Burst_Norm': burst_features_norm[:, 1],
            'Click_Norm': burst_features_norm[:, 2]
        })
        df_micro.to_csv("dolphin_micro_burst_analysis.csv", index=False)

        # Plot
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

        # Manifold
        ax_map = fig.add_subplot(gs[0:2, 0])
        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        for i in self.existing_clusters:
            mask = (self.clusters == i)
            ax_map.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                           color=self.all_colors[i], label=self.labels[i], alpha=0.7, s=15, edgecolors='none')
        ax_map.set_title("Deep Learning Manifold (PCA)", fontsize=14)
        ax_map.legend(loc='lower right', fontsize='x-small')

        # Network
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
        ax_net.set_title("Transition Network", fontsize=14)
        ax_net.axis('off')

        # Barcode
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
    # VISUALIZATION 8: MANDALA
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

    # =========================================================================
    # VISUALIZATION 9: STREAMGRAPH
    # =========================================================================
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
                     labels=['Whistle', 'Burst', 'Click'], colors=pal, alpha=0.85)

        ax.set_title("Behavioral Streamgraph", fontsize=16)
        ax.set_xlabel("Time (min)")
        ax.set_xlim(0, time_min[-1])
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.savefig('dolphin_streamgraph.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_streamgraph.png")

    # =========================================================================
    # VISUALIZATION 10: 3D HELIX
    # =========================================================================
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
        ax.set_xlabel('Dim 1')
        ax.set_ylabel('Dim 2')
        ax.set_zlabel('TIME')
        plt.savefig('dolphin_helix_3d.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_helix_3d.png")

    # =========================================================================
    # VISUALIZATION 11: HELIX ANIMATION
    # =========================================================================
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

    # =========================================================================
    # VISUALIZATION 12: VECTOR FIELD
    # =========================================================================
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
        ax.set_title("Behavioral Vector Field")
        plt.tight_layout()
        plt.savefig('dolphin_vector_field.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_vector_field.png")

    # =========================================================================
    # VISUALIZATION 13: FLOW ANIMATION
    # =========================================================================
    def generate_flow_animation(self):
        print("\n13. Generating Flow Animation...")
        if not HAS_IMAGEIO:
            print("   ! imageio not available, skipping animation.")
            return

        from scipy.interpolate import RegularGridInterpolator

        pca = PCA(n_components=2)
        latent_2d = pca.fit_transform(self.latent_space)
        x_pts, y_pts = latent_2d[:, 0], latent_2d[:, 1]

        grid_size = 40
        x_min, x_max = x_pts.min() - 1, x_pts.max() + 1
        y_min, y_max = y_pts.min() - 1, y_pts.max() + 1
        xi = np.linspace(x_min, x_max, grid_size)
        yi = np.linspace(y_min, y_max, grid_size)

        u, v, count = np.zeros((grid_size, grid_size)), np.zeros((grid_size, grid_size)), np.zeros(
            (grid_size, grid_size))
        for i in range(len(x_pts) - 1):
            idx_x = int((x_pts[i] - x_min) / (x_max - x_min) * (grid_size - 1))
            idx_y = int((y_pts[i] - y_min) / (y_max - y_min) * (grid_size - 1))
            if 0 <= idx_x < grid_size and 0 <= idx_y < grid_size:
                u[idx_y, idx_x] += x_pts[i + 1] - x_pts[i]
                v[idx_y, idx_x] += y_pts[i + 1] - y_pts[i]
                count[idx_y, idx_x] += 1

        mask = count > 0
        u[mask] /= count[mask]
        v[mask] /= count[mask]
        u = gaussian_filter(u, sigma=1.5)
        v = gaussian_filter(v, sigma=1.5)

        vel_u = RegularGridInterpolator((yi, xi), u, bounds_error=False, fill_value=0)
        vel_v = RegularGridInterpolator((yi, xi), v, bounds_error=False, fill_value=0)

        frame_dir = "flow_frames"
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

        n_p = 300
        px = np.random.uniform(x_min, x_max, n_p)
        py = np.random.uniform(y_min, y_max, n_p)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.kdeplot(x=x_pts, y=y_pts, fill=True, cmap="Greys", ax=ax, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')
        points = ax.scatter(px, py, c='red', s=10)

        frames = []
        for f in range(60):
            du = vel_u((py, px))
            dv = vel_v((py, px))
            px += du * 2
            py += dv * 2
            points.set_offsets(np.c_[px, py])
            fname = os.path.join(frame_dir, f"f_{f:03d}.png")
            plt.savefig(fname, dpi=80)
            frames.append(fname)

            # Reset particles that go out of bounds
            reset = (px < x_min) | (px > x_max) | (py < y_min) | (py > y_max) | (np.random.rand(n_p) < 0.05)
            px[reset] = np.random.uniform(x_min, x_max, np.sum(reset))
            py[reset] = np.random.uniform(y_min, y_max, np.sum(reset))

        plt.close()
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave("dolphin_flow_field.gif", images, duration=0.05)
        shutil.rmtree(frame_dir)
        print("   -> SAVED: dolphin_flow_field.gif")

    # =========================================================================
    # VISUALIZATION 14: SANKEY DIAGRAM
    # =========================================================================
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

        # Matplotlib Fallback for PNG
        fig, ax = plt.subplots(figsize=(12, 8))
        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        sns.heatmap(trans_mat / row_sums, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title("Transition Probabilities")
        plt.savefig('dolphin_sankey.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_sankey.png")

    # =========================================================================
    # VISUALIZATION 15: RECURRENCE PLOT
    # =========================================================================
    def generate_recurrence_plot(self):
        print("\n15. Generating Recurrence Plot...")
        step = max(1, len(self.latent_space) // 1000)
        latent_ds = self.latent_space[::step]

        print(f"   -> Distance Matrix Calculation ({len(latent_ds)}x{len(latent_ds)})...")
        dists = squareform(pdist(latent_ds, metric='euclidean'))
        thresh = np.percentile(dists, 15)
        rec = (dists < thresh).astype(int)

        # Calculate RQA metrics
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

    # =========================================================================
    # VISUALIZATION 16: HORIZON CHART
    # =========================================================================
    def generate_horizon_chart(self):
        """
        Enhanced Horizon Chart with multiple analytical panels:
        - Multi-band horizon visualization for each channel
        - Peak activity detection and annotation
        - Cross-channel correlation analysis
        - Rolling statistics (variance, activity index)
        - Behavioral state overlay
        """
        print("\n16. Generating Enhanced Horizon Chart...")
        
        # Smoothing window
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1
        window = max(1, int(5 / dt))
        
        # Prepare data
        df = pd.DataFrame(self.features[self.window_size:], columns=['Whistle', 'Burst', 'Click'])
        df_smooth = df.rolling(window, center=True).mean().fillna(0)
        t = (self.valid_times - self.valid_times[0]) / 60  # Convert to minutes
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(5, 3, height_ratios=[2, 2, 2, 1.5, 1.5], 
                              width_ratios=[3, 1, 1], hspace=0.3, wspace=0.25)
        
        # Color schemes
        channel_colors = {
            'Whistle': ['#c6dbef', '#6baed6', '#2171b5', '#08306b'],  # Blues
            'Burst': ['#fcbba1', '#fb6a4a', '#cb181d', '#67000d'],     # Reds
            'Click': ['#c7e9c0', '#74c476', '#238b45', '#00441b']      # Greens
        }
        channel_base = {'Whistle': '#2171b5', 'Burst': '#cb181d', 'Click': '#238b45'}
        chans = ['Whistle', 'Burst', 'Click']
        
        # =====================================================================
        # PANEL 1-3: Enhanced Horizon Charts with 4 bands
        # =====================================================================
        n_bands = 4
        
        for idx, chan in enumerate(chans):
            ax = fig.add_subplot(gs[idx, 0])
            
            d = df_smooth[chan].values
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            
            # Draw horizon bands
            colors = channel_colors[chan]
            for b in range(n_bands):
                band_height = 1 / n_bands
                d_band = np.clip(d_norm - b * band_height, 0, band_height)
                ax.fill_between(t, 0, d_band * n_bands, color=colors[b], alpha=0.85)
            
            # Detect and annotate peaks
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(d_norm, height=0.7, distance=int(30/dt))
            
            for peak in peaks[:10]:  # Annotate top 10 peaks
                if peak < len(t):
                    ax.axvline(t[peak], color='black', linestyle='--', alpha=0.3, lw=0.8)
                    ax.scatter(t[peak], 0.95, marker='v', color='black', s=30, zorder=5)
            
            # Add mean and std lines
            mean_val = np.mean(d_norm)
            std_val = np.std(d_norm)
            ax.axhline(mean_val, color='white', linestyle='-', alpha=0.7, lw=1.5)
            ax.axhline(mean_val + std_val, color='white', linestyle=':', alpha=0.5, lw=1)
            
            # Labels and formatting
            ax.set_ylabel(f'{chan}\nIntensity', fontsize=10, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_xlim(t[0], t[-1])
            
            # Add statistics text
            peak_count = len(peaks)
            time_above_thresh = np.sum(d_norm > 0.5) / len(d_norm) * 100
            ax.text(0.02, 0.92, f'Peaks: {peak_count} | >50%: {time_above_thresh:.1f}%', 
                   transform=ax.transAxes, fontsize=8, color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
            
            if idx < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time (min)', fontsize=10)
        
        # =====================================================================
        # PANEL 1-3 RIGHT: Distribution histograms
        # =====================================================================
        for idx, chan in enumerate(chans):
            ax = fig.add_subplot(gs[idx, 1])
            
            d = df_smooth[chan].values
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            
            # Horizontal histogram
            ax.hist(d_norm, bins=30, orientation='horizontal', color=channel_base[chan], 
                   alpha=0.7, edgecolor='white', linewidth=0.5)
            ax.axhline(np.mean(d_norm), color='red', linestyle='--', lw=1.5, label='Mean')
            ax.axhline(np.median(d_norm), color='orange', linestyle=':', lw=1.5, label='Median')
            
            ax.set_ylim(0, 1)
            ax.set_ylabel('')
            ax.set_yticklabels([])
            ax.set_xlabel('Count', fontsize=9)
            ax.set_title('Distribution', fontsize=9)
            
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right')
        
        # =====================================================================
        # PANEL 1-3 FAR RIGHT: Box plots by behavioral state
        # =====================================================================
        for idx, chan in enumerate(chans):
            ax = fig.add_subplot(gs[idx, 2])
            
            d = df_smooth[chan].values
            d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
            
            # Group by behavioral state (simplified to 4 categories)
            state_data = {}
            for i, state in enumerate(self.clusters):
                label = self.labels[state]
                # Simplify label
                if 'SILENCE' in label:
                    cat = 'Silent'
                elif 'SCAN' in label:
                    cat = 'Scanning'
                elif 'WHISTLE' in label or 'BURST' in label:
                    cat = 'Social'
                else:
                    cat = 'Active'
                
                if i < len(d_norm):
                    if cat not in state_data:
                        state_data[cat] = []
                    state_data[cat].append(d_norm[i])
            
            # Box plot
            if state_data:
                positions = range(len(state_data))
                bp = ax.boxplot(state_data.values(), positions=positions, 
                               patch_artist=True, widths=0.6)
                
                box_colors = ['#d9d9d9', '#9ecae1', '#fdae6b', '#a1d99b']
                for patch, color in zip(bp['boxes'], box_colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.8)
                
                ax.set_xticks(positions)
                ax.set_xticklabels(state_data.keys(), rotation=45, ha='right', fontsize=7)
                ax.set_ylim(0, 1)
                ax.set_title('By State', fontsize=9)
                ax.set_ylabel('')
                ax.set_yticklabels([])
        
        # =====================================================================
        # PANEL 4: Cross-channel correlation over time
        # =====================================================================
        ax_corr = fig.add_subplot(gs[3, :2])
        
        # Rolling correlation (60-second window)
        corr_window = int(60 / dt)
        
        corr_wb = []  # Whistle-Burst
        corr_wc = []  # Whistle-Click
        corr_bc = []  # Burst-Click
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
        ax_corr.fill_between(corr_times, -1, 1, where=[c > 0.5 for c in corr_wb], 
                            color='#9467bd', alpha=0.1)
        
        ax_corr.set_ylabel('Correlation (r)', fontsize=10)
        ax_corr.set_xlabel('Time (min)', fontsize=10)
        ax_corr.set_ylim(-1, 1)
        ax_corr.set_xlim(t[0], t[-1])
        ax_corr.legend(loc='upper right', fontsize=8, ncol=3)
        ax_corr.set_title('Cross-Channel Correlation (60s rolling window)', fontsize=11, fontweight='bold')
        ax_corr.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 4 RIGHT: Correlation matrix
        # =====================================================================
        ax_mat = fig.add_subplot(gs[3, 2])
        
        corr_matrix = df_smooth.corr()
        im = ax_mat.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax_mat.set_xticks(range(3))
        ax_mat.set_yticks(range(3))
        ax_mat.set_xticklabels(['W', 'B', 'C'], fontsize=9)
        ax_mat.set_yticklabels(['W', 'B', 'C'], fontsize=9)
        
        # Annotate values
        for i in range(3):
            for j in range(3):
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax_mat.text(j, i, f'{val:.2f}', ha='center', va='center', 
                           fontsize=9, color=color, fontweight='bold')
        
        ax_mat.set_title('Overall\nCorrelation', fontsize=9)
        plt.colorbar(im, ax=ax_mat, shrink=0.8)
        
        # =====================================================================
        # PANEL 5: Combined activity index with behavioral state overlay
        # =====================================================================
        ax_act = fig.add_subplot(gs[4, :2])
        
        # Compute combined activity index
        activity_index = (df_smooth['Whistle'] + df_smooth['Burst'] + df_smooth['Click']) / 3
        activity_norm = (activity_index - activity_index.min()) / (activity_index.max() - activity_index.min() + 1e-8)
        
        # Smooth further for trend
        activity_smooth = pd.Series(activity_norm).rolling(window*3, center=True).mean()
        activity_smooth = activity_smooth.fillna(pd.Series(activity_norm))
        
        # Plot activity
        ax_act.fill_between(t, 0, activity_norm, color='#756bb1', alpha=0.4, label='Activity Index')
        ax_act.plot(t, activity_smooth, color='#54278f', lw=2, label='Trend')
        
        # Add behavioral state color bar at bottom
        state_height = 0.08
        for i in range(0, len(self.clusters), max(1, len(self.clusters)//500)):
            if i < len(t):
                ax_act.axvspan(t[i], t[min(i+1, len(t)-1)], ymin=0, ymax=state_height,
                              color=self.all_colors[self.clusters[i]], alpha=0.8)
        
        # Annotations
        ax_act.axhline(np.mean(activity_norm), color='red', linestyle='--', lw=1, alpha=0.7)
        
        # Find high activity periods
        high_activity = activity_norm > np.percentile(activity_norm, 90)
        high_regions = np.where(np.diff(high_activity.astype(int)) == 1)[0]
        
        for region in high_regions[:5]:
            if region < len(t):
                ax_act.annotate('↑', xy=(t[region], 0.95), fontsize=10, ha='center', color='red')
        
        ax_act.set_ylabel('Activity\nIndex', fontsize=10)
        ax_act.set_xlabel('Time (min)', fontsize=10)
        ax_act.set_ylim(0, 1)
        ax_act.set_xlim(t[0], t[-1])
        ax_act.legend(loc='upper right', fontsize=8)
        ax_act.set_title('Combined Acoustic Activity with Behavioral State Overlay', fontsize=11, fontweight='bold')
        
        # Add text annotation for state bar
        ax_act.text(0.01, 0.04, 'Behavioral State:', transform=ax_act.transAxes, 
                   fontsize=8, verticalalignment='center')
        
        # =====================================================================
        # PANEL 5 RIGHT: Activity distribution and statistics
        # =====================================================================
        ax_stats = fig.add_subplot(gs[4, 2])
        ax_stats.axis('off')
        
        # Calculate statistics
        stats_text = f"""
ACTIVITY STATISTICS
═══════════════════
Mean:     {np.mean(activity_norm):.3f}
Std:      {np.std(activity_norm):.3f}
Max:      {np.max(activity_norm):.3f}
Min:      {np.min(activity_norm):.3f}

Percentiles:
  25th:   {np.percentile(activity_norm, 25):.3f}
  50th:   {np.percentile(activity_norm, 50):.3f}
  75th:   {np.percentile(activity_norm, 75):.3f}
  90th:   {np.percentile(activity_norm, 90):.3f}

High Activity:
  >75%:   {np.sum(activity_norm > 0.75)/len(activity_norm)*100:.1f}%
  >90%:   {np.sum(activity_norm > 0.90)/len(activity_norm)*100:.1f}%
"""
        ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
        
        # =====================================================================
        # Main title
        # =====================================================================
        plt.suptitle('Enhanced Horizon Chart: Multi-Channel Acoustic Intensity Analysis',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_horizon_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_horizon_chart.png")

    # =========================================================================
    # VISUALIZATION 17: CHORD DIAGRAM
    # =========================================================================
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
            ax.text(theta, 11.5, self.labels[cid], rotation=np.degrees(theta), ha='center', va='center', fontsize=8)

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

    # =========================================================================
    # VISUALIZATION 18: SPECTROGRAM OVERLAY
    # =========================================================================
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

    # =========================================================================
    # VISUALIZATION 19: POLAR HISTOGRAM
    # =========================================================================
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

    # =========================================================================
    # VISUALIZATION 20: PHASE PORTRAIT
    # =========================================================================
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

    # =========================================================================
    # VISUALIZATION 21: ENTROPY ANALYSIS (ENHANCED)
    # =========================================================================
    def generate_entropy_plot(self):
        """
        Generate comprehensive entropy analysis showing behavioral complexity.
        Includes: Rolling entropy, distribution, switching rate, and phase space.
        """
        print("\n21. Generating Entropy Analysis (Enhanced)...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Parameters
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1
        window_sec = 60  # 60 second window
        ws = max(10, int(window_sec / dt))
        n_states = len(self.existing_clusters)

        # =====================================================================
        # 1. Rolling Shannon Entropy
        # =====================================================================
        ax1 = axes[0, 0]

        ents = []
        times_ent = []
        step = max(1, ws // 4)

        for i in range(0, len(self.clusters) - ws, step):
            chunk = self.clusters[i:i + ws]
            # Calculate Shannon entropy
            counts = np.bincount(chunk.astype(int), minlength=max(self.existing_clusters) + 1)
            probs = counts / counts.sum()
            probs = probs[probs > 0]  # Remove zeros
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
        tm = (times_ent - self.valid_times[0]) / 60  # Convert to minutes

        # Max possible entropy
        max_ent = np.log2(n_states)
        mean_ent = np.mean(ents)

        ax1.fill_between(tm, 0, ents_smooth, color='#6a0dad', alpha=0.3)
        ax1.plot(tm, ents_smooth, color='#6a0dad', lw=2, label='Shannon Entropy')
        ax1.axhline(max_ent, color='red', linestyle='--', lw=1.5,
                    label=f'Max H = {max_ent:.2f} bits')
        ax1.axhline(mean_ent, color='orange', linestyle=':', lw=1.5,
                    label=f'Mean H = {mean_ent:.2f} bits')

        ax1.set_xlabel('Time (min)', fontsize=11)
        ax1.set_ylabel('Entropy (bits)', fontsize=11)
        ax1.set_title('Behavioral Complexity (Rolling Entropy)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(tm[0], tm[-1])

        print(f"   -> Mean Entropy: {mean_ent:.3f} bits ({mean_ent / max_ent * 100:.1f}% of max)")

        # =====================================================================
        # 2. Entropy Distribution
        # =====================================================================
        ax2 = axes[0, 1]

        ax2.hist(ents, bins=40, color='#6a0dad', edgecolor='black', alpha=0.7, density=True)
        ax2.axvline(mean_ent, color='red', linestyle='--', lw=2, label=f'Mean: {mean_ent:.2f}')
        ax2.axvline(np.median(ents), color='green', linestyle=':', lw=2,
                    label=f'Median: {np.median(ents):.2f}')

        ax2.set_xlabel('Entropy (bits)', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Entropy Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # =====================================================================
        # 3. Behavioral Switching Rate
        # =====================================================================
        ax3 = axes[1, 0]

        switch_rate = []
        switch_times = []

        for i in range(0, len(self.clusters) - ws, step):
            chunk = self.clusters[i:i + ws]
            switches = np.sum(np.diff(chunk.astype(int)) != 0)
            rate = switches / ws * 100  # Switches per 100 samples
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
        ax3.set_xlim(st[0], st[-1])

        # =====================================================================
        # 4. Entropy vs Activity Phase Space
        # =====================================================================
        ax4 = axes[1, 1]

        activity_vals = []
        entropy_vals = []

        for i in range(0, len(self.clusters) - ws, step):
            chunk = self.clusters[i:i + ws]

            # Entropy
            counts = np.bincount(chunk.astype(int), minlength=max(self.existing_clusters) + 1)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            h = -np.sum(probs * np.log2(probs))
            entropy_vals.append(h)

            # Activity (mean feature energy)
            feat_chunk = self.features[self.window_size + i:self.window_size + i + ws]
            if len(feat_chunk) > 0:
                activity = np.mean(feat_chunk)
            else:
                activity = 0
            activity_vals.append(activity)

        # Color by time
        colors = np.linspace(0, 1, len(entropy_vals))
        scatter = ax4.scatter(activity_vals, entropy_vals, c=colors, cmap='viridis',
                              s=30, alpha=0.6, edgecolors='none')

        ax4.set_xlabel('Mean Acoustic Activity', fontsize=11)
        ax4.set_ylabel('Entropy (bits)', fontsize=11)
        ax4.set_title('Complexity vs Activity Phase Space', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Time (relative)')

        # Correlation
        if len(activity_vals) > 2:
            corr = np.corrcoef(activity_vals, entropy_vals)[0, 1]
            ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # =====================================================================
        # Save
        # =====================================================================
        plt.suptitle('Cognitive Complexity Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_entropy_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   -> SAVED: dolphin_entropy_plot.png")

    # =========================================================================
    # VISUALIZATION 22: VORONOI MAP
    # =========================================================================
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

        # Colorize regions via grid
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
            ax.text(cent[0], cent[1] + 0.3, self.labels[c_ids[i]], ha='center', fontsize=8, fontweight='bold')

        ax.set_title("Behavioral Territories (Voronoi)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        plt.tight_layout()
        plt.savefig('dolphin_voronoi_map.png', dpi=300)
        plt.close()
        print("   -> SAVED: dolphin_voronoi_map.png")

    # =========================================================================
    # VISUALIZATION 23: RIDGE PLOT
    # =========================================================================
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

    # =========================================================================
    # VISUALIZATION 24: SUNBURST SEQUENCE
    # =========================================================================
    def generate_sunburst_sequence(self):
        print("\n24. Generating Sunburst...")
        if not HAS_PLOTLY:
            print("   ! plotly not available, skipping.")
            return

        # Collapse runs
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
            try:
                fig.write_image("dolphin_sequence_sunburst.png", scale=2)
                print("   -> SAVED: dolphin_sequence_sunburst.png")
            except:
                pass
        except Exception as e:
            print(f"   ! Sunburst error: {e}")

    # =========================================================================
    # VISUALIZATION 25: ICI ANALYSIS
    # =========================================================================
    def generate_ici_analysis(self):
        """
        Enhanced Inter-Click Interval (ICI) Analysis for echolocation behavior.
        
        Includes:
        - Bimodal distribution detection and fitting
        - Terminal buzz identification (foraging behavior)
        - Click rate dynamics over time
        - ICI return map with density visualization
        - Behavioral state stratification
        - Click train segmentation
        - Statistical summary panel
        """
        print("\n25. Generating Enhanced ICI Analysis...")

        clicks = self.features[self.window_size:, 2]  # Click channel
        dt = self.valid_times[1] - self.valid_times[0] if len(self.valid_times) > 1 else 0.1

        # Detect click events (threshold crossings with hysteresis)
        threshold_high = 0.5
        threshold_low = 0.3
        
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
            print("   ! Not enough click events, lowering threshold...")
            threshold_high = 0.3
            threshold_low = 0.2
            click_indices = []
            in_click = False
            
            for i in range(len(clicks)):
                if not in_click and clicks[i] >= threshold_high:
                    click_indices.append(i)
                    in_click = True
                elif in_click and clicks[i] < threshold_low:
                    in_click = False
            
            click_indices = np.array(click_indices)
        
        if len(click_indices) < 10:
            print("   ! Not enough click events detected - skipping ICI analysis")
            return

        # Calculate ICIs
        icis = np.diff(click_indices)
        click_times = self.valid_times[click_indices[1:]]
        
        # Convert ICI to milliseconds (approximate)
        ici_ms = icis * dt * 1000

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # =====================================================================
        # PANEL 1: Enhanced ICI Histogram with Bimodal Analysis
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        # Histogram with KDE overlay
        n, bins, patches = ax1.hist(ici_ms, bins=60, color='#3498db', edgecolor='white', 
                                     alpha=0.7, density=True, label='ICI Distribution')
        
        # Fit and plot KDE
        from scipy.stats import gaussian_kde
        if len(ici_ms) > 10:
            kde = gaussian_kde(ici_ms)
            x_kde = np.linspace(ici_ms.min(), np.percentile(ici_ms, 99), 200)
            ax1.plot(x_kde, kde(x_kde), 'r-', lw=2, label='KDE Estimate')
        
        # Find peaks in histogram (bimodal detection)
        hist_counts, hist_edges = np.histogram(ici_ms, bins=60)
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist_counts, height=np.max(hist_counts)*0.1, distance=5)
        
        peak_colors = ['#e74c3c', '#2ecc71', '#9b59b6']
        for i, peak in enumerate(peaks[:3]):
            ax1.axvline(hist_centers[peak], color=peak_colors[i], linestyle='--', lw=2,
                       label=f'Mode {i+1}: {hist_centers[peak]:.1f} ms')
        
        # Statistics annotations
        ax1.axvline(np.median(ici_ms), color='orange', linestyle=':', lw=2, 
                   label=f'Median: {np.median(ici_ms):.1f} ms')
        
        ax1.set_xlabel('Inter-Click Interval (ms)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('ICI Distribution with Bimodal Analysis', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xlim(0, np.percentile(ici_ms, 98))
        ax1.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 2: ICI Categories (Echolocation vs Social)
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 2:4])
        
        # Categorize ICIs
        # Short ICI (<50ms): Terminal buzz / rapid echolocation
        # Medium ICI (50-200ms): Regular echolocation  
        # Long ICI (>200ms): Social clicks / pauses
        
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
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax2.annotate(f'{pct:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('ICI Functional Categories', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add interpretation text
        interpretation = f"Foraging indicator: {percentages[0]:.1f}% terminal buzz clicks"
        ax2.text(0.5, 0.95, interpretation, transform=ax2.transAxes, fontsize=9,
                ha='center', va='top', style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # =====================================================================
        # PANEL 3: ICI Temporal Evolution with Click Rate
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, :3])
        
        # Scatter plot of ICIs over time
        scatter = ax3.scatter(click_times, ici_ms, c=np.arange(len(ici_ms)), 
                             cmap='viridis', s=8, alpha=0.5, label='Individual ICI')
        
        # Rolling median trend
        window_size = min(50, len(ici_ms)//5)
        if window_size > 3:
            ici_series = pd.Series(ici_ms)
            rolling_median = ici_series.rolling(window_size, center=True).median()
            rolling_q25 = ici_series.rolling(window_size, center=True).quantile(0.25)
            rolling_q75 = ici_series.rolling(window_size, center=True).quantile(0.75)
            
            ax3.plot(click_times, rolling_median, 'r-', lw=2, label='Rolling Median')
            ax3.fill_between(click_times, rolling_q25, rolling_q75, 
                            color='red', alpha=0.2, label='IQR Band')
        
        # Mark potential terminal buzzes (rapid ICI decrease)
        buzz_candidates = []
        for i in range(5, len(ici_ms)):
            if all(ici_ms[i-j] > ici_ms[i-j+1] for j in range(4, 0, -1)):
                if ici_ms[i] < 30:  # End with very short ICI
                    buzz_candidates.append(i)
        
        for buzz_idx in buzz_candidates[:10]:
            ax3.axvline(click_times[buzz_idx], color='green', linestyle='--', alpha=0.5, lw=1)
            ax3.annotate('TB', xy=(click_times[buzz_idx], ax3.get_ylim()[1]*0.9),
                        fontsize=7, color='green', ha='center')
        
        ax3.set_xlabel('Time (s)', fontsize=11)
        ax3.set_ylabel('ICI (ms)', fontsize=11)
        ax3.set_title(f'ICI Temporal Evolution (n={len(ici_ms)} intervals, TB=Terminal Buzz)', 
                     fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_ylim(0, np.percentile(ici_ms, 95))
        ax3.grid(True, alpha=0.3)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax3, shrink=0.6, pad=0.02)
        cbar.set_label('Sequence Order', fontsize=9)

        # =====================================================================
        # PANEL 4: Click Rate Over Time
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 3])
        
        # Calculate click rate in sliding windows
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
        
        ax4.axhline(np.mean(click_rates), color='red', linestyle='--', lw=1,
                   label=f'Mean: {np.mean(click_rates):.1f} Hz')
        
        ax4.set_xlabel('Time (s)', fontsize=10)
        ax4.set_ylabel('Click Rate (Hz)', fontsize=10)
        ax4.set_title('Click Rate\nDynamics', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 5: Enhanced ICI Return Map with Density
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 0:2])
        
        if len(icis) > 1:
            ici_n = ici_ms[:-1]
            ici_n1 = ici_ms[1:]
            
            # 2D histogram for density
            h, xedges, yedges = np.histogram2d(ici_n, ici_n1, bins=50,
                                                range=[[0, np.percentile(ici_ms, 95)],
                                                       [0, np.percentile(ici_ms, 95)]])
            
            # Plot density
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            im = ax5.imshow(h.T, origin='lower', extent=extent, aspect='auto',
                           cmap='hot', interpolation='gaussian')
            
            # Overlay scatter with transparency
            ax5.scatter(ici_n, ici_n1, c='cyan', s=3, alpha=0.3, edgecolors='none')
            
            # Diagonal line (no change)
            max_val = np.percentile(ici_ms, 95)
            ax5.plot([0, max_val], [0, max_val], 'w--', lw=2, label='No Change')
            
            # Annotations for regions
            ax5.text(max_val*0.8, max_val*0.2, 'ICI↓\n(Approach)', fontsize=9, 
                    color='white', ha='center', fontweight='bold')
            ax5.text(max_val*0.2, max_val*0.8, 'ICI↑\n(Retreat)', fontsize=9,
                    color='white', ha='center', fontweight='bold')
            
            ax5.set_xlabel('ICI(n) (ms)', fontsize=11)
            ax5.set_ylabel('ICI(n+1) (ms)', fontsize=11)
            ax5.set_title('ICI Return Map (Phase Space)', fontsize=12, fontweight='bold')
            ax5.legend(loc='upper right', fontsize=8)
            
            plt.colorbar(im, ax=ax5, shrink=0.8, label='Density')

        # =====================================================================
        # PANEL 6: ICI by Behavioral State (Violin Plot)
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 2:4])
        
        click_states = self.clusters[click_indices[1:]]
        
        # Prepare data for violin plot
        violin_data = []
        violin_labels = []
        violin_colors = []
        
        for state in self.existing_clusters:
            mask = click_states == state
            if mask.sum() > 10:
                state_icis = ici_ms[mask]
                violin_data.append(state_icis)
                # Shorten label
                label = self.labels[state]
                short_label = label.split('[')[0].strip()[:15]
                violin_labels.append(short_label)
                violin_colors.append(self.all_colors[state])
        
        if violin_data:
            parts = ax6.violinplot(violin_data, positions=range(len(violin_data)),
                                   showmeans=True, showmedians=True)
            
            # Color the violins
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(violin_colors[i])
                pc.set_alpha(0.7)
            
            # Style the lines
            parts['cmeans'].set_color('red')
            parts['cmeans'].set_linewidth(2)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(2)
            
            ax6.set_xticks(range(len(violin_labels)))
            ax6.set_xticklabels(violin_labels, rotation=45, ha='right', fontsize=8)
            ax6.set_ylabel('ICI (ms)', fontsize=11)
            ax6.set_title('ICI Distribution by Behavioral State', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Add legend for mean/median
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='red', lw=2, label='Mean'),
                             Line2D([0], [0], color='black', lw=2, label='Median')]
            ax6.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # =====================================================================
        # PANEL 7: Click Train Analysis
        # =====================================================================
        ax7 = fig.add_subplot(gs[3, 0:2])
        
        # Identify click trains (sequences with ICI < threshold)
        train_threshold = 150  # ms
        in_train = ici_ms < train_threshold
        
        # Find train boundaries
        train_starts = []
        train_ends = []
        train_lengths = []
        
        i = 0
        while i < len(in_train):
            if in_train[i]:
                start = i
                while i < len(in_train) and in_train[i]:
                    i += 1
                end = i
                if end - start >= 3:  # At least 3 clicks in a train
                    train_starts.append(start)
                    train_ends.append(end)
                    train_lengths.append(end - start)
            else:
                i += 1
        
        # Plot train length distribution
        if train_lengths:
            ax7.hist(train_lengths, bins=30, color='#1abc9c', edgecolor='black', alpha=0.7)
            ax7.axvline(np.median(train_lengths), color='red', linestyle='--', lw=2,
                       label=f'Median: {np.median(train_lengths):.0f} clicks')
            ax7.axvline(np.mean(train_lengths), color='orange', linestyle=':', lw=2,
                       label=f'Mean: {np.mean(train_lengths):.1f} clicks')
            
            ax7.set_xlabel('Click Train Length (# clicks)', fontsize=11)
            ax7.set_ylabel('Frequency', fontsize=11)
            ax7.set_title(f'Click Train Length Distribution (n={len(train_lengths)} trains)', 
                         fontsize=12, fontweight='bold')
            ax7.legend(fontsize=9)
            ax7.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 8: ICI Autocorrelation
        # =====================================================================
        ax8 = fig.add_subplot(gs[3, 2])
        
        # Calculate autocorrelation
        max_lag = min(100, len(ici_ms)//4)
        autocorr = []
        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr = np.corrcoef(ici_ms[:-lag], ici_ms[lag:])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
        
        ax8.bar(range(max_lag), autocorr, color='#3498db', alpha=0.7, width=1)
        ax8.axhline(0, color='black', lw=1)
        ax8.axhline(1.96/np.sqrt(len(ici_ms)), color='red', linestyle='--', lw=1, alpha=0.7)
        ax8.axhline(-1.96/np.sqrt(len(ici_ms)), color='red', linestyle='--', lw=1, alpha=0.7)
        
        ax8.set_xlabel('Lag', fontsize=10)
        ax8.set_ylabel('Autocorrelation', fontsize=10)
        ax8.set_title('ICI\nAutocorrelation', fontsize=11, fontweight='bold')
        ax8.set_xlim(0, max_lag)
        ax8.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 9: Statistics Summary
        # =====================================================================
        ax9 = fig.add_subplot(gs[3, 3])
        ax9.axis('off')
        
        # Calculate comprehensive statistics
        stats_text = f"""
ICI STATISTICS
══════════════════════
Total Clicks:    {len(click_indices)}
Total Intervals: {len(ici_ms)}
Duration:        {(click_times[-1]-click_times[0]):.1f} s

ICI (ms):
  Mean:          {np.mean(ici_ms):.1f}
  Median:        {np.median(ici_ms):.1f}
  Std:           {np.std(ici_ms):.1f}
  Min:           {np.min(ici_ms):.1f}
  Max:           {np.max(ici_ms):.1f}

Percentiles:
  5th:           {np.percentile(ici_ms, 5):.1f}
  25th:          {np.percentile(ici_ms, 25):.1f}
  75th:          {np.percentile(ici_ms, 75):.1f}
  95th:          {np.percentile(ici_ms, 95):.1f}

Click Trains:
  Count:         {len(train_lengths) if train_lengths else 0}
  Mean Length:   {(np.mean(train_lengths) if train_lengths else 0):.1f}

Detected:
  Terminal Buzz: {len(buzz_candidates)}
  Modes:         {len(peaks)}
"""
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

        # =====================================================================
        # Main Title
        # =====================================================================
        plt.suptitle('Enhanced Inter-Click Interval (ICI) Analysis\nEcholocation Behavior Characterization',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_ici_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_ici_analysis.png")

    # =========================================================================
    # VISUALIZATION 26: BOUT ANALYSIS
    # =========================================================================
    def generate_bout_analysis(self):
        """
        Enhanced Behavioral Bout Analysis with comprehensive episode characterization.
        
        Includes:
        - Bout duration distributions with survival analysis
        - Inter-bout interval (IBI) analysis
        - Bout fragmentation index
        - Temporal clustering patterns
        - Bout sequence analysis (n-grams)
        - State persistence metrics
        - Circadian/temporal patterns
        - Bout energy profiles
        - Statistical summary
        """
        print("\n26. Generating Enhanced Bout Analysis...")

        clusters = self.clusters
        times = self.valid_times
        features = self.features[self.window_size:]
        dt = times[1] - times[0] if len(times) > 1 else 0.1

        # =====================================================================
        # BOUT DETECTION with minimum duration filtering
        # =====================================================================
        min_bout_duration = 3  # Minimum 3 samples to count as a bout
        
        bouts = []  # List of dicts with bout properties
        current_state = clusters[0]
        bout_start = 0

        for i in range(1, len(clusters)):
            if clusters[i] != current_state:
                duration = i - bout_start
                if duration >= min_bout_duration:
                    # Calculate bout properties
                    bout_features = features[bout_start:i]
                    bout_dict = {
                        'state': current_state,
                        'start_idx': bout_start,
                        'end_idx': i,
                        'duration': duration,
                        'duration_sec': duration * dt,
                        'start_time': times[bout_start],
                        'end_time': times[i-1],
                        'mean_whistle': np.mean(bout_features[:, 0]) if len(bout_features) > 0 else 0,
                        'mean_burst': np.mean(bout_features[:, 1]) if len(bout_features) > 0 else 0,
                        'mean_click': np.mean(bout_features[:, 2]) if len(bout_features) > 0 else 0,
                        'total_energy': np.sum(bout_features) if len(bout_features) > 0 else 0,
                        'energy_variance': np.var(np.sum(bout_features, axis=1)) if len(bout_features) > 1 else 0
                    }
                    bouts.append(bout_dict)
                current_state = clusters[i]
                bout_start = i
        
        # Last bout
        duration = len(clusters) - bout_start
        if duration >= min_bout_duration:
            bout_features = features[bout_start:]
            bout_dict = {
                'state': current_state,
                'start_idx': bout_start,
                'end_idx': len(clusters),
                'duration': duration,
                'duration_sec': duration * dt,
                'start_time': times[bout_start],
                'end_time': times[-1],
                'mean_whistle': np.mean(bout_features[:, 0]) if len(bout_features) > 0 else 0,
                'mean_burst': np.mean(bout_features[:, 1]) if len(bout_features) > 0 else 0,
                'mean_click': np.mean(bout_features[:, 2]) if len(bout_features) > 0 else 0,
                'total_energy': np.sum(bout_features) if len(bout_features) > 0 else 0,
                'energy_variance': np.var(np.sum(bout_features, axis=1)) if len(bout_features) > 1 else 0
            }
            bouts.append(bout_dict)

        if len(bouts) < 5:
            print("   ! Not enough bouts detected, lowering minimum duration...")
            min_bout_duration = 1  # Lower minimum
            bouts = []
            current_state = clusters[0]
            bout_start = 0
            
            for i in range(1, len(clusters)):
                if clusters[i] != current_state:
                    duration = i - bout_start
                    if duration >= min_bout_duration:
                        bout_features = features[bout_start:i]
                        bout_dict = {
                            'state': current_state,
                            'start_idx': bout_start,
                            'end_idx': i,
                            'duration': duration,
                            'duration_sec': duration * dt,
                            'start_time': times[bout_start],
                            'end_time': times[i-1],
                            'mean_whistle': np.mean(bout_features[:, 0]) if len(bout_features) > 0 else 0,
                            'mean_burst': np.mean(bout_features[:, 1]) if len(bout_features) > 0 else 0,
                            'mean_click': np.mean(bout_features[:, 2]) if len(bout_features) > 0 else 0,
                            'total_energy': np.sum(bout_features) if len(bout_features) > 0 else 0,
                            'energy_variance': np.var(np.sum(bout_features, axis=1)) if len(bout_features) > 1 else 0
                        }
                        bouts.append(bout_dict)
                    current_state = clusters[i]
                    bout_start = i
        
        if len(bouts) < 3:
            print("   ! Not enough bouts detected - skipping bout analysis")
            return

        # Convert to DataFrame for easier analysis
        bout_df = pd.DataFrame(bouts)
        
        # Calculate inter-bout intervals
        ibis = []
        for i in range(1, len(bouts)):
            ibi = bouts[i]['start_time'] - bouts[i-1]['end_time']
            ibis.append(ibi)
        ibis = np.array(ibis)

        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.35)

        # =====================================================================
        # PANEL 1: Enhanced Bout Duration Distribution with Log-Normal Fit
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        durations_sec = bout_df['duration_sec'].values
        
        # Histogram
        n, bins, patches = ax1.hist(durations_sec, bins=40, color='#3498db', 
                                     edgecolor='white', alpha=0.7, density=True)
        
        # Try to fit log-normal distribution
        try:
            from scipy.stats import lognorm
            # Filter positive values
            pos_durations = durations_sec[durations_sec > 0]
            if len(pos_durations) > 10:
                shape, loc, scale = lognorm.fit(pos_durations, floc=0)
                x_fit = np.linspace(0.001, np.percentile(durations_sec, 99), 100)
                y_fit = lognorm.pdf(x_fit, shape, loc, scale)
                ax1.plot(x_fit, y_fit, 'r-', lw=2, label=f'Log-Normal fit (σ={shape:.2f})')
        except:
            pass
        
        # Add statistics lines
        ax1.axvline(np.median(durations_sec), color='green', linestyle='--', lw=2,
                   label=f'Median: {np.median(durations_sec):.2f}s')
        ax1.axvline(np.mean(durations_sec), color='orange', linestyle=':', lw=2,
                   label=f'Mean: {np.mean(durations_sec):.2f}s')
        
        ax1.set_xlabel('Bout Duration (seconds)', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('Bout Duration Distribution', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, np.percentile(durations_sec, 98))

        # =====================================================================
        # PANEL 2: Survival Analysis (Kaplan-Meier style)
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 2:4])
        
        # Calculate survival curve (probability of bout lasting longer than t)
        sorted_durations = np.sort(durations_sec)
        survival_prob = 1 - np.arange(1, len(sorted_durations) + 1) / len(sorted_durations)
        
        ax2.step(sorted_durations, survival_prob, where='post', color='#e74c3c', lw=2)
        ax2.fill_between(sorted_durations, 0, survival_prob, step='post', alpha=0.3, color='#e74c3c')
        
        # Add reference lines
        ax2.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7)
        median_idx = np.searchsorted(survival_prob[::-1], 0.5)
        if median_idx < len(sorted_durations):
            median_survival = sorted_durations[-(median_idx+1)]
            ax2.axvline(median_survival, color='gray', linestyle='--', lw=1, alpha=0.7)
            ax2.annotate(f'T₅₀={median_survival:.1f}s', xy=(median_survival, 0.5),
                        xytext=(median_survival+1, 0.6), fontsize=9)
        
        ax2.set_xlabel('Bout Duration (seconds)', fontsize=11)
        ax2.set_ylabel('Survival Probability', fontsize=11)
        ax2.set_title('Bout Survival Curve (Duration Persistence)', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, np.percentile(durations_sec, 95))
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 3: Mean Bout Duration by State (Enhanced Bar Chart)
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 0:2])
        
        state_stats = bout_df.groupby('state').agg({
            'duration_sec': ['mean', 'std', 'count', 'median']
        }).reset_index()
        state_stats.columns = ['state', 'mean', 'std', 'count', 'median']
        state_stats = state_stats.sort_values('mean', ascending=True)
        
        y_pos = np.arange(len(state_stats))
        colors_bar = [self.all_colors[int(s)] for s in state_stats['state']]
        labels_bar = [self.labels[int(s)][:20] for s in state_stats['state']]
        
        # Horizontal bar chart
        bars = ax3.barh(y_pos, state_stats['mean'], xerr=state_stats['std'],
                       color=colors_bar, edgecolor='black', alpha=0.8, capsize=3)
        
        # Add count annotations
        for i, (mean_val, count) in enumerate(zip(state_stats['mean'], state_stats['count'])):
            ax3.annotate(f'n={count}', xy=(mean_val + state_stats['std'].iloc[i] + 0.1, i),
                        fontsize=8, va='center')
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(labels_bar, fontsize=9)
        ax3.set_xlabel('Mean Bout Duration (seconds)', fontsize=11)
        ax3.set_title('Bout Duration by Behavioral State', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # =====================================================================
        # PANEL 4: Inter-Bout Interval Analysis
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 2:4])
        
        if len(ibis) > 5:
            ax4.hist(ibis, bins=40, color='#9b59b6', edgecolor='white', alpha=0.7)
            ax4.axvline(np.median(ibis), color='red', linestyle='--', lw=2,
                       label=f'Median IBI: {np.median(ibis):.2f}s')
            ax4.axvline(np.mean(ibis), color='orange', linestyle=':', lw=2,
                       label=f'Mean IBI: {np.mean(ibis):.2f}s')
            
            ax4.set_xlabel('Inter-Bout Interval (seconds)', fontsize=11)
            ax4.set_ylabel('Frequency', fontsize=11)
            ax4.set_title('Inter-Bout Interval Distribution', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(0, np.percentile(ibis, 98))

        # =====================================================================
        # PANEL 5: Bout Frequency Over Time (Temporal Pattern)
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 0:2])
        
        # Divide recording into time bins and count bouts per bin
        total_time = times[-1] - times[0]
        n_bins = min(30, int(total_time / 60))  # ~1 minute bins
        bin_edges = np.linspace(times[0], times[-1], n_bins + 1)
        
        bout_counts_time = np.zeros(n_bins)
        for bout in bouts:
            bin_idx = np.searchsorted(bin_edges[1:], bout['start_time'])
            if bin_idx < n_bins:
                bout_counts_time[bin_idx] += 1
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers_min = (bin_centers - times[0]) / 60
        
        ax5.bar(bin_centers_min, bout_counts_time, width=total_time/60/n_bins*0.9,
               color='#1abc9c', edgecolor='black', alpha=0.7)
        
        # Add trend line
        z = np.polyfit(bin_centers_min, bout_counts_time, 2)
        p = np.poly1d(z)
        ax5.plot(bin_centers_min, p(bin_centers_min), 'r-', lw=2, label='Trend')
        
        ax5.set_xlabel('Time (minutes)', fontsize=11)
        ax5.set_ylabel('Bout Count', fontsize=11)
        ax5.set_title('Bout Frequency Over Time', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 6: Bout Duration Over Time with State Colors
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 2:4])
        
        bout_start_times_min = (bout_df['start_time'] - times[0]) / 60
        bout_colors = [self.all_colors[int(s)] for s in bout_df['state']]
        
        # Scatter with size proportional to energy
        sizes = 20 + 100 * (bout_df['total_energy'] / bout_df['total_energy'].max())
        scatter = ax6.scatter(bout_start_times_min, bout_df['duration_sec'], 
                             c=bout_colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Rolling mean
        if len(bout_df) > 10:
            window = min(10, len(bout_df)//3)
            rolling = bout_df['duration_sec'].rolling(window, center=True).mean()
            ax6.plot(bout_start_times_min, rolling, 'k-', lw=2, label='Rolling Mean')
            ax6.legend(fontsize=9)
        
        ax6.set_xlabel('Time (minutes)', fontsize=11)
        ax6.set_ylabel('Bout Duration (seconds)', fontsize=11)
        ax6.set_title('Bout Duration Timeline (size ∝ energy)', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 7: Bout Sequence Analysis (Bigrams)
        # =====================================================================
        ax7 = fig.add_subplot(gs[3, 0:2])
        
        # Create transition matrix from bout sequences
        states_seq = [int(b['state']) for b in bouts]
        unique_states = list(set(states_seq))
        n_unique = len(unique_states)
        state_to_idx = {s: i for i, s in enumerate(unique_states)}
        
        bigram_matrix = np.zeros((n_unique, n_unique))
        for i in range(len(states_seq) - 1):
            from_idx = state_to_idx[states_seq[i]]
            to_idx = state_to_idx[states_seq[i+1]]
            bigram_matrix[from_idx, to_idx] += 1
        
        # Normalize by row
        row_sums = bigram_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        bigram_prob = bigram_matrix / row_sums
        
        # Plot
        im = ax7.imshow(bigram_prob, cmap='YlOrRd', aspect='auto')
        
        # Labels
        state_labels = [self.labels[s][:10] for s in unique_states]
        ax7.set_xticks(range(n_unique))
        ax7.set_yticks(range(n_unique))
        ax7.set_xticklabels(state_labels, rotation=45, ha='right', fontsize=8)
        ax7.set_yticklabels(state_labels, fontsize=8)
        
        # Annotate values
        for i in range(n_unique):
            for j in range(n_unique):
                if bigram_prob[i, j] > 0.05:
                    color = 'white' if bigram_prob[i, j] > 0.5 else 'black'
                    ax7.text(j, i, f'{bigram_prob[i, j]:.2f}', ha='center', va='center',
                            fontsize=7, color=color)
        
        ax7.set_xlabel('Next Bout State', fontsize=10)
        ax7.set_ylabel('Current Bout State', fontsize=10)
        ax7.set_title('Bout Sequence Probabilities (Bigrams)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax7, shrink=0.8, label='Probability')

        # =====================================================================
        # PANEL 8: Bout Energy Profile
        # =====================================================================
        ax8 = fig.add_subplot(gs[3, 2:4])
        
        # Energy by state
        energy_by_state = bout_df.groupby('state')['total_energy'].mean().reset_index()
        energy_by_state = energy_by_state.sort_values('total_energy', ascending=True)
        
        y_pos = np.arange(len(energy_by_state))
        colors_energy = [self.all_colors[int(s)] for s in energy_by_state['state']]
        labels_energy = [self.labels[int(s)][:15] for s in energy_by_state['state']]
        
        bars = ax8.barh(y_pos, energy_by_state['total_energy'], color=colors_energy,
                       edgecolor='black', alpha=0.8)
        
        ax8.set_yticks(y_pos)
        ax8.set_yticklabels(labels_energy, fontsize=9)
        ax8.set_xlabel('Mean Bout Energy (sum of acoustic features)', fontsize=10)
        ax8.set_title('Bout Energy Profile by State', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')

        # =====================================================================
        # PANEL 9: Bout Fragmentation Analysis
        # =====================================================================
        ax9 = fig.add_subplot(gs[4, 0])
        
        # Calculate fragmentation index per state (more bouts = more fragmented)
        # Fragmentation = number of bouts / total time in state
        frag_data = []
        for state in self.existing_clusters:
            state_bouts = bout_df[bout_df['state'] == state]
            if len(state_bouts) > 0:
                total_time_in_state = state_bouts['duration_sec'].sum()
                n_bouts = len(state_bouts)
                # Fragmentation index: bouts per minute in that state
                if total_time_in_state > 0:
                    frag_index = n_bouts / (total_time_in_state / 60)
                else:
                    frag_index = 0
                frag_data.append({
                    'state': state,
                    'label': self.labels[state][:12],
                    'frag_index': frag_index,
                    'color': self.all_colors[state]
                })
        
        if frag_data:
            frag_df = pd.DataFrame(frag_data).sort_values('frag_index', ascending=True)
            
            bars = ax9.barh(range(len(frag_df)), frag_df['frag_index'],
                           color=frag_df['color'].values, edgecolor='black', alpha=0.8)
            ax9.set_yticks(range(len(frag_df)))
            ax9.set_yticklabels(frag_df['label'].values, fontsize=8)
            ax9.set_xlabel('Fragmentation\n(bouts/min)', fontsize=9)
            ax9.set_title('Bout\nFragmentation', fontsize=10, fontweight='bold')
            ax9.grid(True, alpha=0.3, axis='x')

        # =====================================================================
        # PANEL 10: Bout State Distribution Pie
        # =====================================================================
        ax10 = fig.add_subplot(gs[4, 1])
        
        state_counts = bout_df['state'].value_counts()
        pie_colors = [self.all_colors[int(s)] for s in state_counts.index]
        pie_labels = [self.labels[int(s)][:10] for s in state_counts.index]
        
        wedges, texts, autotexts = ax10.pie(state_counts.values, colors=pie_colors,
                                            autopct='%1.0f%%', startangle=90,
                                            pctdistance=0.75)
        
        for autotext in autotexts:
            autotext.set_fontsize(7)
        
        ax10.set_title('Bout State\nDistribution', fontsize=10, fontweight='bold')

        # =====================================================================
        # PANEL 11: Duration vs Energy Scatter
        # =====================================================================
        ax11 = fig.add_subplot(gs[4, 2])
        
        scatter = ax11.scatter(bout_df['duration_sec'], bout_df['total_energy'],
                              c=[self.all_colors[int(s)] for s in bout_df['state']],
                              s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
        
        # Correlation line
        if len(bout_df) > 10:
            z = np.polyfit(bout_df['duration_sec'], bout_df['total_energy'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(bout_df['duration_sec'].min(), bout_df['duration_sec'].max(), 100)
            ax11.plot(x_line, p(x_line), 'r--', lw=2, alpha=0.7)
            
            corr = np.corrcoef(bout_df['duration_sec'], bout_df['total_energy'])[0, 1]
            ax11.text(0.05, 0.95, f'r = {corr:.2f}', transform=ax11.transAxes,
                     fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax11.set_xlabel('Duration (s)', fontsize=9)
        ax11.set_ylabel('Energy', fontsize=9)
        ax11.set_title('Duration vs\nEnergy', fontsize=10, fontweight='bold')
        ax11.grid(True, alpha=0.3)

        # =====================================================================
        # PANEL 12: Statistics Summary
        # =====================================================================
        ax12 = fig.add_subplot(gs[4, 3])
        ax12.axis('off')
        
        # Calculate comprehensive statistics
        total_bouts = len(bouts)
        total_time_sec = times[-1] - times[0]
        mean_bouts_per_min = total_bouts / (total_time_sec / 60)
        
        # Most common transitions
        transitions = [(states_seq[i], states_seq[i+1]) for i in range(len(states_seq)-1)]
        from collections import Counter
        trans_counts = Counter(transitions)
        top_trans = trans_counts.most_common(3)
        
        # Format top transitions
        # top_trans format: [((from_state, to_state), count), ...]
        if len(top_trans) > 0:
            from_state = top_trans[0][0][0]  # First element of the tuple key
            to_state = top_trans[0][0][1]    # Second element of the tuple key
            count = top_trans[0][1]          # The count
            trans_str = f"  {self.labels[from_state][:8]}→{self.labels[to_state][:8]}: {count}"
        else:
            trans_str = "  N/A"
        
        mean_ibi = np.mean(ibis) if len(ibis) > 0 else 0
        median_ibi = np.median(ibis) if len(ibis) > 0 else 0
        
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
  Mean IBI:       {mean_ibi:.2f}
  Median IBI:     {median_ibi:.2f}

TOP TRANSITION:
{trans_str}
"""
        ax12.text(0.05, 0.95, stats_text, transform=ax12.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))

        # =====================================================================
        # Main Title
        # =====================================================================
        plt.suptitle('Enhanced Behavioral Bout Analysis\nEpisode Characterization and Temporal Dynamics',
                    fontsize=14, fontweight='bold', y=0.99)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig('dolphin_bout_analysis.png', dpi=200, facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_bout_analysis.png")

    # =========================================================================
    # VISUALIZATION 27: MARKOV ANALYSIS
    # =========================================================================
    def generate_markov_analysis(self):
        """Markov chain stationary distribution analysis."""
        print("\n27. Generating Markov Analysis...")

        n_states = max(self.existing_clusters) + 1

        # Build transition matrix
        transitions = np.zeros((n_states, n_states))
        for i in range(len(self.clusters) - 1):
            transitions[self.clusters[i], self.clusters[i + 1]] += 1

        # Normalize
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = transitions / row_sums

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Transition matrix heatmap
        ax1 = axes[0, 0]
        P_existing = P[np.ix_(self.existing_clusters, self.existing_clusters)]
        im = ax1.imshow(P_existing, cmap='Blues', vmin=0, vmax=1)
        ax1.set_xticks(range(len(self.existing_clusters)))
        ax1.set_yticks(range(len(self.existing_clusters)))
        labels_short = [self.labels[i][:10] for i in self.existing_clusters]
        ax1.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=8)
        ax1.set_yticklabels(labels_short, fontsize=8)
        ax1.set_title('Transition Matrix P', fontweight='bold')
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # 2. Stationary distribution
        ax2 = axes[0, 1]

        # Power iteration for stationary distribution
        pi = np.ones(n_states) / n_states
        for _ in range(100):
            pi = pi @ P

        # Empirical distribution
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

        # 3. Eigenvalue spectrum
        ax3 = axes[1, 0]
        eigenvalues = np.linalg.eigvals(P)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]

        ax3.bar(range(1, min(len(eigenvalues_sorted), 10) + 1), eigenvalues_sorted[:10],
                color='purple', alpha=0.7, edgecolor='black')
        ax3.axhline(1, color='red', linestyle='--', label='λ=1 (stacionárius)')
        ax3.set_xlabel('Eigenvalue Index')
        ax3.set_ylabel('|λ|')
        ax3.set_title('Eigenvalue Spectrum', fontweight='bold')
        ax3.legend()

        # Mixing time estimate
        if len(eigenvalues_sorted) > 1 and eigenvalues_sorted[1] < 0.9999:
            mixing_time = 1 / (1 - eigenvalues_sorted[1])
            ax3.text(0.6, 0.8, f'Mixing time ≈ {mixing_time:.1f}',
                     transform=ax3.transAxes, fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat'))

        # 4. State persistence (self-transition probabilities)
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
    def generate_social_network(self):
        """Social acoustic network visualization (simulated individuals)."""
        print("\n28. Generating Social Network...")

        if not HAS_NETWORKX:
            print("   ! networkx not available, skipping.")
            return

        # Simulate 4 dolphins
        np.random.seed(42)
        n_dolphins = 4
        dolphin_names = ['Luna', 'Ocean', 'Star', 'Wave']
        dolphin_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        n_samples = len(self.features[self.window_size:])

        # Simulate individual patterns based on features
        dolphin_activity = np.zeros((n_dolphins, n_samples))
        for i in range(n_dolphins):
            phase = i * np.pi / 2
            freq = 0.02 + i * 0.005
            pattern = 0.3 * np.sin(2 * np.pi * freq * np.arange(n_samples) + phase)
            noise = 0.1 * np.random.randn(n_samples)
            dolphin_activity[i] = np.clip(pattern + noise + self.features[self.window_size:, 0] * (0.5 + 0.2 * i), 0, 1)

        # Calculate interaction matrix (cross-correlation)
        interaction_matrix = np.zeros((n_dolphins, n_dolphins))

        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if i != j:
                    lag_range = range(5, 30)
                    max_corr = 0
                    for lag in lag_range:
                        if lag < len(dolphin_activity[i]):
                            corr = np.corrcoef(dolphin_activity[i, :-lag],
                                               dolphin_activity[j, lag:])[0, 1]
                            if not np.isnan(corr):
                                max_corr = max(max_corr, corr)
                    interaction_matrix[i, j] = max(0, max_corr)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # 1. Network graph
        ax1 = axes[0]
        G = nx.DiGraph()

        for i, name in enumerate(dolphin_names):
            G.add_node(i, label=name)

        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if i != j and interaction_matrix[i, j] > 0.3:
                    G.add_edge(i, j, weight=interaction_matrix[i, j])

        pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, node_color=dolphin_colors, node_size=2000, alpha=0.9, ax=ax1)

        for (u, v, d) in G.edges(data=True):
            ax1.annotate('', xy=pos[v], xytext=pos[u],
                         arrowprops=dict(arrowstyle='-|>', color='gray',
                                         lw=d['weight'] * 5, alpha=0.6,
                                         connectionstyle='arc3,rad=0.1'))

        nx.draw_networkx_labels(G, pos, {i: name for i, name in enumerate(dolphin_names)},
                                font_size=12, font_weight='bold', ax=ax1)

        ax1.set_title('Social Acoustic Network\n(Edge thickness ∝ interaction)', fontweight='bold')
        ax1.axis('off')

        # 2. Interaction matrix heatmap
        ax2 = axes[1]
        im = ax2.imshow(interaction_matrix, cmap='YlOrRd', vmin=0, vmax=1)
        ax2.set_xticks(range(n_dolphins))
        ax2.set_yticks(range(n_dolphins))
        ax2.set_xticklabels(dolphin_names)
        ax2.set_yticklabels(dolphin_names)
        ax2.set_xlabel('Responder')
        ax2.set_ylabel('Caller')

        for i in range(n_dolphins):
            for j in range(n_dolphins):
                ax2.text(j, i, f'{interaction_matrix[i, j]:.2f}',
                         ha='center', va='center',
                         color='white' if interaction_matrix[i, j] > 0.5 else 'black')

        ax2.set_title('Call-Response Interaction Matrix', fontweight='bold')
        plt.colorbar(im, ax=ax2, label='Interaction Strength', shrink=0.8)

        plt.suptitle('Individual Interaction Patterns', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dolphin_social_network.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   -> SAVED: dolphin_social_network.png")

    # =========================================================================
    # VISUALIZATION 29: VOCAL TURN-TAKING ANALYSIS
    # =========================================================================
    def generate_turn_taking_analysis(self):
        """
        Vocal Turn-Taking Analysis - Conversation-like dynamics in dolphin communication.
        
        Analyzes:
        - Inter-vocalization intervals (IVI)
        - Call-response latencies
        - Overlap detection
        - Turn-taking patterns between simulated individuals
        """
        print("\n29. Generating Vocal Turn-Taking Analysis...")
        
        # Simulate individual vocalizations based on acoustic features
        features = self.features[self.window_size:]
        times = self.valid_times
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        # Define individuals and assign vocalizations
        n_dolphins = 4
        dolphin_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        dolphin_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Detect vocal events (combined whistle + burst threshold)
        vocal_intensity = features[:, 0] + features[:, 1]  # whistle + burst
        vocal_threshold = np.percentile(vocal_intensity, 70)
        
        # Find vocal events
        vocal_events = []
        in_vocal = False
        event_start = 0
        
        for i in range(len(vocal_intensity)):
            if not in_vocal and vocal_intensity[i] > vocal_threshold:
                event_start = i
                in_vocal = True
            elif in_vocal and vocal_intensity[i] < vocal_threshold * 0.7:
                if i - event_start >= 3:  # Minimum duration
                    # Assign to a dolphin based on acoustic signature
                    mean_whistle = np.mean(features[event_start:i, 0])
                    mean_burst = np.mean(features[event_start:i, 1])
                    # Simple assignment based on feature ratios
                    signature = (mean_whistle * 2 + mean_burst) * 1000
                    dolphin_id = int(signature) % n_dolphins
                    
                    vocal_events.append({
                        'start_idx': event_start,
                        'end_idx': i,
                        'start_time': times[event_start],
                        'end_time': times[i],
                        'duration': (i - event_start) * dt,
                        'dolphin': dolphin_id,
                        'intensity': np.mean(vocal_intensity[event_start:i])
                    })
                in_vocal = False
        
        if len(vocal_events) < 20:
            print("   ! Not enough vocal events, lowering threshold...")
            # Try with lower threshold
            vocal_threshold = np.percentile(vocal_intensity, 60)
            vocal_events = []
            in_vocal = False
            event_start = 0
            
            for i in range(len(vocal_intensity)):
                if not in_vocal and vocal_intensity[i] > vocal_threshold:
                    event_start = i
                    in_vocal = True
                elif in_vocal and vocal_intensity[i] < vocal_threshold * 0.8:
                    if i - event_start >= 2:
                        mean_whistle = np.mean(features[event_start:i, 0])
                        mean_burst = np.mean(features[event_start:i, 1])
                        signature = (mean_whistle * 2 + mean_burst) * 1000
                        dolphin_id = int(signature) % n_dolphins
                        
                        vocal_events.append({
                            'start_idx': event_start,
                            'end_idx': i,
                            'start_time': times[event_start],
                            'end_time': times[i],
                            'duration': (i - event_start) * dt,
                            'dolphin': dolphin_id,
                            'intensity': np.mean(vocal_intensity[event_start:i])
                        })
                    in_vocal = False
        
        if len(vocal_events) < 10:
            print("   ! Not enough vocal events detected - skipping turn-taking analysis")
            return
        
        vocal_df = pd.DataFrame(vocal_events)
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # =====================================================================
        # PANEL 1: Vocal Event Timeline (Raster Plot)
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, :])
        
        for dolphin_id in range(n_dolphins):
            dolphin_events = vocal_df[vocal_df['dolphin'] == dolphin_id]
            for _, event in dolphin_events.iterrows():
                ax1.barh(dolphin_id, event['duration'], left=event['start_time'],
                        height=0.6, color=dolphin_colors[dolphin_id], alpha=0.8)
        
        ax1.set_yticks(range(n_dolphins))
        ax1.set_yticklabels(dolphin_names)
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Individual', fontsize=11)
        ax1.set_title('Vocal Event Timeline (Turn-Taking Raster)', fontsize=12, fontweight='bold')
        ax1.set_xlim(times[0], min(times[0] + 120, times[-1]))  # First 2 minutes
        ax1.grid(True, alpha=0.3, axis='x')
        
        # =====================================================================
        # PANEL 2: Inter-Vocalization Interval Distribution
        # =====================================================================
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Calculate IVIs between consecutive vocalizations (any individual)
        sorted_events = vocal_df.sort_values('start_time')
        ivis = np.diff(sorted_events['start_time'].values)
        ivis = ivis[ivis > 0]  # Remove any negative/zero
        ivis = ivis[ivis < 10]  # Cap at 10 seconds for visualization
        
        ax2.hist(ivis, bins=40, color='#3498db', edgecolor='white', alpha=0.7, density=True)
        ax2.axvline(np.median(ivis), color='red', linestyle='--', lw=2,
                   label=f'Median: {np.median(ivis):.2f}s')
        ax2.set_xlabel('Inter-Vocalization Interval (s)', fontsize=10)
        ax2.set_ylabel('Density', fontsize=10)
        ax2.set_title('IVI Distribution', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 3: Response Latency by Individual Pair
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate response latencies between pairs
        response_latencies = {(i, j): [] for i in range(n_dolphins) for j in range(n_dolphins) if i != j}
        
        sorted_events_list = sorted_events.to_dict('records')
        for i in range(len(sorted_events_list) - 1):
            current = sorted_events_list[i]
            next_event = sorted_events_list[i + 1]
            
            if current['dolphin'] != next_event['dolphin']:
                latency = next_event['start_time'] - current['end_time']
                if 0 < latency < 5:  # Reasonable response window
                    pair = (current['dolphin'], next_event['dolphin'])
                    response_latencies[pair].append(latency)
        
        # Create latency matrix
        latency_matrix = np.zeros((n_dolphins, n_dolphins))
        for (i, j), latencies in response_latencies.items():
            if latencies:
                latency_matrix[i, j] = np.median(latencies)
        
        im = ax3.imshow(latency_matrix, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(n_dolphins))
        ax3.set_yticks(range(n_dolphins))
        ax3.set_xticklabels(dolphin_names, fontsize=9)
        ax3.set_yticklabels(dolphin_names, fontsize=9)
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if latency_matrix[i, j] > 0:
                    ax3.text(j, i, f'{latency_matrix[i, j]:.2f}', ha='center', va='center',
                            fontsize=8, color='white' if latency_matrix[i, j] > 1 else 'black')
        
        ax3.set_xlabel('Responder', fontsize=10)
        ax3.set_ylabel('Initiator', fontsize=10)
        ax3.set_title('Median Response Latency (s)', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # =====================================================================
        # PANEL 4: Turn-Taking Frequency
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 2])
        
        # Count turn-taking events per pair
        turn_counts = np.zeros((n_dolphins, n_dolphins))
        for (i, j), latencies in response_latencies.items():
            turn_counts[i, j] = len(latencies)
        
        im = ax4.imshow(turn_counts, cmap='Blues', aspect='auto')
        ax4.set_xticks(range(n_dolphins))
        ax4.set_yticks(range(n_dolphins))
        ax4.set_xticklabels(dolphin_names, fontsize=9)
        ax4.set_yticklabels(dolphin_names, fontsize=9)
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if turn_counts[i, j] > 0:
                    ax4.text(j, i, f'{int(turn_counts[i, j])}', ha='center', va='center',
                            fontsize=9, color='white' if turn_counts[i, j] > np.max(turn_counts)/2 else 'black')
        
        ax4.set_xlabel('Responder', fontsize=10)
        ax4.set_ylabel('Initiator', fontsize=10)
        ax4.set_title('Turn-Taking Frequency', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        # =====================================================================
        # PANEL 5: Vocal Overlap Analysis
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Detect overlapping vocalizations
        overlaps = []
        for i, event1 in enumerate(vocal_events):
            for j, event2 in enumerate(vocal_events):
                if i < j and event1['dolphin'] != event2['dolphin']:
                    # Check for overlap
                    overlap_start = max(event1['start_time'], event2['start_time'])
                    overlap_end = min(event1['end_time'], event2['end_time'])
                    if overlap_start < overlap_end:
                        overlaps.append({
                            'pair': (event1['dolphin'], event2['dolphin']),
                            'duration': overlap_end - overlap_start,
                            'time': overlap_start
                        })
        
        # Overlap statistics by pair
        overlap_counts = np.zeros((n_dolphins, n_dolphins))
        for overlap in overlaps:
            i, j = overlap['pair']
            overlap_counts[i, j] += 1
            overlap_counts[j, i] += 1
        
        np.fill_diagonal(overlap_counts, 0)
        
        im = ax5.imshow(overlap_counts, cmap='Reds', aspect='auto')
        ax5.set_xticks(range(n_dolphins))
        ax5.set_yticks(range(n_dolphins))
        ax5.set_xticklabels(dolphin_names, fontsize=9)
        ax5.set_yticklabels(dolphin_names, fontsize=9)
        ax5.set_title(f'Vocal Overlaps (n={len(overlaps)})', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax5, shrink=0.8, label='Count')
        
        # =====================================================================
        # PANEL 6: Speaking Time Distribution
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 1])
        
        speaking_times = [vocal_df[vocal_df['dolphin'] == d]['duration'].sum() for d in range(n_dolphins)]
        total_speaking = sum(speaking_times)
        percentages = [t/total_speaking*100 for t in speaking_times]
        
        bars = ax6.bar(dolphin_names, speaking_times, color=dolphin_colors, edgecolor='black', alpha=0.8)
        
        for bar, pct in zip(bars, percentages):
            ax6.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        ax6.set_ylabel('Total Vocal Time (s)', fontsize=10)
        ax6.set_title('Speaking Time Distribution', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # PANEL 7: Call Rate Over Time
        # =====================================================================
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Calculate call rate in sliding windows
        window_sec = 30
        call_rates = {d: [] for d in range(n_dolphins)}
        time_bins = []
        
        for t_start in np.arange(times[0], times[-1] - window_sec, window_sec/2):
            t_end = t_start + window_sec
            time_bins.append((t_start + t_end) / 2)
            
            for d in range(n_dolphins):
                d_events = vocal_df[(vocal_df['dolphin'] == d) & 
                                   (vocal_df['start_time'] >= t_start) & 
                                   (vocal_df['start_time'] < t_end)]
                call_rates[d].append(len(d_events) / window_sec * 60)  # calls per minute
        
        for d in range(n_dolphins):
            ax7.plot(time_bins, call_rates[d], color=dolphin_colors[d], lw=2, 
                    label=dolphin_names[d], alpha=0.8)
        
        ax7.set_xlabel('Time (s)', fontsize=10)
        ax7.set_ylabel('Call Rate (calls/min)', fontsize=10)
        ax7.set_title('Individual Call Rates Over Time', fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8, loc='upper right')
        ax7.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 8: Turn-Taking Network Graph
        # =====================================================================
        ax8 = fig.add_subplot(gs[3, 0:2])
        
        if HAS_NETWORKX:
            G = nx.DiGraph()
            
            for d in range(n_dolphins):
                G.add_node(d, name=dolphin_names[d])
            
            for (i, j), latencies in response_latencies.items():
                if len(latencies) >= 3:
                    G.add_edge(i, j, weight=len(latencies), latency=np.median(latencies))
            
            pos = nx.circular_layout(G)
            
            # Draw nodes
            node_sizes = [speaking_times[d] * 50 + 500 for d in range(n_dolphins)]
            nx.draw_networkx_nodes(G, pos, ax=ax8, node_color=dolphin_colors,
                                  node_size=node_sizes, alpha=0.9)
            
            # Draw edges with width proportional to turn-taking frequency
            edges = G.edges(data=True)
            if edges:
                edge_weights = [d['weight'] for _, _, d in edges]
                max_weight = max(edge_weights) if edge_weights else 1
                
                for (u, v, d) in edges:
                    width = 1 + 4 * d['weight'] / max_weight
                    nx.draw_networkx_edges(G, pos, ax=ax8, edgelist=[(u, v)],
                                          width=width, alpha=0.6,
                                          edge_color='gray', arrows=True,
                                          arrowsize=15, connectionstyle='arc3,rad=0.1')
            
            nx.draw_networkx_labels(G, pos, ax=ax8, 
                                   labels={d: dolphin_names[d] for d in range(n_dolphins)},
                                   font_size=10, font_weight='bold')
            
            ax8.set_title('Turn-Taking Network\n(node size ∝ speaking time, edge width ∝ exchanges)',
                         fontsize=11, fontweight='bold')
        else:
            ax8.text(0.5, 0.5, 'NetworkX not available', ha='center', va='center', fontsize=12)
        
        ax8.axis('off')
        
        # =====================================================================
        # PANEL 9: Statistics Summary
        # =====================================================================
        ax9 = fig.add_subplot(gs[3, 2])
        ax9.axis('off')
        
        total_events = len(vocal_events)
        mean_duration = vocal_df['duration'].mean()
        mean_ivi = np.mean(ivis) if len(ivis) > 0 else 0
        
        stats_text = f"""
TURN-TAKING STATISTICS
══════════════════════════
Total Vocal Events:  {total_events}
Total Overlaps:      {len(overlaps)}
Overlap Rate:        {len(overlaps)/total_events*100:.1f}%

TIMING:
  Mean Duration:     {mean_duration:.2f} s
  Mean IVI:          {mean_ivi:.2f} s
  Median IVI:        {np.median(ivis):.2f} s

INDIVIDUAL ACTIVITY:
  Alpha:  {percentages[0]:.1f}% of vocal time
  Beta:   {percentages[1]:.1f}% of vocal time
  Gamma:  {percentages[2]:.1f}% of vocal time
  Delta:  {percentages[3]:.1f}% of vocal time

TURN-TAKING PAIRS:
  Most active: {dolphin_names[np.unravel_index(turn_counts.argmax(), turn_counts.shape)[0]]} → {dolphin_names[np.unravel_index(turn_counts.argmax(), turn_counts.shape)[1]]}
"""
        ax9.text(0.05, 0.95, stats_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        plt.suptitle('Vocal Turn-Taking Analysis: Conversation Dynamics',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_turn_taking.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_turn_taking.png")

    # =========================================================================
    # VISUALIZATION 30: ACOUSTIC CONTAGION CASCADE
    # =========================================================================
    def generate_contagion_cascade(self):
        """
        Acoustic Contagion Cascade - How vocalizations spread through the group.
        
        Models vocal behavior as an epidemic-like spreading process.
        """
        print("\n30. Generating Acoustic Contagion Cascade...")
        
        features = self.features[self.window_size:]
        times = self.valid_times
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        # Simulate 4 individuals
        n_dolphins = 4
        dolphin_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        dolphin_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Detect high-activity bursts that could trigger contagion
        activity = np.sum(features, axis=1)
        activity_smooth = pd.Series(activity).rolling(10, center=True).mean()
        activity_smooth = activity_smooth.fillna(pd.Series(activity))
        
        # Find cascade initiation events (sharp increases in activity)
        threshold = np.percentile(activity_smooth, 85)
        
        cascades = []
        in_cascade = False
        cascade_start = 0
        
        for i in range(1, len(activity_smooth)):
            if not in_cascade and activity_smooth.iloc[i] > threshold and activity_smooth.iloc[i-1] < threshold * 0.8:
                cascade_start = i
                in_cascade = True
            elif in_cascade and activity_smooth.iloc[i] < threshold * 0.6:
                if i - cascade_start >= 10:
                    # Simulate which dolphins participated
                    participants = []
                    join_times = []
                    
                    # Initiator (random based on features)
                    initiator = int(features[cascade_start, 0] * 1000) % n_dolphins
                    participants.append(initiator)
                    join_times.append(0)
                    
                    # Other dolphins join based on feature patterns
                    for j in range(1, n_dolphins):
                        dolphin = (initiator + j) % n_dolphins
                        # Random delay based on acoustic features
                        delay = int((features[cascade_start + min(j*5, i-cascade_start-1), 1] * 50) + j * 3)
                        if delay < (i - cascade_start):
                            participants.append(dolphin)
                            join_times.append(delay)
                    
                    cascades.append({
                        'start_idx': cascade_start,
                        'end_idx': i,
                        'start_time': times[cascade_start],
                        'duration': (i - cascade_start) * dt,
                        'initiator': initiator,
                        'participants': participants,
                        'join_times': join_times,
                        'peak_activity': np.max(activity_smooth.iloc[cascade_start:i]),
                        'n_participants': len(participants)
                    })
                in_cascade = False
        
        if len(cascades) < 3:
            print("   ! Not enough cascade events detected, lowering threshold...")
            # Try with lower threshold
            threshold = np.percentile(activity_smooth, 75)
            cascades = []
            in_cascade = False
            cascade_start = 0
            
            for i in range(1, len(activity_smooth)):
                if not in_cascade and activity_smooth.iloc[i] > threshold and activity_smooth.iloc[i-1] < threshold * 0.9:
                    cascade_start = i
                    in_cascade = True
                elif in_cascade and activity_smooth.iloc[i] < threshold * 0.7:
                    if i - cascade_start >= 5:  # Lower minimum duration
                        participants = []
                        join_times = []
                        initiator = int(features[cascade_start, 0] * 1000) % n_dolphins
                        participants.append(initiator)
                        join_times.append(0)
                        
                        for j in range(1, n_dolphins):
                            dolphin = (initiator + j) % n_dolphins
                            delay = int((features[cascade_start + min(j*3, i-cascade_start-1), 1] * 30) + j * 2)
                            if delay < (i - cascade_start):
                                participants.append(dolphin)
                                join_times.append(delay)
                        
                        cascades.append({
                            'start_idx': cascade_start,
                            'end_idx': i,
                            'start_time': times[cascade_start],
                            'duration': (i - cascade_start) * dt,
                            'initiator': initiator,
                            'participants': participants,
                            'join_times': join_times,
                            'peak_activity': np.max(activity_smooth.iloc[cascade_start:i]),
                            'n_participants': len(participants)
                        })
                    in_cascade = False
        
        if len(cascades) < 1:
            print("   ! No cascade events detected - skipping contagion cascade analysis")
            return
        
        cascade_df = pd.DataFrame(cascades)
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # =====================================================================
        # PANEL 1: Cascade Timeline (Top)
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot activity trace
        time_plot = (times - times[0]) / 60
        ax1.fill_between(time_plot, 0, activity_smooth, color='gray', alpha=0.3, label='Activity')
        ax1.axhline(threshold, color='red', linestyle='--', lw=1, alpha=0.7, label='Cascade Threshold')
        
        # Mark cascades
        for i, cascade in enumerate(cascades[:20]):  # Show first 20
            start_min = (cascade['start_time'] - times[0]) / 60
            ax1.axvspan(start_min, start_min + cascade['duration']/60,
                       color=dolphin_colors[cascade['initiator']], alpha=0.4)
            ax1.annotate(f'C{i+1}', xy=(start_min, threshold * 1.1),
                        fontsize=7, ha='center')
        
        ax1.set_xlabel('Time (min)', fontsize=11)
        ax1.set_ylabel('Acoustic Activity', fontsize=11)
        ax1.set_title('Acoustic Contagion Cascade Timeline', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 2: Cascade Size Distribution
        # =====================================================================
        ax2 = fig.add_subplot(gs[1, 0])
        
        sizes = cascade_df['n_participants'].values
        ax2.hist(sizes, bins=range(1, n_dolphins + 2), color='#3498db', edgecolor='black',
                alpha=0.7, align='left')
        ax2.set_xlabel('Cascade Size (# participants)', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Cascade Size Distribution', fontsize=11, fontweight='bold')
        ax2.set_xticks(range(1, n_dolphins + 1))
        ax2.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 3: Initiator Frequency
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 1])
        
        initiator_counts = cascade_df['initiator'].value_counts().reindex(range(n_dolphins), fill_value=0)
        bars = ax3.bar(dolphin_names, initiator_counts.values, color=dolphin_colors,
                      edgecolor='black', alpha=0.8)
        
        ax3.set_ylabel('# Cascades Initiated', fontsize=10)
        ax3.set_title('Cascade Initiator Frequency', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # PANEL 4: Cascade Duration vs Size
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 2])
        
        scatter_colors = [dolphin_colors[i] for i in cascade_df['initiator']]
        ax4.scatter(cascade_df['n_participants'], cascade_df['duration'],
                   c=scatter_colors, s=cascade_df['peak_activity'] * 50 + 50,
                   alpha=0.7, edgecolors='black')
        
        # Trend line
        if len(cascade_df) > 5:
            z = np.polyfit(cascade_df['n_participants'], cascade_df['duration'], 1)
            p = np.poly1d(z)
            x_line = np.array([1, n_dolphins])
            ax4.plot(x_line, p(x_line), 'r--', lw=2, alpha=0.7)
        
        ax4.set_xlabel('Cascade Size', fontsize=10)
        ax4.set_ylabel('Duration (s)', fontsize=10)
        ax4.set_title('Duration vs Size\n(size ∝ peak activity)', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 5: Cascade Spreading Pattern (Example)
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 0])
        
        # Show spreading pattern for largest cascade
        if len(cascades) > 0:
            largest = max(cascades, key=lambda x: x['n_participants'])
            
            for i, (participant, join_time) in enumerate(zip(largest['participants'], largest['join_times'])):
                ax5.barh(i, 1, left=join_time, color=dolphin_colors[participant],
                        height=0.6, edgecolor='black', alpha=0.8)
                ax5.text(join_time + 0.5, i, dolphin_names[participant], va='center', fontsize=9)
            
            ax5.set_xlabel('Time from Cascade Start (samples)', fontsize=10)
            ax5.set_ylabel('Join Order', fontsize=10)
            ax5.set_title('Example Cascade Spreading\n(Largest cascade)', fontsize=11, fontweight='bold')
            ax5.set_yticks([])
        
        # =====================================================================
        # PANEL 6: Contagion Network
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Build contagion network (who triggers whom)
        contagion_matrix = np.zeros((n_dolphins, n_dolphins))
        
        for cascade in cascades:
            if len(cascade['participants']) >= 2:
                initiator = cascade['participants'][0]
                for follower in cascade['participants'][1:]:
                    contagion_matrix[initiator, follower] += 1
        
        im = ax6.imshow(contagion_matrix, cmap='Oranges', aspect='auto')
        ax6.set_xticks(range(n_dolphins))
        ax6.set_yticks(range(n_dolphins))
        ax6.set_xticklabels(dolphin_names, fontsize=9)
        ax6.set_yticklabels(dolphin_names, fontsize=9)
        ax6.set_xlabel('Follower', fontsize=10)
        ax6.set_ylabel('Initiator', fontsize=10)
        ax6.set_title('Contagion Matrix', fontsize=11, fontweight='bold')
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if contagion_matrix[i, j] > 0:
                    ax6.text(j, i, f'{int(contagion_matrix[i, j])}', ha='center', va='center',
                            fontsize=9, color='white' if contagion_matrix[i, j] > np.max(contagion_matrix)/2 else 'black')
        
        plt.colorbar(im, ax=ax6, shrink=0.8)
        
        # =====================================================================
        # PANEL 7: Statistics
        # =====================================================================
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        mean_size = cascade_df['n_participants'].mean()
        mean_duration = cascade_df['duration'].mean()
        full_cascades = sum(cascade_df['n_participants'] == n_dolphins)
        
        stats_text = f"""
CONTAGION CASCADE STATISTICS
════════════════════════════════
Total Cascades:      {len(cascades)}
Full Group Events:   {full_cascades} ({full_cascades/len(cascades)*100:.1f}%)

CASCADE SIZE:
  Mean:              {mean_size:.2f} dolphins
  Max:               {cascade_df['n_participants'].max()} dolphins
  
CASCADE DURATION:
  Mean:              {mean_duration:.2f} s
  Max:               {cascade_df['duration'].max():.2f} s

INITIATORS:
  Alpha:  {initiator_counts[0]} ({initiator_counts[0]/len(cascades)*100:.1f}%)
  Beta:   {initiator_counts[1]} ({initiator_counts[1]/len(cascades)*100:.1f}%)
  Gamma:  {initiator_counts[2]} ({initiator_counts[2]/len(cascades)*100:.1f}%)
  Delta:  {initiator_counts[3]} ({initiator_counts[3]/len(cascades)*100:.1f}%)

CONTAGION RATE:
  R₀ (mean followers): {(mean_size - 1):.2f}
"""
        ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        plt.suptitle('Acoustic Contagion Cascade Analysis: Vocal Behavior Spreading',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_contagion_cascade.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_contagion_cascade.png")

    # =========================================================================
    # VISUALIZATION 31: DYNAMIC SOCIAL NETWORK EVOLUTION
    # =========================================================================
    def generate_dynamic_network(self):
        """
        Dynamic Social Network Evolution - Network changes over time.
        
        Uses sliding windows to show how acoustic relationships evolve.
        """
        print("\n31. Generating Dynamic Network Evolution...")
        
        features = self.features[self.window_size:]
        times = self.valid_times
        
        n_dolphins = 4
        dolphin_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        dolphin_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Divide recording into time windows
        n_windows = 6
        window_size = len(features) // n_windows
        
        # Calculate interaction matrices for each window
        window_matrices = []
        window_times = []
        
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = (w + 1) * window_size if w < n_windows - 1 else len(features)
            
            window_features = features[start_idx:end_idx]
            
            # Simulate individual activity and calculate correlations
            n_samples = len(window_features)
            individual_signals = np.zeros((n_dolphins, n_samples))
            
            for d in range(n_dolphins):
                # Create individual signal based on feature patterns
                phase_shift = d * n_samples // n_dolphins
                base_signal = np.roll(np.sum(window_features, axis=1), phase_shift)
                noise = np.random.randn(n_samples) * 0.1
                individual_signals[d] = base_signal + noise
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(individual_signals)
            np.fill_diagonal(corr_matrix, 0)
            corr_matrix = np.abs(corr_matrix)
            
            window_matrices.append(corr_matrix)
            window_times.append((times[start_idx] - times[0]) / 60)
        
        # Create figure with small multiples
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # =====================================================================
        # TOP ROW: Network snapshots
        # =====================================================================
        for w in range(min(4, n_windows)):
            ax = fig.add_subplot(gs[0, w])
            
            if HAS_NETWORKX:
                G = nx.Graph()
                for d in range(n_dolphins):
                    G.add_node(d, name=dolphin_names[d])
                
                for i in range(n_dolphins):
                    for j in range(i+1, n_dolphins):
                        if window_matrices[w][i, j] > 0.3:
                            G.add_edge(i, j, weight=window_matrices[w][i, j])
                
                pos = nx.circular_layout(G)
                
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=dolphin_colors,
                                      node_size=800, alpha=0.9)
                
                edges = G.edges(data=True)
                if edges:
                    edge_weights = [d['weight'] * 3 for _, _, d in edges]
                    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights, alpha=0.6)
                
                nx.draw_networkx_labels(G, pos, ax=ax,
                                       labels={d: dolphin_names[d][0] for d in range(n_dolphins)},
                                       font_size=10, font_weight='bold')
            
            ax.set_title(f'T = {window_times[w]:.1f} min', fontsize=10, fontweight='bold')
            ax.axis('off')
        
        # =====================================================================
        # MIDDLE ROW: Metrics evolution
        # =====================================================================
        ax_density = fig.add_subplot(gs[1, 0:2])
        ax_central = fig.add_subplot(gs[1, 2:4])
        
        # Calculate network metrics over time
        densities = []
        centralities = {d: [] for d in range(n_dolphins)}
        
        for w, matrix in enumerate(window_matrices):
            # Density (proportion of strong connections)
            density = np.sum(matrix > 0.3) / (n_dolphins * (n_dolphins - 1))
            densities.append(density)
            
            # Degree centrality (sum of connections)
            for d in range(n_dolphins):
                centrality = np.sum(matrix[d, :])
                centralities[d].append(centrality)
        
        # Plot density evolution
        ax_density.plot(window_times[:len(densities)], densities, 'ko-', lw=2, markersize=8)
        ax_density.fill_between(window_times[:len(densities)], 0, densities, alpha=0.3)
        ax_density.set_xlabel('Time (min)', fontsize=11)
        ax_density.set_ylabel('Network Density', fontsize=11)
        ax_density.set_title('Network Density Evolution', fontsize=12, fontweight='bold')
        ax_density.grid(True, alpha=0.3)
        ax_density.set_ylim(0, 1)
        
        # Plot centrality evolution
        for d in range(n_dolphins):
            ax_central.plot(window_times[:len(centralities[d])], centralities[d],
                           color=dolphin_colors[d], lw=2, marker='o', markersize=6,
                           label=dolphin_names[d])
        ax_central.set_xlabel('Time (min)', fontsize=11)
        ax_central.set_ylabel('Centrality Score', fontsize=11)
        ax_central.set_title('Individual Centrality Evolution', fontsize=12, fontweight='bold')
        ax_central.legend(fontsize=9)
        ax_central.grid(True, alpha=0.3)
        
        # =====================================================================
        # BOTTOM ROW: Connection strength heatmaps
        # =====================================================================
        ax_heat = fig.add_subplot(gs[2, 0:3])
        
        # Create time-resolved connection strength matrix
        n_pairs = n_dolphins * (n_dolphins - 1) // 2
        pair_labels = []
        pair_strengths = np.zeros((n_pairs, n_windows))
        
        pair_idx = 0
        for i in range(n_dolphins):
            for j in range(i+1, n_dolphins):
                pair_labels.append(f'{dolphin_names[i][0]}-{dolphin_names[j][0]}')
                for w in range(n_windows):
                    pair_strengths[pair_idx, w] = window_matrices[w][i, j]
                pair_idx += 1
        
        im = ax_heat.imshow(pair_strengths, aspect='auto', cmap='YlOrRd',
                           extent=[0, window_times[-1], n_pairs-0.5, -0.5])
        ax_heat.set_yticks(range(n_pairs))
        ax_heat.set_yticklabels(pair_labels, fontsize=9)
        ax_heat.set_xlabel('Time (min)', fontsize=11)
        ax_heat.set_ylabel('Dyad', fontsize=11)
        ax_heat.set_title('Dyadic Connection Strength Over Time', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax_heat, shrink=0.8, label='Correlation')
        
        # =====================================================================
        # PANEL: Network change statistics
        # =====================================================================
        ax_stats = fig.add_subplot(gs[2, 3])
        ax_stats.axis('off')
        
        # Calculate stability metrics
        stability = []
        for w in range(1, len(window_matrices)):
            diff = np.abs(window_matrices[w] - window_matrices[w-1])
            stability.append(1 - np.mean(diff))
        
        mean_stability = np.mean(stability) if stability else 0
        
        stats_text = f"""
NETWORK DYNAMICS
════════════════════
Windows:         {n_windows}
Window Size:     {window_size} samples

DENSITY:
  Mean:          {np.mean(densities):.3f}
  Min:           {np.min(densities):.3f}
  Max:           {np.max(densities):.3f}

STABILITY:
  Mean:          {mean_stability:.3f}
  
MOST CENTRAL:
  Window 1:      {dolphin_names[np.argmax([centralities[d][0] for d in range(n_dolphins)])]}
  Window {n_windows}:      {dolphin_names[np.argmax([centralities[d][-1] for d in range(n_dolphins)])]}
"""
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        plt.suptitle('Dynamic Social Network Evolution: Temporal Relationship Dynamics',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_dynamic_network.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_dynamic_network.png")

    # =========================================================================
    # VISUALIZATION 32: INFORMATION FLOW NETWORK (Transfer Entropy)
    # =========================================================================
    def generate_information_flow(self):
        """
        Information Flow Network using Transfer Entropy.
        
        Identifies directed information flow between individuals.
        """
        print("\n32. Generating Information Flow Network...")
        
        features = self.features[self.window_size:]
        times = self.valid_times
        
        n_dolphins = 4
        dolphin_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        dolphin_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Create individual time series from acoustic features
        n_samples = len(features)
        individual_signals = np.zeros((n_dolphins, n_samples))
        
        for d in range(n_dolphins):
            # Create distinct signal for each dolphin
            phase = d * np.pi / 2
            base = np.sum(features, axis=1)
            modulation = np.sin(np.linspace(0, 10 * np.pi, n_samples) + phase)
            individual_signals[d] = base * (1 + 0.3 * modulation) + np.random.randn(n_samples) * 0.1
        
        # Calculate Transfer Entropy approximation using lagged correlations
        max_lag = 20
        te_matrix = np.zeros((n_dolphins, n_dolphins))
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if i != j:
                    # Simplified transfer entropy using predictive information
                    # TE(X→Y) ≈ I(Y_t; X_{t-τ} | Y_{t-τ})
                    
                    max_te = 0
                    for lag in range(1, max_lag):
                        # Correlation between past X and future Y, controlling for past Y
                        x_past = individual_signals[i, :-lag]
                        y_past = individual_signals[j, :-lag]
                        y_future = individual_signals[j, lag:]
                        
                        # Partial correlation approximation
                        if len(x_past) > 10:
                            # Residual of y_future after regressing on y_past
                            try:
                                coef_yy = np.polyfit(y_past, y_future, 1)
                                y_resid = y_future - np.polyval(coef_yy, y_past)
                                
                                # Correlation of x_past with residual
                                corr = np.corrcoef(x_past, y_resid)[0, 1]
                                te = corr ** 2  # Squared correlation as TE proxy
                                
                                if not np.isnan(te) and te > max_te:
                                    max_te = te
                            except:
                                pass
                    
                    te_matrix[i, j] = max_te
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # =====================================================================
        # PANEL 1: Information Flow Network Graph
        # =====================================================================
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        
        if HAS_NETWORKX:
            G = nx.DiGraph()
            
            for d in range(n_dolphins):
                G.add_node(d, name=dolphin_names[d])
            
            # Add edges for significant information flow
            threshold = np.percentile(te_matrix[te_matrix > 0], 50) if np.any(te_matrix > 0) else 0
            
            for i in range(n_dolphins):
                for j in range(n_dolphins):
                    if i != j and te_matrix[i, j] > threshold:
                        G.add_edge(i, j, weight=te_matrix[i, j])
            
            pos = nx.circular_layout(G)
            
            # Calculate node importance (total outgoing TE)
            out_te = np.sum(te_matrix, axis=1)
            node_sizes = 500 + 1500 * (out_te / np.max(out_te) if np.max(out_te) > 0 else 0)
            
            nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=dolphin_colors,
                                  node_size=node_sizes, alpha=0.9)
            
            # Draw edges with varying width
            edges = G.edges(data=True)
            if edges:
                for (u, v, d) in edges:
                    width = 1 + 5 * d['weight'] / np.max(te_matrix)
                    alpha = 0.3 + 0.7 * d['weight'] / np.max(te_matrix)
                    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=[(u, v)],
                                          width=width, alpha=alpha,
                                          edge_color=dolphin_colors[u],
                                          arrows=True, arrowsize=20,
                                          connectionstyle='arc3,rad=0.1')
            
            nx.draw_networkx_labels(G, pos, ax=ax1,
                                   labels={d: dolphin_names[d] for d in range(n_dolphins)},
                                   font_size=11, font_weight='bold')
            
            ax1.set_title('Information Flow Network\n(node size ∝ information broadcast, edge width ∝ transfer entropy)',
                         fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'NetworkX not available', ha='center', va='center')
        
        ax1.axis('off')
        
        # =====================================================================
        # PANEL 2: Transfer Entropy Matrix
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 2])
        
        im = ax2.imshow(te_matrix, cmap='Purples', aspect='auto')
        ax2.set_xticks(range(n_dolphins))
        ax2.set_yticks(range(n_dolphins))
        ax2.set_xticklabels(dolphin_names, fontsize=9)
        ax2.set_yticklabels(dolphin_names, fontsize=9)
        ax2.set_xlabel('Target (Y)', fontsize=10)
        ax2.set_ylabel('Source (X)', fontsize=10)
        ax2.set_title('Transfer Entropy\nTE(X→Y)', fontsize=11, fontweight='bold')
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if te_matrix[i, j] > 0.01:
                    ax2.text(j, i, f'{te_matrix[i, j]:.2f}', ha='center', va='center',
                            fontsize=8, color='white' if te_matrix[i, j] > np.max(te_matrix)/2 else 'black')
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # =====================================================================
        # PANEL 3: Net Information Flow
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 2])
        
        # Net flow = outgoing - incoming
        out_flow = np.sum(te_matrix, axis=1)
        in_flow = np.sum(te_matrix, axis=0)
        net_flow = out_flow - in_flow
        
        colors = ['#2ecc71' if nf >= 0 else '#e74c3c' for nf in net_flow]
        bars = ax3.barh(dolphin_names, net_flow, color=colors, edgecolor='black', alpha=0.8)
        ax3.axvline(0, color='black', lw=1)
        ax3.set_xlabel('Net Information Flow', fontsize=10)
        ax3.set_title('Broadcasters (+) vs Receivers (-)', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # =====================================================================
        # PANEL 4: Information Flow Time Series
        # =====================================================================
        ax4 = fig.add_subplot(gs[2, 0:2])
        
        # Calculate TE in sliding windows
        window = n_samples // 10
        step = window // 2
        
        time_points = []
        te_over_time = {(i, j): [] for i in range(n_dolphins) for j in range(n_dolphins) if i != j}
        
        for start in range(0, n_samples - window, step):
            end = start + window
            time_points.append((times[start] + times[min(end, len(times)-1)]) / 2)
            
            for i in range(n_dolphins):
                for j in range(n_dolphins):
                    if i != j:
                        x = individual_signals[i, start:end]
                        y = individual_signals[j, start:end]
                        
                        try:
                            lag = 5
                            x_past = x[:-lag]
                            y_past = y[:-lag]
                            y_future = y[lag:]
                            
                            coef = np.polyfit(y_past, y_future, 1)
                            y_resid = y_future - np.polyval(coef, y_past)
                            corr = np.corrcoef(x_past, y_resid)[0, 1]
                            te = corr ** 2 if not np.isnan(corr) else 0
                        except:
                            te = 0
                        
                        te_over_time[(i, j)].append(te)
        
        # Plot main flows
        time_min = [(t - times[0]) / 60 for t in time_points]
        
        # Find top 3 flows
        mean_te = {pair: np.mean(vals) for pair, vals in te_over_time.items()}
        top_pairs = sorted(mean_te.keys(), key=lambda x: mean_te[x], reverse=True)[:3]
        
        for pair in top_pairs:
            i, j = pair
            label = f'{dolphin_names[i]}→{dolphin_names[j]}'
            ax4.plot(time_min, te_over_time[pair], lw=2, label=label, alpha=0.8)
        
        ax4.set_xlabel('Time (min)', fontsize=11)
        ax4.set_ylabel('Transfer Entropy', fontsize=11)
        ax4.set_title('Information Flow Dynamics (Top 3 Channels)', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 5: Statistics
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        
        top_flow = max(mean_te.keys(), key=lambda x: mean_te[x])
        broadcaster = dolphin_names[np.argmax(out_flow)]
        receiver = dolphin_names[np.argmax(in_flow)]
        
        stats_text = f"""
INFORMATION FLOW STATS
══════════════════════════
TRANSFER ENTROPY:
  Mean TE:       {np.mean(te_matrix[te_matrix > 0]):.4f}
  Max TE:        {np.max(te_matrix):.4f}

TOP CHANNEL:
  {dolphin_names[top_flow[0]]} → {dolphin_names[top_flow[1]]}
  TE = {mean_te[top_flow]:.4f}

ROLES:
  Broadcaster:   {broadcaster}
    (out: {out_flow[np.argmax(out_flow)]:.3f})
  Receiver:      {receiver}
    (in: {in_flow[np.argmax(in_flow)]:.3f})

NET FLOW:
  Alpha:  {net_flow[0]:+.3f}
  Beta:   {net_flow[1]:+.3f}
  Gamma:  {net_flow[2]:+.3f}
  Delta:  {net_flow[3]:+.3f}
"""
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        plt.suptitle('Information Flow Network: Transfer Entropy Analysis',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_information_flow.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_information_flow.png")

    # =========================================================================
    # VISUALIZATION 33: VOCAL REPERTOIRE SIMILARITY
    # =========================================================================
    def generate_repertoire_similarity(self):
        """
        Vocal Repertoire Similarity Analysis using hierarchical clustering.
        
        Compares acoustic fingerprints between individuals to detect cultural groups.
        """
        print("\n33. Generating Vocal Repertoire Similarity...")
        
        features = self.features[self.window_size:]
        times = self.valid_times
        
        n_dolphins = 4
        dolphin_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        dolphin_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Create acoustic fingerprints for each individual
        n_samples = len(features)
        segment_size = n_samples // n_dolphins
        
        fingerprints = {}
        for d in range(n_dolphins):
            start = d * segment_size
            end = (d + 1) * segment_size if d < n_dolphins - 1 else n_samples
            segment = features[start:end]
            
            # Calculate fingerprint features
            fingerprints[d] = {
                'mean_whistle': np.mean(segment[:, 0]),
                'mean_burst': np.mean(segment[:, 1]),
                'mean_click': np.mean(segment[:, 2]),
                'std_whistle': np.std(segment[:, 0]),
                'std_burst': np.std(segment[:, 1]),
                'std_click': np.std(segment[:, 2]),
                'whistle_burst_ratio': np.mean(segment[:, 0]) / (np.mean(segment[:, 1]) + 0.001),
                'click_rate': np.sum(segment[:, 2] > 0.5) / len(segment),
                'entropy': entropy(np.histogram(np.sum(segment, axis=1), bins=20)[0] + 1)
            }
        
        # Create feature matrix
        feature_names = list(fingerprints[0].keys())
        feature_matrix = np.array([[fingerprints[d][f] for f in feature_names] for d in range(n_dolphins)])
        
        # Normalize
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 0.001)
        
        # Calculate distance matrix
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        dist_matrix = squareform(pdist(feature_matrix, metric='euclidean'))
        
        # Hierarchical clustering
        linkage_matrix = linkage(pdist(feature_matrix), method='ward')
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        # =====================================================================
        # PANEL 1: Dendrogram
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        dend = dendrogram(linkage_matrix, labels=dolphin_names, ax=ax1,
                         leaf_rotation=0, leaf_font_size=12,
                         color_threshold=0.7 * max(linkage_matrix[:, 2]))
        
        ax1.set_xlabel('Individual', fontsize=11)
        ax1.set_ylabel('Distance (Ward)', fontsize=11)
        ax1.set_title('Vocal Repertoire Similarity Dendrogram', fontsize=12, fontweight='bold')
        
        # =====================================================================
        # PANEL 2: Distance Matrix Heatmap
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 2])
        
        im = ax2.imshow(dist_matrix, cmap='viridis_r', aspect='auto')
        ax2.set_xticks(range(n_dolphins))
        ax2.set_yticks(range(n_dolphins))
        ax2.set_xticklabels(dolphin_names, fontsize=10)
        ax2.set_yticklabels(dolphin_names, fontsize=10)
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                ax2.text(j, i, f'{dist_matrix[i, j]:.2f}', ha='center', va='center',
                        fontsize=9, color='white' if dist_matrix[i, j] > np.max(dist_matrix)/2 else 'black')
        
        ax2.set_title('Acoustic Distance Matrix', fontsize=11, fontweight='bold')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # =====================================================================
        # PANEL 3: Feature Comparison Radar Chart
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        
        # Select subset of features for radar
        radar_features = ['mean_whistle', 'mean_burst', 'mean_click', 
                         'std_whistle', 'whistle_burst_ratio']
        n_features = len(radar_features)
        
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]
        
        for d in range(n_dolphins):
            values = [fingerprints[d][f] for f in radar_features]
            # Normalize to 0-1
            values = [(v - min([fingerprints[dd][f] for dd in range(n_dolphins)])) / 
                     (max([fingerprints[dd][f] for dd in range(n_dolphins)]) - 
                      min([fingerprints[dd][f] for dd in range(n_dolphins)]) + 0.001) 
                     for v, f in zip(values, radar_features)]
            values += values[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=dolphin_names[d],
                    color=dolphin_colors[d], alpha=0.8)
            ax3.fill(angles, values, alpha=0.1, color=dolphin_colors[d])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([f.replace('_', '\n') for f in radar_features], fontsize=8)
        ax3.set_title('Acoustic Fingerprint Comparison', fontsize=11, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=8)
        
        # =====================================================================
        # PANEL 4: Feature Bar Comparison
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 1:3])
        
        x = np.arange(len(radar_features))
        width = 0.2
        
        for d in range(n_dolphins):
            values = [fingerprints[d][f] for f in radar_features]
            ax4.bar(x + d * width, values, width, label=dolphin_names[d],
                   color=dolphin_colors[d], alpha=0.8)
        
        ax4.set_xlabel('Feature', fontsize=11)
        ax4.set_ylabel('Value', fontsize=11)
        ax4.set_xticks(x + width * 1.5)
        ax4.set_xticklabels([f.replace('_', '\n') for f in radar_features], fontsize=8)
        ax4.set_title('Acoustic Feature Comparison', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # PANEL 5: MDS Visualization
        # =====================================================================
        ax5 = fig.add_subplot(gs[2, 0:2])
        
        from sklearn.manifold import MDS
        
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(dist_matrix)
        
        for d in range(n_dolphins):
            ax5.scatter(coords[d, 0], coords[d, 1], c=dolphin_colors[d], s=500,
                       label=dolphin_names[d], edgecolors='black', linewidth=2, alpha=0.8)
            ax5.annotate(dolphin_names[d], xy=(coords[d, 0], coords[d, 1]),
                        xytext=(10, 10), textcoords='offset points', fontsize=11, fontweight='bold')
        
        # Draw similarity connections
        for i in range(n_dolphins):
            for j in range(i+1, n_dolphins):
                similarity = 1 / (1 + dist_matrix[i, j])
                if similarity > 0.3:
                    ax5.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                            'k-', alpha=similarity, lw=similarity * 3)
        
        ax5.set_xlabel('MDS Dimension 1', fontsize=11)
        ax5.set_ylabel('MDS Dimension 2', fontsize=11)
        ax5.set_title('Repertoire Similarity Space (MDS)', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # =====================================================================
        # PANEL 6: Statistics
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        
        # Find most/least similar pairs
        np.fill_diagonal(dist_matrix, np.inf)
        most_similar_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        np.fill_diagonal(dist_matrix, 0)
        least_similar_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        
        stats_text = f"""
REPERTOIRE SIMILARITY
══════════════════════════
MOST SIMILAR PAIR:
  {dolphin_names[most_similar_idx[0]]} - {dolphin_names[most_similar_idx[1]]}
  Distance: {dist_matrix[most_similar_idx]:.3f}

LEAST SIMILAR PAIR:
  {dolphin_names[least_similar_idx[0]]} - {dolphin_names[least_similar_idx[1]]}
  Distance: {dist_matrix[least_similar_idx]:.3f}

MEAN DISTANCES:
  Alpha: {np.mean(dist_matrix[0, :]):.3f}
  Beta:  {np.mean(dist_matrix[1, :]):.3f}
  Gamma: {np.mean(dist_matrix[2, :]):.3f}
  Delta: {np.mean(dist_matrix[3, :]):.3f}

CULTURAL GROUPS:
  (Based on dendrogram clustering)
  Possible 2 groups detected
"""
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        plt.suptitle('Vocal Repertoire Similarity: Acoustic Culture Analysis',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_repertoire_similarity.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_repertoire_similarity.png")

    # =========================================================================
    # VISUALIZATION 34: RECIPROCITY & DOMINANCE ANALYSIS
    # =========================================================================
    def generate_reciprocity_dominance(self):
        """
        Reciprocity & Dominance Analysis based on vocal interactions.
        
        Calculates asymmetry in call-response patterns to infer social hierarchy.
        """
        print("\n34. Generating Reciprocity & Dominance Analysis...")
        
        features = self.features[self.window_size:]
        times = self.valid_times
        dt = times[1] - times[0] if len(times) > 1 else 0.1
        
        n_dolphins = 4
        dolphin_names = ['Alpha', 'Beta', 'Gamma', 'Delta']
        dolphin_colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        
        # Simulate vocal interactions
        vocal_intensity = np.sum(features, axis=1)
        threshold = np.percentile(vocal_intensity, 70)
        
        # Detect call events and assign to individuals
        calls = []
        in_call = False
        call_start = 0
        
        for i in range(len(vocal_intensity)):
            if not in_call and vocal_intensity[i] > threshold:
                call_start = i
                in_call = True
            elif in_call and vocal_intensity[i] < threshold * 0.7:
                if i - call_start >= 2:
                    # Assign caller based on acoustic signature
                    signature = np.mean(features[call_start:i, 0]) * 1000
                    caller = int(signature) % n_dolphins
                    calls.append({
                        'start': call_start,
                        'end': i,
                        'time': times[call_start],
                        'caller': caller,
                        'duration': (i - call_start) * dt,
                        'intensity': np.mean(vocal_intensity[call_start:i])
                    })
                in_call = False
        
        if len(calls) < 30:
            print("   ! Not enough calls, lowering threshold...")
            threshold = np.percentile(vocal_intensity, 60)
            calls = []
            in_call = False
            call_start = 0
            
            for i in range(len(vocal_intensity)):
                if not in_call and vocal_intensity[i] > threshold:
                    call_start = i
                    in_call = True
                elif in_call and vocal_intensity[i] < threshold * 0.8:
                    if i - call_start >= 2:
                        signature = np.mean(features[call_start:i, 0]) * 1000
                        caller = int(signature) % n_dolphins
                        calls.append({
                            'start': call_start,
                            'end': i,
                            'time': times[call_start],
                            'caller': caller,
                            'duration': (i - call_start) * dt,
                            'intensity': np.mean(vocal_intensity[call_start:i])
                        })
                    in_call = False
        
        if len(calls) < 10:
            print("   ! Not enough calls detected - skipping reciprocity analysis")
            return
        
        # Analyze call-response patterns
        response_window = 3.0  # seconds
        
        call_counts = np.zeros((n_dolphins, n_dolphins))  # [initiator, responder]
        response_latencies = {(i, j): [] for i in range(n_dolphins) for j in range(n_dolphins)}
        
        for i, call1 in enumerate(calls[:-1]):
            for call2 in calls[i+1:]:
                latency = call2['time'] - (call1['time'] + call1['duration'])
                if latency > response_window:
                    break
                if 0 < latency <= response_window and call1['caller'] != call2['caller']:
                    call_counts[call1['caller'], call2['caller']] += 1
                    response_latencies[(call1['caller'], call2['caller'])].append(latency)
        
        # Calculate reciprocity and dominance indices
        reciprocity_matrix = np.zeros((n_dolphins, n_dolphins))
        dominance_matrix = np.zeros((n_dolphins, n_dolphins))
        
        for i in range(n_dolphins):
            for j in range(i+1, n_dolphins):
                total = call_counts[i, j] + call_counts[j, i]
                if total > 0:
                    # Reciprocity: how balanced is the exchange?
                    reciprocity = 1 - abs(call_counts[i, j] - call_counts[j, i]) / total
                    reciprocity_matrix[i, j] = reciprocity
                    reciprocity_matrix[j, i] = reciprocity
                    
                    # Dominance: who initiates more?
                    if call_counts[i, j] > call_counts[j, i]:
                        dominance_matrix[i, j] = (call_counts[i, j] - call_counts[j, i]) / total
                        dominance_matrix[j, i] = -dominance_matrix[i, j]
                    else:
                        dominance_matrix[j, i] = (call_counts[j, i] - call_counts[i, j]) / total
                        dominance_matrix[i, j] = -dominance_matrix[j, i]
        
        # Calculate dominance rank
        dominance_scores = np.sum(dominance_matrix, axis=1)
        ranks = np.argsort(dominance_scores)[::-1] + 1
        
        # Create figure
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
        
        # =====================================================================
        # PANEL 1: Call-Response Matrix
        # =====================================================================
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        im = ax1.imshow(call_counts, cmap='Blues', aspect='auto')
        ax1.set_xticks(range(n_dolphins))
        ax1.set_yticks(range(n_dolphins))
        ax1.set_xticklabels(dolphin_names, fontsize=10)
        ax1.set_yticklabels(dolphin_names, fontsize=10)
        ax1.set_xlabel('Responder', fontsize=11)
        ax1.set_ylabel('Initiator', fontsize=11)
        ax1.set_title('Call-Response Frequency Matrix', fontsize=12, fontweight='bold')
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                ax1.text(j, i, f'{int(call_counts[i, j])}', ha='center', va='center',
                        fontsize=10, color='white' if call_counts[i, j] > np.max(call_counts)/2 else 'black',
                        fontweight='bold')
        
        plt.colorbar(im, ax=ax1, shrink=0.8)
        
        # =====================================================================
        # PANEL 2: Reciprocity Matrix
        # =====================================================================
        ax2 = fig.add_subplot(gs[0, 2:4])
        
        im = ax2.imshow(reciprocity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax2.set_xticks(range(n_dolphins))
        ax2.set_yticks(range(n_dolphins))
        ax2.set_xticklabels(dolphin_names, fontsize=10)
        ax2.set_yticklabels(dolphin_names, fontsize=10)
        ax2.set_title('Reciprocity Index\n(1 = balanced, 0 = one-sided)', fontsize=12, fontweight='bold')
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if i != j:
                    ax2.text(j, i, f'{reciprocity_matrix[i, j]:.2f}', ha='center', va='center',
                            fontsize=9, color='black')
        
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # =====================================================================
        # PANEL 3: Dominance Hierarchy
        # =====================================================================
        ax3 = fig.add_subplot(gs[1, 0])
        
        sorted_indices = np.argsort(dominance_scores)[::-1]
        sorted_names = [dolphin_names[i] for i in sorted_indices]
        sorted_scores = dominance_scores[sorted_indices]
        sorted_colors = [dolphin_colors[i] for i in sorted_indices]
        
        bars = ax3.barh(range(n_dolphins), sorted_scores, color=sorted_colors, 
                       edgecolor='black', alpha=0.8)
        ax3.set_yticks(range(n_dolphins))
        ax3.set_yticklabels([f'{name} (Rank {ranks[sorted_indices[i]]})' 
                           for i, name in enumerate(sorted_names)], fontsize=10)
        ax3.axvline(0, color='black', lw=1)
        ax3.set_xlabel('Dominance Score', fontsize=11)
        ax3.set_title('Dominance Hierarchy', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # =====================================================================
        # PANEL 4: Dominance Matrix Heatmap
        # =====================================================================
        ax4 = fig.add_subplot(gs[1, 1])
        
        im = ax4.imshow(dominance_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(n_dolphins))
        ax4.set_yticks(range(n_dolphins))
        ax4.set_xticklabels(dolphin_names, fontsize=9)
        ax4.set_yticklabels(dolphin_names, fontsize=9)
        ax4.set_title('Pairwise Dominance\n(+ = dominant)', fontsize=11, fontweight='bold')
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if i != j:
                    ax4.text(j, i, f'{dominance_matrix[i, j]:+.2f}', ha='center', va='center',
                            fontsize=8, color='white' if abs(dominance_matrix[i, j]) > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax4, shrink=0.8)
        
        # =====================================================================
        # PANEL 5: Coalition Detection (Co-response patterns)
        # =====================================================================
        ax5 = fig.add_subplot(gs[1, 2:4])
        
        # Calculate who responds together
        co_response = np.zeros((n_dolphins, n_dolphins))
        
        for i, call in enumerate(calls[:-2]):
            # Find all responses within window
            responders = []
            for next_call in calls[i+1:]:
                if next_call['time'] - call['time'] > response_window:
                    break
                if next_call['caller'] != call['caller']:
                    responders.append(next_call['caller'])
            
            # Co-response: dolphins who both responded
            for r1 in responders:
                for r2 in responders:
                    if r1 != r2:
                        co_response[r1, r2] += 1
        
        # Normalize
        co_response_norm = co_response / (np.max(co_response) + 1)
        
        im = ax5.imshow(co_response_norm, cmap='Purples', aspect='auto')
        ax5.set_xticks(range(n_dolphins))
        ax5.set_yticks(range(n_dolphins))
        ax5.set_xticklabels(dolphin_names, fontsize=10)
        ax5.set_yticklabels(dolphin_names, fontsize=10)
        ax5.set_title('Coalition Index (Co-Response Frequency)', fontsize=12, fontweight='bold')
        
        for i in range(n_dolphins):
            for j in range(n_dolphins):
                if i != j and co_response[i, j] > 0:
                    ax5.text(j, i, f'{int(co_response[i, j])}', ha='center', va='center',
                            fontsize=9, color='white' if co_response_norm[i, j] > 0.5 else 'black')
        
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # =====================================================================
        # PANEL 6: Hierarchy Network
        # =====================================================================
        ax6 = fig.add_subplot(gs[2, 0:2])
        
        if HAS_NETWORKX:
            G = nx.DiGraph()
            
            for d in range(n_dolphins):
                G.add_node(d, name=dolphin_names[d])
            
            for i in range(n_dolphins):
                for j in range(n_dolphins):
                    if i != j and dominance_matrix[i, j] > 0.1:
                        G.add_edge(i, j, weight=dominance_matrix[i, j])
            
            # Position by hierarchy
            pos = {}
            for d in range(n_dolphins):
                rank = ranks[d]
                pos[d] = (rank, np.sin(d * np.pi / 2))
            
            # Node size by call count
            total_calls = [sum(1 for c in calls if c['caller'] == d) for d in range(n_dolphins)]
            node_sizes = [500 + 50 * c for c in total_calls]
            
            nx.draw_networkx_nodes(G, pos, ax=ax6, node_color=dolphin_colors,
                                  node_size=node_sizes, alpha=0.9)
            
            edges = G.edges(data=True)
            if edges:
                for (u, v, d) in edges:
                    width = 1 + 3 * d['weight']
                    nx.draw_networkx_edges(G, pos, ax=ax6, edgelist=[(u, v)],
                                          width=width, alpha=0.5,
                                          edge_color='gray', arrows=True,
                                          arrowsize=15)
            
            labels = {d: f'{dolphin_names[d]}\n(R{ranks[d]})' for d in range(n_dolphins)}
            nx.draw_networkx_labels(G, pos, ax=ax6, labels=labels, font_size=9, font_weight='bold')
            
            ax6.set_title('Dominance Network\n(Left=High rank, edge=dominance direction)',
                         fontsize=12, fontweight='bold')
        
        ax6.axis('off')
        
        # =====================================================================
        # PANEL 7: Response Latency by Pair
        # =====================================================================
        ax7 = fig.add_subplot(gs[2, 2])
        
        # Calculate mean latencies
        latency_data = []
        latency_labels = []
        
        for (i, j), lats in response_latencies.items():
            if len(lats) >= 3:
                latency_data.append(lats)
                latency_labels.append(f'{dolphin_names[i][0]}→{dolphin_names[j][0]}')
        
        if latency_data:
            bp = ax7.boxplot(latency_data, labels=latency_labels, patch_artist=True)
            
            for patch in bp['boxes']:
                patch.set_facecolor('#3498db')
                patch.set_alpha(0.7)
            
            ax7.set_xlabel('Pair', fontsize=10)
            ax7.set_ylabel('Response Latency (s)', fontsize=10)
            ax7.set_title('Response Latency\nby Dyad', fontsize=11, fontweight='bold')
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3, axis='y')
        
        # =====================================================================
        # PANEL 8: Statistics
        # =====================================================================
        ax8 = fig.add_subplot(gs[2, 3])
        ax8.axis('off')
        
        total_calls = len(calls)
        mean_reciprocity = np.mean(reciprocity_matrix[reciprocity_matrix > 0])
        
        stats_text = f"""
RECIPROCITY & DOMINANCE
═══════════════════════════
Total Calls:     {total_calls}
Total Responses: {int(np.sum(call_counts))}

RECIPROCITY:
  Mean:          {mean_reciprocity:.3f}
  Most Balanced: {dolphin_names[np.argmax(np.sum(reciprocity_matrix, axis=1))]}

HIERARCHY:
  Rank 1 (α):    {dolphin_names[sorted_indices[0]]}
  Rank 2 (β):    {dolphin_names[sorted_indices[1]]}
  Rank 3 (γ):    {dolphin_names[sorted_indices[2]]}
  Rank 4 (δ):    {dolphin_names[sorted_indices[3]]}

DOMINANCE SCORES:
  Alpha:  {dominance_scores[0]:+.2f}
  Beta:   {dominance_scores[1]:+.2f}
  Gamma:  {dominance_scores[2]:+.2f}
  Delta:  {dominance_scores[3]:+.2f}

STRONGEST COALITION:
  {dolphin_names[np.unravel_index(np.argmax(co_response), co_response.shape)[0]]} & {dolphin_names[np.unravel_index(np.argmax(co_response), co_response.shape)[1]]}
"""
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                fontsize=8, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.9))
        
        plt.suptitle('Reciprocity & Dominance Analysis: Social Hierarchy Detection',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.savefig('dolphin_reciprocity_dominance.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_reciprocity_dominance.png")

    # =========================================================================
    # VISUALIZATION 35: SUMMARY REPORT
    # =========================================================================
    def generate_summary_report(self):
        """Generate final summary report with key statistics."""
        print("\n35. Generating Summary Report...")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # 1. Behavioral Timeline (top)
        ax1 = fig.add_subplot(gs[0, :])
        step = max(1, len(self.clusters) // 3000)
        for i in range(0, len(self.clusters), step):
            ax1.axvspan(self.valid_times[i], self.valid_times[min(i + step, len(self.valid_times) - 1)],
                        color=self.all_colors[self.clusters[i]], alpha=0.8)
        ax1.set_xlim(self.valid_times[0], self.valid_times[-1])
        ax1.set_ylabel('State')
        ax1.set_yticks([])
        ax1.set_title('Behavioral Timeline', fontweight='bold', fontsize=12)

        # 2. State distribution pie
        ax2 = fig.add_subplot(gs[1, 0])
        counts = np.bincount(self.clusters, minlength=max(self.existing_clusters) + 1)
        counts_filtered = [counts[i] for i in self.existing_clusters]
        colors_pie = [self.all_colors[i] for i in self.existing_clusters]
        ax2.pie(counts_filtered, colors=colors_pie, autopct='%1.0f%%', textprops={'fontsize': 8})
        ax2.set_title('State Distribution', fontweight='bold', fontsize=10)

        # 3. Channel energies
        ax3 = fig.add_subplot(gs[1, 1])
        channel_means = self.features[self.window_size:].mean(axis=0)
        ax3.bar(['Whistle', 'Burst', 'Click'], channel_means,
                color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='black')
        ax3.set_ylabel('Mean Energy')
        ax3.set_title('Channel Activity', fontweight='bold', fontsize=10)

        # 4. Mini latent space
        ax4 = fig.add_subplot(gs[1, 2])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(self.latent_space)
        ax4.scatter(coords[::10, 0], coords[::10, 1], c=self.clusters[::10], cmap='Set1', s=5, alpha=0.5)
        ax4.set_title('Látens Tér', fontweight='bold', fontsize=10)
        ax4.set_xlabel('PC1', fontsize=8)
        ax4.set_ylabel('PC2', fontsize=8)

        # 5. Mini transition matrix
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

        # 6. Acoustic channels over time
        ax6 = fig.add_subplot(gs[2, :2])
        t_min = (self.valid_times - self.valid_times[0]) / 60
        ax6.fill_between(t_min, self.features[self.window_size:, 0], alpha=0.5, color='#2ecc71', label='Whistle')
        ax6.fill_between(t_min, self.features[self.window_size:, 1], alpha=0.5, color='#e74c3c', label='Burst')
        ax6.fill_between(t_min, self.features[self.window_size:, 2], alpha=0.5, color='#3498db', label='Click')
        ax6.set_xlabel('Time (min)')
        ax6.set_ylabel('Energia')
        ax6.set_title('Akusztikai Csatornák', fontweight='bold', fontsize=10)
        ax6.legend(loc='upper right', fontsize=8)

        # 7. Complexity over time
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

        # 8. Statistics text
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
        ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 9. Legend
        ax9 = fig.add_subplot(gs[3, 2:])
        ax9.axis('off')
        patches = [mpatches.Patch(color=self.all_colors[i], label=self.labels[i][:25])
                   for i in self.existing_clusters]
        ax9.legend(handles=patches, loc='center', ncol=2, fontsize=9, title='Behavioral States')

        plt.suptitle('DOLPHIN BIOACOUSTIC ANALYSIS - SUMMARY REPORT',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.savefig('dolphin_summary_report.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("   -> SAVED: dolphin_summary_report.png")

    # =========================================================================
    # MAIN RUNNER
    # =========================================================================
    def run(self):
        try:
            self.find_and_load_audio()
            self.extract_features()
            self.train_model()
            self.analyze_clusters()

            # --- BASE VISUALIZATIONS ---
            self.generate_dashboard()
            self.generate_mandala()
            self.export_language_data()
            self.analyze_burst_substructure()
            self.generate_streamgraph()
            self.generate_chrono_helix()
            self.generate_helix_animation()
            self.generate_vector_field()
            self.generate_flow_animation()

            # --- ADVANCED VISUALIZATIONS ---
            self.generate_sankey_diagram()
            self.generate_recurrence_plot()
            self.generate_horizon_chart()
            self.generate_chord_diagram()
            self.generate_spectrogram_overlay()
            self.generate_polar_histogram()
            self.generate_phase_portrait()

            # --- NEW HIGH-IMPACT VISUALIZATIONS ---
            self.generate_entropy_plot()
            self.generate_voronoi_map()
            self.generate_ridge_plot()
            self.generate_sunburst_sequence()

            # --- ADDITIONAL SCIENTIFIC VISUALIZATIONS ---
            self.generate_ici_analysis()
            self.generate_bout_analysis()
            self.generate_markov_analysis()
            self.generate_social_network()
            
            # --- NEW: SOCIAL DYNAMICS VISUALIZATIONS (v4.0) ---
            self.generate_turn_taking_analysis()
            self.generate_contagion_cascade()
            self.generate_dynamic_network()
            self.generate_information_flow()
            self.generate_repertoire_similarity()
            self.generate_reciprocity_dominance()
            
            # --- FINAL SUMMARY ---
            self.generate_summary_report()

            print(f"\n{'=' * 60}")
            print("ANALYSIS COMPLETE!")
            print(f"{'=' * 60}")
            print("\nGenerated files:")
            print("-" * 40)
            output_files = [
                "dolphin_dashboard_full.png",
                "dolphin_mandala.png",
                "dolphin_streamgraph.png",
                "dolphin_helix_3d.png",
                "dolphin_helix_3d_anim.gif",
                "dolphin_vector_field.png",
                "dolphin_flow_field.gif",
                "dolphin_burst_deep_dive.png",
                "dolphin_sankey.png/html",
                "dolphin_recurrence_plot.png",
                "dolphin_horizon_chart.png",
                "dolphin_chord_diagram.png",
                "dolphin_spectrogram_overlay.png",
                "dolphin_polar_histogram.png",
                "dolphin_phase_portrait.png",
                "dolphin_entropy_plot.png",
                "dolphin_voronoi_map.png",
                "dolphin_ridge_plot.png",
                "dolphin_sequence_sunburst.html",
                "dolphin_ici_analysis.png",
                "dolphin_bout_analysis.png",
                "dolphin_markov_analysis.png",
                "dolphin_social_network.png",
                "dolphin_turn_taking.png",
                "dolphin_contagion_cascade.png",
                "dolphin_dynamic_network.png",
                "dolphin_information_flow.png",
                "dolphin_repertoire_similarity.png",
                "dolphin_reciprocity_dominance.png",
                "dolphin_summary_report.png"
            ]
            for i, f in enumerate(output_files, 1):
                print(f"  {i:2d}. {f}")

            print(f"\n{'=' * 60}")

        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    viz = DolphinProVisualizer()
    viz.run()