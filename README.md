# Multimodal Music Clustering with Variational Autoencoders (CSE 425)

This repository contains the implementation for **Easy, Medium, and Hard tasks**
of the CSE 425 project on unsupervised representation learning and clustering of music.

The project explores **Variational Autoencoders (VAEs)** for music feature learning,
including **multimodal fusion of audio and lyrics** and **conditional β-VAEs**.

---

## 📂 Dataset
- **Audio:** Free Music Archive (FMA-small)
- **Genres:** Rock, Pop, Hip-Hop, Folk, Experimental
- **Text:**  
  - Whisper-based speech transcriptions  
  - Public lyric APIs (when available)

> Raw audio, lyrics, and extracted features are **not included** in the repository  
> and can be regenerated using the provided scripts.

---

## ✅ Tasks Implemented

### 🟢 Easy Task
- Mel-spectrogram extraction
- Basic VAE / Conv-VAE for audio feature learning
- K-Means clustering on latent features
- PCA + K-Means baseline
- Evaluation: Silhouette Score, Calinski–Harabasz Index
- Visualization: UMAP, t-SNE

### 🟡 Medium Task
- **Multimodal Conv-VAE (Audio + Lyrics)**
- Lyrics embeddings using TF-IDF + SVD
- Clustering: K-Means, Agglomerative, DBSCAN
- Evaluation: Silhouette, Davies–Bouldin, ARI
- Comparison with PCA baseline

### 🔴 Hard Task
- **Conditional β-VAE (CVAE)** with genre conditioning
- Disentangled latent representations
- Multimodal clustering (audio + lyrics + genre)
- Metrics: Silhouette, NMI, ARI, Cluster Purity
- Comparisons:
  - PCA + K-Means
  - Autoencoder + K-Means
  - Raw spectral feature clustering
- Latent space visualizations and reconstructions

---


