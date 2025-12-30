from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap

LATENTS = Path("data/features/conv_vae_latents.npy")
META = Path("data/features/conv_vae_latents_meta.csv")
MELS_INDEX = Path("data/features/mels_index.csv")

OUT_DIR = Path("data/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FIXED_T = 256  # must match Conv-VAE fixed time


def fix_time(m: np.ndarray, T: int) -> np.ndarray:
    if m.shape[1] > T:
        return m[:, :T]
    if m.shape[1] < T:
        pad = np.zeros((m.shape[0], T - m.shape[1]), dtype=m.dtype)
        return np.concatenate([m, pad], axis=1)
    return m


def load_flat_mels(mels_index_csv: Path, fixed_t: int) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(mels_index_csv)
    mats = []
    for p in df["mel_path"]:
        m = np.load(p)
        m = fix_time(m, fixed_t)
        mats.append(m.reshape(-1))
    X = np.stack(mats, axis=0).astype(np.float32)
    X = StandardScaler().fit_transform(X)
    return X, df[["track_id", "genre"]].copy()


def eval_kmeans(X: np.ndarray, k: int, name: str):
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    print(f"{name}: Silhouette={sil:.4f}  CH={ch:.2f}")
    return labels, sil, ch


def plot_2d(Y: np.ndarray, labels: np.ndarray, title: str, outpath: Path):
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=8, c=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    if not LATENTS.exists() or not META.exists():
        raise FileNotFoundError("Missing VAE latents. Run 06_train_conv_vae.py first.")

    Z = np.load(LATENTS)
    meta = pd.read_csv(META)

    K = meta["genre"].nunique()  # 5
    print("K (num genres):", K)

    # VAE latent clustering
    Zs = StandardScaler().fit_transform(Z)
    labels_vae, sil_vae, ch_vae = eval_kmeans(Zs, K, "Conv-VAE Latent + KMeans")

    # Baseline: PCA + KMeans on mel features (flattened)
    X_mel, mel_meta = load_flat_mels(MELS_INDEX, fixed_t=FIXED_T)
    pca = PCA(n_components=32, random_state=SEED)
    X_pca = pca.fit_transform(X_mel)
    labels_pca, sil_pca, ch_pca = eval_kmeans(X_pca, K, "PCA(32) + KMeans")

    metrics = pd.DataFrame([
        {"method": "Conv-VAE Latent + KMeans", "silhouette": sil_vae, "calinski_harabasz": ch_vae},
        {"method": "PCA(32) + KMeans", "silhouette": sil_pca, "calinski_harabasz": ch_pca},
    ])
    metrics_path = OUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print("\nSaved metrics:", metrics_path)

    # UMAP + t-SNE on VAE latents
    reducer = umap.UMAP(n_components=2, random_state=SEED)
    Y_umap = reducer.fit_transform(Zs)
    plot_2d(Y_umap, labels_vae, "UMAP (Conv-VAE Latent) colored by KMeans cluster", OUT_DIR / "umap_vae_kmeans.png")

    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto", perplexity=30)
    Y_tsne = tsne.fit_transform(Zs)
    plot_2d(Y_tsne, labels_vae, "t-SNE (Conv-VAE Latent) colored by KMeans cluster", OUT_DIR / "tsne_vae_kmeans.png")

    print("\n✅ Done. Plots in:", OUT_DIR)
    print(" - umap_vae_kmeans.png")
    print(" - tsne_vae_kmeans.png")

if __name__ == "__main__":
    main()
