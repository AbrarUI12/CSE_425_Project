from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Multimodal VAE outputs
LATENTS = Path("data/features/mm_vae_latents.npy")
META = Path("data/features/mm_vae_latents_meta.csv")

# Baseline features: PCA on flattened mel
MELS_INDEX = Path("data/features/mels_index.csv")

OUT_DIR = Path("data/results_medium")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
FIXED_T = 256

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

def eval_all(X: np.ndarray, y_true: np.ndarray | None, k: int, prefix: str) -> list[dict]:
    results = []

    # KMeans
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    lab = km.fit_predict(X)
    results.append(score_row(X, lab, y_true, f"{prefix} + KMeans"))

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    lab = agg.fit_predict(X)
    results.append(score_row(X, lab, y_true, f"{prefix} + Agglomerative"))

    # DBSCAN (unsupervised density; may label noise as -1)
    db = DBSCAN(eps=1.6, min_samples=10)  # you can tune eps later
    lab = db.fit_predict(X)
    results.append(score_row(X, lab, y_true, f"{prefix} + DBSCAN"))

    return results

def score_row(X, labels, y_true, name):
    # Some clustering outputs may assign 1 cluster only -> silhouette undefined
    unique = len(set(labels))
    sil = np.nan
    dbi = np.nan
    if unique >= 2 and unique < len(labels):
        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)

    ari = np.nan
    if y_true is not None:
        # ARI can handle -1 labels fine
        ari = adjusted_rand_score(y_true, labels)

    print(f"{name}: clusters={unique} sil={sil} dbi={dbi} ari={ari}")
    return {
        "method": name,
        "n_clusters": unique,
        "silhouette": sil,
        "davies_bouldin": dbi,
        "ARI": ari
    }

def plot_2d(Y: np.ndarray, labels: np.ndarray, title: str, outpath: Path):
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=8, c=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def main():
    if not LATENTS.exists() or not META.exists():
        raise FileNotFoundError("Missing mm VAE latents. Run 09_train_multimodal_vae.py first.")

    Z = np.load(LATENTS)
    meta = pd.read_csv(META)

    # True labels available (genres) -> ARI
    le = LabelEncoder()
    y_true = le.fit_transform(meta["genre"].astype(str))
    k = len(le.classes_)
    print("K:", k, "classes:", list(le.classes_))

    # VAE latent features
    Zs = StandardScaler().fit_transform(Z)
    res_vae = eval_all(Zs, y_true, k, "MM-VAE Latent")

    # Baseline: PCA on mel flattened -> 32 dims
    X_mel, mel_meta = load_flat_mels(MELS_INDEX, fixed_t=FIXED_T)

    # Align baseline to same track_ids as meta (multimodal subset)
    mm_ids = set(meta["track_id"].astype(int).tolist())
    keep = mel_meta["track_id"].astype(int).isin(mm_ids)
    X_mel = X_mel[keep.values]
    mel_meta = mel_meta[keep].reset_index(drop=True)

    X_pca = PCA(n_components=32, random_state=SEED).fit_transform(X_mel)
    X_pca = StandardScaler().fit_transform(X_pca)

    y_true_base = le.transform(mel_meta["genre"].astype(str))
    res_pca = eval_all(X_pca, y_true_base, k, "PCA(32)")

    results = pd.DataFrame(res_vae + res_pca)
    out_csv = OUT_DIR / "metrics_medium.csv"
    results.to_csv(out_csv, index=False)
    print("\n✅ Saved:", out_csv)

    # Visualize MM-VAE KMeans clusters
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(Zs)

    reducer = umap.UMAP(n_components=2, random_state=SEED)
    Y_umap = reducer.fit_transform(Zs)
    plot_2d(Y_umap, km_labels, "UMAP (MM-VAE Latent) + KMeans", OUT_DIR / "umap_mmvae_kmeans.png")

    tsne = TSNE(n_components=2, random_state=SEED, init="pca", learning_rate="auto", perplexity=30)
    Y_tsne = tsne.fit_transform(Zs)
    plot_2d(Y_tsne, km_labels, "t-SNE (MM-VAE Latent) + KMeans", OUT_DIR / "tsne_mmvae_kmeans.png")

    print("✅ Plots saved in:", OUT_DIR)

if __name__ == "__main__":
    main()
