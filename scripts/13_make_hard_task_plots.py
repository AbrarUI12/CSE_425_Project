from __future__ import annotations

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from sklearn.manifold import TSNE
import umap


# -----------------------------
# INPUTS (Hard Task outputs)
# -----------------------------
LATENTS = Path("data/features/mm_cvae_latents.npy")
META = Path("data/features/mm_cvae_latents_meta.csv")

# -----------------------------
# OUTPUTS
# -----------------------------
OUT_DIR = Path("data/results_hard")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLUSTERS = 5


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        total += Counter(y_true[idx]).most_common(1)[0][1]
    return total / len(y_true)


def plot_2d(Y: np.ndarray, labels: np.ndarray, title: str, outpath: Path) -> None:
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], s=10, c=labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_genre_distribution(df: pd.DataFrame, outpath: Path) -> None:
    # stacked bar: cluster -> genre proportions
    pivot = (
        df.groupby(["cluster", "genre"])
        .size()
        .reset_index(name="count")
        .pivot(index="cluster", columns="genre", values="count")
        .fillna(0)
    )
    prop = pivot.div(pivot.sum(axis=1), axis=0)

    ax = prop.plot(kind="bar", stacked=True)
    ax.set_title("Cluster distribution over genres (MM-CVAE)")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    if not LATENTS.exists():
        raise FileNotFoundError(f"Missing: {LATENTS} (run scripts/11_train_mm_cvae.py first)")
    if not META.exists():
        raise FileNotFoundError(f"Missing: {META} (run scripts/11_train_mm_cvae.py first)")

    Z = np.load(LATENTS).astype(np.float32)
    meta = pd.read_csv(META)

    if "genre" not in meta.columns:
        raise ValueError("mm_cvae_latents_meta.csv must contain a 'genre' column")

    # Encode genres to integers (for ARI/NMI/purity)
    genres = meta["genre"].astype("category")
    y_true = genres.cat.codes.to_numpy()

    # Standardize latents (important for clustering + embeddings)
    Zs = StandardScaler().fit_transform(Z)

    # KMeans clusters
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    y_pred = kmeans.fit_predict(Zs)

    # Metrics
    sil = silhouette_score(Zs, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)

    metrics = pd.DataFrame(
        [{
            "method": "MM-CVAE Latent + KMeans",
            "n_clusters": N_CLUSTERS,
            "silhouette": sil,
            "NMI": nmi,
            "ARI": ari,
            "purity": pur,
        }]
    )
    metrics.to_csv(OUT_DIR / "metrics_hard.csv", index=False)
    print("✅ metrics saved:", OUT_DIR / "metrics_hard.csv")
    print(metrics.to_string(index=False))

    # Save a merged df for plotting distributions
    plot_df = meta.copy()
    plot_df["cluster"] = y_pred
    plot_df.to_csv(OUT_DIR / "mm_cvae_clusters.csv", index=False)

    # UMAP embeddings
    reducer = umap.UMAP(n_components=2, random_state=SEED)
    Y_umap = reducer.fit_transform(Zs)

    # t-SNE embeddings
    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        init="pca",
        learning_rate="auto",
        perplexity=30,
    )
    Y_tsne = tsne.fit_transform(Zs)

    # ---- plots colored by KMeans clusters ----
    plot_2d(
        Y_umap, y_pred,
        "UMAP (MM-CVAE Latent) colored by KMeans cluster",
        OUT_DIR / "umap_mmcvae_kmeans.png"
    )
    plot_2d(
        Y_tsne, y_pred,
        "t-SNE (MM-CVAE Latent) colored by KMeans cluster",
        OUT_DIR / "tsne_mmcvae_kmeans.png"
    )

    # ---- plots colored by true genre ----
    plot_2d(
        Y_umap, y_true,
        "UMAP (MM-CVAE Latent) colored by Genre",
        OUT_DIR / "umap_mmcvae_genre.png"
    )
    plot_2d(
        Y_tsne, y_true,
        "t-SNE (MM-CVAE Latent) colored by Genre",
        OUT_DIR / "tsne_mmcvae_genre.png"
    )

    # ---- cluster distribution over genres (stacked bar) ----
    plot_genre_distribution(plot_df, OUT_DIR / "cluster_genre_distribution.png")

    print("\n✅ Hard task plots saved in:", OUT_DIR)
    print(" - umap_mmcvae_kmeans.png")
    print(" - tsne_mmcvae_kmeans.png")
    print(" - umap_mmcvae_genre.png")
    print(" - tsne_mmcvae_genre.png")
    print(" - cluster_genre_distribution.png")


if __name__ == "__main__":
    main()
