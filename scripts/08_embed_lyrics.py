from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

MANIFEST = Path("data/fma_manifest_combined_text_only_clean.csv")  # 2863 rows, has text_path_combined
OUT_EMB = Path("data/features/lyrics_tfidf_svd_256.npy")
OUT_META = Path("data/features/lyrics_embed_meta.csv")

SVD_DIM = 256
MAX_FEATURES = 50000
MIN_DF = 2
MAX_DF = 0.9
SEED = 42

def read_text(p: str) -> str:
    try:
        return Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    if "text_path_combined" not in df.columns:
        raise ValueError("Manifest must contain text_path_combined")

    texts = []
    keep_rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Reading lyrics text"):
        p = str(r["text_path_combined"])
        if p and Path(p).exists():
            texts.append(read_text(p))
            keep_rows.append({"track_id": int(r["track_id"]), "genre": str(r["genre"])})

    meta = pd.DataFrame(keep_rows)
    print("Texts loaded:", len(texts))

    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words="english"
    )
    X = vec.fit_transform(texts)  # sparse

    svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
    E = svd.fit_transform(X).astype(np.float32)

    # Standardize embeddings (helps fusion models)
    E = StandardScaler().fit_transform(E).astype(np.float32)

    OUT_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMB, E)
    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT_META, index=False)

    print("\n✅ Lyrics embeddings done")
    print("Embeddings:", OUT_EMB, "shape:", E.shape)
    print("Meta:", OUT_META)

if __name__ == "__main__":
    main()
