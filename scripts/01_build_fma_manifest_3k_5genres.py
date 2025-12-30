from __future__ import annotations

from pathlib import Path
import pandas as pd

META_DIR = Path("data/fma_metadata")
AUDIO_DIR = Path("data/fma_small")

OUT_MANIFEST = Path("data/fma_manifest_3k_5genres_test.csv")

TOTAL_TRACKS = 10
N_GENRES = 5
SEED = 42


def find_file(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return hits[0]


def find_audio_root() -> Path:
    # Usually: data/fma_small/fma_small/000/000002.mp3
    for candidate in [AUDIO_DIR / "fma_small", AUDIO_DIR]:
        if candidate.exists() and list(candidate.rglob("*.mp3")):
            return candidate
    raise FileNotFoundError("Could not locate extracted mp3 files under data/fma_small")


def main():
    tracks_csv = find_file(META_DIR, "tracks.csv")
    genres_csv = find_file(META_DIR, "genres.csv")
    audio_root = find_audio_root()

    print("Using:")
    print(" tracks.csv:", tracks_csv)
    print(" genres.csv:", genres_csv)
    print(" audio_root:", audio_root)

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Filter to FMA-small
    df = tracks[tracks[("set", "subset")] == "small"].copy()

    # Required fields
    df = df[[("track", "title"), ("artist", "name"), ("track", "genre_top")]].copy()
    df.columns = ["title", "artist", "genre_top"]
    df = df.dropna(subset=["title", "artist", "genre_top"])

    # Keep only sensible strings
    df = df[df["title"].apply(lambda x: isinstance(x, str))]
    df = df[df["artist"].apply(lambda x: isinstance(x, str))]
    df["genre_top"] = df["genre_top"].astype(str).str.strip()

    print(f"Eligible tracks (small subset) with title/artist/genre_top: {len(df)}")

    # Pick top 5 genre_top categories
    top_genres = df["genre_top"].value_counts().head(N_GENRES).index.tolist()
    print(f"Top {N_GENRES} genres selected:")
    for g in top_genres:
        print(" ", g)

    df = df[df["genre_top"].isin(top_genres)].copy()

    # Balanced sample: 3000 / 5 = 600 each
    per_genre = TOTAL_TRACKS // N_GENRES
    sampled_parts = []

    for g in top_genres:
        gdf = df[df["genre_top"] == g]
        if len(gdf) < per_genre:
            raise RuntimeError(
                f"Genre '{g}' has only {len(gdf)} tracks, less than required {per_genre}. "
                "Reduce TOTAL_TRACKS or pick different genres."
            )
        sampled_parts.append(gdf.sample(n=per_genre, random_state=SEED))

    sampled = pd.concat(sampled_parts).sample(frac=1.0, random_state=SEED)

    # Build audio_path for each track_id (index is track_id)
    rows = []
    for track_id, r in sampled.iterrows():
        tid = int(track_id)
        tid_str = f"{tid:06d}"
        audio_path = audio_root / tid_str[:3] / f"{tid_str}.mp3"
        if not audio_path.exists():
            continue

        rows.append({
            "track_id": tid,
            "title": r["title"].strip(),
            "artist": r["artist"].strip(),
            "genre": r["genre_top"],
            "audio_path": str(audio_path),
            "lyrics_path": "",
            "lyrics_source": ""
        })

    out_df = pd.DataFrame(rows)

    # Ensure exactly 3000 rows (in case of rare missing mp3 path issues)
    if len(out_df) < TOTAL_TRACKS:
        raise RuntimeError(
            f"Only built {len(out_df)} rows, expected {TOTAL_TRACKS}. "
            "This usually means some audio paths are missing."
        )

    out_df = out_df.head(TOTAL_TRACKS)

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_MANIFEST, index=False)

    print("\n✅ Wrote manifest:", OUT_MANIFEST)
    print("Total tracks:", len(out_df))
    print("Tracks per genre:")
    print(out_df["genre"].value_counts())


if __name__ == "__main__":
    main()
