from __future__ import annotations

from pathlib import Path
import pandas as pd

IN_MANIFEST = Path("data/fma_manifest_3k_5genres_lyrics.csv")
OUT_LYRICS_ONLY = Path("data/fma_manifest_lyrics_only.csv")
OUT_AUDIO_ONLY = Path("data/fma_manifest_audio_only.csv")  # optional


def main() -> None:
    if not IN_MANIFEST.exists():
        raise FileNotFoundError(f"Missing {IN_MANIFEST}")

    df = pd.read_csv(IN_MANIFEST)

    # Normalize to strings
    df["lyrics_path"] = df.get("lyrics_path", "").astype(str)
    df["audio_path"] = df.get("audio_path", "").astype(str)

    # Check file existence
    df["lyrics_exists"] = df["lyrics_path"].apply(lambda p: bool(p) and Path(p).exists())
    df["audio_exists"] = df["audio_path"].apply(lambda p: bool(p) and Path(p).exists())

    # Keep only rows with audio that exists (should be all)
    df_audio_ok = df[df["audio_exists"]].copy()

    # Lyrics-only subset: must have lyrics file existing
    df_lyrics = df_audio_ok[df_audio_ok["lyrics_exists"]].copy()

    # Write outputs
    OUT_AUDIO_ONLY.parent.mkdir(parents=True, exist_ok=True)
    df_audio_ok.drop(columns=["lyrics_exists", "audio_exists"]).to_csv(OUT_AUDIO_ONLY, index=False)
    df_lyrics.drop(columns=["lyrics_exists", "audio_exists"]).to_csv(OUT_LYRICS_ONLY, index=False)

    # Print summary
    total = len(df)
    audio_ok = len(df_audio_ok)
    lyrics_ok = len(df_lyrics)

    print("\n✅ Manifest check complete")
    print(f"Input rows:                 {total}")
    print(f"Audio file exists:          {audio_ok}/{total}")
    print(f"Lyrics file exists:         {lyrics_ok}/{total}")
    print(f"Wrote lyrics-only manifest:  {OUT_LYRICS_ONLY}")
    print(f"Wrote audio-only manifest:   {OUT_AUDIO_ONLY}")


if __name__ == "__main__":
    main()
