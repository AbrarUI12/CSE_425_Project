from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

# ---------- CONFIG ----------
# Choose your "master" manifest to start from:
# - If you are working with 3k/5genres, use this:
MASTER_MANIFEST = Path("data/fma_manifest_3k_5genres_lyrics.csv")
# - If you want to use your 5k whisper manifest instead, comment above and uncomment:
# MASTER_MANIFEST = Path("data/fma_manifest_5k_5genres_lyrics_whisper.csv")

WHISPER_DIR = Path(r"data/whisper_transcriptions")
OUT_DIR = Path("data/lyrics_combined")

OUT_MANIFEST_ALL = Path("data/fma_manifest_combined.csv")
OUT_MANIFEST_TEXT_ONLY = Path("data/fma_manifest_combined_text_only.csv")

# Behavior:
# "prefer_whisper" -> use whisper text if exists, else api text
# "concat_both"    -> if both exist, concatenate whisper + api
COMBINE_MODE = "concat_both"  # or "prefer_whisper"
# ----------------------------


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\. ]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160] if s else "unknown"


def build_whisper_map(folder: Path) -> dict[int, Path]:
    """
    Map track_id -> whisper transcript file path.
    Assumes the filename contains the numeric track_id somewhere.
    Examples:
      123456.txt
      whisper_123456.txt
      Artist - Title (123456).txt
    """
    mapping: dict[int, Path] = {}
    for p in folder.rglob("*.txt"):
        m = re.search(r"\b(\d{3,7})\b", p.stem)
        if not m:
            continue
        tid = int(m.group(1))
        mapping.setdefault(tid, p)
    return mapping


def main():
    if not MASTER_MANIFEST.exists():
        raise FileNotFoundError(f"Missing master manifest: {MASTER_MANIFEST}")
    if not WHISPER_DIR.exists():
        raise FileNotFoundError(f"Missing whisper folder: {WHISPER_DIR}")

    df = pd.read_csv(MASTER_MANIFEST)

    if "track_id" not in df.columns:
        raise ValueError("Manifest must contain a 'track_id' column.")

    # Normalize columns
    if "lyrics_path" not in df.columns:
        df["lyrics_path"] = ""
    if "lyrics_source" not in df.columns:
        df["lyrics_source"] = ""

    df["lyrics_path"] = df["lyrics_path"].astype(str).fillna("")
    df["lyrics_source"] = df["lyrics_source"].astype(str).fillna("")

    whisper_map = build_whisper_map(WHISPER_DIR)
    print(f"Found whisper files mapped to track_id: {len(whisper_map)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create new columns
    df["lyrics_path_api"] = df["lyrics_path"]
    df["lyrics_source_api"] = df["lyrics_source"]
    df["lyrics_path_whisper"] = ""
    df["text_path_combined"] = ""
    df["text_source_combined"] = ""

    have_any_text = 0
    have_both = 0
    have_whisper = 0
    have_api = 0

    for i, row in df.iterrows():
        tid = int(row["track_id"])

        api_path = Path(row["lyrics_path_api"]) if row["lyrics_path_api"] else None
        api_ok = bool(api_path) and api_path.exists()

        w_path = whisper_map.get(tid)
        w_ok = bool(w_path) and w_path.exists()

        if w_ok:
            df.at[i, "lyrics_path_whisper"] = str(w_path)

        # Read texts
        api_text = read_text(api_path) if api_ok else ""
        whisper_text = read_text(w_path) if w_ok else ""

        if w_ok and api_ok:
            have_both += 1
        if w_ok:
            have_whisper += 1
        if api_ok:
            have_api += 1

        if not (w_ok or api_ok):
            continue

        have_any_text += 1

        # Combine
        if COMBINE_MODE == "prefer_whisper":
            combined = whisper_text if whisper_text else api_text
            source = "whisper" if whisper_text else "api"
        else:  # concat_both
            if whisper_text and api_text:
                combined = whisper_text + "\n\n---\n\n" + api_text
                source = "both"
            elif whisper_text:
                combined = whisper_text
                source = "whisper"
            else:
                combined = api_text
                source = "api"

        # Save combined text to file
        artist = str(row.get("artist", "")).strip()
        title = str(row.get("title", "")).strip()

        fname = safe_filename(f"{artist} - {title} ({tid}).txt")
        out_path = OUT_DIR / fname
        out_path.write_text(combined, encoding="utf-8")

        df.at[i, "text_path_combined"] = str(out_path)
        df.at[i, "text_source_combined"] = source

    # Save manifests
    OUT_MANIFEST_ALL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MANIFEST_ALL, index=False)

    df_text = df[df["text_path_combined"].astype(str).str.len() > 0].copy()
    df_text.to_csv(OUT_MANIFEST_TEXT_ONLY, index=False)

    print("\n✅ Combined manifest created")
    print(f"Master rows:                  {len(df)}")
    print(f"Tracks with ANY text:         {have_any_text}/{len(df)}")
    print(f"Tracks with Whisper text:     {have_whisper}/{len(df)}")
    print(f"Tracks with API lyrics:       {have_api}/{len(df)}")
    print(f"Tracks with BOTH:             {have_both}/{len(df)}")
    print(f"Combined text files folder:   {OUT_DIR}")
    print(f"Wrote manifest (all):         {OUT_MANIFEST_ALL}")
    print(f"Wrote manifest (text-only):   {OUT_MANIFEST_TEXT_ONLY}")


if __name__ == "__main__":
    main()
