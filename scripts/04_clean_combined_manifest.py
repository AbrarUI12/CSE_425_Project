from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

INP = Path("data/fma_manifest_combined.csv")
OUT = Path("data/fma_manifest_combined_clean.csv")
OUT_TEXT_ONLY = Path("data/fma_manifest_combined_text_only_clean.csv")

def to_empty_if_nan(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

def main():
    df = pd.read_csv(INP)

    # Clean these columns if they exist
    for col in ["lyrics_path", "lyrics_source", "lyrics_path_api", "lyrics_source_api",
                "lyrics_path_whisper", "text_path_combined", "text_source_combined"]:
        if col in df.columns:
            df[col] = df[col].apply(to_empty_if_nan)

    # Verify file existence for combined text
    df["text_exists"] = df["text_path_combined"].apply(lambda p: bool(p) and Path(p).exists())

    df.to_csv(OUT, index=False)
    df_text = df[df["text_exists"]].copy()
    df_text.to_csv(OUT_TEXT_ONLY, index=False)

    print("✅ Cleaned manifest written:", OUT)
    print("✅ Cleaned text-only written:", OUT_TEXT_ONLY)
    print("Text exists:", df_text["text_exists"].sum(), "/", len(df))

if __name__ == "__main__":
    main()
