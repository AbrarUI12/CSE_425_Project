import pandas as pd
from pathlib import Path

df = pd.read_csv("data/fma_manifest_combined_text_only_clean.csv")

print("Total rows:", len(df))
print("Audio exists:", sum(Path(p).exists() for p in df["audio_path"]))
print("Text exists:", sum(Path(p).exists() for p in df["text_path_combined"]))

print("\nGenre distribution:")
print(df["genre"].value_counts())

print("\nText source distribution:")
print(df["text_source_combined"].value_counts())
