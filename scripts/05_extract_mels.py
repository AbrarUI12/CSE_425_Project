from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

MANIFEST = Path("data/fma_manifest_combined_text_only_clean.csv")
OUT_DIR = Path("data/features/mels")
OUT_INDEX = Path("data/features/mels_index.csv")

# Mel settings
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512

CLIP_SECONDS = 30.0
MAX_SAMPLES = int(SR * CLIP_SECONDS)

def pad_or_trim(y: np.ndarray, max_len: int) -> np.ndarray:
    if len(y) > max_len:
        return y[:max_len]
    if len(y) < max_len:
        return np.pad(y, (0, max_len - len(y)))
    return y

def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    failed = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting log-mels"):
        track_id = int(r["track_id"])
        audio_path = Path(r["audio_path"])
        if not audio_path.exists():
            failed += 1
            continue

        out_path = OUT_DIR / f"{track_id}.npy"
        if out_path.exists():
            rows.append({"track_id": track_id, "mel_path": str(out_path), "genre": r["genre"]})
            continue

        try:
            y, sr = librosa.load(str(audio_path), sr=SR, mono=True)
            y = pad_or_trim(y, MAX_SAMPLES)

            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP, power=2.0
            )
            logmel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

            np.save(out_path, logmel)  # shape [128, T]
            rows.append({"track_id": track_id, "mel_path": str(out_path), "genre": r["genre"]})

        except Exception:
            failed += 1

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_INDEX, index=False)

    print("\n✅ Mel extraction done")
    print("Index:", OUT_INDEX)
    print("Rows:", len(out_df))
    print("Failed:", failed)
    if len(out_df):
        x = np.load(out_df.iloc[0]["mel_path"])
        print("Example mel shape:", x.shape)

if __name__ == "__main__":
    main()
