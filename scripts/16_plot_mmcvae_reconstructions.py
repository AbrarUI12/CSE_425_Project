from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder


# ---------- FILES (must match your training script) ----------
MELS_INDEX = Path("data/features/mels_index.csv")
LYR_EMB = Path("data/features/lyrics_tfidf_svd_256.npy")
LYR_META = Path("data/features/lyrics_embed_meta.csv")
CKPT = Path("data/models/mm_cvae.pt")  # <- saved by your training script

OUT_DIR = Path("data/results_hard")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "mm_cvae_reconstructions.png"

# ---------- CONFIG ----------
LATENT_DIM = 64
FIXED_T = 256
SEED = 42
N_EXAMPLES = 5   # show 5 (enough for report)
BATCH_SIZE = 64


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fix_time(m: np.ndarray, T: int) -> np.ndarray:
    if m.shape[1] > T:
        return m[:, :T]
    if m.shape[1] < T:
        return np.pad(m, ((0, 0), (0, T - m.shape[1])))
    return m


class DatasetMMCVAE(Dataset):
    """
    EXACTLY mirrors your training dataset:
    - intersects track_id between mels_index.csv and lyrics_embed_meta.csv
    - loads mel .npy from mel_path
    - normalizes using mean/std from first 200 samples
    - loads lyrics embedding (tfidf+svd)
    - label-encodes genre
    """
    def __init__(self, mel_df: pd.DataFrame, lyr_emb: np.ndarray, lyr_meta: pd.DataFrame):
        common = sorted(set(mel_df.track_id) & set(lyr_meta.track_id))
        mel_df = mel_df[mel_df.track_id.isin(common)].sort_values("track_id")
        lyr_meta = lyr_meta[lyr_meta.track_id.isin(common)].sort_values("track_id")

        self.mel_df = mel_df.reset_index(drop=True)
        self.lyr_emb = lyr_emb
        self.tid2idx = {tid: i for i, tid in enumerate(lyr_meta.track_id)}

        self.le = LabelEncoder()
        self.genres = self.le.fit_transform(mel_df.genre)

        sample = [fix_time(np.load(p), FIXED_T) for p in mel_df.mel_path[:200]]
        stack = np.stack(sample)
        self.mean, self.std = stack.mean(), stack.std() + 1e-6

    def __len__(self):
        return len(self.mel_df)

    def __getitem__(self, idx):
        r = self.mel_df.iloc[idx]
        tid = int(r.track_id)

        mel = fix_time(np.load(r.mel_path), FIXED_T)
        mel = (mel - self.mean) / self.std
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)  # [1, 128, 256]

        txt = torch.tensor(self.lyr_emb[self.tid2idx[tid]], dtype=torch.float32)  # [256]
        genre = torch.tensor(self.genres[idx], dtype=torch.long)

        return mel, txt, genre, tid


class MMCVAE(nn.Module):
    """
    Same architecture as your training script, but with one important fix:
    - use a fixed num_classes (n_genres) for one-hot, not y.max()+1
      (this makes inference always stable)
    """
    def __init__(self, n_genres: int, latent_dim: int):
        super().__init__()
        self.n_genres = n_genres

        self.enc_audio = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU()
        )
        self.flat = 256 * 16 * 32  # must match your training

        self.enc_text = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU()
        )

        self.mu = nn.Linear(self.flat + 128 + n_genres, latent_dim)
        self.logvar = nn.Linear(self.flat + 128 + n_genres, latent_dim)

        self.fc = nn.Linear(latent_dim + n_genres, self.flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1)
        )

    def forward(self, x_a, x_t, y):
        y_oh = F.one_hot(y, num_classes=self.n_genres).float()

        ha = self.enc_audio(x_a).view(x_a.size(0), -1)
        ht = self.enc_text(x_t)
        h = torch.cat([ha, ht, y_oh], 1)

        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        zc = torch.cat([z, y_oh], 1)
        d = self.fc(zc).view(x_a.size(0), 256, 16, 32)
        return self.dec(d), mu, logvar


@torch.no_grad()
def make_reconstruction_figure(model: nn.Module, ds: DatasetMMCVAE, device: torch.device):
    # choose examples: try to cover multiple genres
    df = ds.mel_df.copy()
    df["genre_id"] = ds.genres
    df["genre_name"] = ds.le.inverse_transform(ds.genres)

    picks = []
    for gname in sorted(df["genre_name"].unique()):
        row = df[df["genre_name"] == gname].sample(1, random_state=SEED).index[0]
        picks.append(int(row))
        if len(picks) >= N_EXAMPLES:
            break
    while len(picks) < N_EXAMPLES:
        picks.append(int(df.sample(1, random_state=SEED + len(picks)).index[0]))

    originals = []
    recons = []
    titles = []

    for idx in picks:
        xa, xt, y, tid = ds[idx]
        xa = xa.unsqueeze(0).to(device)   # [1,1,128,256]
        xt = xt.unsqueeze(0).to(device)   # [1,256]
        y = y.unsqueeze(0).to(device)     # [1]

        xr, _, _ = model(xa, xt, y)

        # back to numpy
        x0 = xa[0, 0].detach().cpu().numpy()   # normalized mel
        r0 = xr[0, 0].detach().cpu().numpy()

        # denormalize to look more like real spectrogram scale
        x0 = x0 * ds.std + ds.mean
        r0 = r0 * ds.std + ds.mean

        originals.append(x0)
        recons.append(r0)
        titles.append(f"{ds.le.inverse_transform([int(y.item())])[0]}\nID:{int(tid)}")

    # plot grid: 2 rows (orig / recon), N columns
    n = len(originals)
    plt.figure(figsize=(3.2 * n, 6.0))

    for i in range(n):
        ax1 = plt.subplot(2, n, i + 1)
        ax1.imshow(originals[i], origin="lower", aspect="auto")
        ax1.set_title("Original\n" + titles[i], fontsize=10)
        ax1.set_xticks([]); ax1.set_yticks([])

        ax2 = plt.subplot(2, n, n + i + 1)
        ax2.imshow(recons[i], origin="lower", aspect="auto")
        ax2.set_title("Reconstruction", fontsize=10)
        ax2.set_xticks([]); ax2.set_yticks([])

    plt.suptitle("MM-CVAE Reconstruction Examples (Hard Task)", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print(f"✅ Saved reconstruction figure: {OUT_FIG}")


def main():
    set_seed(SEED)

    if not CKPT.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {CKPT}\n"
            "Run your MM-CVAE training script first. It should save data/models/mm_cvae.pt"
        )

    mel_df = pd.read_csv(MELS_INDEX)
    lyr_emb = np.load(LYR_EMB)
    lyr_meta = pd.read_csv(LYR_META)

    ds = DatasetMMCVAE(mel_df, lyr_emb, lyr_meta)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MMCVAE(n_genres=len(ds.le.classes_), latent_dim=LATENT_DIM).to(device)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    print(f"Model loaded from {CKPT} on {device}. Building reconstructions...")
    make_reconstruction_figure(model, ds, device)


if __name__ == "__main__":
    main()
