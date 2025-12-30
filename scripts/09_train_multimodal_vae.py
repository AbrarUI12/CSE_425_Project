from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Inputs
MELS_INDEX = Path("data/features/mels_index.csv")  # from 05_extract_mels.py
LYR_EMB = Path("data/features/lyrics_tfidf_svd_256.npy")
LYR_META = Path("data/features/lyrics_embed_meta.csv")

# Outputs
OUT_MODEL = Path("data/models/mm_vae.pt")
OUT_LATENTS = Path("data/features/mm_vae_latents.npy")
OUT_META = Path("data/features/mm_vae_latents_meta.csv")

# Training config (your requested values)
LATENT_DIM = 64
KL_BETA = 0.0001
EPOCHS = 40
BATCH_SIZE = 64
LR = 1e-3
SEED = 42

# Mel fixed size
FIXED_T = 256

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fix_time(m: np.ndarray, T: int) -> np.ndarray:
    if m.shape[1] > T:
        return m[:, :T]
    if m.shape[1] < T:
        pad = np.zeros((m.shape[0], T - m.shape[1]), dtype=m.dtype)
        return np.concatenate([m, pad], axis=1)
    return m

class MMDataset(Dataset):
    def __init__(self, mel_df: pd.DataFrame, lyr_emb: np.ndarray, lyr_meta: pd.DataFrame):
        # align by track_id intersection
        lyr_meta = lyr_meta.copy()
        lyr_meta["track_id"] = lyr_meta["track_id"].astype(int)
        mel_df = mel_df.copy()
        mel_df["track_id"] = mel_df["track_id"].astype(int)

        common = sorted(set(mel_df["track_id"]).intersection(set(lyr_meta["track_id"])))
        mel_df = mel_df[mel_df["track_id"].isin(common)].sort_values("track_id")
        lyr_meta = lyr_meta[lyr_meta["track_id"].isin(common)].sort_values("track_id")

        # build map for embeddings
        id_to_idx = {tid: i for i, tid in enumerate(lyr_meta["track_id"].tolist())}

        self.mel_df = mel_df.reset_index(drop=True)
        self.lyr_emb = lyr_emb
        self.id_to_idx = id_to_idx

        # normalization for mel
        sample_paths = self.mel_df["mel_path"].head(min(200, len(self.mel_df))).tolist()
        mats = []
        for p in sample_paths:
            m = np.load(p)
            m = fix_time(m, FIXED_T)
            mats.append(m)
        stack = np.stack(mats, axis=0)
        self.mel_mean = float(stack.mean())
        self.mel_std = float(stack.std() + 1e-6)

    def __len__(self):
        return len(self.mel_df)

    def __getitem__(self, idx):
        r = self.mel_df.iloc[idx]
        tid = int(r["track_id"])

        m = np.load(r["mel_path"])
        m = fix_time(m, FIXED_T)
        m = ((m - self.mel_mean) / self.mel_std).astype(np.float32)
        x_audio = torch.from_numpy(m).unsqueeze(0)  # [1,128,256]

        e_idx = self.id_to_idx[tid]
        x_text = torch.from_numpy(self.lyr_emb[e_idx].astype(np.float32))  # [256]

        genre = str(r["genre"])
        return x_audio, x_text, tid, genre

class MMVAE(nn.Module):
    def __init__(self, latent_dim: int, text_dim: int = 256):
        super().__init__()
        # audio encoder
        self.enc_a = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # [32,64,128]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # [64,32,64]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # [128,16,32]
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),# [256,16,32]
            nn.ReLU(),
        )
        self.flat_a = 256 * 16 * 32

        # text encoder
        self.enc_t = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.mu = nn.Linear(self.flat_a + 128, latent_dim)
        self.logvar = nn.Linear(self.flat_a + 128, latent_dim)

        # decoder (reconstruct audio)
        self.fc = nn.Linear(latent_dim, self.flat_a)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_audio, x_text):
        ha = self.enc_a(x_audio).reshape(x_audio.size(0), -1)
        ht = self.enc_t(x_text)
        h = torch.cat([ha, ht], dim=1)

        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparam(mu, logvar)

        d = self.fc(z).view(x_audio.size(0), 256, 16, 32)
        x_hat = self.dec(d)
        return x_hat, mu, logvar

def loss_fn(x, x_hat, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + KL_BETA * kl, recon.detach(), kl.detach()

def main():
    set_seed(SEED)

    for p in [MELS_INDEX, LYR_EMB, LYR_META]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    mel_df = pd.read_csv(MELS_INDEX)
    lyr_emb = np.load(LYR_EMB)
    lyr_meta = pd.read_csv(LYR_META)

    ds = MMDataset(mel_df, lyr_emb, lyr_meta)
    print("Multimodal pairs:", len(ds))

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MMVAE(latent_dim=LATENT_DIM, text_dim=lyr_emb.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for x_a, x_t, _, _ in dl:
            x_a = x_a.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)

            opt.zero_grad()
            x_hat, mu, logvar = model(x_a, x_t)
            loss, recon, kl = loss_fn(x_a, x_hat, mu, logvar)
            loss.backward()
            opt.step()

            losses.append([float(loss), float(recon), float(kl)])

        avg = np.mean(losses, axis=0)
        print(f"Epoch {epoch:02d}/{EPOCHS} loss={avg[0]:.4f} recon={avg[1]:.4f} kl={avg[2]:.4f}")

    # Extract latents
    model.eval()
    latents = []
    meta_rows = []
    with torch.no_grad():
        for x_a, x_t, tid, genre in tqdm(
            DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True),
            desc="Extracting latents"
        ):
            x_a = x_a.to(device, non_blocking=True)
            x_t = x_t.to(device, non_blocking=True)
            _, mu, _ = model(x_a, x_t)
            latents.append(mu.cpu().numpy())
            for t, g in zip(tid.numpy().tolist(), genre):
                meta_rows.append({"track_id": t, "genre": g})

    Z = np.vstack(latents).astype(np.float32)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LATENTS.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), OUT_MODEL)
    np.save(OUT_LATENTS, Z)
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)

    print("\n✅ Multimodal VAE done")
    print("Latents:", OUT_LATENTS, "shape:", Z.shape)
    print("Meta:", OUT_META)
    print("Model:", OUT_MODEL)

if __name__ == "__main__":
    main()
