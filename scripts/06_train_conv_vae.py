from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

MELS_INDEX = Path("data/features/mels_index.csv")
OUT_MODEL = Path("data/models/conv_vae.pt")
OUT_LATENTS = Path("data/features/conv_vae_latents.npy")
OUT_META = Path("data/features/conv_vae_latents_meta.csv")

# Training config
SEED = 42
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3

LATENT_DIM = 32
KL_BETA = 0.001

# Fix mel size to [1, 128, 256] (pad/crop time)
N_MELS = 128
FIXED_T = 256

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def fix_time(m: np.ndarray, T: int) -> np.ndarray:
    # m: [128, time]
    if m.shape[1] > T:
        return m[:, :T]
    if m.shape[1] < T:
        pad = np.zeros((m.shape[0], T - m.shape[1]), dtype=m.dtype)
        return np.concatenate([m, pad], axis=1)
    return m

class MelDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # Estimate mean/std from a subset for normalization
        sample = self.df["mel_path"].head(min(200, len(self.df))).tolist()
        mats = []
        for p in sample:
            m = np.load(p)
            m = fix_time(m, FIXED_T)
            mats.append(m)
        stack = np.stack(mats, axis=0)
        self.mean = float(stack.mean())
        self.std = float(stack.std() + 1e-6)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        m = np.load(row["mel_path"])
        m = fix_time(m, FIXED_T)
        x = (m - self.mean) / self.std
        x = x.astype(np.float32)
        x = torch.from_numpy(x).unsqueeze(0)  # [1, 128, T]
        return x, int(row["track_id"]), str(row["genre"])

class ConvVAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        # Encoder: [B,1,128,256] -> [B,256,16,32]
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),   # -> [32,64,128]
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> [64,32,64]
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # -> [128,16,32]
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),# -> [256,16,32]
            nn.ReLU(),
        )
        self.flat_dim = 256 * 16 * 32
        self.mu = nn.Linear(self.flat_dim, latent_dim)
        self.logvar = nn.Linear(self.flat_dim, latent_dim)

        # Decoder
        self.fc = nn.Linear(latent_dim, self.flat_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # -> [128,32,64]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # -> [64,64,128]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # -> [32,128,256]
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),             # -> [1,128,256]
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        h = h.reshape(x.size(0), -1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparam(mu, logvar)

        d = self.fc(z).view(x.size(0), 256, 16, 32)
        x_hat = self.dec(d)
        return x_hat, mu, logvar

def loss_fn(x, x_hat, mu, logvar):
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + KL_BETA * kl, recon.detach(), kl.detach()

def main():
    set_seed(SEED)

    if not MELS_INDEX.exists():
        raise FileNotFoundError(f"Missing {MELS_INDEX}. Run 05_extract_mels.py first.")

    df = pd.read_csv(MELS_INDEX)
    ds = MelDataset(df)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = ConvVAE(LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for x, _, _ in dl:
            x = x.to(device, non_blocking=True)
            opt.zero_grad()
            x_hat, mu, logvar = model(x)
            loss, recon, kl = loss_fn(x, x_hat, mu, logvar)
            loss.backward()
            opt.step()
            losses.append([float(loss), float(recon), float(kl)])

        avg = np.mean(losses, axis=0)
        print(f"Epoch {epoch:02d}/{EPOCHS} loss={avg[0]:.4f} recon={avg[1]:.4f} kl={avg[2]:.4f}")

    # Extract latents (mu)
    model.eval()
    latents = []
    meta_rows = []
    with torch.no_grad():
        for x, tid, genre in tqdm(DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True),
                                  desc="Extracting latents"):
            x = x.to(device, non_blocking=True)
            _, mu, _ = model(x)
            latents.append(mu.cpu().numpy())
            for t, g in zip(tid.numpy().tolist(), genre):
                meta_rows.append({"track_id": t, "genre": g})

    Z = np.vstack(latents).astype(np.float32)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_LATENTS.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), OUT_MODEL)
    np.save(OUT_LATENTS, Z)
    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)

    print("\n✅ Conv-VAE done")
    print("Latents:", OUT_LATENTS, "shape:", Z.shape)
    print("Meta:", OUT_META)
    print("Model:", OUT_MODEL)

if __name__ == "__main__":
    main()
