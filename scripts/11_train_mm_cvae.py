from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

# ---------- FILES ----------
MELS_INDEX = Path("data/features/mels_index.csv")
LYR_EMB = Path("data/features/lyrics_tfidf_svd_256.npy")
LYR_META = Path("data/features/lyrics_embed_meta.csv")

OUT_MODEL = Path("data/models/mm_cvae.pt")
OUT_LATENTS = Path("data/features/mm_cvae_latents.npy")
OUT_META = Path("data/features/mm_cvae_latents_meta.csv")

# ---------- CONFIG ----------
LATENT_DIM = 64
KL_BETA = 0.002          # stronger beta for disentanglement
EPOCHS = 50              # more training
BATCH_SIZE = 64
LR = 1e-3
FIXED_T = 256
SEED = 42
# ---------------------------

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def fix_time(m, T):
    if m.shape[1] > T:
        return m[:, :T]
    if m.shape[1] < T:
        return np.pad(m, ((0,0),(0,T-m.shape[1])))
    return m

class DatasetMMCVAE(Dataset):
    def __init__(self, mel_df, lyr_emb, lyr_meta):
        common = sorted(set(mel_df.track_id) & set(lyr_meta.track_id))
        mel_df = mel_df[mel_df.track_id.isin(common)].sort_values("track_id")
        lyr_meta = lyr_meta[lyr_meta.track_id.isin(common)].sort_values("track_id")

        self.mel_df = mel_df.reset_index(drop=True)
        self.lyr_emb = lyr_emb
        self.tid2idx = {tid:i for i,tid in enumerate(lyr_meta.track_id)}

        self.le = LabelEncoder()
        self.genres = self.le.fit_transform(mel_df.genre)

        sample = [fix_time(np.load(p), FIXED_T) for p in mel_df.mel_path[:200]]
        stack = np.stack(sample)
        self.mean, self.std = stack.mean(), stack.std() + 1e-6

    def __len__(self): return len(self.mel_df)

    def __getitem__(self, idx):
        r = self.mel_df.iloc[idx]
        tid = int(r.track_id)

        mel = fix_time(np.load(r.mel_path), FIXED_T)
        mel = (mel - self.mean) / self.std
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        txt = torch.tensor(self.lyr_emb[self.tid2idx[tid]], dtype=torch.float32)
        genre = torch.tensor(self.genres[idx], dtype=torch.long)

        return mel, txt, genre, tid

class MMCVAE(nn.Module):
    def __init__(self, n_genres, latent_dim):
        super().__init__()

        self.enc_audio = nn.Sequential(
            nn.Conv2d(1,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,64,4,2,1), nn.ReLU(),
            nn.Conv2d(64,128,4,2,1), nn.ReLU(),
            nn.Conv2d(128,256,3,1,1), nn.ReLU()
        )
        self.flat = 256*16*32

        self.enc_text = nn.Sequential(
            nn.Linear(256,128), nn.ReLU()
        )

        self.mu = nn.Linear(self.flat+128+n_genres, latent_dim)
        self.logvar = nn.Linear(self.flat+128+n_genres, latent_dim)

        self.fc = nn.Linear(latent_dim+n_genres, self.flat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.Conv2d(32,1,3,1,1)
        )

    def forward(self, x_a, x_t, y):
        y_oh = F.one_hot(y, num_classes=y.max()+1).float()

        ha = self.enc_audio(x_a).view(x_a.size(0), -1)
        ht = self.enc_text(x_t)
        h = torch.cat([ha, ht, y_oh], 1)

        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5*logvar)
        z = mu + torch.randn_like(std)*std

        zc = torch.cat([z, y_oh], 1)
        d = self.fc(zc).view(x_a.size(0),256,16,32)
        return self.dec(d), mu, logvar

def loss_fn(x, xr, mu, logvar):
    recon = F.mse_loss(xr, x)
    kl = -0.5*torch.mean(1+logvar-mu.pow(2)-logvar.exp())
    return recon + KL_BETA*kl

def main():
    set_seed(SEED)

    mel_df = pd.read_csv(MELS_INDEX)
    lyr_emb = np.load(LYR_EMB)
    lyr_meta = pd.read_csv(LYR_META)

    ds = DatasetMMCVAE(mel_df, lyr_emb, lyr_meta)
    dl = DataLoader(ds, BATCH_SIZE, shuffle=True)

    device = torch.device("cuda")
    model = MMCVAE(len(set(ds.genres)), LATENT_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for e in range(EPOCHS):
        losses=[]
        for xa, xt, y, _ in dl:
            xa, xt, y = xa.to(device), xt.to(device), y.to(device)
            xr, mu, lv = model(xa, xt, y)
            loss = loss_fn(xa, xr, mu, lv)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"Epoch {e+1}/{EPOCHS} loss={np.mean(losses):.4f}")

    model.eval()
    Z, meta = [], []
    with torch.no_grad():
        for xa, xt, y, tid in DataLoader(ds,128):
            xa, xt, y = xa.to(device), xt.to(device), y.to(device)
            _, mu, _ = model(xa, xt, y)
            Z.append(mu.cpu().numpy())
            for t,g in zip(tid.numpy(), y.cpu().numpy()):
                meta.append({"track_id":t,"genre":ds.le.inverse_transform([g])[0]})

    np.save(OUT_LATENTS, np.vstack(Z))
    pd.DataFrame(meta).to_csv(OUT_META, index=False)
    torch.save(model.state_dict(), OUT_MODEL)

if __name__=="__main__":
    main()
