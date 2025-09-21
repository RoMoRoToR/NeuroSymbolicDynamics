# train_transformer.py
# -*- coding: utf-8 -*-
import argparse, os, json, math, random
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ------------- utils/dataset -------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_air_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    cand = [c for c in df.columns if str(c).strip().lower() in
            ["passengers", "#passengers", "value"]]
    if cand:
        s = pd.to_numeric(df[cand[0]], errors="coerce")
    else:
        num = df.select_dtypes(include=[np.number])
        assert num.shape[1] >= 1, "Нужна числовая колонка."
        s = pd.to_numeric(num.iloc[:, 0], errors="coerce")
    s = s.dropna().astype("float32").to_numpy().reshape(-1,1)
    return s

class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, mean=None, std=None):
        assert series.ndim == 2
        self.seq_len = seq_len
        if mean is None: mean = series.mean(axis=0)
        if std  is None: std  = series.std(axis=0) + 1e-8
        self.mean, self.std = mean, std
        x = (series - mean) / std
        X, y = [], []
        for i in range(len(x) - seq_len):
            X.append(x[i:i+seq_len]); y.append(x[i+seq_len])
        self.X = np.array(X, np.float32); self.y = np.array(y, np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

# ------------- model ---------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,D]
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

def build_causal_mask(T, device):
    m = torch.full((T, T), float("-inf"), device=device)
    return torch.triu(m, diagonal=1)

class Block(nn.Module):
    def __init__(self, d_model=64, n_head=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x, mask):
        h, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + h)
        h = self.ff(x)
        x = self.ln2(x + h)
        return x

class TinyDecoderOnly(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_head=4, n_layer=3, d_ff=128, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim)
    def forward(self, x):                     # [B,T,1]
        h = self.inp(x); h = self.pos(h)
        mask = build_causal_mask(h.size(1), h.device)
        for blk in self.blocks: h = blk(h, mask)
        h = self.norm(h)
        last = h[:, -1, :]
        return self.head(last)                # [B,1]
    @torch.no_grad()
    def token_embeddings(self, x):
        h = self.inp(x); h = self.pos(h)
        mask = build_causal_mask(h.size(1), h.device)
        for blk in self.blocks: h = blk(h, mask)
        return self.norm(h)                   # [B,T,d_model]

# ------------- training ------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    series = load_air_csv(args.data)
    T = len(series); split = int(T * args.train_frac)
    train_raw, val_raw = series[:split], series[split - args.seq_len:]

    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0) + 1e-8
    dtrain = SeqDataset(train_raw, args.seq_len, mean, std)
    dval   = SeqDataset(val_raw,   args.seq_len, mean, std)

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(dval,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    os.makedirs(args.outdir, exist_ok=True)

    model = TinyDecoderOnly(d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer,
                            d_ff=args.d_ff, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    save_ckpt(model, opt, 0, math.inf, args)
    dump_token_states(model, dval, device, epoch=0, outdir=args.outdir, max_batches=args.state_batches)

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); total += loss.item() * xb.size(0)
        train_loss = total / len(dtrain)

        model.eval(); vtot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtot += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss = vtot / len(dval)
        print(f"[Transformer] epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if epoch in args.save_epochs or epoch == args.epochs:
            save_ckpt(model, opt, epoch, val_loss, args)
            dump_token_states(model, dval, device, epoch=epoch, outdir=args.outdir, max_batches=args.state_batches)
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(model, opt, epoch, val_loss, args, name="best")

def save_ckpt(model, opt, epoch, val_loss, args, name=None):
    tag = name if name else f"epoch_{epoch:03d}"
    torch.save({
        "epoch": epoch, "val_loss": float(val_loss),
        "model_state": model.state_dict(), "opt_state": opt.state_dict(),
        "config": vars(args)
    }, os.path.join(args.outdir, f"{tag}.pt"))
    with open(os.path.join(args.outdir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f, ensure_ascii=False, indent=2)

@torch.no_grad()
def dump_token_states(model, dval, device, epoch, outdir, max_batches=8):
    model.eval()
    loader = DataLoader(dval, batch_size=1, shuffle=False)
    H_seq, Y_true, Y_pred = [], [], []
    for i, (xb, yb) in enumerate(loader):
        xb = xb.to(device)
        h = model.token_embeddings(xb)           # [1,T,d_model]
        yhat = model(xb)                         # [1,1]
        H_seq.append(h.squeeze(0).cpu().numpy())
        Y_true.append(yb.squeeze(0).cpu().numpy())
        Y_pred.append(yhat.squeeze(0).cpu().numpy())
        if i + 1 >= max_batches: break
    np.savez_compressed(os.path.join(outdir, f"states_epoch_{epoch:03d}.npz"),
                        H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

# ------------- CLI -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="AirPassengers.csv")
    p.add_argument("--outdir", type=str, default="runs/air/Transformer")
    p.add_argument("--seq_len", type=int, default=12)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=3)
    p.add_argument("--d_ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_epochs", type=int, nargs="*", default=[1,5,10,25,50,100,150,200])
    p.add_argument("--state_batches", type=int, default=8)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
