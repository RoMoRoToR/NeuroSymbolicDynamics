# train_transformer.py
# -*- coding: utf-8 -*-
import argparse, os, json, math, random, glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 0) Утилиты
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# -----------------------------
# 1) Датасет со скользящими окнами
# -----------------------------
class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, mean=None, std=None):
        assert series.ndim == 2
        self.seq_len = seq_len
        if mean is None: mean = series.mean(axis=0)
        if std  is None: std  = series.std(axis=0) + 1e-8
        self.mean, self.std = mean, std
        self.series = (series - mean) / std
        X, y = [], []
        for i in range(len(self.series) - seq_len):
            X.append(self.series[i:i+seq_len])
            y.append(self.series[i+seq_len])
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

# -----------------------------
# 2) Загрузчики данных
# -----------------------------
def _read_all_csvs(path_or_dir: str) -> pd.DataFrame:
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"Нет CSV в папке: {path_or_dir}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(path_or_dir)
    return df

def load_lorenz_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    assert num.shape[1] >= 3
    return num.iloc[:, :3].values.astype("float32")

def load_bitcoin_series(path_or_dir: str, resample: str = "1H") -> np.ndarray:
    df = _read_all_csvs(path_or_dir)
    cols_map = {c.lower().strip(): c for c in df.columns}
    ts_col = None
    for cand in ["timestamp","date","time","datetime"]:
        if cand in cols_map:
            ts_col = cols_map[cand]; break
    if ts_col is None:
        raise ValueError("Нет Timestamp/Date/Time/Datetime.")

    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        dt = pd.to_datetime(ts, unit="s", utc=True)
    else:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
    df = df.copy()
    df["__dt__"] = dt
    df = df.dropna(subset=["__dt__"]).sort_values("__dt__").set_index("__dt__")

    def find(opts):
        for cand in opts:
            c = cand.lower()
            if c in cols_map: return cols_map[c]
            for col in df.columns:
                if col.lower().replace(" ", "").replace("(", "").replace(")", "") == c.replace(" ", "").replace("(", "").replace(")", ""):
                    return col
        return None

    c_open  = find(["open"])
    c_high  = find(["high"])
    c_low   = find(["low"])
    c_close = find(["close", "weighted price"])
    c_vol   = find(["volume_btc", "volume (btc)", "volume_(btc)", "volume", "volume currency", "volume_(currency)"])
    if c_close is None:
        raise ValueError("Не найдена колонка Close/Weighted Price.")

    if resample:
        agg = {}
        if c_open:  agg[c_open]  = "first"
        if c_high:  agg[c_high]  = "max"
        if c_low:   agg[c_low]   = "min"
        if c_close: agg[c_close] = "last"
        if c_vol:   agg[c_vol]   = "sum"
        df = df.resample(resample).agg(agg)
    df = df.ffill()

    close = df[c_close].astype("float64")
    high  = df[c_high].astype("float64") if c_high in df else close
    low   = df[c_low].astype("float64") if c_low in df else close
    vol   = df[c_vol].astype("float64") if c_vol in df else pd.Series(0.0, index=df.index)

    r_close = np.log(close).diff().fillna(0.0).values
    hl_range = ((high - low) / (close.shift(1).replace(0, np.nan))).fillna(0.0).values
    v_log = np.log1p(vol).diff().fillna(0.0).values
    feats = np.stack([r_close, hl_range, v_log], axis=1).astype("float32")
    feats = np.nan_to_num(feats, copy=False)
    return feats

def load_series(path: str, data_kind: str, resample: str) -> np.ndarray:
    if data_kind == "lorenz":  return load_lorenz_csv(path)
    if data_kind == "bitcoin": return load_bitcoin_series(path, resample=resample)
    raise ValueError(f"unknown data_kind={data_kind}")

# -----------------------------
# 3) Модель: tiny decoder-only Transformer
# -----------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class CausalSelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        h, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + h)
        h2 = self.ff(x)
        x = self.ln2(x + h2)
        return x

def build_causal_mask(T, device):
    mask = torch.full((T, T), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

class TinyDecoderOnly(nn.Module):
    def __init__(self, input_dim=3, d_model=128, n_head=4, n_layer=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([CausalSelfAttentionBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, input_dim)

    def forward(self, x):                # x: [B,T,D]
        h = self.inp(x)
        h = self.pos(h)
        attn_mask = build_causal_mask(h.size(1), h.device)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        h = self.norm(h)
        last = h[:, -1, :]
        pred = self.head(last)           # [B,D]
        return pred

    @torch.no_grad()
    def token_embeddings(self, x):
        h = self.inp(x)
        h = self.pos(h)
        attn_mask = build_causal_mask(h.size(1), h.device)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        h = self.norm(h)                 # [B,T,d_model]
        return h

# -----------------------------
# 4) Тренировка + state dumps
# -----------------------------
def save_checkpoint(model, opt, epoch, val_loss, args, name=None):
    tag = name if name is not None else f"epoch_{epoch:03d}"
    os.makedirs(args.outdir, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "config": vars(args)
    }, os.path.join(args.outdir, f"{tag}.pt"))
    with open(os.path.join(args.outdir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f, ensure_ascii=False, indent=2)

@torch.no_grad()
def dump_token_states(model, dval, device, epoch, outdir, max_batches=4):
    model.eval()
    loader = DataLoader(dval, batch_size=1, shuffle=False)
    H_seq, Y_true, Y_pred = [], [], []
    cnt = 0
    for xb, yb in loader:
        xb = xb.to(device)
        h = model.token_embeddings(xb)           # [1,T,d_model]
        yhat = model(xb)                         # [1,D]
        H_seq.append(h.squeeze(0).cpu().numpy()) # [T,d_model]
        Y_true.append(yb.squeeze(0).cpu().numpy())
        Y_pred.append(yhat.squeeze(0).cpu().numpy())
        cnt += 1
        if cnt >= max_batches: break
    np.savez_compressed(os.path.join(outdir, f"states_epoch_{epoch:03d}.npz"),
                        H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    raw = load_series(args.data, args.data_kind, args.resample)   # [T,3]
    T = len(raw); split = int(T * args.train_frac)
    train_raw, val_raw = raw[:split], raw[split - args.seq_len:]

    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0)+1e-8
    dtrain = SeqDataset(train_raw, args.seq_len, mean, std)
    dval   = SeqDataset(val_raw,   args.seq_len, mean, std)

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(dval,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TinyDecoderOnly(input_dim=raw.shape[1], d_model=args.d_model, n_head=args.n_head,
                            n_layer=args.n_layer, d_ff=args.d_ff, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    save_checkpoint(model, opt, 0, math.inf, args)
    dump_token_states(model, dval, device, epoch=0, outdir=args.outdir, max_batches=args.state_batches)

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(dtrain)

        model.eval(); vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtotal += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss = vtotal / len(dval)
        print(f"[Transformer] Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if epoch in args.save_epochs or epoch == args.epochs:
            save_checkpoint(model, opt, epoch, val_loss, args)
            dump_token_states(model, dval, device, epoch=epoch, outdir=args.outdir, max_batches=args.state_batches)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, opt, epoch, val_loss, args, name="best")

# -----------------------------
# 5) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Путь к CSV или папке с CSV (Kaggle bitcoin).")
    p.add_argument("--data_kind", type=str, choices=["lorenz","bitcoin"], default="bitcoin")
    p.add_argument("--resample", type=str, default="1H", help="Напр., 15min, 1H, 4H, 1D.")
    p.add_argument("--outdir", type=str, default="runs/bitcoin/Transformer")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_epochs", type=int, nargs="*", default=[1,5,10,25,50,100,150,200])
    p.add_argument("--state_batches", type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
