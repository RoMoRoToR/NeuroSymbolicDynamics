# train_rnn_bi.py
# -*- coding: utf-8 -*-
import os, math, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- utils ----------
def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_series_csv(path: str) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, autostrip=True)
    if arr.ndim == 1: arr = arr[:, None]
    arr = arr[~np.any(~np.isfinite(arr), axis=1)]
    if arr.size == 0: raise RuntimeError("empty/non-numeric CSV")
    return arr.astype("float32")

# ---------- dataset ----------
class SeqDataset(Dataset):
    def __init__(self, raw: np.ndarray, L: int, mean: np.ndarray, std: np.ndarray):
        self.L = int(L)
        self.mean = mean.astype("float32")
        self.std  = (std + 1e-8).astype("float32")
        X = ((raw - self.mean) / self.std).astype("float32")
        T = len(X)
        N = max(0, T - self.L)
        self.X = np.lib.stride_tricks.sliding_window_view(X, (self.L, X.shape[1])) \
                   .reshape(T - self.L + 1, self.L, X.shape[1])[:N]
        self.Y = X[self.L:self.L + N]
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# ---------- model ----------
class BiLSTMNet(nn.Module):
    def __init__(self, in_dim=1, hid=96, n_layer=2, dropout=0.1):
        super().__init__()
        self.in_dim, self.hid = in_dim, hid
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hid, num_layers=n_layer,
            batch_first=True, dropout=dropout if n_layer > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2*hid, in_dim)  # bi: concat(fwd,bwd)

    @torch.no_grad()
    def lstm_outputs(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)                   # [B,T,2H]
        return self.dropout(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.head(out[:, -1, :])         # [B,D]

# ---------- io helpers ----------
def save_checkpoint(model, opt, epoch, val_loss, args, name=None):
    tag = name if name is not None else f"epoch_{epoch:03d}"
    ensure_dir(args.outdir)
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
def dump_hidden_states(model, dval: Dataset, device, epoch: int, outdir: str, max_batches: int = 4):
    model.eval()
    loader = DataLoader(dval, batch_size=1, shuffle=False)
    H_seq, Y_true, Y_pred = [], [], []
    cnt = 0
    for xb, yb in loader:
        xb = xb.to(device)
        h = model.lstm_outputs(xb)                    # [1,T,2H]
        yhat = model(xb)
        H_seq.append(h.squeeze(0).cpu().numpy())      # [T,2H]
        Y_true.append(yb.squeeze(0).cpu().numpy())
        Y_pred.append(yhat.squeeze(0).cpu().numpy())
        cnt += 1
        if (max_batches is not None) and (max_batches > 0) and (cnt >= max_batches):
            break
    np.savez_compressed(os.path.join(outdir, f"states_epoch_{epoch:03d}.npz"),
                        H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

@torch.no_grad()
def dump_hidden_states_dense(model, val_raw: np.ndarray, mean: np.ndarray, std: np.ndarray,
                             device, epoch: int, outdir: str, dense_stride: int = 1):
    """
    Непрерывный прогон BiLSTM по всей валидации.
    (формально bi некаузален; для съёма аттрактора это ок — получаем 2H на шаг)
    """
    model.eval()
    X = ((val_raw - mean) / (std + 1e-8)).astype("float32")
    x = torch.from_numpy(X)[None, ...].to(device)     # [1,T,D]
    out = model.lstm_outputs(x).squeeze(0).cpu().numpy()  # [T,2H]
    step = max(1, int(dense_stride))
    H_long = out[::step]

    path = os.path.join(outdir, f"states_epoch_{epoch:03d}.npz")
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        H_seq = list(old["H_seq"]); Y_true = list(old["Y_true"]); Y_pred = list(old["Y_pred"])
    else:
        H_seq, Y_true, Y_pred = [], [], []
    H_seq.append(H_long)
    np.savez_compressed(path, H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

# ---------- training ----------
def train(args):
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    raw = load_series_csv(args.data)
    T = len(raw); T_tr = int(T * args.train_frac)
    train_raw, val_raw = raw[:T_tr], raw[T_tr:]
    mean, std = train_raw.mean(0), train_raw.std(0)

    dtrain = SeqDataset(train_raw, args.seq_len, mean, std)
    dval   = SeqDataset(val_raw,   args.seq_len, mean, std)

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(dval,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = BiLSTMNet(in_dim=raw.shape[1], hid=args.hid, n_layer=args.n_layer, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    ensure_dir(args.outdir)

    # epoch 0
    save_checkpoint(model, opt, 0, math.inf, args)
    dump_hidden_states(model, dval, device, epoch=0, outdir=args.outdir, max_batches=args.state_batches)
    if args.dense_states:
        dump_hidden_states_dense(model, val_raw, mean, std, device, epoch=0, outdir=args.outdir, dense_stride=args.dense_stride)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / max(1, len(dtrain))

        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtot += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss = vtot / max(1, len(dval))
        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if epoch in args.save_epochs or epoch == args.epochs:
            save_checkpoint(model, opt, epoch, val_loss, args)
            dump_hidden_states(model, dval, device, epoch=epoch, outdir=args.outdir, max_batches=args.state_batches)
            if args.dense_states:
                dump_hidden_states_dense(model, val_raw, mean, std, device, epoch=epoch, outdir=args.outdir, dense_stride=args.dense_stride)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, opt, epoch, val_loss, args, name="best")

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="lorenz.csv")
    p.add_argument("--outdir", type=str, default="runs/lorenz/RNN_BiLSTM")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--hid", type=int, default=96)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--save_epochs", type=int, nargs="*", default=[1,5,10,25,50,100])
    p.add_argument("--state_batches", type=int, default=4,
                   help="сколько первых вал-окон писать; 0/отрицательное — писать все")
    p.add_argument("--dense_states", action="store_true",
                   help="непрерывный поток скрытых состояний по всей валидации")
    p.add_argument("--dense_stride", type=int, default=1,
                   help="шаг даунсемплинга по времени для dense-режима (>=1)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
