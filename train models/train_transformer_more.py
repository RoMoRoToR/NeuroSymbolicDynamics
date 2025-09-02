# train_transformer.py
# -*- coding: utf-8 -*-
"""
Decoder-only Transformer для временного ряда + съём скрытых состояний для
символьной/топологической диагностики.

НОВОЕ:
- --state_batches <= 0 : записывать все вал-окна в states_epoch_XXX.npz
- --dense_states (--dense_stride): «густой» поток скрытых состояний по всей валидации.
"""

import os, math, json, argparse, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 0) Утилиты
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_series_csv(path: str) -> np.ndarray:
    """
    Загружает CSV с числовыми столбцами (T x D).
    Если есть заголовок — пропустим.
    """
    try:
        arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, autostrip=True)
        # genfromtxt вернёт nan на заголовок — удалим строки с nan
        if arr.ndim == 1:
            arr = arr[:, None]
        arr = arr[~np.any(~np.isfinite(arr), axis=1)]
        if arr.size == 0:
            raise ValueError("CSV parsed but empty / non-numeric.")
        return arr.astype("float32")
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать {path}: {e}")

# -----------------------------
# 1) Датасет
# -----------------------------
class SeqDataset(Dataset):
    """
    Окна длины L из нормализованного ряда X (T x D):
      x[i] = X[i:i+L],   y[i] = X[i+L]  (next-step)
    """
    def __init__(self, raw: np.ndarray, L: int, mean: np.ndarray, std: np.ndarray):
        assert raw.ndim == 2, "raw must be [T,D]"
        self.L = int(L)
        self.mean = mean.astype("float32")
        self.std  = (std + 1e-8).astype("float32")
        X = ((raw - self.mean) / self.std).astype("float32")
        T = len(X)
        N = max(0, T - self.L)
        self.X = np.lib.stride_tricks.sliding_window_view(X, (self.L, X.shape[1])) \
                   .reshape(T - self.L + 1, self.L, X.shape[1])[:N]
        self.Y = X[self.L: self.L + N]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.Y[i])

# -----------------------------
# 2) Модель: Tiny decoder-only Transformer
# -----------------------------
class TinyDecoderOnly(nn.Module):
    def __init__(self, in_dim=1, d_model=128, n_head=4, n_layer=4, d_ff=256, dropout=0.1, max_len=1024):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.max_len = max_len

        self.in_proj = nn.Linear(in_dim, d_model)
        # позиционные эмбеддинги (обучаемые)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.drop = nn.Dropout(dropout)

        # «голова» регрессии на следующий шаг
        self.head = nn.Linear(d_model, in_dim)

    # нужно для съёма состояний
    @torch.no_grad()
    def token_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D_in] → вернёт [B, T, d_model] (pos+proj, идущие на encoder).
        Без применения self.encoder — это то, что мы хотим писать в H_seq.
        """
        b, t, _ = x.shape
        z = self.in_proj(x) + self.pos[:, :t, :]
        return self.drop(z)

    def _causal_mask(self, t: int, device=None):
        # верхнетреугольная матрица True над диагональю (маскируем «будущее»)
        m = torch.triu(torch.ones(t, t, device=device), diagonal=1).bool()
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D_in] → y: [B, D_in] (прогноз следующей точки)
        """
        b, t, _ = x.shape
        h = self.token_embeddings(x)                 # [B,T,d_model]
        mask = self._causal_mask(t, x.device)        # [T,T]
        h = self.encoder(h, mask=mask)               # [B,T,d_model]
        h_last = h[:, -1, :]                         # [B,d_model]
        return self.head(h_last)                     # [B,in_dim]

# -----------------------------
# 3) Тренировка
# -----------------------------
def train(args):
    seed_everything(args.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # данные
    raw = load_series_csv(args.data)                # [T,D]
    T = len(raw)
    T_tr = int(T * args.train_frac)
    train_raw, val_raw = raw[:T_tr], raw[T_tr:]
    mean = train_raw.mean(axis=0); std = train_raw.std(axis=0)

    dtrain = SeqDataset(train_raw, args.seq_len, mean, std)
    dval   = SeqDataset(val_raw,   args.seq_len, mean, std)

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(dval,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = TinyDecoderOnly(
        in_dim=raw.shape[1],
        d_model=args.d_model, n_head=args.n_head, n_layer=args.n_layer,
        d_ff=args.d_ff, dropout=args.dropout, max_len=args.seq_len
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    ensure_dir(args.outdir)

    # Эпоха 0: до обучения
    save_checkpoint(model, opt, 0, math.inf, args)
    dump_token_states(model, dval, device, epoch=0, outdir=args.outdir, max_batches=args.state_batches)
    if args.dense_states:
        dump_token_states_dense(model, val_raw, mean, std, args.seq_len, device,
                                epoch=0, outdir=args.outdir, dense_stride=args.dense_stride)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
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

        # ---- val ----
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vtotal += loss.item() * xb.size(0)
        val_loss = vtotal / max(1, len(dval))
        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        # чекпоинты и съём скрытых состояний
        if epoch in args.save_epochs or epoch == args.epochs:
            save_checkpoint(model, opt, epoch, val_loss, args)
            dump_token_states(model, dval, device, epoch=epoch, outdir=args.outdir, max_batches=args.state_batches)
            if args.dense_states:
                dump_token_states_dense(model, val_raw, mean, std, args.seq_len, device,
                                        epoch=epoch, outdir=args.outdir, dense_stride=args.dense_stride)

        # лучший
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, opt, epoch, val_loss, args, name="best")

def save_checkpoint(model, opt, epoch, val_loss, args, name=None):
    tag = name if name is not None else f"epoch_{epoch:03d}"
    ckpt_path = os.path.join(args.outdir, f"{tag}.pt")
    torch.save({
        "epoch": epoch,
        "val_loss": float(val_loss),
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "config": vars(args)
    }, ckpt_path)
    with open(os.path.join(args.outdir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f, ensure_ascii=False, indent=2)

# -----------------------------
# 3a) Съём скрытых состояний (обычные окна)
# -----------------------------
@torch.no_grad()
def dump_token_states(model, dval: Dataset, device, epoch: int, outdir: str, max_batches: int = 4):
    """
    Снимает эмбеддинги токенов для первых max_batches вал-окон (batch_size=1).
    Если max_batches <= 0 или None — снимаем все окна.
    Сохраняем: states_epoch_XXX.npz с ключами H_seq, Y_true, Y_pred.
    """
    model.eval()
    loader = DataLoader(dval, batch_size=1, shuffle=False)
    H_seq, Y_true, Y_pred = [], [], []
    cnt = 0
    for xb, yb in loader:
        xb = xb.to(device)
        h = model.token_embeddings(xb)                 # [1,T,d_model]
        yhat = model(xb)                               # [1,D]
        H_seq.append(h.squeeze(0).cpu().numpy())       # [T,d_model]
        Y_true.append(yb.squeeze(0).cpu().numpy())
        Y_pred.append(yhat.squeeze(0).cpu().numpy())
        cnt += 1
        if (max_batches is not None) and (max_batches > 0) and (cnt >= max_batches):
            break
    np.savez_compressed(os.path.join(outdir, f"states_epoch_{epoch:03d}.npz"),
                        H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

# -----------------------------
# 3b) «Густой» поток (скользящее окно по всей валидации)
# -----------------------------
@torch.no_grad()
def dump_token_states_dense(model, val_raw: np.ndarray, mean: np.ndarray, std: np.ndarray,
                            seq_len: int, device, epoch: int, outdir: str, dense_stride: int = 1):
    """
    Скользящее окно длины L=seq_len по всей валидации.
    На каждом шаге берём эмбеддинг последнего токена — получаем ~T-L+1 точек.
    Результат ДОзаписывается в states_epoch_XXX.npz как дополнительная последовательность.
    """
    model.eval()
    series = ((val_raw - mean) / (std + 1e-8)).astype("float32")   # [T,D]
    T = len(series)
    H_list = []
    step = max(1, int(dense_stride))
    for s in range(0, max(0, T - seq_len + 1), step):
        xw = torch.from_numpy(series[s:s+seq_len])[None, ...].to(device)  # [1,L,D]
        emb_last = model.token_embeddings(xw)[0, -1, :].detach().cpu().numpy()  # [d_model]
        H_list.append(emb_last)
    H_long = np.stack(H_list, axis=0) if len(H_list) else np.zeros((0, model.d_model), dtype="float32")

    path = os.path.join(outdir, f"states_epoch_{epoch:03d}.npz")
    if os.path.exists(path):
        old = np.load(path, allow_pickle=True)
        H_seq = list(old["H_seq"]); Y_true = list(old["Y_true"]); Y_pred = list(old["Y_pred"])
    else:
        H_seq, Y_true, Y_pred = [], [], []
    H_seq.append(H_long)
    np.savez_compressed(path, H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

# -----------------------------
# 4) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="lorenz.csv",
                   help="CSV с числовыми столбцами (T x D).")
    p.add_argument("--outdir", type=str, default="runs/lorenz/Transformer")
    p.add_argument("--seq_len", type=int, default=20)
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
    p.add_argument("--state_batches", type=int, default=4,
                   help="сколько первых вал-окон писать; 0/отрицательное — писать все")
    p.add_argument("--dense_states", action="store_true",
                   help="сохранять непрерывный поток скрытых состояний по всей валидации")
    p.add_argument("--dense_stride", type=int, default=1,
                   help="шаг скользящего окна (или даунсемплинг по времени) для dense-режима")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
