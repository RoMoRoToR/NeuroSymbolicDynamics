import os, math, argparse, numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------- Данные: окна из CSV ----------
class SeqWindows(Dataset):
    def __init__(self, csv_path, cols, win=64, stride=1, norm=True):
        df = pd.read_csv(csv_path)
        X = df[cols].values.astype(np.float32)
        if norm:
            self.mu, self.sig = X.mean(0), X.std(0)+1e-8
            X = (X - self.mu)/self.sig
        self.X = X
        self.win, self.stride = win, stride
        self.idxs = list(range(0, len(X)-win-1, stride))
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        j = self.idxs[i]
        x = self.X[j:j+self.win]          # [T,D]
        y = self.X[j+1:j+self.win+1]      # next-step prediction
        return torch.from_numpy(x), torch.from_numpy(y)

# ---------- Модели ----------
class LSTMModel(nn.Module):
    def __init__(self, d_in, d_hid=128, n_layers=1, bidir=False):
        super().__init__()
        self.rnn = nn.LSTM(d_in, d_hid, num_layers=n_layers, batch_first=True, bidirectional=bidir)
        self.head = nn.Linear(d_hid*(2 if bidir else 1), d_in)
        self.dhid = d_hid*(2 if bidir else 1)
    def forward(self, x):
        # x: [B,T,D]
        h, _ = self.rnn(x)               # [B,T,H]
        y = self.head(h)                 # [B,T,D]
        return y, h

class TinyDecoder(nn.Module):
    """минимальный transformer decoder-style для последовательностей"""
    def __init__(self, d_in, d_model=128, nhead=4, nlayers=2, dim_ff=512):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, d_in)
    def forward(self, x):
        # causal mask
        B,T,D = x.shape
        m = torch.triu(torch.ones(T,T, device=x.device), diagonal=1).bool()
        z = self.proj(x)
        h = self.enc(z, mask=m)    # [B,T,d_model]
        y = self.head(h)
        return y, h

# ---------- Обучение + съём скрытых состояний ----------
@torch.no_grad()
def dump_hidden_states(model, loader_val, out_dir, epoch, device):
    model.eval()
    H_seq = []
    for xb, _ in loader_val:
        xb = xb.to(device)
        _, h = model(xb)
        # собираем по батчам в список окон [T,H]
        for i in range(h.shape[0]):
            H_seq.append(h[i].detach().cpu().numpy())
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"states_epoch_{epoch}.npz"), H_seq=np.array(H_seq, dtype=object))
    # мини-лог
    with open(os.path.join(out_dir, f"epoch_{epoch:03d}.json"), "w") as f:
        f.write('{"epoch": %d}'%epoch)

def train_one(model, train_csv, val_csv, cols, out_dir, epochs=(0,10,25,50,100,150,200),
              win=64, lr=1e-3, batch=64, device="cuda" if torch.cuda.is_available() else "cpu"):
    ds_tr = SeqWindows(train_csv, cols, win=win, stride=1)
    ds_va = SeqWindows(val_csv,   cols, win=win, stride=win)  # неперекрывающиеся окна
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # epoch 0 (до обучения) — сразу снимем скрытые состояния
    dump_hidden_states(model, dl_va, out_dir, epoch=0, device=device)

    for ep in range(1, max(epochs)+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            yp, _ = model(xb)
            loss = loss_fn(yp, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep in epochs:
            dump_hidden_states(model, dl_va, out_dir, epoch=ep, device=device)
    print(f"[ok] saved states to {out_dir}")

def main():
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="data/karman_train.csv")
    ap.add_argument("--val_csv",   default="data/karman_val.csv")
    ap.add_argument("--cols", nargs="+", default=["CL","CD","dP","S1","S2"])
    ap.add_argument("--win", type=int, default=64)
    ap.add_argument("--out_root", default="runs/karman")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)

    # RNN (LSTM)
    rnn = LSTMModel(d_in=len(args.cols), d_hid=128, bidir=False)
    train_one(rnn, args.train_csv, args.val_csv, args.cols,
              out_dir=os.path.join(args.out_root, "RNN_LSTM"), win=args.win)

    # BiLSTM
    bi = LSTMModel(d_in=len(args.cols), d_hid=128, bidir=True)
    train_one(bi, args.train_csv, args.val_csv, args.cols,
              out_dir=os.path.join(args.out_root, "RNN_BiLSTM"), win=args.win)

    # Transformer (decoder-style)
    tr = TinyDecoder(d_in=len(args.cols), d_model=128, nhead=4, nlayers=4, dim_ff=512)
    train_one(tr, args.train_csv, args.val_csv, args.cols,
              out_dir=os.path.join(args.out_root, "Transformer"), win=args.win)

if __name__ == "__main__":
    main()
