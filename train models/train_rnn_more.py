# train_rnn_more.py
# -*- coding: utf-8 -*-
import os, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def lorenz(T=20000, dt=0.01, sigma=10., rho=28., beta=8/3):
    x = np.zeros((T, 3), dtype=np.float32)
    x[0] = np.array([1.0, 1.0, 1.0], np.float32)
    for t in range(T-1):
        dx = sigma*(x[t,1]-x[t,0])
        dy = x[t,0]*(rho-x[t,2]) - x[t,1]
        dz = x[t,0]*x[t,1] - beta*x[t,2]
        x[t+1] = x[t] + dt*np.array([dx, dy, dz], np.float32)
    return x

def make_sliding_windows(x, win=64, stride=1):
    xs, ys = [], []
    for i in range(0, len(x)-win, stride):
        xs.append(x[i:i+win])
        ys.append(x[i+1:i+win+1])
    xs = torch.from_numpy(np.stack(xs))
    ys = torch.from_numpy(np.stack(ys))
    return xs, ys

class LSTMReg(nn.Module):
    def __init__(self, d_in=3, d_hid=64, n_layers=2):
        super().__init__()
        self.rnn = nn.LSTM(d_in, d_hid, n_layers, batch_first=True)
        self.readout = nn.Linear(d_hid, d_in)
        self.n_layers = n_layers
        self.d_hid = d_hid
    def forward(self, x, return_states=False):
        h, _ = self.rnn(x)                # [B,T,H]
        y = self.readout(h)               # [B,T,3]
        if return_states:
            return y, h                   # берём h_t верхнего слоя
        return y

@torch.no_grad()
def collect_states_rnn(model, loader, max_points=50000):
    model.eval()
    H_all = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        _, h = model(xb, return_states=True)      # [B,T,H]
        s = h.reshape(-1, h.size(-1)).cpu().numpy()
        H_all.append(s)
        if sum(len(a) for a in H_all) >= max_points:
            break
    H = np.concatenate(H_all, axis=0)
    return [H]

def save_states(run_dir, epoch, H_seq, train_loss=None, val_loss=None):
    os.makedirs(run_dir, exist_ok=True)
    np.savez_compressed(os.path.join(run_dir, f"states_epoch_{epoch:03d}.npz"), H_seq=H_seq)
    with open(os.path.join(run_dir, f"epoch_{epoch:03d}.json"), "w", encoding="utf-8") as f:
        json.dump({"train_loss": float(train_loss) if train_loss is not None else None,
                   "val_loss": float(val_loss) if val_loss is not None else None}, f)

def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    series = lorenz(T=30000)
    xtr, ytr = make_sliding_windows(series[:20000], win=64, stride=1)
    xva, yva = make_sliding_windows(series[20000:],  win=64, stride=1)

    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(xva, yva), batch_size=64, shuffle=False)

    model = LSTMReg().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    run_dir = "../Karman/runs/lorenz/RNN_LSTM"
    os.makedirs(run_dir, exist_ok=True)

    H0 = collect_states_rnn(model, val_loader, max_points=60000)
    save_states(run_dir, 0, H0)

    EPOCHS = 200
    for epoch in range(1, EPOCHS+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yhat = model(xb)
                va_loss += loss_fn(yhat, yb).item()
        va_loss /= len(val_loader)

        H_seq = collect_states_rnn(model, val_loader, max_points=60000)
        save_states(run_dir, epoch, H_seq, tr_loss, va_loss)
        if epoch % 20 == 0:
            print(f"epoch {epoch:03d}: train {tr_loss:.4f} | val {va_loss:.4f} | saved {len(H_seq[0])} points")

if __name__ == "__main__":
    main()
