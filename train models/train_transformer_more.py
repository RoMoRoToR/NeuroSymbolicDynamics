# train_transformer_more.py
# -*- coding: utf-8 -*-
import os, json, math, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# --------------------- данные Лоренца (пример) ---------------------
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
        ys.append(x[i+1:i+win+1])  # one-step ahead
    xs = torch.from_numpy(np.stack(xs))        # [N, win, 3]
    ys = torch.from_numpy(np.stack(ys))
    return xs, ys

# --------------------- Transformer_old ---------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # [L,1,D]
    def forward(self, x):
        # x: [L,B,D]
        L = x.size(0)
        return x + self.pe[:L]

class TinyTransformer(nn.Module):
    def __init__(self, d_in=3, d_model=64, nhead=4, num_layers=4, d_ff=128):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=False, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pe = PositionalEncoding(d_model)
        self.readout = nn.Linear(d_model, d_in)

    def forward(self, x, return_states=False, hook_dict=None):
        # x: [B,T,3] -> transformer uses [T,B,D]
        z = self.proj(x)                       # [B,T,D]
        z = z.transpose(0,1)                   # [T,B,D]
        z = self.pe(z)                         # [T,B,D]
        if return_states and hook_dict is not None:
            hook_dict.clear()
            def _hook(module, inp, out):
                # out: [T,B,D] из ПОСЛЕДНЕГО encoder блока
                hook_dict["enc_last"] = out.detach()
            handle = self.encoder.layers[-1].register_forward_hook(_hook)
            h = self.encoder(z)                # [T,B,D]
            handle.remove()
        else:
            h = self.encoder(z)
        y = self.readout(h).transpose(0,1)     # [B,T,3]
        if return_states and hook_dict is not None:
            # возьмём выход ПОСЛЕДНЕГО слоя: [T,B,D] -> [B,T,D]
            states = hook_dict["enc_last"].transpose(0,1)
            return y, states
        return y

# --------------------- utils ---------------------
@torch.no_grad()
def collect_states_transformer(model, loader, max_points=50000):
    model.eval()
    hook = {}
    H_all = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        _, states = model(xb, return_states=True, hook_dict=hook)  # [B,T,D]
        s = states.reshape(-1, states.size(-1)).cpu().numpy()      # [B*T, D]
        H_all.append(s)
        if sum(len(a) for a in H_all) >= max_points:
            break
    H = np.concatenate(H_all, axis=0)
    return [H]  # список последовательностей (одна длинная)

def save_states(run_dir, epoch, H_seq, train_loss=None, val_loss=None):
    os.makedirs(run_dir, exist_ok=True)
    np.savez_compressed(os.path.join(run_dir, f"states_epoch_{epoch:03d}.npz"), H_seq=H_seq)
    with open(os.path.join(run_dir, f"epoch_{epoch:03d}.json"), "w", encoding="utf-8") as f:
        json.dump({"train_loss": float(train_loss) if train_loss is not None else None,
                   "val_loss": float(val_loss) if val_loss is not None else None}, f)

# --------------------- main ---------------------
def main():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)  # лишь один раз, не каждый эпох
    series = lorenz(T=30000)
    xtr, ytr = make_sliding_windows(series[:20000], win=64, stride=1)
    xva, yva = make_sliding_windows(series[20000:],  win=64, stride=1)

    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=64, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(xva, yva), batch_size=64, shuffle=False)

    model = TinyTransformer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    run_dir = "../Karman/runs/lorenz/Transformer"
    os.makedirs(run_dir, exist_ok=True)

    # начальное состояние (epoch 0)
    H0 = collect_states_transformer(model, val_loader, max_points=60000)
    save_states(run_dir, 0, H0, train_loss=None, val_loss=None)

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

        # валидация
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yhat = model(xb)
                va_loss += loss_fn(yhat, yb).item()
        va_loss /= len(val_loader)

        # скрытые состояния на всём валид. датасете (много точек!)
        H_seq = collect_states_transformer(model, val_loader, max_points=60000)
        save_states(run_dir, epoch, H_seq, tr_loss, va_loss)
        if epoch % 20 == 0:
            print(f"epoch {epoch:03d}: train {tr_loss:.4f} | val {va_loss:.4f} | saved {len(H_seq[0])} points")

if __name__ == "__main__":
    main()
