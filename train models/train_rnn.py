# train_rnn.py
import argparse, os, json, math, random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 1) Утилиты и датасет
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class SeqDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, mean=None, std=None):
        """
        series: np.ndarray [T, D] (ожидаем D=3 для (X,Y,Z))
        """
        assert series.ndim == 2, "series must be [T, D]"
        self.seq_len = seq_len
        # нормируем по train-статистике
        if mean is None: mean = series.mean(axis=0)
        if std is None:  std  = series.std(axis=0) + 1e-8
        self.mean, self.std = mean, std
        self.series = (series - mean) / std

        self.X, self.y = [], []
        for i in range(len(self.series) - seq_len):
            self.X.append(self.series[i:i+seq_len])
            self.y.append(self.series[i+seq_len])  # прогноз следующего шага
        self.X, self.y = np.array(self.X, dtype=np.float32), np.array(self.y, dtype=np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

def load_lorenz_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    # берем X,Y,Z если есть; иначе — первые 3 числовых столбца
    cols = [c for c in df.columns if c.strip().lower() in ["x","y","z"]]
    if len(cols) >= 3:
        data = df[cols[:3]].values.astype("float32")
    else:
        num = df.select_dtypes(include=[np.number])
        assert num.shape[1] >= 3, "Нужно хотя бы 3 числовых столбца (X,Y,Z)."
        data = num.iloc[:, :3].values.astype("float32")
    return data

# -----------------------------
# 2) Модель LSTM
# -----------------------------
class LSTMNextStep(nn.Module):
    def __init__(self, input_dim=3, hidden=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, dropout=dropout if num_layers>1 else 0.0, batch_first=True)
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x):
        # x: [B, T, D]
        out, _ = self.lstm(x)          # out: [B, T, H]
        last = out[:, -1, :]           # [B, H]
        pred = self.head(last)         # [B, D]
        return pred

    @torch.no_grad()
    def hidden_sequence(self, x):
        # для съема скрытых состояний по времени
        out, _ = self.lstm(x)          # [B, T, H]
        return out

# -----------------------------
# 3) Обучение
# -----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    raw = load_lorenz_csv(args.data)
    T = len(raw)
    split = int(T * args.train_frac)
    train_raw, val_raw = raw[:split], raw[split - args.seq_len:]  # небольшое перекрытие, чтобы окна не обрывались

    # считаем статистики по train, применяем одинаково к train/val
    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0)+1e-8
    dtrain = SeqDataset(train_raw, args.seq_len, mean, std)
    dval   = SeqDataset(val_raw,   args.seq_len, mean, std)

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(dval,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LSTMNextStep(hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    os.makedirs(args.outdir, exist_ok=True)
    # сохраним "эпоха 0" (инициализация) и снимем состояния
    save_checkpoint(model, opt, 0, math.inf, args)
    dump_hidden_states(model, dval, device, epoch=0, outdir=args.outdir, max_batches=args.state_batches)

    best_val = math.inf
    for epoch in range(1, args.epochs+1):
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
        train_loss = total / len(dtrain)

        # вал
        model.eval()
        vtotal = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                vtotal += loss.item() * xb.size(0)
        val_loss = vtotal / len(dval)

        print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if epoch in args.save_epochs or epoch == args.epochs:
            save_checkpoint(model, opt, epoch, val_loss, args)
            dump_hidden_states(model, dval, device, epoch=epoch, outdir=args.outdir, max_batches=args.state_batches)

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
    # также положим короткий json
    with open(os.path.join(args.outdir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f, ensure_ascii=False, indent=2)

@torch.no_grad()
def dump_hidden_states(model, dval, device, epoch, outdir, max_batches=2):
    model.eval()
    loader = DataLoader(dval, batch_size=1, shuffle=False)
    H_seq, Y_true, Y_pred = [], [], []
    cnt = 0
    for xb, yb in loader:
        xb = xb.to(device)
        h = model.hidden_sequence(xb)         # [1, T, H]
        yhat = model(xb)
        H_seq.append(h.squeeze(0).cpu().numpy())  # [T, H]
        Y_true.append(yb.squeeze(0).cpu().numpy())
        Y_pred.append(yhat.squeeze(0).cpu().numpy())
        cnt += 1
        if cnt >= max_batches: break
    np.savez_compressed(os.path.join(outdir, f"states_epoch_{epoch:03d}.npz"),
                        H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

# -----------------------------
# 4) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="lorenz.csv")
    p.add_argument("--outdir", type=str, default="runs/lorenz/RNN_LSTM")
    p.add_argument("--seq_len", type=int, default=20)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_epochs", type=int, nargs="*", default=[1,5,10,25,50,100,150,200])
    p.add_argument("--state_batches", type=int, default=4, help="сколько первых вал-окон писать скрытых состояний")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
