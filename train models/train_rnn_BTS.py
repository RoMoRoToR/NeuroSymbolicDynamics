# train_rnn.py
# -*- coding: utf-8 -*-
import argparse, os, json, math, random, glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 0) Утилиты, воспроизводимость
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# -----------------------------
# 1) Датасет (скользящие окна)
# -----------------------------
class SeqDataset(Dataset):
    """
    Берёт массив [T, D], нормирует по train-статистике (mean/std),
    разрезает на окна длины seq_len и таргет = следующий шаг (next-step).
    """
    def __init__(self, series: np.ndarray, seq_len: int, mean=None, std=None):
        assert series.ndim == 2, "series must be [T, D]"
        self.seq_len = seq_len
        if mean is None: mean = series.mean(axis=0)
        if std  is None: std  = series.std(axis=0) + 1e-8
        self.mean, self.std = mean, std
        self.series = (series - mean) / std
        X, y = [], []
        for i in range(len(self.series) - seq_len):
            X.append(self.series[i:i+seq_len])
            y.append(self.series[i+seq_len])
        self.X = np.asarray(X, np.float32)
        self.y = np.asarray(y, np.float32)

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

# -----------------------------
# 2) Загрузчики данных
# -----------------------------
def load_lorenz_csv(path: str) -> np.ndarray:
    """
    Ожидает CSV с числовыми колонками (X,Y,Z, ...). Возвращаем первые 3 столбца [T,3].
    """
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    assert num.shape[1] >= 3, "Нужно >=3 числовых столбцов в Lorenz CSV."
    arr = num.iloc[:, :3].values.astype("float32")
    return arr

def _read_all_csvs(path_or_dir: str) -> pd.DataFrame:
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(f"В папке нет CSV: {path_or_dir}")
        dfs = [pd.read_csv(f) for f in files]
        df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        df = pd.read_csv(path_or_dir)
    return df

def load_bitcoin_series(path_or_dir: str, resample: str = "1H") -> np.ndarray:
    """
    Kaggle: mczielinski/bitcoin-historical-data
    Преобразуем OHLCV → 3 фичи:
      1) r_close = diff(log(Close))
      2) hl_range = (High - Low) / Close_{t-1}
      3) v_log = diff(log1p(Volume_BTC))   (если нет Volume_BTC — пробуем Volume или Volume_(BTC))
    Реземплинг по времени (по умолчанию 1H). Возвращаем [T,3] (float32).
    """
    df = _read_all_csvs(path_or_dir)

    # --- Нормализуем названия столбцов
    cols_map = {c.lower().strip(): c for c in df.columns}
    # Timestamp может быть unix (секунды) или строка
    ts_col = None
    for cand in ["timestamp", "date", "time", "datetime"]:
        if cand in cols_map:
            ts_col = cols_map[cand]; break
    if ts_col is None:
        raise ValueError("Не нашли колонку с временем (Timestamp/Date/Time/Datetime).")

    # Преобразуем в datetime
    ts = df[ts_col]
    if np.issubdtype(ts.dtype, np.number):
        # unix seconds
        dt = pd.to_datetime(ts, unit="s", utc=True)
    else:
        dt = pd.to_datetime(ts, errors="coerce", utc=True)
    df = df.copy()
    df["__dt__"] = dt
    df = df.dropna(subset=["__dt__"]).sort_values("__dt__")
    df = df.set_index("__dt__")

    # Поиск OHLC/Volume
    def find(name_opts):
        for cand in name_opts:
            c = cand.lower()
            if c in cols_map:
                return cols_map[c]
            # иногда встречаются пробелы/скобки: Volume BTC / Volume_(BTC)
            for col in df.columns:
                if col.lower().replace(" ", "").replace("(", "").replace(")", "") == c.replace(" ", "").replace("(", "").replace(")", ""):
                    return col
        return None

    c_open  = find(["open"])
    c_high  = find(["high"])
    c_low   = find(["low"])
    c_close = find(["close", "weighted price"])  # на всякий случай
    c_vol   = find(["volume_btc", "volume (btc)", "volume_(btc)", "volume", "volume currency", "volume_(currency)"])

    if c_close is None:
        raise ValueError("Не найдена колонка Close (или Weighted Price).")

    # Если задан реземплинг — агрегируем OHLC, объём суммируем/mean при отсутствии
    if resample:
        agg = {}
        if c_open:  agg[c_open]  = "first"
        if c_high:  agg[c_high]  = "max"
        if c_low:   agg[c_low]   = "min"
        if c_close: agg[c_close] = "last"
        if c_vol:   agg[c_vol]   = "sum"
        df = df.resample(resample).agg(agg)

    # Заполняем пропуски по времени (например, выходные) вперёд
    df = df.ffill()

    # Вычисляем фичи
    close = df[c_close].astype("float64")
    high  = df[c_high].astype("float64") if c_high in df else close
    low   = df[c_low].astype("float64") if c_low in df else close
    vol   = df[c_vol].astype("float64") if c_vol in df else pd.Series(0.0, index=df.index)

    r_close = np.log(close).diff().fillna(0.0).values
    hl_range = ((high - low) / (close.shift(1).replace(0, np.nan))).fillna(0.0).values
    v_log = np.log1p(vol).diff().fillna(0.0).values

    feats = np.stack([r_close, hl_range, v_log], axis=1).astype("float32")
    # Удалим возможные NaN/Inf
    feats = np.nan_to_num(feats, copy=False)
    return feats

def load_series(path: str, data_kind: str, resample: str) -> np.ndarray:
    if data_kind == "lorenz":
        return load_lorenz_csv(path)
    elif data_kind == "bitcoin":
        return load_bitcoin_series(path, resample=resample)
    else:
        raise ValueError(f"Неизвестный data_kind: {data_kind}")

# -----------------------------
# 3) Модель: односторонний LSTM
# -----------------------------
class LSTMNextStep(nn.Module):
    """
    Односторонний LSTM. Выход по последней временной позиции -> Linear -> прогноз следующего шага (D-мерный).
    """
    def __init__(self, input_dim=3, hidden=128, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False, batch_first=True
        )
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x):                 # x: [B,T,D]
        out, _ = self.lstm(x)            # [B,T,H]
        last = out[:, -1, :]             # [B,H]
        return self.head(last)           # [B,D]

    @torch.no_grad()
    def hidden_sequence(self, x):
        out, _ = self.lstm(x)            # [B,T,H]
        return out

# -----------------------------
# 4) Обучение и логгинг
# -----------------------------
def save_ckpt(model, opt, epoch, val_loss, args, name=None):
    tag = name if name else f"epoch_{epoch:03d}"
    os.makedirs(args.outdir, exist_ok=True)
    torch.save({
        "epoch": epoch, "val_loss": float(val_loss),
        "model_state": model.state_dict(), "opt_state": opt.state_dict(),
        "config": vars(args)
    }, os.path.join(args.outdir, f"{tag}.pt"))
    with open(os.path.join(args.outdir, f"{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f, ensure_ascii=False, indent=2)

@torch.no_grad()
def dump_states(model, dval, device, epoch, outdir, max_batches=4):
    model.eval()
    loader = DataLoader(dval, batch_size=1, shuffle=False)
    H_seq, Y_true, Y_pred = [], [], []
    cnt = 0
    for xb, yb in loader:
        xb = xb.to(device)
        h = model.hidden_sequence(xb)               # [1,T,H]
        yhat = model(xb)                            # [1,D]
        H_seq.append(h.squeeze(0).cpu().numpy())    # [T,H]
        Y_true.append(yb.squeeze(0).cpu().numpy())  # [D]
        Y_pred.append(yhat.squeeze(0).cpu().numpy())
        cnt += 1
        if cnt >= max_batches: break
    np.savez_compressed(os.path.join(outdir, f"states_epoch_{epoch:03d}.npz"),
                        H_seq=H_seq, Y_true=Y_true, Y_pred=Y_pred)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    raw = load_series(args.data, args.data_kind, args.resample)  # [T,3]
    T = len(raw); split = int(T * args.train_frac)
    # важный момент: валидацию начинаем на seq_len раньше, чтобы были полные окна
    train_raw = raw[:split]
    val_raw   = raw[split - args.seq_len:]

    mean, std = train_raw.mean(axis=0), train_raw.std(axis=0) + 1e-8
    dtrain = SeqDataset(train_raw, args.seq_len, mean, std)
    dval   = SeqDataset(val_raw,   args.seq_len, mean, std)

    train_loader = DataLoader(dtrain, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(dval,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = LSTMNextStep(input_dim=raw.shape[1], hidden=args.hidden, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    # epoch 0: снимем состояния и json
    save_ckpt(model, opt, 0, math.inf, args)
    dump_states(model, dval, device, epoch=0, outdir=args.outdir, max_batches=args.state_batches)

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(dtrain)

        model.eval(); vtot = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                vtot += loss_fn(model(xb), yb).item() * xb.size(0)
        val_loss = vtot / len(dval)
        print(f"[LSTM] Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if epoch in args.save_epochs or epoch == args.epochs:
            save_ckpt(model, opt, epoch, val_loss, args)
            dump_states(model, dval, device, epoch=epoch, outdir=args.outdir, max_batches=args.state_batches)

        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(model, opt, epoch, val_loss, args, name="best")

# -----------------------------
# 5) CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Путь к CSV или папке с CSV.")
    p.add_argument("--data_kind", type=str, choices=["lorenz", "bitcoin"], default="bitcoin")
    p.add_argument("--resample", type=str, default="1H", help="Период реземплинга для bitcoin (например, 15min, 1H, 4H, 1D).")
    p.add_argument("--outdir", type=str, default="runs/bitcoin/RNN_LSTM")
    p.add_argument("--seq_len", type=int, default=64)
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
    p.add_argument("--state_batches", type=int, default=4)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
