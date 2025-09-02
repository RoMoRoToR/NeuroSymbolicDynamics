# plot_raw_data.py
# -*- coding: utf-8 -*-
"""
Простая визуализация исходных данных из CSV:
- Временной ряд
- Фазовые портреты (2D/3D)
- Такенс-эмбеддинг для 1D сигнала
- PCA 2D/3D (для многомерных)
- Спектр (periodogram/PSD)

Зависимости: numpy, pandas, matplotlib, scikit-learn (только для PCA).
"""

import os, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from sklearn.decomposition import PCA

# ----------------- утилиты -----------------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def read_csv_smart(path: str) -> pd.DataFrame:
    """
    Пытаемся прочитать с заголовком; если он «плохой», падаем на header=None.
    Удаляем строки с NaN/inf.
    """
    try:
        df = pd.read_csv(path)
        # если все имена — числа/Unnamed — считаем, что заголовка по сути нет
        bad = all([(str(c).startswith("Unnamed") or str(c).isdigit()) for c in df.columns])
        if bad:
            df = pd.read_csv(path, header=None)
    except Exception:
        df = pd.read_csv(path, header=None)
    # в числовой формат:
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df.astype("float32")

def parse_cols(df: pd.DataFrame, cols_arg: Optional[str]) -> List[int]:
    """
    Возвращает индексы столбцов, которые будем визуализировать.
    cols_arg может быть вида "0,1" (индексы) или "x,y" (имена).
    Если не задано — используем все столбцы.
    """
    if not cols_arg:
        return list(range(df.shape[1]))
    out = []
    tokens = [t.strip() for t in cols_arg.split(",") if t.strip() != ""]
    for t in tokens:
        if t.isdigit():
            i = int(t)
            if i < 0 or i >= df.shape[1]:
                raise ValueError(f"Индекс столбца вне диапазона: {i}")
            out.append(i)
        else:
            if t not in df.columns:
                raise ValueError(f"Столбец по имени не найден: {t}")
            out.append(int(df.columns.get_loc(t)))
    return out

def normalize_arr(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return X
    if mode == "zscore":
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-8
        return (X - mu) / sd
    if mode == "minmax":
        lo = np.nanmin(X, axis=0); hi = np.nanmax(X, axis=0); rng = (hi - lo); rng[rng == 0] = 1.0
        return (X - lo) / rng
    raise ValueError("normalize must be one of: none, zscore, minmax")

def moving_average_1d(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) == 0:
        return x
    # «valid» + паддинг краёв
    k = int(w)
    pad = k // 2
    xp = np.pad(x, (pad, k - 1 - pad), mode="edge")
    ker = np.ones(k, dtype=float) / k
    return np.convolve(xp, ker, mode="valid").astype(x.dtype)

def smooth_matrix(X: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return X
    out = np.zeros_like(X)
    for j in range(X.shape[1]):
        out[:, j] = moving_average_1d(X[:, j], window)
    return out

def downsample(X: np.ndarray, step: int) -> np.ndarray:
    step = max(1, int(step))
    return X[::step]

def safe_pca(X: np.ndarray, n_components: int) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0:
        return np.zeros((0, n_components))
    ncomp = int(min(n_components, X.shape[1], len(X)))
    if ncomp == 0:
        return np.zeros((0, n_components))
    Z = PCA(n_components=ncomp, random_state=0).fit_transform(X)
    if ncomp < n_components:
        Z = np.c_[Z, np.zeros((len(Z), n_components - ncomp))]
    return Z

def takens_embed_1d(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    if m < 2: m = 2
    if tau < 1: tau = 1
    N = len(x) - (m - 1) * tau
    if N <= 1:
        return np.zeros((0, m), float)
    idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
    return x[idx].T

# ----------------- графики -----------------
def plot_time_series(T: np.ndarray, X: np.ndarray, labels: List[str], out_png: str, title: str):
    plt.figure(figsize=(8, 4.8))
    for j in range(X.shape[1]):
        plt.plot(T, X[:, j], label=labels[j], lw=1.0)
    plt.title(title)
    plt.xlabel("t"); plt.ylabel("value"); plt.grid(True, alpha=0.3)
    if len(labels) <= 10:
        plt.legend(loc="best", fontsize=9)
    plt.tight_layout(); plt.savefig(out_png, dpi=170); plt.close()
    print(f"[ok] saved: {out_png}")

def plot_phase_2d(x: np.ndarray, y: np.ndarray, out_png: str, title: str):
    if len(x) < 2:
        return
    plt.figure(figsize=(5.6, 5.2))
    # линия по времени
    t = np.linspace(0, 1, len(x))
    plt.plot(x, y, lw=0.8, alpha=0.7)
    sc = plt.scatter(x, y, c=t, s=8, alpha=0.9, cmap=plt.cm.viridis)
    cb = plt.colorbar(sc); cb.set_label("time")
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.savefig(out_png, dpi=170); plt.close()
    print(f"[ok] saved: {out_png}")

def plot_phase_3d(X3: np.ndarray, out_png: str, title: str):
    if len(X3) < 2:
        return
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.add_subplot(111, projection='3d')
    t = np.linspace(0, 1, len(X3))
    ax.plot(X3[:,0], X3[:,1], X3[:,2], lw=0.9, alpha=0.8)
    sc = ax.scatter(X3[:,0], X3[:,1], X3[:,2], c=t, s=6, alpha=0.9, cmap=plt.cm.viridis)
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.06, label="time")
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("x3")
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=165); plt.close()
    print(f"[ok] saved: {out_png}")

def plot_psd(x: np.ndarray, dt: float, out_png: str, title: str):
    """
    Простейший periodogram: |FFT|^2, ось частоты в Гц (1/ед.времени).
    """
    x = np.asarray(x, float).ravel()
    if len(x) < 4:
        return
    x = x - np.mean(x)
    n = int(2**math.floor(math.log2(len(x))))  # ближайшая степень двойки вниз
    xf = np.fft.rfft(x[:n] * np.hanning(n))
    psd = (np.abs(xf)**2) / n
    freqs = np.fft.rfftfreq(n, d=dt if dt > 0 else 1.0)
    plt.figure(figsize=(7.2, 4.2))
    plt.semilogy(freqs[1:], psd[1:] + 1e-16)  # пропускаем DC для масштаба
    plt.xlabel("frequency"); plt.ylabel("PSD")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=170); plt.close()
    print(f"[ok] saved: {out_png}")

# ----------------- основной поток -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Путь к CSV (T x D). Если первая колонка — время, используйте --time_col.")
    ap.add_argument("--cols", type=str, default=None, help="Список столбцов по индексам/именам, напр. '0,1' или 'x,y'. По умолчанию — все.")
    ap.add_argument("--time_col", type=str, default=None, help="Имя или индекс колонки времени (иначе время = 0..T-1).")
    ap.add_argument("--normalize", type=str, default="none", choices=["none", "zscore", "minmax"], help="Нормализация признаков.")
    ap.add_argument("--smooth", type=int, default=1, help="Окно сглаживания (moving average). 1 — выкл.")
    ap.add_argument("--downsample", type=int, default=1, help="Шаг даунсемплинга по времени (>=1).")
    ap.add_argument("--takens_m", type=int, default=3, help="Размерность Такенса (для 1D сигнала).")
    ap.add_argument("--takens_tau", type=int, default=2, help="Задержка Такенса.")
    ap.add_argument("--plot3d", action="store_true", help="Делать 3D-фазовый портрет (если возможно).")
    ap.add_argument("--do_pca", action="store_true", help="Строить PCA 2D/3D для многомерного ряда.")
    ap.add_argument("--dt", type=float, default=1.0, help="Шаг дискретизации (если нет явной временной колонки).")
    ap.add_argument("--do_psd", action="store_true", help="Строить спектр (PSD) для первой выбранной колонки.")
    ap.add_argument("--outdir", type=str, default="results/raw_plots")
    ap.add_argument("--title", type=str, default=None)
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = read_csv_smart(args.data)

    # Время
    time = None
    if args.time_col is not None:
        # парсим индекс/имя
        try:
            t_idx = int(args.time_col) if str(args.time_col).isdigit() else df.columns.get_loc(args.time_col)
        except Exception:
            raise ValueError(f"--time_col '{args.time_col}' не найден")
        time = df.iloc[:, t_idx].to_numpy()
        Xdf = df.drop(df.columns[t_idx], axis=1)
    else:
        Xdf = df
        time = np.arange(len(df), dtype=float)

    # Столбцы данных
    idxs = parse_cols(Xdf, args.cols)
    X = Xdf.iloc[:, idxs].to_numpy()  # [T, Dsel]
    names = [str(Xdf.columns[i]) for i in idxs]

    # Предобработка
    X = normalize_arr(X, args.normalize)
    X = smooth_matrix(X, args.smooth)
    if args.downsample > 1:
        step = int(args.downsample)
        X = downsample(X, step)
        time = time[::step]

    # Основной заголовок
    title = args.title if args.title else os.path.basename(os.path.normpath(args.data))

    # 1) Временные ряды
    plot_time_series(time, X, names, os.path.join(args.outdir, "timeseries.png"),
                     f"{title} — time series")

    # 2) Фазовый портрет/Такенс
    if X.shape[1] >= 2:
        # 2D фазовый портрет (первая и вторая колонки)
        plot_phase_2d(X[:, 0], X[:, 1],
                      os.path.join(args.outdir, "phase2d.png"),
                      f"{title} — phase portrait (col0 vs col1)")
        # 3D (если запросили и есть >=3 столбцов)
        if args.plot3d and X.shape[1] >= 3:
            plot_phase_3d(X[:, :3], os.path.join(args.outdir, "phase3d.png"),
                          f"{title} — phase 3D (col0,col1,col2)")
    else:
        # 1D сигнал → Такенс
        Xtak = takens_embed_1d(X[:, 0], args.takens_m, args.takens_tau)
        if Xtak.shape[1] >= 2 and len(Xtak) > 1:
            plot_phase_2d(Xtak[:, 0], Xtak[:, 1],
                          os.path.join(args.outdir, "takens2d.png"),
                          f"{title} — Takens (m={args.takens_m}, tau={args.takens_tau})")
        if args.plot3d and Xtak.shape[1] >= 3 and len(Xtak) > 1:
            plot_phase_3d(Xtak[:, :3],
                          os.path.join(args.outdir, "takens3d.png"),
                          f"{title} — Takens 3D (m={args.takens_m}, tau={args.takens_tau})")

    # 3) PCA (для многомерного)
    if args.do_pca and X.shape[1] >= 2:
        Z2 = safe_pca(X, 2)
        if len(Z2) > 1:
            plt.figure(figsize=(5.8, 5.3))
            t = np.linspace(0, 1, len(Z2))
            plt.plot(Z2[:, 0], Z2[:, 1], lw=0.9, alpha=0.8)
            sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=t, s=8, alpha=0.9, cmap=plt.cm.viridis)
            plt.colorbar(sc, label="time")
            plt.title(f"{title} — PCA 2D"); plt.xlabel("PC1"); plt.ylabel("PC2")
            plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "pca2d.png"), dpi=170); plt.close()
            print(f"[ok] saved: {os.path.join(args.outdir, 'pca2d.png')}")
        if args.plot3d:
            Z3 = safe_pca(X, 3)
            if len(Z3) > 1:
                plot_phase_3d(Z3, os.path.join(args.outdir, "pca3d.png"),
                              f"{title} — PCA 3D")

    # 4) Спектр (PSD) для первой выбранной колонки
    if args.do_psd:
        # если явная шкала времени дана, оценим dt по медиане разностей
        if time is not None and len(time) >= 2:
            dt = float(np.median(np.diff(time)))
            if not np.isfinite(dt) or dt <= 0:
                dt = float(args.dt)
        else:
            dt = float(args.dt)
        plot_psd(X[:, 0], dt, os.path.join(args.outdir, "psd_col0.png"),
                 f"{title} — PSD (col0), dt={dt:g}")

if __name__ == "__main__":
    main()
