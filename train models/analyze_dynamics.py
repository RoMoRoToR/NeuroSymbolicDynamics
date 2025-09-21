# analyze_dynamics.py
# -*- coding: utf-8 -*-
r"""
Анализ скрытой динамики по чекпоинтам эпох для RNN/Transformer_old.

Что делает:
- Сканирует папки run_dir/*/states_epoch_XXX.npz (или run_dir/states_epoch_XXX.npz)
- Для каждого эпох-чекпоинта: загружает H_seq (скрытые состояния), конкатенирует все окна по времени
- Делает PCA(2D) для визуализации траектории
- Символизация: k-means по скрытым состояниям → строка меток
- Метрики: Lempel–Ziv complexity (raw и нормированная), permutation entropy, block entropies H(n) и оценка h_KS, энтропия Марковской цепи (по матрице переходов), оценка корреляционной размерности D2
- Сохраняет per-epoch фигуры (траектория, heatmap P)
- Собирает GIF по эпохам (опция)
- Собирает CSV со всеми метриками и строит графики метрик vs эпоха (для нескольких моделей — на одном графике)

Зависимости: numpy, pandas, scikit-learn, matplotlib, imageio
Установка: pip install numpy pandas scikit-learn matplotlib imageio

Пример запуска (Windows PowerShell):
python .\analyze_dynamics.py `
  --run_dirs "runs\lorenz\RNN_LSTM" "runs\lorenz\Transformer_old" `
  --outdir "results\lorenz" `
  --k 16 `
  --make_gif
"""

import os, re, glob, math, argparse, json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import imageio.v2 as imageio

# -----------------------------
# Утилиты
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    if not m:
        return -1
    return int(m.group(1))


# -----------------------------
# Метрики символической динамики
# -----------------------------

def lz_complexity_int(seq: np.ndarray) -> int:
    """Классическая LZ76 (очень простая реализация для целочисленной последовательности).
    Возвращает число фраз (complexity c(n)).
    """
    s = seq.tolist()
    n = len(s)
    if n == 0:
        return 0
    c = 1
    i = 0
    k = 1
    l = 1
    while True:
        if i + k > n:
            c += 0
            break
        sub = s[i:i+k]
        found = False
        # поиск подстроки sub в префиксе s[0:i]
        for j in range(0, i):
            if s[j:j+k] == sub:
                found = True
                break
        if found:
            k += 1
            if i + k > n:
                c += 1
                break
        else:
            c += 1
            i += k
            k = 1
            if i + 1 > n:
                break
    return c


def lz_normalized(seq: np.ndarray) -> float:
    """Нормировка c(n) / (n / log n) — популярная аппроксимация.
    Возвращает 0, если n < 2 для устойчивости.
    """
    n = len(seq)
    if n < 2:
        return 0.0
    c = lz_complexity_int(seq)
    return c / (n / math.log(n))


def ngram_entropy(labels: np.ndarray, n: int) -> float:
    if len(labels) < n:
        return 0.0
    from collections import Counter
    tuples = [tuple(labels[i:i+n]) for i in range(len(labels)-n+1)]
    cnt = Counter(tuples)
    total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], dtype=float)
    # Shannon entropy (log2)
    eps = 1e-12
    return float((-probs * np.log2(probs + eps)).sum())


def block_entropies(labels: np.ndarray, nmax: int = 6) -> Tuple[List[int], List[float]]:
    ns, Hs = [], []
    for n in range(1, nmax+1):
        ns.append(n)
        Hs.append(ngram_entropy(labels, n))
    return ns, Hs


def ks_entropy_from_blocks(ns: List[int], Hs: List[float]) -> float:
    """Оценка h_KS как наклон H(n) по n (линейная регрессия на хвосте)."""
    if len(ns) < 2:
        return 0.0
    x = np.array(ns, dtype=float)
    y = np.array(Hs, dtype=float)
    # берём от n=2 до n_max (если точек мало — берём как есть)
    mask = x >= max(2, int(np.floor(len(ns)/2)))
    if mask.sum() < 2:
        mask = x >= 2
    X = np.vstack([x[mask], np.ones(mask.sum())]).T
    a, b = np.linalg.lstsq(X, y[mask], rcond=None)[0]
    return float(a)


def transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    P = np.zeros((K, K), dtype=float)
    for i in range(len(labels)-1):
        P[labels[i], labels[i+1]] += 1.0
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums>0)
    # небольшая сглаживающая добавка, чтобы избежать -inf при логах
    P = (P + 1e-8)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def stationary_dist(P: np.ndarray, tol: float = 1e-10, iters: int = 10000) -> np.ndarray:
    K = P.shape[0]
    pi = np.ones(K, dtype=float) / K
    for _ in range(iters):
        new = pi @ P
        if np.linalg.norm(new - pi, 1) < tol:
            return new
        pi = new
    return pi


def markov_entropy_rate(P: np.ndarray) -> float:
    pi = stationary_dist(P)
    eps = 1e-12
    h = 0.0
    for i in range(P.shape[0]):
        row = P[i]
        h += pi[i] * (-(row * np.log2(row + eps)).sum())
    return float(h)


def permutation_entropy(x: np.ndarray, m: int = 5, tau: int = 1) -> float:
    """Permutation entropy (Bandt & Pompe). Возвращает H_perm в битах.
    Нормализованная версия = H_perm / log2(m!).
    """
    n = len(x)
    if n < (m - 1) * tau + 1:
        return 0.0
    # формируем векторы длины m, с лагом tau
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = x[i:(i + m * tau):tau]
        order = np.argsort(window, kind='quicksort')
        patterns.append(tuple(order))
    from collections import Counter
    cnt = Counter(patterns)
    total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], dtype=float)
    eps = 1e-12
    H = float((-probs * np.log2(probs + eps)).sum())
    return H


def correlation_dimension(X: np.ndarray, quantiles=(0.1, 0.6), n_eps=10) -> float:
    """Очень грубая оценка D2 по корреляционной сумме (анализ скейлинга).
    Берём сетку eps по квантилям расстояний.
    """
    if len(X) < 5:
        return 0.0
    # попарные расстояния (евклидовы)
    from scipy.spatial.distance import pdist
    d = pdist(X, metric='euclidean')
    if len(d) == 0:
        return 0.0
    lo = np.quantile(d, quantiles[0])
    hi = np.quantile(d, quantiles[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0 or lo <= 0 or hi <= lo:
        return 0.0
    eps_grid = np.geomspace(lo, hi, num=n_eps)
    C = []
    for eps in eps_grid:
        C.append((d < eps).mean())
    x = np.log(eps_grid)
    y = np.log(np.maximum(C, 1e-12))
    # линейная регрессия на центральной части
    mid = slice(max(1, n_eps//4), -max(1, n_eps//4))
    if n_eps < 6:
        mid = slice(1, -1)
    Xmat = np.vstack([x[mid], np.ones_like(x[mid])]).T
    a, b = np.linalg.lstsq(Xmat, y[mid], rcond=None)[0]
    return float(a)


# -----------------------------
# Основной анализ по одной модели (run_dir)
# -----------------------------

def analyze_run(run_dir: str, outdir: str, k: int = 16, nmax_block: int = 6,
                perm_m: int = 5, perm_tau: int = 1, make_gif: bool = False,
                model_name: str = None, which_seq: int = None) -> pd.DataFrame:
    """Возвращает таблицу метрик по эпохам и сохраняет графики/гифки в outdir/model_name."""
    # собираем все npz
    pattern_a = os.path.join(run_dir, "states_epoch_*.npz")
    pattern_b = os.path.join(run_dir, "*", "states_epoch_*.npz")
    files = glob.glob(pattern_a) + glob.glob(pattern_b)
    files = sorted(files, key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Не найдены npz-файлы в {run_dir}")

    if model_name is None:
        model_name = os.path.basename(os.path.normpath(run_dir))

    od = os.path.join(outdir, model_name)
    figs_dir = os.path.join(od, "figs")
    ensure_dir(figs_dir)

    rows = []
    frame_paths_traj = []
    frame_paths_heat = []

    for f in files:
        epoch = parse_epoch_from_name(f)
        dat = np.load(f, allow_pickle=True)
        H_list = dat['H_seq']  # это список массивов [T, D]
        # конкатенируем все окна по времени (или берём конкретное which_seq)
        if which_seq is not None and 0 <= which_seq < len(H_list):
            H = H_list[which_seq]
        else:
            H = np.concatenate([np.array(h, dtype=float) for h in H_list], axis=0)
        H = np.asarray(H, dtype=float)
        T, D = H.shape

        # PCA для 2D-визуализации и для perm-entropy по PC1
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(H)
        pc1 = Z[:, 0]

        # кластеризация для символизации
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(H)

        # матрица переходов и энтропия Маркова
        P = transition_matrix(labels, k)
        H_markov = markov_entropy_rate(P)

        # Lempel–Ziv
        LZ_raw = float(lz_complexity_int(labels))
        LZ_norm = float(lz_normalized(labels))

        # Block entropies и h_KS
        ns, Hs = block_entropies(labels, nmax=nmax_block)
        h_KS = ks_entropy_from_blocks(ns, Hs)

        # Permutation entropy по PC1 (норм. и ненорм.)
        H_perm = permutation_entropy(pc1, m=perm_m, tau=perm_tau)
        H_perm_norm = H_perm / math.log2(math.factorial(perm_m)) if perm_m >= 2 else 0.0

        # Корреляционная размерность (по Z для устойчивости масштаба)
        try:
            D2 = correlation_dimension(Z)
        except Exception:
            D2 = 0.0

        # Фигуры для GIF
        # 1) траектория в PCA2D, цвет — кластеры
        fig1 = plt.figure(figsize=(5, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=8)
        plt.title(f"{model_name}: PCA traj (epoch {epoch})")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        path_traj = os.path.join(figs_dir, f"traj_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(path_traj, dpi=150); plt.close(fig1)

        # 2) heatmap матрицы переходов
        fig2 = plt.figure(figsize=(5, 4))
        plt.imshow(P, aspect='auto')
        plt.colorbar(label='P(i→j)')
        plt.title(f"{model_name}: Transition P (epoch {epoch})")
        plt.xlabel("j"); plt.ylabel("i")
        path_heat = os.path.join(figs_dir, f"P_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(path_heat, dpi=150); plt.close(fig2)

        frame_paths_traj.append(path_traj)
        frame_paths_heat.append(path_heat)

        # собираем строку метрик
        rows.append({
            "model": model_name,
            "epoch": epoch,
            "T_points": int(T),
            "k": int(k),
            "LZ_raw": LZ_raw,
            "LZ_norm": LZ_norm,
            "H_markov": H_markov,
            "H_perm": H_perm,
            "H_perm_norm": H_perm_norm,
            "h_KS": h_KS,
            "D2": D2,
            "H_blocks": json.dumps({int(n): float(h) for n, h in zip(ns, Hs)})
        })

    # сохраняем GIF'ы
    if make_gif and len(frame_paths_traj) > 1:
        gif_traj = os.path.join(od, "pca_traj.gif")
        with imageio.get_writer(gif_traj, mode='I', duration=0.8) as w:
            for p in frame_paths_traj:
                w.append_data(imageio.imread(p))
    if make_gif and len(frame_paths_heat) > 1:
        gif_heat = os.path.join(od, "P_heatmap.gif")
        with imageio.get_writer(gif_heat, mode='I', duration=0.8) as w:
            for p in frame_paths_heat:
                w.append_data(imageio.imread(p))

    df = pd.DataFrame(rows).sort_values(["epoch"]).reset_index(drop=True)
    # сохраняем CSV и графики метрик
    ensure_dir(od)
    csv_path = os.path.join(od, "metrics.csv")
    df.to_csv(csv_path, index=False)

    # Графики метрик vs эпоха
    def plot_metric(name: str, ylabel: str, fname: str):
        fig = plt.figure(figsize=(6, 4))
        plt.plot(df['epoch'], df[name], marker='o')
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(f"{model_name} — {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(od, fname), dpi=150); plt.close(fig)

    plot_metric('LZ_norm', 'LZ (norm)', 'LZ_norm_vs_epoch.png')
    plot_metric('H_markov', 'Entropy rate (bits/step)', 'Hmarkov_vs_epoch.png')
    plot_metric('h_KS', 'h_KS (slope of H(n))', 'hKS_vs_epoch.png')
    plot_metric('H_perm_norm', 'Permutation entropy (norm)', 'PermEnt_norm_vs_epoch.png')
    plot_metric('D2', 'Correlation dimension D2', 'D2_vs_epoch.png')

    return df


# -----------------------------
# Сравнительные графики для нескольких моделей
# -----------------------------

def plot_compare(dfs: List[pd.DataFrame], outdir: str, metric: str, ylabel: str, fname: str):
    fig = plt.figure(figsize=(7, 4))
    for df in dfs:
        name = str(df['model'].iloc[0])
        plt.plot(df['epoch'], df[metric], marker='o', label=name)
    plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=150); plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dirs', nargs='+', required=True, help='Одна или несколько папок модели с states_epoch_*.npz')
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--nmax_block', type=int, default=6)
    ap.add_argument('--perm_m', type=int, default=5)
    ap.add_argument('--perm_tau', type=int, default=1)
    ap.add_argument('--make_gif', action='store_true')
    ap.add_argument('--which_seq', type=int, default=None, help='Если задано — брать только это окно из H_seq; иначе конкатенировать все')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    dfs = []
    for rd in args.run_dirs:
        df = analyze_run(rd, args.outdir, k=args.k, nmax_block=args.nmax_block,
                         perm_m=args.perm_m, perm_tau=args.perm_tau,
                         make_gif=args.make_gif, model_name=None,
                         which_seq=args.which_seq)
        dfs.append(df)

    # Совмещённые графики метрик (если моделей > 1)
    if len(dfs) >= 1:
        plot_compare(dfs, args.outdir, 'LZ_norm', 'LZ (norm)', 'compare_LZ_norm.png')
        plot_compare(dfs, args.outdir, 'H_markov', 'Entropy rate (bits/step)', 'compare_Hmarkov.png')
        plot_compare(dfs, args.outdir, 'h_KS', 'h_KS (slope of H(n))', 'compare_hKS.png')
        plot_compare(dfs, args.outdir, 'H_perm_norm', 'Permutation entropy (norm)', 'compare_PermEnt_norm.png')
        plot_compare(dfs, args.outdir, 'D2', 'Correlation dimension D2', 'compare_D2.png')

if __name__ == '__main__':
    main()
