#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Считает наши метрики (LZ_norm, h_Markov, PermEnt_norm, D2, hKS)
и CKA vs epoch0 для нескольких моделей (run_dirs), кладёт всё в:
  results/<ds>/metrics.csv
  results/<ds>/CKA_vs_epoch0.csv
и рисует:
  compare_LZ_norm.png, compare_Hmarkov.png, compare_PermEnt_norm.png,
  compare_D2.png, compare_hKS.png, CKA_vs_epoch0.png

Ожидаемый формат стейтов: в каждой папке run_dir лежат файлы
states_epoch_<E>.npz с ключом 'H_seq' (список окон T×d).
"""
import math
import os, re, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.signal import welch
import matplotlib.pyplot as plt

# ---------------------- утилиты загрузки ----------------------
EPOCH_RE = re.compile(r"states_epoch_(\d+)\.npz$")

def load_epoch_states(path_npz):
    z = np.load(path_npz, allow_pickle=True)
    H_seq = z["H_seq"]  # dtype=object (список окон)
    # конкатенируем окна по времени: [sum_T, d]
    mats = []
    for H in H_seq:
        H = np.asarray(H)
        if H.ndim != 2: continue
        mats.append(H)
    if not mats:
        return None
    X = np.vstack(mats).astype(np.float32, copy=False)
    return X

def load_all_epochs(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")))
    out = []
    for f in files:
        m = EPOCH_RE.search(os.path.basename(f))
        if not m: continue
        ep = int(m.group(1))
        X = load_epoch_states(f)
        if X is None: continue
        out.append((ep, X))
    out.sort(key=lambda t: t[0])
    return out  # list of (epoch, X: [N,d])

# ---------------------- метрики символико-топ. ----------------------
def pca2(X, max_points=50000, seed=0):
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_points:
        idx = rng.choice(X.shape[0], size=max_points, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    # центрирование
    Xs = Xs - Xs.mean(axis=0, keepdims=True)
    p = PCA(n_components=2, random_state=seed).fit(Xs)
    Z2 = p.transform(Xs)
    pc1_full = (X - X.mean(axis=0, keepdims=True)) @ p.components_.T[:,0]
    return Z2, pc1_full

def kmeans_labels(Z, k=16, seed=0):
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    y = km.fit_predict(Z)
    return y

def lz_complexity_norm(labels, K):
    """
    LZ76 на дискретной последовательности, нормированная оценка скорости энтропии:
    C_hat = c(n)*log(n)/(n*log(K))
    """
    s = np.array(labels, dtype=int).tolist()
    n = len(s)
    if n < 4 or K < 2:
        return np.nan
    # классический LZ76 (кол-во "фраз")
    i, c, k, l = 0, 1, 1, 1
    while True:
        if i + k >= n:
            c += 1
            break
        if s[i:i+k] == s[l:l+k]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            i += 1
            if i == l:
                c += 1
                l += k
                if l + 1 > n:
                    break
                i, k = 0, 1
            else:
                k = 1
    return (c * np.log(n)) / (n * np.log(K))

def markov_entropy_rate(labels, K):
    """ h_Markov = - sum_i pi_i sum_j P_ij log2 P_ij """
    lbl = np.asarray(labels, dtype=int)
    if lbl.size < 4: return np.nan
    P = np.zeros((K, K), dtype=np.float64)
    for a, b in zip(lbl[:-1], lbl[1:]):
        if 0 <= a < K and 0 <= b < K:
            P[a, b] += 1.0
    row_sum = P.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.divide(P, row_sum, where=row_sum>0)
    # стационарка по частотам посещения
    pi = np.zeros(K, dtype=np.float64)
    for a in lbl:
        if 0 <= a < K: pi[a] += 1
    pi = pi / max(1.0, pi.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        Hrows = -np.nansum(P * (np.log2(P + 1e-12)), axis=1)
    return float(np.nansum(pi * Hrows))

def perm_entropy_norm(x, m=5, tau=1):
    """ Перестановочная энтропия (норм.), по Bandt-Pompe. """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < (m-1)*tau + 2: return np.nan
    patterns = {}
    for i in range(0, n - (m-1)*tau):
        window = x[i:i+tau*m:tau]
        order = tuple(np.argsort(window))
        patterns[order] = patterns.get(order, 0) + 1
    counts = np.array(list(patterns.values()), dtype=float)
    p = counts / counts.sum()
    H = -np.sum(p * np.log2(p + 1e-12))
    Hmax = np.log2(math.factorial(m))
    return H / Hmax

def corr_dimension(Z2, r_min_q=0.1, r_max_q=0.9, n_r=20, theiler=2):
    """
    Корреляционная размерность D2 по корреляционной сумме:
      C(r) = (1/(N^2)) * #{(i,j): ||x_i-x_j||<r, |i-j|>Theiler}
    Возвращаем наклон регрессии log C(r) ~ D2 log r на среднем участке.
    """
    if Z2.shape[0] < 100:
        return np.nan
    D = squareform(pdist(Z2, metric="euclidean"))
    N = D.shape[0]
    # Theiler window
    for i in range(N):
        D[i, max(0, i-theiler):i+theiler+1] = np.inf
    # радиусы по квантилям дистанций (исключая inf)
    flat = D[np.isfinite(D)]
    if flat.size < 10: return np.nan
    rmin = np.quantile(flat, r_min_q)
    rmax = np.quantile(flat, r_max_q)
    if not np.isfinite(rmin) or not np.isfinite(rmax) or rmin <= 0 or rmax <= rmin:
        return np.nan
    radii = np.exp(np.linspace(np.log(rmin), np.log(rmax), n_r))
    Cs = []
    for r in radii:
        Cs.append(np.count_nonzero(flat < r) / (N*N))
    Cs = np.array(Cs) + 1e-12
    # берём средний участок (исключим крайние 20%)
    s = slice(int(0.2*n_r), int(0.8*n_r))
    xs = np.log(radii[s])
    ys = np.log(Cs[s])
    # линрегресс без scipy.stats
    A = np.vstack([xs, np.ones_like(xs)]).T
    try:
        slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        return float(slope)
    except Exception:
        return np.nan

def ks_entropy_estimate(labels, n_max=6):
    """
    h_KS ~ slope(H(n)) с коррекцией Миллера–Мэдау и монотонизацией H(n).
    """
    lbl = np.asarray(labels, dtype=int)
    n = len(lbl)
    if n < 200:
        return np.nan
    Hn = []
    for m in range(1, n_max+1):
        # частоты блоков длины m
        counts = {}
        for i in range(0, n - m + 1):
            key = tuple(lbl[i:i+m])
            counts[key] = counts.get(key, 0) + 1
        cnt = np.array(list(counts.values()), dtype=float)
        N = cnt.sum()
        p = cnt / N
        Hm = -np.sum(p * np.log2(p + 1e-12))
        # Miller-Madow correction ~ (S-1)/(2N ln 2)
        S = len(cnt)
        Hm += (S - 1) / (2 * N * np.log(2))
        Hn.append(Hm)
    Hn = np.maximum.accumulate(Hn)  # монотонизация
    x = np.arange(1, len(Hn)+1, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        slope, _ = np.linalg.lstsq(A, Hn, rcond=None)[0]
        return max(0.0, float(slope))
    except Exception:
        return np.nan

# ---------------------- CKA (линейная) ----------------------
def center(X):
    return X - X.mean(axis=0, keepdims=True)

def hsic_linear(X, Y):
    # HSIC для линейных ядер: tr(X X^T Y Y^T)
    return np.sum((X @ X.T) * (Y @ Y.T))

def linear_cka(X, Y, max_points=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = min(X.shape[0], Y.shape[0], max_points)
    if X.shape[0] > n:
        ix = rng.choice(X.shape[0], n, replace=False)
        X = X[ix]
    if Y.shape[0] > n:
        iy = rng.choice(Y.shape[0], n, replace=False)
        Y = Y[iy]
    Xc = center(X.astype(np.float64, copy=False))
    Yc = center(Y.astype(np.float64, copy=False))
    hsic_xy = hsic_linear(Xc, Yc)
    hsic_xx = hsic_linear(Xc, Xc)
    hsic_yy = hsic_linear(Yc, Yc)
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    return float(hsic_xy / denom)

# ---------------------- основной расчёт ----------------------
def analyze_runs(run_dirs, model_names, outdir, k=16, seed=0,
                 max_points_pca=50000, max_points_cka=20000,
                 theiler=2, m_perm=5, takens=False):
    """
    run_dirs/model_names — одинаковой длины списки.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # загрузим эпохи по моделям
    epochs_all = {}
    data_all = {}
    for rd, name in zip(run_dirs, model_names):
        pairs = load_all_epochs(rd)
        if not pairs:
            print(f"[warn] {name}: нет states_epoch_*.npz в {rd}")
            continue
        epochs_all[name] = [ep for ep,_ in pairs]
        data_all[name]   = {ep:X for ep,X in pairs}

    # общий список эпох (пересечение по всем моделям)
    if not data_all:
        raise RuntimeError("Не найдено стейтов ни в одном run_dir.")
    common_epochs = None
    for name in data_all:
        eps = set(data_all[name].keys())
        common_epochs = eps if common_epochs is None else (common_epochs & eps)
    epochs = sorted(common_epochs)
    if not epochs:
        raise RuntimeError("Нет общих эпох между моделями.")

    # подготовим контейнер CSV
    rows = []
    # репрезентации для CKA: хранить по моделям
    repr_by_model = {name:{} for name in model_names}

    for ep in epochs:
        row = {"epoch": ep}
        for name in model_names:
            if ep not in data_all[name]:
                # пропустим, если у модели нет этой эпохи
                continue
            X = data_all[name][ep]  # [N,d]
            # PCA2 + PC1 (для PermEnt)
            Z2, pc1 = pca2(X, max_points=max_points_pca, seed=seed)
            # символизация — kmeans по Z2
            labels = kmeans_labels(Z2, k=k, seed=seed)

            # метрики
            lz = lz_complexity_norm(labels, K=k)
            hM = markov_entropy_rate(labels, K=k)
            pEnt = perm_entropy_norm(pc1, m=m_perm, tau=1)
            D2 = corr_dimension(Z2, theiler=theiler)
            hKS = ks_entropy_estimate(labels, n_max=6)

            row[f"{name}_LZ_norm"] = lz
            row[f"{name}_Hmarkov"] = hM
            row[f"{name}_PermEnt_norm"] = pEnt
            row[f"{name}_D2"] = D2
            row[f"{name}_hKS"] = hKS

            # сохраним компактную репрезентацию для CKA
            # используем исходные скрытые состояния с центровкой/даунсэмплингом
            rng = np.random.default_rng(seed)
            n = min(X.shape[0], max_points_cka)
            if X.shape[0] > n:
                ix = rng.choice(X.shape[0], n, replace=False)
                Xi = X[ix]
            else:
                Xi = X
            repr_by_model[name][ep] = Xi.astype(np.float32, copy=False)

        rows.append(row)

    # metrics.csv
    df = pd.DataFrame(rows).sort_values("epoch")
    df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    print(f"[ok] metrics.csv -> {outdir}")

    # CKA vs epoch0 для каждой модели
    cka_rows = []
    ep0 = epochs[0]
    for name in model_names:
        if ep0 not in repr_by_model[name]:
            continue
        X0 = repr_by_model[name][ep0]
        for ep in epochs:
            Xe = repr_by_model[name].get(ep)
            if Xe is None:
                continue
            cka = linear_cka(X0, Xe, max_points=max_points_cka, seed=seed)
            cka_rows.append({"epoch": ep, f"{name}_CKA_vs_epoch0": cka})
    # Скомбинируем по столбцам
    df_cka = pd.DataFrame(cka_rows).groupby("epoch").first().reset_index()
    df_cka.to_csv(os.path.join(outdir, "CKA_vs_epoch0.csv"), index=False)
    print(f"[ok] CKA_vs_epoch0.csv -> {outdir}")

    # ----------- графики -----------
    def plot_metric(key_suffix, fname, ylabel):
        plt.figure(figsize=(7,4))
        for name in model_names:
            col = f"{name}_{key_suffix}"
            if col in df.columns:
                plt.plot(df["epoch"], df[col], label=name, linewidth=2)
        plt.xlabel("epoch"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close()

    plot_metric("LZ_norm", "compare_LZ_norm.png", r"$\widehat{C}_{\mathrm{LZ}}$")
    plot_metric("Hmarkov", "compare_Hmarkov.png", r"$h_{\mathrm{Markov}}$")
    plot_metric("PermEnt_norm", "compare_PermEnt_norm.png", "Permutation Entropy (norm.)")
    plot_metric("D2", "compare_D2.png", r"$D_2$")
    plot_metric("hKS", "compare_hKS.png", r"$h_{\mathrm{KS}}$")

    # CKA plot
    if not df_cka.empty:
        plt.figure(figsize=(7,4))
        for name in model_names:
            col = f"{name}_CKA_vs_epoch0"
            if col in df_cka.columns:
                plt.plot(df_cka["epoch"], df_cka[col], label=name, linewidth=2)
        plt.xlabel("epoch"); plt.ylabel("CKA vs epoch0"); plt.ylim(0,1.02); plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "CKA_vs_epoch0.png"), dpi=150)
        plt.close()

    print("[ok] plots saved in", outdir)

# ---------------------- CLI ----------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True,
                    help="Список папок с states_epoch_*.npz (по одной на модель)")
    ap.add_argument("--model_names", nargs="+", default=None,
                    help="Отображаемые имена моделей (если не заданы — имена папок)")
    ap.add_argument("--outdir", required=True, help="Куда складывать результаты")
    ap.add_argument("--k", type=int, default=16, help="Число символов (k-means)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_points_pca", type=int, default=50000)
    ap.add_argument("--max_points_cka", type=int, default=20000)
    ap.add_argument("--theiler", type=int, default=2)
    ap.add_argument("--m_perm", type=int, default=5)
    ap.add_argument("--takens", action="store_true",
                    help="(зарезервировано, сейчас PCA2; Takens можно добавить позже)")
    args = ap.parse_args()

    if args.model_names is None or len(args.model_names) != len(args.run_dirs):
        args.model_names = [Path(rd).name for rd in args.run_dirs]

    analyze_runs(run_dirs=args.run_dirs,
                 model_names=args.model_names,
                 outdir=args.outdir,
                 k=args.k, seed=args.seed,
                 max_points_pca=args.max_points_pca,
                 max_points_cka=args.max_points_cka,
                 theiler=args.theiler,
                 m_perm=args.m_perm,
                 takens=args.takens)
