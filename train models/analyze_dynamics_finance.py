# -*- coding: utf-8 -*-
r"""
Повторный анализ скрытой динамики с робастными h_KS и D2 и отрисовкой по кластерам.

Что делает:
- Грузит states_epoch_*.npz (ключ 'H_seq') из нескольких run_dirs (RNN, BiLSTM, Transformer).
- (Опц.) Takens: подбирает tau (по первому минимуму AMI) и m (FNN) по PC1 исходной траектории.
- Символизация: k-means по выбранному источнику (H или Takens-вложение) со стандартизацией.
- Метрики: LZ_norm, Markov entropy rate, permutation entropy (norm), h_KS (MM-хвост + fallback), D2 (Takens).
- Визуализации:
    * barplot частот кластеров (вместо точек);
    * теплокарта матрицы переходов P;
    * сравнение метрик по моделям (png);
    * (опц.) GIF теплокарт по эпохам (скорость --gif_duration).
"""

import os, re, glob, math, argparse, json
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree
import imageio.v2 as imageio

# =============================
# Утилиты
# =============================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def safe_json_load(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

# =============================
# Символьные метрики
# =============================

def lz_complexity_int(seq: np.ndarray) -> int:
    """Очень простая LZ76 (число фраз) для целочисленной последовательности."""
    s = seq.tolist(); n = len(s)
    if n == 0: return 0
    c, i, k = 1, 0, 1
    while True:
        if i + k > n: break
        sub = s[i:i+k]
        found = any(s[j:j+k] == sub for j in range(0, i))
        if found:
            k += 1
            if i + k > n:
                c += 1; break
        else:
            c += 1; i += k; k = 1
            if i + 1 > n: break
    return c

def lz_normalized(seq: np.ndarray) -> float:
    n = len(seq)
    if n < 2: return 0.0
    return lz_complexity_int(seq) / (n / math.log(n))

def transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    P = np.zeros((K, K), float)
    for i in range(len(labels)-1):
        P[labels[i], labels[i+1]] += 1.0
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums>0)
    P = (P + 1e-12)
    P = P / P.sum(axis=1, keepdims=True)
    return P

def stationary_dist(P: np.ndarray, tol: float = 1e-10, iters: int = 10000) -> np.ndarray:
    K = P.shape[0]
    pi = np.ones(K) / K
    for _ in range(iters):
        new = pi @ P
        if np.linalg.norm(new - pi, 1) < tol: return new
        pi = new
    return pi

def markov_entropy_rate(P: np.ndarray) -> float:
    pi = stationary_dist(P)
    h = 0.0
    for i in range(P.shape[0]):
        row = P[i]
        h += pi[i] * (-(row * np.log2(row + 1e-12)).sum())
    return float(h)

def permutation_entropy(x: np.ndarray, m: int = 5, tau: int = 1) -> float:
    """Permutation entropy (Bandt & Pompe) в битах."""
    n = len(x)
    if n < (m - 1) * tau + 1: return 0.0
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = x[i:(i + m * tau):tau]
        patterns.append(tuple(np.argsort(window, kind='quicksort')))
    cnt = Counter(patterns); total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], float)
    return float((-probs * np.log2(probs + 1e-12)).sum())

# ------ Блочные энтропии + h_KS (робастно) ------

def _ngram_counts(labels: np.ndarray, n: int):
    if len(labels) < n:
        return Counter(), 0, 0
    tuples = [tuple(labels[i:i+n]) for i in range(len(labels)-n+1)]
    cnt = Counter(tuples)
    return cnt, len(tuples), len(cnt)

def ngram_entropy(labels: np.ndarray, n: int) -> float:
    """MLE-энтропия слов длины n (для fallback; без учёта не наблюдавшихся слов)."""
    cnt, Nn, _ = _ngram_counts(labels, n)
    if Nn <= 0: return 0.0
    p = np.array([v / Nn for v in cnt.values()], float)
    return float((-p * np.log2(p + 1e-12)).sum())

def block_entropies_mm(labels: np.ndarray, nmax: int = 8,
                       min_samples: int = 1000, min_unique: int = 30) -> Tuple[List[int], List[float]]:
    """H_n с поправкой Миллера–Мэдоу. Прерываемся, если n-грамм мало/короткий ряд."""
    ns, Hs = [], []
    for n in range(1, nmax+1):
        cnt, Nn, kn = _ngram_counts(labels, n)
        if Nn < min_samples or kn < min_unique:
            break
        p = np.array([v / Nn for v in cnt.values()], float)
        H_mle = float((-p * np.log2(p + 1e-12)).sum())
        H_mm  = H_mle + (max(kn,1) - 1) / (2.0 * Nn * math.log(2.0))  # Miller–Madow
        ns.append(n); Hs.append(H_mm)
    return ns, Hs

def ks_entropy_from_blocks_tail(ns: List[int], Hs: List[float], tail: int = 3) -> float:
    """Наклон хвоста H(n): h ≈ average ΔH на последних tail шагах (монотонизированная H)."""
    if len(ns) < 2:
        return 0.0
    Hs = np.maximum.accumulate(np.asarray(Hs, float))  # делаем монотонной
    d  = np.diff(Hs)
    k  = int(max(1, min(tail, len(d))))
    return float(max(0.0, np.mean(d[-k:])))

# =============================
# Takens: AMI (tau), FNN (m), D2 на реконструкции
# =============================

def delayed_mutual_information(x: np.ndarray, max_tau: int = 50, bins: int = 64) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    I = []
    hist_x, edges = np.histogram(x, bins=bins, density=True)
    p_x = hist_x / (hist_x.sum() + 1e-12)
    for tau in range(1, max_tau+1):
        x1 = x[:-tau]; x2 = x[tau:]
        H, _, _ = np.histogram2d(x1, x2, bins=bins, density=True)
        p12 = H / (H.sum() + 1e-12)
        p2, _ = np.histogram(x2, bins=edges, density=True)
        p2 = p2 / (p2.sum() + 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            logterm = np.log((p12 + 1e-12) / ((p_x[:, None] * p2[None, :]) + 1e-12))
            I_tau = float(np.nansum(p12 * logterm))
        I.append(I_tau)
    return np.array(I)

def choose_tau_by_first_min_ami(x: np.ndarray, max_tau: int = 50, bins: int = 64) -> int:
    I = delayed_mutual_information(x, max_tau=max_tau, bins=bins)
    for t in range(1, len(I)-1):
        if I[t] < I[t-1] and I[t] <= I[t+1]:
            return t+1
    return int(max(1, np.argmin(I)+1))

def false_nearest_neighbors(pc1: np.ndarray, tau: int, m_max: int = 10,
                            Rtol: float = 15.0, Atol: float = 2.0) -> int:
    x = np.asarray(pc1, float).ravel()
    N = len(x) - (m_max * tau)
    if N <= 1: return 3
    def embed(m):
        idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
        return x[idx].T
    from sklearn.neighbors import NearestNeighbors
    fracs = []
    for m in range(1, m_max+1):
        Xm = embed(m)
        nbr = NearestNeighbors(n_neighbors=2).fit(Xm)
        dists, idxs = nbr.kneighbors(Xm, return_distance=True)
        d1 = dists[:, 1] + 1e-12
        if m < m_max:
            Xm1 = embed(m+1)
            j = idxs[:, 1]
            num = np.abs(Xm1[:, -1] - Xm1[j, -1])
            R = num / d1
            A = num / np.std(x)
            fnn = np.mean((R > Rtol) | (A > Atol))
        else:
            fnn = fracs[-1] if fracs else 1.0
        fracs.append(fnn)
        if m >= 2 and fnn < 0.1:
            return m
    return int(np.argmin(fracs) + 1)

def takens_embedding_1d(pc1: np.ndarray, m: int, tau: int) -> np.ndarray:
    x = np.asarray(pc1, float).ravel()
    N = len(x) - (m - 1) * tau
    idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
    return x[idx].T

def correlation_dimension_takens(pc1: np.ndarray, tau: int, m: int,
                                 q_lo=0.05, q_hi=0.95, n_eps=18, theiler=5) -> float:
    """
    D2 по корреляционной сумме на реконструкции Такенса (по pc1).
    """
    x = np.asarray(pc1, float).ravel()
    N = len(x) - (m - 1) * tau
    if N < max(80, 5*m):
        return 0.0
    idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
    X = x[idx].T  # [N, m]

    tree = KDTree(X, metric='euclidean')

    # сетка eps из квантилей ближайших соседей
    samp = min(2000, N)
    sel = np.random.RandomState(0).choice(N, size=samp, replace=False)
    dists, _ = tree.query(X[sel], k=2)
    d1 = dists[:, 1]
    lo = np.quantile(d1, q_lo); hi = np.quantile(d1, q_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.0
    eps_grid = np.geomspace(lo, hi, n_eps)

    counts = []
    W = int(max(0, theiler))
    batch = 2048
    for eps in eps_grid:
        cnt = 0
        den = 0
        for start in range(0, N, batch):
            stop = min(N, start+batch)
            Xi = X[start:stop]
            ind = tree.query_radius(Xi, r=eps, count_only=False, return_distance=False)
            for i_local, neigh in enumerate(ind):
                i = start + i_local
                neigh = neigh[(neigh != i) & (np.abs(neigh - i) > W)]
                cnt += neigh.size
                den += (N - 1 - 2*W) if (N - 1 - 2*W) > 0 else 0
        C = (cnt / max(den, 1)) if den > 0 else 0.0
        counts.append(max(C, 1e-12))

    xlog = np.log(eps_grid); ylog = np.log(np.array(counts))
    n = len(xlog); a = max(1, n//5); b = n - a - 1
    sl = slice(a, b if b > a else n-1)
    Xlr = np.vstack([xlog[sl], np.ones_like(xlog[sl])]).T
    slope, _ = np.linalg.lstsq(Xlr, ylog[sl], rcond=None)[0]
    return float(max(0.0, slope))

# =============================
# Основной анализ одного run_dir
# =============================

def analyze_run(run_dir: str, outdir: str, k: int = 16, nmax_block: int = 8,
                perm_m: int = 5, perm_tau: int = 1, make_gif: bool = False,
                which_seq: int = None, use_takens: bool = True, ami_max_tau: int = 50,
                fnn_mmax: int = 10, hks_min_samples: int = 1000, hks_min_unique: int = 30,
                hks_tail: int = 3, gif_duration: float = 1.2) -> pd.DataFrame:
    """
    Возвращает DataFrame метрик по эпохам; сохраняет фигуры и CSV.
    """
    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Не найдены npz-файлы в {run_dir}")

    model_name = os.path.basename(os.path.normpath(run_dir))
    od = os.path.join(outdir, model_name)
    figs_dir = os.path.join(od, "figs"); ensure_dir(figs_dir)

    rows = []
    frame_paths_heat = []

    for f in files:
        epoch = parse_epoch_from_name(f)
        dat = np.load(f, allow_pickle=True)
        H_list = dat['H_seq']
        H = H_list[which_seq] if (which_seq is not None and 0 <= which_seq < len(H_list)) \
            else np.concatenate([np.asarray(h, float) for h in H_list], axis=0)
        H = np.asarray(H, float)
        T, D = H.shape

        # Источник анализа
        pc1_init = PCA(n_components=2, random_state=0).fit_transform(H)[:, 0]
        src = H
        takens_info = {"used": False}
        if use_takens:
            tau = choose_tau_by_first_min_ami(pc1_init, max_tau=ami_max_tau)
            m   = false_nearest_neighbors(pc1_init, tau=tau, m_max=fnn_mmax)
            src = takens_embedding_1d(pc1_init, m=m, tau=tau)
            takens_info = {"used": True, "tau": int(tau), "m": int(m)}
        else:
            tau = 1; m = 2

        # Символизация (стандартизация → k-means)
        Z = StandardScaler().fit_transform(src)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Z)

        # Матрица переходов и метрики
        P = transition_matrix(labels, k)
        H_markov = markov_entropy_rate(P)
        LZ_raw = float(lz_complexity_int(labels))
        LZ_norm = float(lz_normalized(labels))

        # H(n) с MM + fallback
        ns, Hs = block_entropies_mm(labels, nmax=nmax_block,
                                    min_samples=hks_min_samples, min_unique=hks_min_unique)
        if len(ns) >= 2:
            h_KS = ks_entropy_from_blocks_tail(ns, Hs, tail=hks_tail)
        else:
            # Fallback: как минимум H2-H1; если нельзя — берём марковскую скорость
            H1 = ngram_entropy(labels, 1)
            H2 = ngram_entropy(labels, 2)
            if H2 > 0 and H1 > 0:
                h_KS = max(0.0, H2 - H1)
            else:
                h_KS = float(H_markov)

        # Permutation entropy по PC1 согласованного источника
        pc1_src = PCA(n_components=2, random_state=0).fit_transform(src)[:, 0]
        H_perm = permutation_entropy(pc1_src, m=perm_m, tau=perm_tau)
        H_perm_norm = H_perm / math.log2(math.factorial(perm_m)) if perm_m >= 2 else 0.0

        # Корреляционная размерность D2 — на реконструкции Такенса по pc1_init
        D2 = correlation_dimension_takens(pc1_init, tau=tau, m=m, q_lo=0.05, q_hi=0.95, n_eps=18, theiler=5)

        # --- ВИЗУАЛИЗАЦИИ (без точек) ---
        # (1) Частоты кластеров (barplot)
        cnt = Counter(labels.tolist()); total = max(1, sum(cnt.values()))
        states = sorted(cnt.keys())
        freqs  = [cnt[s]/total for s in states]
        plt.figure(figsize=(6,4))
        plt.bar([str(s) for s in states], freqs)
        ttxt = f"{model_name} — cluster frequencies (epoch {epoch})"
        if takens_info["used"]:
            ttxt += f" | Takens m={takens_info['m']}, τ={takens_info['tau']}"
        plt.title(ttxt); plt.xlabel("Cluster id"); plt.ylabel("Frequency")
        p_bar = os.path.join(figs_dir, f"clusters_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_bar, dpi=150); plt.close()

        # (2) Теплокарта P
        plt.figure(figsize=(5,4))
        plt.imshow(P, aspect='auto', origin='lower')
        plt.colorbar(label='P(i→j)')
        plt.title(f"{model_name}: Transition P (epoch {epoch})")
        plt.xlabel("j"); plt.ylabel("i")
        p_heat = os.path.join(figs_dir, f"P_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_heat, dpi=150); plt.close()
        frame_paths_heat.append(p_heat)

        # Лоссы (если есть epoch_XXX.json рядом)
        json_path = os.path.join(run_dir, f"epoch_{epoch:03d}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(f), f"epoch_{epoch:03d}.json")
        j = safe_json_load(json_path) or {}

        rows.append({
            "model": model_name, "epoch": int(epoch), "T_points": int(T), "k": int(k),
            "LZ_raw": float(LZ_raw), "LZ_norm": float(LZ_norm), "H_markov": float(H_markov),
            "H_perm": float(H_perm), "H_perm_norm": float(H_perm_norm),
            "h_KS": float(h_KS), "D2": float(D2),
            "Takens_used": int(takens_info["used"]),
            "Takens_tau": int(takens_info["tau"]) if takens_info["used"] else 0,
            "Takens_m": int(takens_info["m"]) if takens_info["used"] else 0,
            "val_loss": float(j.get("val_loss", np.nan)), "train_loss": float(j.get("train_loss", np.nan))
        })

    # GIF теплокарт
    if make_gif and len(frame_paths_heat) > 1:
        ensure_dir(od)
        with imageio.get_writer(os.path.join(od, "P_heatmap.gif"), mode='I', duration=gif_duration) as w:
            for p in frame_paths_heat:
                w.append_data(imageio.imread(p))

    # CSV
    df = pd.DataFrame(rows).sort_values(["epoch"]).reset_index(drop=True)
    ensure_dir(od)
    df.to_csv(os.path.join(od, "metrics.csv"), index=False)

    # Локальные графики метрик для одной модели
    def plot_metric(df_loc: pd.DataFrame, name: str, ylabel: str, fname: str):
        plt.figure(figsize=(6,4))
        plt.plot(df_loc['epoch'], df_loc[name], marker='o')
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(f"{model_name} — {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(od, fname), dpi=150); plt.close()

    plot_metric(df, 'LZ_norm', 'LZ (norm)', 'LZ_norm_vs_epoch.png')
    plot_metric(df, 'H_markov', 'Entropy rate (bits/step)', 'Hmarkov_vs_epoch.png')
    plot_metric(df, 'H_perm_norm', 'Permutation entropy (norm)', 'PermEnt_norm_vs_epoch.png')
    plot_metric(df, 'h_KS', 'h_KS (slope of H(n))', 'hKS_vs_epoch.png')
    plot_metric(df, 'D2', 'Correlation dimension D2', 'D2_vs_epoch.png')

    return df

# =============================
# Сравнение нескольких моделей
# =============================

def plot_compare(dfs: List[pd.DataFrame], outdir: str, metric: str, ylabel: str, fname: str):
    plt.figure(figsize=(7,4))
    for df in dfs:
        name = str(df['model'].iloc[0])
        plt.plot(df['epoch'], df[metric], marker='o', label=name)
    plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(ylabel)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=150); plt.close()

# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dirs', nargs='+', required=True, help='Папки модели со states_epoch_*.npz')
    ap.add_argument('--outdir', type=str, default='results')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--nmax_block', type=int, default=8)
    ap.add_argument('--perm_m', type=int, default=5)
    ap.add_argument('--perm_tau', type=int, default=1)
    ap.add_argument('--make_gif', action='store_true')
    ap.add_argument('--gif_duration', type=float, default=1.2, help='Пауза между кадрами GIF (сек.)')
    ap.add_argument('--which_seq', type=int, default=None, help='Если задано — брать только это окно из H_seq; иначе конкатенировать все')
    ap.add_argument('--takens', dest='use_takens', action='store_true', help='Использовать Takens для источника символизации и D2')
    ap.add_argument('--no_takens', dest='use_takens', action='store_false')
    ap.set_defaults(use_takens=True)
    ap.add_argument('--ami_max_tau', type=int, default=50)
    ap.add_argument('--fnn_mmax', type=int, default=10)
    ap.add_argument('--hks_min_samples', type=int, default=1000, help='Минимум N_n для учёта H(n)')
    ap.add_argument('--hks_min_unique', type=int, default=30, help='Минимум уникальных n-грамм k_n')
    ap.add_argument('--hks_tail', type=int, default=3, help='Длина хвоста для усреднения ΔH')

    args = ap.parse_args()

    ensure_dir(args.outdir)
    dfs = []
    for rd in args.run_dirs:
        df = analyze_run(
            rd, args.outdir,
            k=args.k, nmax_block=args.nmax_block,
            perm_m=args.perm_m, perm_tau=args.perm_tau,
            make_gif=args.make_gif, which_seq=args.which_seq,
            use_takens=args.use_takens, ami_max_tau=args.ami_max_tau, fnn_mmax=args.fnn_mmax,
            hks_min_samples=args.hks_min_samples, hks_min_unique=args.hks_min_unique,
            hks_tail=args.hks_tail, gif_duration=args.gif_duration
        )
        dfs.append(df)

    if len(dfs) >= 1:
        plot_compare(dfs, args.outdir, 'LZ_norm', 'LZ (norm)', 'compare_LZ_norm.png')
        plot_compare(dfs, args.outdir, 'H_markov', 'Entropy rate (bits/step)', 'compare_Hmarkov.png')
        plot_compare(dfs, args.outdir, 'H_perm_norm', 'Permutation entropy (norm)', 'compare_PermEnt_norm.png')
        plot_compare(dfs, args.outdir, 'h_KS', 'h_KS (slope of H(n))', 'compare_hKS.png')
        plot_compare(dfs, args.outdir, 'D2', 'Correlation dimension D2', 'compare_D2.png')

if __name__ == '__main__':
    main()
