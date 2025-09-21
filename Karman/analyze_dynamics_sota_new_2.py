# analyze_dynamics_sota.py
# -*- coding: utf-8 -*-
r"""
Быстрый анализ скрытой динамики по чекпоинтам эпох (RNN/Transformer_old).

Что считает (всё опционально флагами):
1) Символьные метрики: LZ (raw/norm), блоковые энтропии и наклон h_KS, permutation entropy,
   энтропия Маркова, грубая корреляционная размерность D2. + GIF теплокарт P.
2) Takens-embedding по PC1: τ (AMI) + m (FNN) — на подвыборке/даунсэмпле.
3) Mapper-граф (упрощённо) — MiniBatchKMeans в бинах; рёбра по пересечениям.
4) CKA (linear) и SVCCA между эпохами — на ограниченном числе точек.
5) 3D PCA кадры/GIF (без 2D-картинок по запросу пользователя).

Основные ускорения по умолчанию:
- --time_stride 2, --max_points 20000
- AMI: --ami_bins 32, --ami_stride 2, --ami_max_tau 40
- FNN: --fnn_sample 5000
- Корр. размерность: --corr_max_pairs 100000
- Mapper: --mapper_max_points 4000
- KMeans: --kmeans_impl minibatch

Пример:
python analyze_dynamics_sota.py ^
  --run_dirs runs/lorenz/Transformer_old runs/lorenz/RNN_LSTM ^
  --outdir results/lorenz --k 16 --gif3d --takens --mapper --prefer_dense
"""

import os, re, glob, math, argparse, json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
import imageio.v2 as imageio
import networkx as nx

# =============================
# УТИЛИТЫ / ВВОД-ВЫВОД / СЭЙФТИ
# =============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def safe_json_load(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def to_float_or_nan(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')

def sanitize_seq(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, float)
    if H.ndim != 2 or len(H) == 0: return np.zeros((0, 0), float)
    return H[np.all(np.isfinite(H), axis=1)]

def choose_sequence(H_list, which_seq: Optional[int], prefer_dense: bool) -> Optional[np.ndarray]:
    valid = []
    for h in H_list:
        hh = sanitize_seq(h)
        if len(hh) > 0:
            valid.append(hh)
    if not valid:
        return None
    if which_seq is not None and 0 <= which_seq < len(valid):
        return valid[which_seq]
    if prefer_dense:
        idx = int(np.argmax([len(v) for v in valid]))
        return valid[idx]
    return np.concatenate(valid, axis=0)

def even_subsample_idx(n: int, max_n: int) -> np.ndarray:
    if n <= max_n: return np.arange(n)
    return np.linspace(0, n-1, max_n).astype(int)

def downsample_time(X: np.ndarray, stride: int) -> np.ndarray:
    return X[::max(1, int(stride))]

# =============================
# СИМВОЛИЧЕСКИЕ МЕТРИКИ
# =============================

def lz_complexity_int(seq: np.ndarray) -> int:
    s = seq.tolist(); n = len(s)
    if n == 0: return 0
    c = 1; i = 0; k = 1
    while True:
        if i + k > n: break
        sub = s[i:i+k]
        found = any(s[j:j+k] == sub for j in range(0, i))
        if found:
            k += 1
            if i + k > n: c += 1; break
        else:
            c += 1; i += k; k = 1
            if i + 1 > n: break
    return c

def lz_normalized(seq: np.ndarray) -> float:
    n = len(seq)
    if n < 2: return 0.0
    return lz_complexity_int(seq) / (n / math.log(n))

def ngram_entropy(labels: np.ndarray, n: int) -> float:
    if len(labels) < n: return 0.0
    from collections import Counter
    tuples = [tuple(labels[i:i+n]) for i in range(len(labels)-n+1)]
    cnt = Counter(tuples); total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], dtype=float)
    return float((-probs * np.log2(probs + 1e-12)).sum())

def block_entropies(labels: np.ndarray, nmax: int = 6) -> Tuple[List[int], List[float]]:
    ns, Hs = [], []
    for n in range(1, nmax+1):
        ns.append(n); Hs.append(ngram_entropy(labels, n))
    return ns, Hs

def ks_entropy_from_blocks(ns: List[int], Hs: List[float]) -> float:
    ns = np.asarray(ns, dtype=float); Hs = np.asarray(Hs, dtype=float)
    if ns.size < 2: return 0.0
    msk = np.isfinite(ns) & np.isfinite(Hs)
    ns, Hs = ns[msk], Hs[msk]
    if ns.size < 2: return 0.0
    ord_idx = np.argsort(ns)
    ns, Hs = ns[ord_idx], Hs[ord_idx]
    Hs = np.maximum.accumulate(Hs)
    dH = np.diff(Hs); dn = np.diff(ns)
    inc = dH / (dn + 1e-12)
    if inc.size == 0: return 0.0
    k = 3 if inc.size >= 3 else inc.size
    return float(max(0.0, np.mean(inc[-k:])))

def transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    P = np.zeros((K, K), float)
    for i in range(len(labels)-1):
        P[labels[i], labels[i+1]] += 1.0
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums>0)
    P = (P + 1e-8); P = P / P.sum(axis=1, keepdims=True)
    return P

def stationary_dist(P: np.ndarray, tol: float = 1e-10, iters: int = 10000) -> np.ndarray:
    K = P.shape[0]; pi = np.ones(K) / K
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
    x = np.asarray(x, float).ravel()
    if len(x) < (m - 1) * tau + 1: return 0.0
    if not np.isfinite(x).any() or np.allclose(np.var(x), 0.0): return 0.0
    patterns = []
    for i in range(len(x) - (m - 1) * tau):
        window = x[i:(i + m * tau):tau]
        patterns.append(tuple(np.argsort(window, kind='quicksort')))
    from collections import Counter
    cnt = Counter(patterns); total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], float)
    return float((-probs * np.log2(probs + 1e-12)).sum())

def correlation_dimension_random(Z: np.ndarray, max_pairs: int = 100_000,
                                 quantiles=(0.1, 0.6), n_eps=10) -> float:
    """Быстрая D2: расстояния только по случайной подвыборке пар."""
    Z = np.asarray(Z, float)
    n = len(Z)
    if n < 5: return 0.0
    # выберем пары
    m = min(max_pairs, n*(n-1)//2)
    # равномерно по индексу времени, чтобы не брать почти одинаковые точки
    rng = np.random.default_rng(0)
    i = rng.integers(0, n-1, size=m, endpoint=False)
    j = rng.integers(i+1, n, size=m, endpoint=True)
    diffs = Z[i] - Z[j]
    d = np.sqrt((diffs * diffs).sum(axis=1))
    if len(d) == 0: return 0.0
    lo = np.quantile(d, quantiles[0]); hi = np.quantile(d, quantiles[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or hi <= 0 or lo <= 0: return 0.0
    eps = np.geomspace(lo, hi, num=n_eps)
    C = [(d < e).mean() for e in eps]
    x = np.log(eps); y = np.log(np.maximum(C, 1e-12))
    mid = slice(1, -1) if n_eps < 6 else slice(n_eps//4, -n_eps//4)
    A = np.vstack([x[mid], np.ones_like(x[mid])]).T
    a, _ = np.linalg.lstsq(A, y[mid], rcond=None)[0]
    return float(a)

# =============================
# TAKENS: τ через AMI, m через FNN (на подвыборке)
# =============================

def delayed_mutual_information_fast(x: np.ndarray, max_tau: int = 40, bins: int = 32,
                                    stride: int = 2) -> np.ndarray:
    """AMI на даунсэмпленном сигнале (быстро)."""
    x = np.asarray(x, float).ravel()
    if stride > 1:
        x = x[::stride]
    if len(x) < 2:
        return np.zeros(max_tau, float)
    hist_x, edges = np.histogram(x, bins=bins, density=False)
    tot = hist_x.sum()
    if tot == 0:
        return np.zeros(max_tau, float)
    p_x = hist_x / (tot + 1e-12)
    I = []
    for tau in range(1, max_tau+1):
        if tau >= len(x):
            I.append(0.0); continue
        x1 = x[:-tau]; x2 = x[tau:]
        H, _, _ = np.histogram2d(x1, x2, bins=bins, density=False)
        s = H.sum()
        if s == 0:
            I.append(0.0); continue
        p12 = H / (s + 1e-12)
        p2, _ = np.histogram(x2, bins=edges, density=False)
        s2 = p2.sum()
        p2 = p2 / (s2 + 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            logterm = np.log((p12 + 1e-12) / ((p_x[:, None] * p2[None, :]) + 1e-12))
            I_tau = float(np.nansum(p12 * logterm))
        I.append(I_tau)
    return np.array(I, float)

def choose_tau_by_first_min_ami(x: np.ndarray, max_tau: int = 40, bins: int = 32, stride: int = 2) -> int:
    I = delayed_mutual_information_fast(x, max_tau=max_tau, bins=bins, stride=stride)
    for t in range(1, len(I)-1):
        if I[t] < I[t-1] and I[t] <= I[t+1]:
            return t+1
    return int(max(1, np.argmin(I)+1))

def false_nearest_neighbors_fast(pc1: np.ndarray, tau: int, m_max: int = 10,
                                 Rtol: float = 15.0, Atol: float = 2.0,
                                 sample: int = 5000) -> int:
    """FNN на подвыборке точек."""
    x = np.asarray(pc1, float).ravel()
    Nfull = len(x) - (m_max * tau)
    if Nfull <= 1: return 3

    # индексы по времени, равномерно
    idx_time = even_subsample_idx(Nfull, sample)
    def embed(m):
        base = np.add.outer(idx_time, np.arange(m)*tau)   # [n_samp, m]
        return x[base]

    fracs = []
    for m in range(1, m_max+1):
        Xm = embed(m)
        nbr = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(Xm)
        dists, idxs = nbr.kneighbors(Xm, return_distance=True)
        d1 = dists[:, 1] + 1e-12
        if m < m_max:
            Xm1 = embed(m+1)
            j = idxs[:, 1]
            num = np.abs(Xm1[:, -1] - Xm1[j, -1])
            R = num / d1
            A = num / (np.std(x) + 1e-12)
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
    if N <= 1: return np.zeros((0, max(1, m)), float)
    idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
    return x[idx].T

# =============================
# CKA и SVCCA (linear)
# =============================

def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    unit = np.ones((n, n)) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def _gram_linear(X: np.ndarray) -> np.ndarray:
    return X @ X.T

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Kx = _center_gram(_gram_linear(X))
    Ky = _center_gram(_gram_linear(Y))
    hsic = (Kx * Ky).sum()
    norm = np.sqrt((Kx * Kx).sum() * (Ky * Ky).sum()) + 1e-12
    return float(hsic / norm)

def svcca_similarity(X: np.ndarray, Y: np.ndarray, dim: int = 20) -> float:
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Ux, Sx, _ = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y, full_matrices=False)
    Xr = (Ux[:, :dim] * Sx[:dim]); Yr = (Uy[:, :dim] * Sy[:dim])
    Cxy = Xr.T @ Yr
    _, S, _ = np.linalg.svd(Cxy, full_matrices=False)
    from numpy.linalg import norm
    corr = S / (norm(Xr, 'fro') * norm(Yr, 'fro') + 1e-12) * (Xr.shape[0])
    corr = np.clip(corr, 0.0, 1.0)
    k = min(dim, len(corr))
    return float(np.mean(corr[:k]))

# =============================
# MAPPER (упрощённо, minibatch)
# =============================

def mapper_graph(Z: np.ndarray, filter_vals: np.ndarray, n_bins: int = 12,
                 overlap: float = 0.3, k_per_bin: int = 2, k_impl: str = "minibatch"):
    N = len(filter_vals)
    if N == 0: return nx.Graph(), {}
    fmin, fmax = np.min(filter_vals), np.max(filter_vals)
    step = (fmax - fmin) / (n_bins - (n_bins-1)*overlap + 1e-12)
    bins = []
    left = fmin
    for _ in range(n_bins):
        right = left + step
        bins.append((left, right))
        left = left + step * (1 - overlap)

    bin_indices = []
    for (lo, hi) in bins:
        idx = np.where((filter_vals >= lo) & (filter_vals <= hi))[0]
        bin_indices.append(idx)

    G = nx.Graph(); node_pts = {}; node_id = 0
    for b, idx in enumerate(bin_indices):
        if len(idx) < max(5, k_per_bin):  # малые бины пропускаем
            continue
        Xb = Z[idx]
        k_here = min(k_per_bin, len(idx))
        if k_impl == "minibatch":
            km = MiniBatchKMeans(n_clusters=k_here, n_init=3, batch_size=1024, max_iter=50, random_state=0)
        else:
            km = KMeans(n_clusters=k_here, n_init=5, random_state=0)
        labs = km.fit_predict(Xb)
        for c in range(k_here):
            pts = idx[labs == c]
            if len(pts) == 0: continue
            G.add_node(node_id, bin=b, size=int(len(pts)))
            node_pts[node_id] = set(int(p) for p in pts)
            node_id += 1

    nodes = list(node_pts.keys())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            if len(node_pts[nodes[i]] & node_pts[nodes[j]]) > 0:
                G.add_edge(nodes[i], nodes[j])

    return G, node_pts

# =============================
# ОСНОВНОЙ АНАЛИЗ ОДНОГО RUN_DIR (БЫСТРЫЙ)
# =============================

def analyze_run(run_dir: str, outdir: str, k: int = 16, nmax_block: int = 6,
                perm_m: int = 5, perm_tau: int = 1, make_gif: bool = False,
                model_name: str = None, which_seq: int = None,
                takens: bool = False, ami_max_tau: int = 40, fnn_mmax: int = 10,
                do_mapper: bool = False, mapper_bins: int = 12, mapper_overlap: float = 0.3, mapper_k: int = 2,
                do_cka: bool = False, do_svcca: bool = False, cca_dim: int = 20,
                gif3d: bool = False, max_points_repr: int = 8000,
                # ускорители:
                prefer_dense: bool = False, time_stride: int = 2, max_points: int = 20_000,
                ami_bins: int = 32, ami_stride: int = 2, fnn_sample: int = 5000,
                corr_max_pairs: int = 100_000, kmeans_impl: str = "minibatch",
                mapper_max_points: int = 4000, cka_max_points: int = 8000,
                seed: int = 0) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Не найдены npz-файлы в {run_dir}")

    model_name = model_name or os.path.basename(os.path.normpath(run_dir))
    od = os.path.join(outdir, model_name)
    figs_dir = os.path.join(od, "figs"); ensure_dir(figs_dir)

    rep_by_epoch: Dict[int, np.ndarray] = {}
    rows = []
    frame_paths_heat = []
    frame_paths_traj3d = []

    for f in files:
        epoch = parse_epoch_from_name(f)
        dat = np.load(f, allow_pickle=True)
        H_list = dat['H_seq']
        H = choose_sequence(H_list, which_seq, prefer_dense)
        if H is None or len(H) < 3:
            print(f"[skip] {model_name} epoch {epoch}: мало валидных точек."); continue
        H = sanitize_seq(H)

        # быстрый даунсэмпл по времени + ограничение числа точек
        if time_stride > 1:
            H = downsample_time(H, time_stride)
        idx_keep = even_subsample_idx(len(H), max_points)
        H = H[idx_keep]

        if len(H) < 3:
            print(f"[skip] {model_name} epoch {epoch}: после даунсэмпла <3 точек."); continue

        # PCA2 для получения pc1 (быстро)
        pca2_init = PCA(n_components=min(2, H.shape[1], len(H)), random_state=seed)
        Z2_init = pca2_init.fit_transform(H)
        if Z2_init.ndim == 1: Z2_init = Z2_init[:, None]
        pc1_init = Z2_init[:, 0]

        # Takens (быстро, на подвыборке/stride)
        emb_X = H
        takens_info = {"used": False}
        if takens and len(pc1_init) >= 10:
            tau = choose_tau_by_first_min_ami(pc1_init, max_tau=ami_max_tau, bins=ami_bins, stride=ami_stride)
            m   = false_nearest_neighbors_fast(pc1_init, tau=tau, m_max=fnn_mmax, sample=fnn_sample)
            emb_tmp = takens_embedding_1d(pc1_init, m=m, tau=tau)
            if len(emb_tmp) >= 3:
                emb_X = emb_tmp
                takens_info = {"used": True, "tau": int(tau), "m": int(m)}

        # Источник для метрик/визуализаций
        src = emb_X if takens_info["used"] else H

        # 2D PCA только для метрик (Permutation Entropy и Mapper/D2); ОТРИСОВКИ 2D НЕТ
        pca2 = PCA(n_components=min(2, src.shape[1], len(src)), random_state=seed)
        Z2 = pca2.fit_transform(src)
        if Z2.ndim == 1: Z2 = Z2[:, None]
        pc1 = Z2[:, 0]

        # Кластеризация (быстро)
        n_samples = len(src)
        k_eff = int(min(k, max(1, n_samples)))
        if k_eff == 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            if kmeans_impl == "minibatch":
                km = MiniBatchKMeans(n_clusters=k_eff, n_init=3, batch_size=2048, max_iter=50, random_state=seed)
            else:
                km = KMeans(n_clusters=k_eff, n_init=10, random_state=seed)
            labels = km.fit_predict(src)

        # Метрики
        P = transition_matrix(labels, k_eff)
        H_markov = markov_entropy_rate(P)
        LZ_raw = float(lz_complexity_int(labels)); LZ_norm = float(lz_normalized(labels))
        ns, Hs = block_entropies(labels, nmax=nmax_block); h_KS = ks_entropy_from_blocks(ns, Hs)
        H_perm = permutation_entropy(pc1, m=perm_m, tau=perm_tau)
        H_perm_norm = H_perm / math.log2(math.factorial(perm_m)) if perm_m >= 2 else 0.0
        try:
            # D2 на подвыборке пар
            D2 = correlation_dimension_random(Z2, max_pairs=corr_max_pairs)
        except Exception:
            D2 = 0.0

        # Теплокарта P
        plt.figure(figsize=(5.2, 4.2))
        plt.imshow(P, aspect='auto'); plt.colorbar(label='P(i→j)')
        title = f"{model_name}: Transition P (epoch {epoch})"
        if takens_info["used"]: title += f" | Takens m={takens_info['m']}, τ={takens_info['tau']}"
        plt.title(title); plt.xlabel("j"); plt.ylabel("i")
        p_heat = os.path.join(figs_dir, f"P_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_heat, dpi=150); plt.close()
        frame_paths_heat.append(p_heat)

        # 3D PCA кадр (опц., быстро)
        if gif3d:
            ncomp = int(min(3, src.shape[1], len(src)))
            if ncomp >= 2:
                Z3 = PCA(n_components=ncomp, random_state=seed).fit_transform(src)
                if ncomp == 2: Z3 = np.c_[Z3, np.zeros((len(Z3), 1))]
                from mpl_toolkits.mplot3d import Axes3D  # noqa
                fig = plt.figure(figsize=(6.0, 6.0)); ax = fig.add_subplot(111, projection='3d')
                t = np.linspace(0, 1, len(Z3))
                ax.plot(Z3[:,0], Z3[:,1], Z3[:,2], lw=0.9, alpha=0.8)
                sc3 = ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], s=5, c=t, alpha=0.9)
                ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3'); ax.view_init(elev=20, azim=40)
                fig.colorbar(sc3, ax=ax, fraction=0.035, pad=0.06, label="time")
                ax.set_title(f"{model_name} | epoch {epoch}")
                p3d = os.path.join(figs_dir, f"traj3d_epoch_{epoch:03d}.png")
                plt.tight_layout(); plt.savefig(p3d, dpi=140); plt.close()
                frame_paths_traj3d.append(p3d)

        # Mapper (опц., быстро + подвыборка)
        mapper_stats = {"nodes": 0, "edges": 0, "cyclomatic": 0}
        if do_mapper:
            idx_m = even_subsample_idx(len(Z2), mapper_max_points)
            Z2m = Z2[idx_m]; pc1m = pc1[idx_m]
            try:
                G, node_pts = mapper_graph(Z2m, pc1m, n_bins=mapper_bins, overlap=mapper_overlap,
                                           k_per_bin=mapper_k, k_impl=kmeans_impl)
                C = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
                mapper_stats["nodes"] = int(G.number_of_nodes())
                mapper_stats["edges"] = int(G.number_of_edges())
                mapper_stats["cyclomatic"] = int(G.number_of_edges() - G.number_of_nodes() + C)
                plt.figure(figsize=(6, 5))
                pos = nx.spring_layout(G, seed=seed)
                sizes = [max(50, 20*np.log10(5+G.nodes[n]['size'])) for n in G.nodes()]
                nx.draw_networkx(G, pos=pos, with_labels=False, node_size=sizes)
                plt.title(f"Mapper (epoch {epoch}) | V={G.number_of_nodes()}, E={G.number_of_edges()}")
                p_map = os.path.join(figs_dir, f"mapper_epoch_{epoch:03d}.png")
                plt.tight_layout(); plt.savefig(p_map, dpi=150); plt.close()
            except Exception as e:
                print(f"[warn] Mapper failed at epoch {epoch}: {e}")

        # Репрезентации для CKA/SVCCA (подрезаем до cka_max_points)
        Xrepr = H
        if len(Xrepr) > cka_max_points:
            idx = even_subsample_idx(len(Xrepr), cka_max_points)
            Xrepr = Xrepr[idx]
        rep_by_epoch[epoch] = Xrepr

        # Лоссы
        json_path = os.path.join(run_dir, f"epoch_{epoch:03d}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(f), f"epoch_{epoch:03d}.json")
        j = safe_json_load(json_path) or {}

        rows.append({
            "model": model_name, "epoch": epoch, "T_points": int(len(H)), "k": int(k_eff),
            "LZ_raw": float(LZ_raw), "LZ_norm": float(LZ_norm), "H_markov": float(H_markov),
            "H_perm": float(H_perm), "H_perm_norm": float(H_perm_norm), "h_KS": float(h_KS), "D2": float(D2),
            "Takens_used": int(takens_info.get("used", False)),
            "Takens_tau": int(takens_info.get("tau", 0)) if takens_info.get("used") else 0,
            "Takens_m": int(takens_info.get("m", 0)) if takens_info.get("used") else 0,
            "Mapper_nodes": mapper_stats["nodes"], "Mapper_edges": mapper_stats["edges"],
            "Mapper_cyclomatic": mapper_stats["cyclomatic"],
            "val_loss": to_float_or_nan(j.get("val_loss")), "train_loss": to_float_or_nan(j.get("train_loss"))
        })

    # GIF'ы
    ensure_dir(od)
    if make_gif and len(frame_paths_heat) > 1:
        with imageio.get_writer(os.path.join(od, "P_heatmap.gif"), mode='I', duration=1) as w:
            for p in frame_paths_heat: w.append_data(imageio.imread(p))
    if gif3d and len(frame_paths_traj3d) > 1:
        with imageio.get_writer(os.path.join(od, "pca3d_traj.gif"), mode='I', duration=1) as w:
            for p in frame_paths_traj3d: w.append_data(imageio.imread(p))

    # CSV
    df = pd.DataFrame(rows).sort_values(["epoch"]).reset_index(drop=True)
    df.to_csv(os.path.join(od, "metrics.csv"), index=False)

    # CKA/SVCCA vs epoch0 (на подрезанных представлениях)
    if (do_cka or do_svcca) and len(rep_by_epoch) >= 2:
        epochs_sorted = sorted(rep_by_epoch.keys())
        base = rep_by_epoch[epochs_sorted[0]]
        p_cka = []; p_sv = []
        for e in epochs_sorted:
            X = rep_by_epoch[e]
            n = min(len(base), len(X))
            Xb = base[:n]; Xc = X[:n]
            if do_cka:
                p_cka.append({"epoch": e, "CKA_vs_epoch0": linear_cka(Xb, Xc)})
            if do_svcca:
                p_sv.append({"epoch": e, "SVCCA_vs_epoch0": svcca_similarity(Xb, Xc, dim=cca_dim)})
        if p_cka:
            df_cka = pd.DataFrame(p_cka); df_cka.to_csv(os.path.join(od, "cka_vs_epoch0.csv"), index=False)
            plt.figure(figsize=(6,4)); plt.plot(df_cka['epoch'], df_cka['CKA_vs_epoch0'], marker='o')
            plt.xlabel('epoch'); plt.ylabel('CKA vs epoch0'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(od, 'CKA_vs_epoch0.png'), dpi=150); plt.close()
        if p_sv:
            df_sv = pd.DataFrame(p_sv); df_sv.to_csv(os.path.join(od, "svcca_vs_epoch0.csv"), index=False)
            plt.figure(figsize=(6,4)); plt.plot(df_sv['epoch'], df_sv['SVCCA_vs_epoch0'], marker='o')
            plt.xlabel('epoch'); plt.ylabel('SVCCA vs epoch0'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(od, 'SVCCA_vs_epoch0.png'), dpi=150); plt.close()

    # Графики основных метрик
    def plot_metric(df_loc: pd.DataFrame, name: str, ylabel: str, fname: str):
        if name not in df_loc.columns or df_loc.empty: return
        plt.figure(figsize=(6,4))
        plt.plot(df_loc['epoch'], df_loc[name], marker='o')
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(f"{model_name} — {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(od, fname), dpi=150); plt.close()

    plot_metric(df, 'LZ_norm', 'LZ (norm)', 'LZ_norm_vs_epoch.png')
    plot_metric(df, 'H_markov', 'Entropy rate (bits/step)', 'Hmarkov_vs_epoch.png')
    plot_metric(df, 'h_KS', 'h_KS (slope of H(n))', 'hKS_vs_epoch.png')
    plot_metric(df, 'H_perm_norm', 'Permutation entropy (norm)', 'PermEnt_norm_vs_epoch.png')
    plot_metric(df, 'D2', 'Correlation dimension D2', 'D2_vs_epoch.png')
    if do_mapper:
        plot_metric(df, 'Mapper_nodes', 'Mapper: nodes', 'Mapper_nodes_vs_epoch.png')
        plot_metric(df, 'Mapper_cyclomatic', 'Mapper: cyclomatic', 'Mapper_cyclomatic_vs_epoch.png')

    return df

# =============================
# СРАВНЕНИЕ НЕСКОЛЬКИХ МОДЕЛЕЙ
# =============================

def plot_compare(dfs: List[pd.DataFrame], outdir: str, metric: str, ylabel: str, fname: str):
    plt.figure(figsize=(7,4))
    for df in dfs:
        if df.empty or metric not in df.columns: continue
        name = str(df['model'].iloc[0]) if 'model' in df.columns else 'model'
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
    ap.add_argument('--nmax_block', type=int, default=6)
    ap.add_argument('--perm_m', type=int, default=5)
    ap.add_argument('--perm_tau', type=int, default=1)
    ap.add_argument('--make_gif', action='store_true')
    ap.add_argument('--gif3d', action='store_true')
    ap.add_argument('--which_seq', type=int, default=None)
    ap.add_argument('--takens', action='store_true')
    ap.add_argument('--ami_max_tau', type=int, default=40)
    ap.add_argument('--fnn_mmax', type=int, default=10)
    ap.add_argument('--mapper', dest='do_mapper', action='store_true')
    ap.add_argument('--mapper_bins', type=int, default=12)
    ap.add_argument('--mapper_overlap', type=float, default=0.3)
    ap.add_argument('--mapper_k', type=int, default=2)
    ap.add_argument('--do_cka', action='store_true')
    ap.add_argument('--do_svcca', action='store_true')
    ap.add_argument('--cca_dim', type=int, default=20)

    # ускорения (дефолт: быстрые значения)
    ap.add_argument('--prefer_dense', action='store_true')
    ap.add_argument('--time_stride', type=int, default=2)
    ap.add_argument('--max_points', type=int, default=20_000)
    ap.add_argument('--ami_bins', type=int, default=32)
    ap.add_argument('--ami_stride', type=int, default=2)
    ap.add_argument('--fnn_sample', type=int, default=5000)
    ap.add_argument('--corr_max_pairs', type=int, default=100_000)
    ap.add_argument('--kmeans_impl', type=str, default='minibatch', choices=['kmeans','minibatch'])
    ap.add_argument('--mapper_max_points', type=int, default=4000)
    ap.add_argument('--cka_max_points', type=int, default=8000)
    ap.add_argument('--seed', type=int, default=0)

    args = ap.parse_args()

    ensure_dir(args.outdir)
    dfs = []
    for rd in args.run_dirs:
        df = analyze_run(
            rd, args.outdir,
            k=args.k, nmax_block=args.nmax_block, perm_m=args.perm_m, perm_tau=args.perm_tau,
            make_gif=args.make_gif, which_seq=args.which_seq, takens=args.takens,
            ami_max_tau=args.ami_max_tau, fnn_mmax=args.fnn_mmax,
            do_mapper=args.do_mapper, mapper_bins=args.mapper_bins, mapper_overlap=args.mapper_overlap,
            mapper_k=args.mapper_k, do_cka=args.do_cka, do_svcca=args.do_svcca, cca_dim=args.cca_dim,
            gif3d=args.gif3d, prefer_dense=args.prefer_dense,
            time_stride=args.time_stride, max_points=args.max_points,
            ami_bins=args.ami_bins, ami_stride=args.ami_stride, fnn_sample=args.fnn_sample,
            corr_max_pairs=args.corr_max_pairs, kmeans_impl=args.kmeans_impl,
            mapper_max_points=args.mapper_max_points, cka_max_points=args.cka_max_points,
            seed=args.seed
        )
        dfs.append(df)

    if len(dfs) >= 1:
        out = args.outdir
        plot_compare(dfs, out, 'LZ_norm', 'LZ (norm)', 'compare_LZ_norm.png')
        plot_compare(dfs, out, 'H_markov', 'Entropy rate (bits/step)', 'compare_Hmarkov.png')
        plot_compare(dfs, out, 'h_KS', 'h_KS (slope of H(n))', 'compare_hKS.png')
        plot_compare(dfs, out, 'H_perm_norm', 'Permutation entropy (norm)', 'compare_PermEnt_norm.png')
        plot_compare(dfs, out, 'D2', 'Correlation dimension D2', 'compare_D2.png')

if __name__ == '__main__':
    main()
