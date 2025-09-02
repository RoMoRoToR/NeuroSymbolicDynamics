# analyze_dynamics_sota.py
# -*- coding: utf-8 -*-
r"""
Анализ скрытой динамики по чекпоинтам эпох для RNN/Transformer **с расширениями SOTA**.

Что теперь умеет (всё опционально, можно включать флагами):
1) Символическая динамика (как раньше): k-means/ordinal → матрица переходов P →
   Lempel–Ziv (raw/норма), блоковые энтропии H(n) и наклон h_KS, permutation entropy,
   энтропия Маркова, грубая корреляционная размерность D2. + GIF траекторий и P.
2) Takens-embedding по PC1 скрытой траектории: автоматический подбор τ (первый минимум
   time-delayed MI) и m (false nearest neighbors); символизация и метрики считаются на
   реконструированном аттракторе. (Флаг --takens)
3) Mapper-граф (упрощённый): фильтр=PC1, интервалы с перекрытием, кластеризация в бинах
   и рёбра по пересечениям. Сохраняются картинка и граф-статистики (#узлов/#рёбер/цикломатич.
   число). (Флаг --mapper)
4) CKA (linear CKA) и SVCCA между эпохами: сходство представлений скрытых состояний
   (с epoch0 и с предыдущей эпохой). (Флаги --do_cka, --do_svcca)
5) 3D PCA GIF вращающийся (для наглядности). (Флаг --gif3d)

Пример запуска (Linux/PowerShell одинаково):
    python ./analyze_dynamics_sota.py \
      --run_dirs "runs/lorenz/RNN_LSTM" "runs/lorenz/Transformer" \
      --outdir "results/lorenz" \
      --k 16 --make_gif --gif3d --do_cka --do_svcca --takens --mapper

Замечания:
- Все тяжёлые блоки можно отключать флагами. CKA/SVCCA считаем на одинаковом числе точек
  (делается выравнивание / downsample).
"""

import os, re, glob, math, argparse, json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import imageio.v2 as imageio
import networkx as nx

# =============================
# БАЗОВЫЕ УТИЛИТЫ
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

# =============================
# СИМВОЛИЧЕСКИЕ МЕТРИКИ
# =============================

def lz_complexity_int(seq: np.ndarray) -> int:
    """Очень простая LZ76 для целочисленной последовательности (число фраз)."""
    s = seq.tolist(); n = len(s)
    if n == 0: return 0
    c = 1; i = 0; k = 1
    while True:
        if i + k > n: break
        sub = s[i:i+k]
        # встречается ли sub в s[0:i]?
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
    """
    Робастная оценка h_KS из последовательности H(n).
    1) Гарантируем монотонность H(n) (H_{n+1} >= H_n).
    2) Берём среднее приращение на хвосте (последние 2–3 шага).
    3) Гарантируем неотрицательность результата.
    """
    ns = np.asarray(ns, dtype=float)
    Hs = np.asarray(Hs, dtype=float)
    if ns.size < 2:
        return 0.0

    # убрать нечисловые и упорядочить по n
    msk = np.isfinite(ns) & np.isfinite(Hs)
    ns, Hs = ns[msk], Hs[msk]
    if ns.size < 2:
        return 0.0
    ord_idx = np.argsort(ns)
    ns, Hs = ns[ord_idx], Hs[ord_idx]

    # 1) монотонность H(n)
    Hs = np.maximum.accumulate(Hs)

    # 2) приращения (нормируем на Δn на всякий случай)
    dH = np.diff(Hs)
    dn = np.diff(ns)
    inc = dH / (dn + 1e-12)   # обычно dn==1

    if inc.size == 0:
        return 0.0

    # хвост: последние до 3 приращений
    k = 3 if inc.size >= 3 else inc.size
    h = float(np.mean(inc[-k:]))

    # 3) неотрицательность
    return max(0.0, h)



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
    """Permutation entropy (Bandt & Pompe). Возвращает H_perm в битах."""
    n = len(x)
    if n < (m - 1) * tau + 1: return 0.0
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = x[i:(i + m * tau):tau]
        patterns.append(tuple(np.argsort(window, kind='quicksort')))
    from collections import Counter
    cnt = Counter(patterns); total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], float)
    return float((-probs * np.log2(probs + 1e-12)).sum())


def correlation_dimension(X: np.ndarray, quantiles=(0.1, 0.6), n_eps=10) -> float:
    """Грубая оценка D2 по корреляционной сумме."""
    if len(X) < 5: return 0.0
    from scipy.spatial.distance import pdist
    d = pdist(X, metric='euclidean')
    if len(d) == 0: return 0.0
    lo = np.quantile(d, quantiles[0]); hi = np.quantile(d, quantiles[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or hi <= 0 or lo <= 0: return 0.0
    eps_grid = np.geomspace(lo, hi, num=n_eps)
    C = [(d < eps).mean() for eps in eps_grid]
    x = np.log(eps_grid); y = np.log(np.maximum(C, 1e-12))
    mid = slice(max(1, n_eps//4), -max(1, n_eps//4)) if n_eps >= 6 else slice(1, -1)
    Xmat = np.vstack([x[mid], np.ones_like(x[mid])]).T
    a, b = np.linalg.lstsq(Xmat, y[mid], rcond=None)[0]
    return float(a)

# =============================
# TAKENS: τ через AMI, m через FNN; embed по PC1
# =============================

def delayed_mutual_information(x: np.ndarray, max_tau: int = 50, bins: int = 64) -> np.ndarray:
    """Оценка I(x_t; x_{t-τ}) по τ (грубая гистограммная)."""
    x = np.asarray(x, float).ravel()
    I = []
    hist_x, edges = np.histogram(x, bins=bins, density=True)
    p_x = hist_x / (hist_x.sum() + 1e-12)
    for tau in range(1, max_tau+1):
        x1 = x[:-tau]; x2 = x[tau:]
        H, _, _ = np.histogram2d(x1, x2, bins=bins, density=True)
        p12 = H / (H.sum() + 1e-12)
        p1 = p_x
        p2, _ = np.histogram(x2, bins=edges, density=True)
        p2 = p2 / (p2.sum() + 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            logterm = np.log((p12 + 1e-12) / ((p1[:, None] * p2[None, :]) + 1e-12))
            I_tau = float(np.nansum(p12 * logterm))
        I.append(I_tau)
    return np.array(I)


def choose_tau_by_first_min_ami(x: np.ndarray, max_tau: int = 50, bins: int = 64) -> int:
    I = delayed_mutual_information(x, max_tau=max_tau, bins=bins)
    for t in range(1, len(I)-1):
        if I[t] < I[t-1] and I[t] <= I[t+1]:
            return t+1
    return int(max(1, np.argmin(I)+1))


def false_nearest_neighbors(pc1: np.ndarray, tau: int, m_max: int = 10, Rtol: float = 15.0, Atol: float = 2.0) -> int:
    """Грубая FNN (Kennel et al.). Минимальное m, где доля ложных соседей < 10%."""
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
    Ux, Sx, Vtx = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, Vty = np.linalg.svd(Y, full_matrices=False)
    Xr = (Ux[:, :dim] * Sx[:dim]); Yr = (Uy[:, :dim] * Sy[:dim])
    Cxy = Xr.T @ Yr
    U, S, Vt = np.linalg.svd(Cxy, full_matrices=False)
    corr = S / (np.linalg.norm(Xr, 'fro') * np.linalg.norm(Yr, 'fro') + 1e-12) * (Xr.shape[0])
    corr = np.clip(corr, 0.0, 1.0)
    k = min(dim, len(corr))
    return float(np.mean(corr[:k]))

# =============================
# MAPPER (упрощённо)
# =============================

def mapper_graph(Z: np.ndarray, filter_vals: np.ndarray, n_bins: int = 12, overlap: float = 0.3, k_per_bin: int = 2):
    """Простой Mapper: фильтр=filter_vals (обычно PC1). Бьём на интервалы с перекрытием; в каждом
    интервале делаем k-means; ребро между узлами, если наборы точек пересекаются."""
    N = len(filter_vals)
    fmin, fmax = np.min(filter_vals), np.max(filter_vals)
    step = (fmax - fmin) / (n_bins - (n_bins-1)*overlap)
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
        if len(idx) < max(5, k_per_bin):
            continue
        Xb = Z[idx]
        k_here = min(k_per_bin, len(idx))
        labs = KMeans(n_clusters=k_here, n_init=5, random_state=0).fit_predict(Xb)
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
# ОСНОВНОЙ АНАЛИЗ ОДНОГО RUN_DIR
# =============================

def analyze_run(run_dir: str, outdir: str, k: int = 16, nmax_block: int = 6,
                perm_m: int = 5, perm_tau: int = 1, make_gif: bool = False,
                model_name: str = None, which_seq: int = None,
                takens: bool = False, ami_max_tau: int = 50, fnn_mmax: int = 10,
                do_mapper: bool = False, mapper_bins: int = 12, mapper_overlap: float = 0.3, mapper_k: int = 2,
                do_cka: bool = False, do_svcca: bool = False, cca_dim: int = 20,
                gif3d: bool = False, max_points_repr: int = 8000) -> pd.DataFrame:
    """Таблица метрик по эпохам + графики/гифы в outdir/model_name. Все тяжёлые блоки — флаги."""
    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Не найдены npz-файлы в {run_dir}")

    if model_name is None:
        model_name = os.path.basename(os.path.normpath(run_dir))

    od = os.path.join(outdir, model_name)
    figs_dir = os.path.join(od, "figs"); ensure_dir(figs_dir)

    rep_by_epoch = {}
    rows = []
    frame_paths_traj = []; frame_paths_heat = []
    frame_paths_traj3d = []

    for f in files:
        epoch = parse_epoch_from_name(f)
        dat = np.load(f, allow_pickle=True)
        H_list = dat['H_seq']
        H = H_list[which_seq] if (which_seq is not None and 0 <= which_seq < len(H_list)) else np.concatenate([np.asarray(h, float) for h in H_list], axis=0)
        H = np.asarray(H, float)
        T, D = H.shape

        # 2D PCA + PC1 для первичного анализа (на H)
        pca2_init = PCA(n_components=2, random_state=0)
        Z2_init = pca2_init.fit_transform(H)
        pc1_init = Z2_init[:, 0]

        # Takens (опц.) — параметры подбираем по pc1_init, а визуализируем на согласованном источнике
        emb_X = H
        takens_info = {"used": False}
        if takens:
            tau = choose_tau_by_first_min_ami(pc1_init, max_tau=ami_max_tau)
            m   = false_nearest_neighbors(pc1_init, tau=tau, m_max=fnn_mmax)
            emb_X = takens_embedding_1d(pc1_init, m=m, tau=tau)
            takens_info = {"used": True, "tau": int(tau), "m": int(m)}

        # Источник для визуализации и символизации: если включён Takens — берём emb_X; иначе — исходные H
        src_for_vis = emb_X if takens_info.get("used") else H
        pca2 = PCA(n_components=2, random_state=0)
        Z2 = pca2.fit_transform(src_for_vis)
        pc1 = Z2[:, 0]

        # Символизация → метрики
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(emb_X)
        P = transition_matrix(labels, k)
        H_markov = markov_entropy_rate(P)
        LZ_raw = float(lz_complexity_int(labels))
        LZ_norm = float(lz_normalized(labels))
        ns, Hs = block_entropies(labels, nmax=nmax_block)
        h_KS = ks_entropy_from_blocks(ns, Hs)
        H_perm = permutation_entropy(pc1, m=perm_m, tau=perm_tau)
        H_perm_norm = H_perm / math.log2(math.factorial(perm_m)) if perm_m >= 2 else 0.0
        try:
            D2 = correlation_dimension(Z2)
        except Exception:
            D2 = 0.0

        # Фигуры 2D
        fig1 = plt.figure(figsize=(5, 5))
        plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, s=8)
        ttxt = f"{model_name}: PCA traj (epoch {epoch})"
        if takens_info.get("used"):
            ttxt += f" | Takens m={takens_info['m']}, τ={takens_info['tau']}"
        plt.title(ttxt); plt.xlabel("PC1"); plt.ylabel("PC2")
        p_traj = os.path.join(figs_dir, f"traj_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_traj, dpi=150); plt.close(fig1)

        fig2 = plt.figure(figsize=(5, 4))
        plt.imshow(P, aspect='auto'); plt.colorbar(label='P(i→j)')
        plt.title(f"{model_name}: Transition P (epoch {epoch})")
        plt.xlabel("j"); plt.ylabel("i")
        p_heat = os.path.join(figs_dir, f"P_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_heat, dpi=150); plt.close(fig2)

        frame_paths_traj.append(p_traj); frame_paths_heat.append(p_heat)

        # 3D PCA (кадры для GIF)
        if gif3d:
            src3d = src_for_vis  # согласованный источник, чтобы размеры совпадали с labels
            n_features = src3d.shape[1]
            n_samples  = len(src3d)
            ncomp = int(min(3, n_features, n_samples))
            if ncomp >= 2:
                Z3 = PCA(n_components=ncomp, random_state=0).fit_transform(src3d)
                # если компонент всего 2 — добавим нулевую ось, чтобы всё равно сделать 3D-график
                if ncomp == 2:
                    Z3 = np.c_[Z3, np.zeros((len(Z3), 1))]
                lim = np.percentile(np.abs(Z3), 99); lim = max(lim, 1e-3)
                fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection='3d')
                t = np.linspace(0, 1, len(Z3))
                ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
                ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], s=5, c=t)
                ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3'); ax.view_init(elev=20, azim=40)
                p3d = os.path.join(figs_dir, f"traj3d_epoch_{epoch:03d}.png")
                plt.tight_layout(); plt.savefig(p3d, dpi=130); plt.close(fig)
                frame_paths_traj3d.append(p3d)
            # если ncomp < 2 — точек/признаков слишком мало; пропускаем 3D для этой эпохи

        # Mapper (опц.)
        mapper_stats = {"nodes": 0, "edges": 0, "cyclomatic": 0}
        if do_mapper:
            G, node_pts = mapper_graph(Z2, filter_vals=pc1, n_bins=mapper_bins, overlap=mapper_overlap, k_per_bin=mapper_k)
            mapper_stats["nodes"] = int(G.number_of_nodes())
            mapper_stats["edges"] = int(G.number_of_edges())
            C = nx.number_connected_components(G)
            mapper_stats["cyclomatic"] = int(G.number_of_edges() - G.number_of_nodes() + C)
            plt.figure(figsize=(6, 5))
            pos = nx.spring_layout(G, seed=0)
            sizes = [max(50, 20*np.log10(5+G.nodes[n]['size'])) for n in G.nodes()]
            nx.draw_networkx(G, pos=pos, with_labels=False, node_size=sizes)
            plt.title(f"Mapper (epoch {epoch}) | V={G.number_of_nodes()}, E={G.number_of_edges()}")
            p_map = os.path.join(figs_dir, f"mapper_epoch_{epoch:03d}.png")
            plt.tight_layout(); plt.savefig(p_map, dpi=150); plt.close()

        # Репрезентации для CKA/SVCCA (равняем по длине при расчёте)
        Xrepr = H
        if len(Xrepr) > max_points_repr:
            idx = np.linspace(0, len(Xrepr)-1, max_points_repr).astype(int)
            Xrepr = Xrepr[idx]
        rep_by_epoch[epoch] = Xrepr

        # Лоссы (если есть epoch_XXX.json)
        json_path = os.path.join(run_dir, f"epoch_{epoch:03d}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(f), f"epoch_{epoch:03d}.json")
        j = safe_json_load(json_path) or {}

        rows.append({
            "model": model_name, "epoch": epoch, "T_points": int(T), "k": int(k),
            "LZ_raw": float(LZ_raw), "LZ_norm": float(LZ_norm), "H_markov": float(H_markov),
            "H_perm": float(H_perm), "H_perm_norm": float(H_perm_norm), "h_KS": float(h_KS), "D2": float(D2),
            "Takens_used": int(takens_info.get("used", False)),
            "Takens_tau": int(takens_info.get("tau", 0)) if takens_info.get("used") else 0,
            "Takens_m": int(takens_info.get("m", 0)) if takens_info.get("used") else 0,
            "Mapper_nodes": mapper_stats["nodes"], "Mapper_edges": mapper_stats["edges"],
            "Mapper_cyclomatic": mapper_stats["cyclomatic"],
            "val_loss": float(j.get("val_loss", np.nan)), "train_loss": float(j.get("train_loss", np.nan))
        })

    # GIF'ы
    if make_gif and len(frame_paths_traj) > 1:
        with imageio.get_writer(os.path.join(od, "pca_traj.gif"), mode='I', duration=1) as w:
            for p in frame_paths_traj: w.append_data(imageio.imread(p))
    if make_gif and len(frame_paths_heat) > 1:
        with imageio.get_writer(os.path.join(od, "P_heatmap.gif"), mode='I', duration=1) as w:
            for p in frame_paths_heat: w.append_data(imageio.imread(p))
    if gif3d and len(frame_paths_traj3d) > 1:
        with imageio.get_writer(os.path.join(od, "pca3d_traj.gif"), mode='I', duration=1) as w:
            for p in frame_paths_traj3d: w.append_data(imageio.imread(p))

    # CSV
    df = pd.DataFrame(rows).sort_values(["epoch"]).reset_index(drop=True)
    ensure_dir(od)
    df.to_csv(os.path.join(od, "metrics.csv"), index=False)

    # CKA/SVCCA vs epoch0
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
        if do_cka:
            df_cka = pd.DataFrame(p_cka); df_cka.to_csv(os.path.join(od, "cka_vs_epoch0.csv"), index=False)
            plt.figure(figsize=(6,4)); plt.plot(df_cka['epoch'], df_cka['CKA_vs_epoch0'], marker='o')
            plt.xlabel('epoch'); plt.ylabel('CKA vs epoch0'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(od, 'CKA_vs_epoch0.png'), dpi=150); plt.close()
        if do_svcca:
            df_sv = pd.DataFrame(p_sv); df_sv.to_csv(os.path.join(od, "svcca_vs_epoch0.csv"), index=False)
            plt.figure(figsize=(6,4)); plt.plot(df_sv['epoch'], df_sv['SVCCA_vs_epoch0'], marker='o')
            plt.xlabel('epoch'); plt.ylabel('SVCCA vs epoch0'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(od, 'SVCCA_vs_epoch0.png'), dpi=150); plt.close()

    # Графики основных метрик
    def plot_metric(df_loc: pd.DataFrame, name: str, ylabel: str, fname: str):
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
    ap.add_argument('--nmax_block', type=int, default=6)
    ap.add_argument('--perm_m', type=int, default=5)
    ap.add_argument('--perm_tau', type=int, default=1)
    ap.add_argument('--make_gif', action='store_true')
    ap.add_argument('--gif3d', action='store_true')
    ap.add_argument('--which_seq', type=int, default=None, help='Если задано — брать только это окно из H_seq; иначе конкатенировать все')
    ap.add_argument('--takens', action='store_true')
    ap.add_argument('--ami_max_tau', type=int, default=50)
    ap.add_argument('--fnn_mmax', type=int, default=10)
    ap.add_argument('--mapper', dest='do_mapper', action='store_true')
    ap.add_argument('--mapper_bins', type=int, default=12)
    ap.add_argument('--mapper_overlap', type=float, default=0.3)
    ap.add_argument('--mapper_k', type=int, default=2)
    ap.add_argument('--do_cka', action='store_true')
    ap.add_argument('--do_svcca', action='store_true')
    ap.add_argument('--cca_dim', type=int, default=20)

    args = ap.parse_args()

    ensure_dir(args.outdir)
    dfs = []
    for rd in args.run_dirs:
        df = analyze_run(
            rd, args.outdir,
            k=args.k, nmax_block=args.nmax_block, perm_m=args.perm_m, perm_tau=args.perm_tau,
            make_gif=args.make_gif, which_seq=args.which_seq,
            takens=args.takens, ami_max_tau=args.ami_max_tau, fnn_mmax=args.fnn_mmax,
            do_mapper=args.do_mapper, mapper_bins=args.mapper_bins, mapper_overlap=args.mapper_overlap, mapper_k=args.mapper_k,
            do_cka=args.do_cka, do_svcca=args.do_svcca, cca_dim=args.cca_dim,
            gif3d=args.gif3d
        )
        dfs.append(df)

    if len(dfs) >= 1:
        plot_compare(dfs, args.outdir, 'LZ_norm', 'LZ (norm)', 'compare_LZ_norm.png')
        plot_compare(dfs, args.outdir, 'H_markov', 'Entropy rate (bits/step)', 'compare_Hmarkov.png')
        plot_compare(dfs, args.outdir, 'h_KS', 'h_KS (slope of H(n))', 'compare_hKS.png')
        plot_compare(dfs, args.outdir, 'H_perm_norm', 'Permutation entropy (norm)', 'compare_PermEnt_norm.png')
        plot_compare(dfs, args.outdir, 'D2', 'Correlation dimension D2', 'compare_D2.png')

if __name__ == '__main__':
    main()
