# analyze_dynamics_sota_new.py
# -*- coding: utf-8 -*-
import os, re, glob, math, argparse, json
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import imageio.v2 as imageio
import networkx as nx
from scipy.spatial.distance import pdist

# =============================
# Утилиты / защита от пустых данных
# =============================

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def safe_json_load(path: str) -> Optional[dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

def sanitize_seq(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, float)
    if H.ndim != 2 or len(H) == 0:
        return np.zeros((0, 0), float)
    H = H[np.all(np.isfinite(H), axis=1)]
    return H

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

def safe_pca_2d(X: np.ndarray):
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0:
        return np.zeros((0, 2), float), np.zeros((0,), float)
    ncomp = int(min(2, X.shape[1], len(X)))
    if ncomp == 0:
        return np.zeros((0, 2), float), np.zeros((0,), float)
    Z = PCA(n_components=ncomp, random_state=0).fit_transform(X)
    if ncomp == 1:
        Z2 = np.c_[Z, np.zeros((len(Z), 1), float)]
    else:
        Z2 = Z
    return Z2, Z2[:, 0]

# =============================
# Символьные метрики
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
    ord_idx = np.argsort(ns); ns, Hs = ns[ord_idx], Hs[ord_idx]
    Hs = np.maximum.accumulate(Hs)
    inc = np.diff(Hs) / (np.diff(ns) + 1e-12)
    if inc.size == 0: return 0.0
    k = 3 if inc.size >= 3 else inc.size
    h = float(np.mean(inc[-k:]))
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

def correlation_dimension(Z: np.ndarray, quantiles=(0.1, 0.6), n_eps=10) -> float:
    if len(Z) < 5: return 0.0
    d = pdist(Z, metric='euclidean')
    if len(d) == 0: return 0.0
    lo = np.quantile(d, quantiles[0]); hi = np.quantile(d, quantiles[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or hi <= 0 or lo <= 0: return 0.0
    eps_grid = np.geomspace(lo, hi, num=n_eps)
    C = [(d < eps).mean() for eps in eps_grid]
    x = np.log(eps_grid); y = np.log(np.maximum(C, 1e-12))
    mid = slice(max(1, n_eps//4), -max(1, n_eps//4)) if n_eps >= 6 else slice(1, -1)
    Xmat = np.vstack([x[mid], np.ones_like(x[mid])]).T
    a, _ = np.linalg.lstsq(Xmat, y[mid], rcond=None)[0]
    return float(a)

# =============================
# Takens: AMI/FNN с защитами
# =============================

def delayed_mutual_information(x: np.ndarray, max_tau: int = 50, bins: int = 64) -> np.ndarray:
    x = np.asarray(x, float).ravel()
    if len(x) < 2 or not np.isfinite(x).any():
        return np.zeros(max_tau, float)
    hist_x, edges = np.histogram(x, bins=bins, density=False)
    tot = hist_x.sum()
    if tot == 0:
        return np.zeros(max_tau, float)
    p_x = hist_x / tot
    I = []
    for tau in range(1, max_tau+1):
        if tau >= len(x):
            I.append(0.0); continue
        x1 = x[:-tau]; x2 = x[tau:]
        H, _, _ = np.histogram2d(x1, x2, bins=bins, density=False)
        s = H.sum()
        if s == 0:
            I.append(0.0); continue
        p12 = H / s
        p2, _ = np.histogram(x2, bins=edges, density=False)
        s2 = p2.sum()
        if s2 == 0:
            I.append(0.0); continue
        p2 = p2 / s2
        with np.errstate(divide='ignore', invalid='ignore'):
            logterm = np.log((p12 + 1e-12) / ((p_x[:, None] * p2[None, :]) + 1e-12))
            I_tau = float(np.nansum(p12 * logterm))
        I.append(I_tau)
    return np.array(I, float)

def choose_tau_by_first_min_ami(x: np.ndarray, max_tau: int = 50, bins: int = 64) -> int:
    I = delayed_mutual_information(x, max_tau=max_tau, bins=bins)
    for t in range(1, len(I)-1):
        if I[t] < I[t-1] and I[t] <= I[t+1]:
            return t+1
    return int(max(1, np.argmin(I)+1))

def false_nearest_neighbors(pc1: np.ndarray, tau: int, m_max: int = 10, Rtol: float = 15.0, Atol: float = 2.0) -> int:
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
            A = num / np.std(x) if np.std(x) > 0 else np.inf
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
    if N <= 1:
        return np.zeros((0, max(1, m)), float)
    idx = np.arange(m)[:, None] * tau + np.arange(N)[None, :]
    return x[idx].T

# =============================
# Mapper (как раньше)
# =============================

def mapper_graph(Z: np.ndarray, filter_vals: np.ndarray, n_bins: int = 12, overlap: float = 0.3, k_per_bin: int = 2):
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
# Основной анализ одного run_dir
# =============================

def analyze_run(run_dir: str, outdir: str, k: int = 16, nmax_block: int = 6,
                perm_m: int = 5, perm_tau: int = 1, make_gif: bool = False,
                model_name: str = None, which_seq: int = None,
                takens: bool = False, ami_max_tau: int = 50, fnn_mmax: int = 10,
                do_mapper: bool = False, mapper_bins: int = 12, mapper_overlap: float = 0.3, mapper_k: int = 2,
                do_cka: bool = False, do_svcca: bool = False, cca_dim: int = 20,
                gif3d: bool = False, max_points_repr: int = 8000,
                prefer_dense: bool = False,
                color_by: str = "time") -> pd.DataFrame:

    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Не найдены npz-файлы в {run_dir}")

    if model_name is None:
        model_name = os.path.basename(os.path.normpath(run_dir))

    od = os.path.join(outdir, model_name)
    figs_dir = os.path.join(od, "figs"); ensure_dir(figs_dir)

    rep_by_epoch: Dict[int, np.ndarray] = {}
    rows = []
    frame_paths_traj, frame_paths_heat, frame_paths_traj3d = [], [], []

    for f in files:
        epoch = parse_epoch_from_name(f)
        dat = np.load(f, allow_pickle=True)
        H_list = dat['H_seq']
        H = choose_sequence(H_list, which_seq, prefer_dense)
        if H is None or len(H) < 2:
            print(f"[skip] {model_name} epoch {epoch}: нет валидных точек (H is None/len<2).")
            continue
        H = sanitize_seq(H)
        if len(H) < 2:
            print(f"[skip] {model_name} epoch {epoch}: после очистки осталось <2 точек.")
            continue

        # первичный PCA для Takens
        Z2_init, pc1_init = safe_pca_2d(H)
        if len(pc1_init) < 3:
            print(f"[skip] {model_name} epoch {epoch}: недостаточно точек для анализа (pc1_init<3).")
            continue

        # Takens (опц.) с откатом, если точек мало
        emb_X = H
        takens_info = {"used": False}
        if takens:
            tau = choose_tau_by_first_min_ami(pc1_init, max_tau=ami_max_tau)
            m   = false_nearest_neighbors(pc1_init, tau=tau, m_max=fnn_mmax)
            emb_tmp = takens_embedding_1d(pc1_init, m=m, tau=tau)
            if len(emb_tmp) >= 3:
                emb_X = emb_tmp
                takens_info = {"used": True, "tau": int(tau), "m": int(m)}
            else:
                print(f"[info] {model_name} epoch {epoch}: Takens дал мало точек → анализируем исходный H.")

        # PCA для визуализации
        Z2, pc1 = safe_pca_2d(emb_X if takens_info.get("used") else H)
        if len(Z2) < 2:
            print(f"[skip] {model_name} epoch {epoch}: после Takens/PCA точек <2.")
            continue

        # Кластеризация (для метрик; цвета НЕ используем)
        n_samples = len(emb_X)
        k_eff = int(min(k, max(1, n_samples)))
        if k_eff == 1:
            labels = np.zeros(n_samples, dtype=int)
        else:
            labels = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit_predict(emb_X)

        # Матрица переходов/метрики
        P = transition_matrix(labels, k_eff)
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

        # -------- ВИЗУАЛИЗАЦИЯ АТТРАКТОРА (все точки) --------
        # Цвета: время (0..1) или кластеры (опционально)
        if color_by == "cluster":
            cvals = labels
            cmap = plt.cm.tab20
        else:
            cvals = np.linspace(0, 1, len(Z2))
            cmap = plt.cm.viridis

        # 2D
        plt.figure(figsize=(5.6, 5.2))
        plt.plot(Z2[:, 0], Z2[:, 1], lw=0.8, alpha=0.7)
        sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=cvals, s=8, alpha=0.9, cmap=cmap)
        cb = plt.colorbar(sc); cb.set_label("time" if color_by=="time" else "cluster")
        ttxt = f"{model_name}: PCA traj (epoch {epoch})"
        if takens_info.get("used"): ttxt += f" | Takens m={takens_info['m']}, τ={takens_info['tau']}"
        if color_by == "time": ttxt += " | colored by time"
        plt.title(ttxt); plt.xlabel("PC1"); plt.ylabel("PC2")
        p_traj = os.path.join(figs_dir, f"traj_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_traj, dpi=160); plt.close()

        # Теплокарта P
        plt.figure(figsize=(5.2, 4.2))
        plt.imshow(P, aspect='auto'); plt.colorbar(label='P(i→j)')
        plt.title(f"{model_name}: Transition P (epoch {epoch})")
        plt.xlabel("j"); plt.ylabel("i")
        p_heat = os.path.join(figs_dir, f"P_epoch_{epoch:03d}.png")
        plt.tight_layout(); plt.savefig(p_heat, dpi=160); plt.close()

        frame_paths_traj.append(p_traj); frame_paths_heat.append(p_heat)

        # 3D (опционально)
        if gif3d:
            n_features = (emb_X if takens_info.get("used") else H).shape[1]
            ncomp = int(min(3, n_features, len(Z2)))
            if ncomp >= 2:
                Z3 = PCA(n_components=ncomp, random_state=0).fit_transform(emb_X if takens_info.get("used") else H)
                if ncomp == 2:
                    Z3 = np.c_[Z3, np.zeros((len(Z3), 1))]
                from mpl_toolkits.mplot3d import Axes3D  # noqa
                fig = plt.figure(figsize=(6.0, 6.0)); ax = fig.add_subplot(111, projection='3d')
                ax.plot(Z3[:,0], Z3[:,1], Z3[:,2], lw=1.0, alpha=0.85)
                sc3 = ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], s=5, c=(labels if color_by=="cluster" else np.linspace(0,1,len(Z3))), cmap=(plt.cm.tab20 if color_by=="cluster" else plt.cm.viridis), alpha=0.85)
                ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3'); ax.view_init(elev=20, azim=40)
                if color_by == "time":
                    fig.colorbar(sc3, ax=ax, fraction=0.035, pad=0.06, label="time")
                else:
                    fig.colorbar(sc3, ax=ax, fraction=0.035, pad=0.06, label="cluster")
                ax.set_title(f"{model_name} | epoch {epoch}")
                p3d = os.path.join(figs_dir, f"traj3d_epoch_{epoch:03d}.png")
                plt.tight_layout(); plt.savefig(p3d, dpi=140); plt.close()
                frame_paths_traj3d.append(p3d)

        # Репрезентация для CKA/SVCCA
        Xrepr = H
        if len(Xrepr) > max_points_repr:
            idx = np.linspace(0, len(Xrepr)-1, max_points_repr).astype(int)
            Xrepr = Xrepr[idx]
        rep_by_epoch[epoch] = Xrepr

        # Лоссы
        json_path = os.path.join(run_dir, f"epoch_{epoch:03d}.json")
        if not os.path.exists(json_path):
            json_path = os.path.join(os.path.dirname(f), f"epoch_{epoch:03d}.json")
        j = safe_json_load(json_path) or {}

        rows.append({
            "model": model_name, "epoch": epoch, "T_points": int(len(H)), "k_eff": int(k_eff),
            "LZ_raw": float(LZ_raw), "LZ_norm": float(LZ_norm), "H_markov": float(H_markov),
            "H_perm": float(H_perm), "H_perm_norm": float(H_perm_norm), "h_KS": float(h_KS), "D2": float(D2),
            "Takens_used": int(takens_info.get("used", False)),
            "Takens_tau": int(takens_info.get("tau", 0)) if takens_info.get("used") else 0,
            "Takens_m": int(takens_info.get("m", 0)) if takens_info.get("used") else 0,
            "Mapper_nodes": 0, "Mapper_edges": 0, "Mapper_cyclomatic": 0,
            "val_loss": float(j.get("val_loss", np.nan)), "train_loss": float(j.get("train_loss", np.nan))
        })

        # Mapper (кластеры только здесь)
        if do_mapper:
            try:
                G, node_pts = mapper_graph(Z2, filter_vals=Z2[:,0], n_bins=mapper_bins, overlap=mapper_overlap, k_per_bin=mapper_k)
                C = nx.number_connected_components(G)
                rows[-1]["Mapper_nodes"] = int(G.number_of_nodes())
                rows[-1]["Mapper_edges"] = int(G.number_of_edges())
                rows[-1]["Mapper_cyclomatic"] = int(G.number_of_edges() - G.number_of_nodes() + C)
                plt.figure(figsize=(6, 5))
                pos = nx.spring_layout(G, seed=0)
                sizes = [max(50, 20*np.log10(5+G.nodes[n]['size'])) for n in G.nodes()]
                nx.draw_networkx(G, pos=pos, with_labels=False, node_size=sizes)
                plt.title(f"Mapper (epoch {epoch}) | V={G.number_of_nodes()}, E={G.number_of_edges()}")
                p_map = os.path.join(figs_dir, f"mapper_epoch_{epoch:03d}.png")
                plt.tight_layout(); plt.savefig(p_map, dpi=160); plt.close()
            except Exception as e:
                print(f"[warn] Mapper failed at epoch {epoch}: {e}")

    # GIF'ы
    ensure_dir(od)
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
        if p_cka:
            df_cka = pd.DataFrame(p_cka); df_cka.to_csv(os.path.join(od, "cka_vs_epoch0.csv"), index=False)
            plt.figure(figsize=(6,4)); plt.plot(df_cka['epoch'], df_cka['CKA_vs_epoch0'], marker='o')
            plt.xlabel('epoch'); plt.ylabel('CKA vs epoch0'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(od, 'CKA_vs_epoch0.png'), dpi=160); plt.close()
        if p_sv:
            df_sv = pd.DataFrame(p_sv); df_sv.to_csv(os.path.join(od, "svcca_vs_epoch0.csv"), index=False)
            plt.figure(figsize=(6,4)); plt.plot(df_sv['epoch'], df_sv['SVCCA_vs_epoch0'], marker='o')
            plt.xlabel('epoch'); plt.ylabel('SVCCA vs epoch0'); plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(os.path.join(od, 'SVCCA_vs_epoch0.png'), dpi=160); plt.close()

    # Графики основных метрик
    def plot_metric(df_loc: pd.DataFrame, name: str, ylabel: str, fname: str):
        if name not in df_loc.columns or df_loc.empty: return
        plt.figure(figsize=(6,4))
        plt.plot(df_loc['epoch'], df_loc[name], marker='o')
        plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(f"{model_name} — {ylabel}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(od, fname), dpi=160); plt.close()

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
# CKA/SVCCA (linear)
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
    U, S, Vt = np.linalg.svd(Cxy, full_matrices=False)
    corr = S / (np.linalg.norm(Xr, 'fro') * (np.linalg.norm(Yr, 'fro') + 1e-12) + 1e-12) * (Xr.shape[0])
    corr = np.clip(corr, 0.0, 1.0)
    k = min(dim, len(corr))
    return float(np.mean(corr[:k]))

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
    ap.add_argument('--ami_max_tau', type=int, default=50)
    ap.add_argument('--fnn_mmax', type=int, default=10)
    ap.add_argument('--mapper', dest='do_mapper', action='store_true')
    ap.add_argument('--mapper_bins', type=int, default=12)
    ap.add_argument('--mapper_overlap', type=float, default=0.3)
    ap.add_argument('--mapper_k', type=int, default=2)
    ap.add_argument('--do_cka', action='store_true')
    ap.add_argument('--do_svcca', action='store_true')
    ap.add_argument('--cca_dim', type=int, default=20)
    ap.add_argument('--prefer_dense', action='store_true')
    ap.add_argument('--color_by', type=str, default='time', choices=['time', 'cluster'],
                    help='Чем красить траектории: time (по времени, по умолчанию) или cluster.')
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
            gif3d=args.gif3d, prefer_dense=args.prefer_dense,
            color_by=args.color_by
        )
        dfs.append(df)

    if len(dfs) >= 1:
        def plot_compare(dfs: List[pd.DataFrame], outdir: str, metric: str, ylabel: str, fname: str):
            plt.figure(figsize=(7,4))
            for df in dfs:
                if df.empty or metric not in df.columns: continue
                name = str(df['model'].iloc[0]) if 'model' in df.columns else 'model'
                plt.plot(df['epoch'], df[metric], marker='o', label=name)
            plt.xlabel('epoch'); plt.ylabel(ylabel); plt.title(ylabel)
            plt.grid(True, alpha=0.3); plt.legend()
            plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=160); plt.close()

        plot_compare(dfs, args.outdir, 'LZ_norm', 'LZ (norm)', 'compare_LZ_norm.png')
        plot_compare(dfs, args.outdir, 'H_markov', 'Entropy rate (bits/step)', 'compare_Hmarkov.png')
        plot_compare(dfs, args.outdir, 'h_KS', 'h_KS (slope of H(n))', 'compare_hKS.png')
        plot_compare(dfs, args.outdir, 'H_perm_norm', 'Permutation entropy (norm)', 'compare_PermEnt_norm.png')
        plot_compare(dfs, args.outdir, 'D2', 'Correlation dimension D2', 'compare_D2.png')

if __name__ == '__main__':
    main()
