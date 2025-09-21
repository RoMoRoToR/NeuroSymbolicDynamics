# -*- coding: utf-8 -*-
import os, re, glob, math, argparse, json
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
import imageio.v2 as imageio

# ---------- utils ----------
def ensure_dir(p: str):
    if p and not os.path.exists(p): os.makedirs(p, exist_ok=True)

def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def list_state_files(run_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    return files

def load_H_seq(path: str) -> List[np.ndarray]:
    dat = np.load(path, allow_pickle=True)
    # ожидается список окон скрытых состояний
    return [np.asarray(h, float) for h in dat["H_seq"]]

def pick_dense_idx(H_list: List[np.ndarray]) -> int:
    # берём самое длинное окно, чтобы не было «стыков»
    lengths = [h.shape[0] for h in H_list]
    return int(np.argmax(lengths))

def downsample_time(H: np.ndarray, stride: int) -> np.ndarray:
    return H[::max(1, stride), :]

# ---------- metrics ----------
def lz_complexity_int(seq: np.ndarray) -> int:
    s = seq.tolist(); n = len(s)
    if n == 0: return 0
    c, i, k = 1, 0, 1
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

def transition_matrix(labels: np.ndarray, K: int) -> np.ndarray:
    P = np.zeros((K, K), float)
    for i in range(len(labels)-1):
        P[labels[i], labels[i+1]] += 1.0
    row_sums = P.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        P = np.divide(P, row_sums, out=np.zeros_like(P), where=row_sums>0)
    P = (P + 1e-12); P = P / P.sum(axis=1, keepdims=True)
    return P

def stationary_dist(P: np.ndarray, tol: float = 1e-10, iters: int = 10000) -> np.ndarray:
    K = P.shape[0]; pi = np.ones(K)/K
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
    n = len(x)
    if n < (m-1)*tau + 1: return 0.0
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = x[i:(i + m * tau):tau]
        patterns.append(tuple(np.argsort(window, kind='quicksort')))
    from collections import Counter
    cnt = Counter(patterns); total = sum(cnt.values())
    probs = np.array([v/total for v in cnt.values()], float)
    return float((-probs * np.log2(probs + 1e-12)).sum())

def correlation_dimension_D2(Z: np.ndarray, n_r: int = 12) -> float:
    """
    Очень грубая D2: считаем C(r) на радиусах в лог-шкале и берём наклон
    на средней трети диапазона.
    """
    from scipy.spatial.distance import pdist
    d = pdist(Z, metric='euclidean')
    if len(d) == 0: return 0.0
    d = d[d > 0]
    lo, hi = np.percentile(d, 5), np.percentile(d, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0: return 0.0
    rs = np.logspace(np.log10(lo), np.log10(hi), n_r)
    C = [np.mean(d < r) for r in rs]
    x = np.log(rs + 1e-12); y = np.log(np.array(C) + 1e-12)
    i1, i2 = n_r//3, 2*n_r//3
    a = np.polyfit(x[i1:i2], y[i1:i2], 1)[0]
    return float(max(a, 0.0))

# ---------- viz ----------
def plot_3d(Z3: np.ndarray, title: str, out_path: str):
    T = Z3.shape[0]
    t = np.linspace(0, 1, T)
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap('viridis')

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    # линия
    ax.plot(Z3[:,0], Z3[:,1], Z3[:,2], alpha=0.7)
    # точки
    ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=cmap(norm(t)), s=18)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_P_heatmap(P: np.ndarray, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(P, cmap='magma', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

# ---------- main pipeline ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Папка с states_epoch_*.npz")
    ap.add_argument("--epochs", nargs="+", type=int,
                    default=[0,1,5,10,25,50,150,200],
                    help="Список эпох для снятия метрик")
    ap.add_argument("--k", type=int, default=12, help="k для k-means символизации")
    ap.add_argument("--which_seq", type=str, default="densest",
                    help="'densest' или индекс окна (int)")
    ap.add_argument("--time_stride", type=int, default=1, help="даунсэмпл по времени")
    ap.add_argument("--out_dir", type=str, default="dynviz_results")
    ap.add_argument("--gif", action="store_true", help="Собрать GIF из 3D кадров")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # --- соберём карту epoch->file ---
    files = list_state_files(args.run_dir)
    if not files: raise FileNotFoundError(f"Нет states_epoch_*.npz в {args.run_dir}")
    files_map = {parse_epoch_from_name(f): f for f in files}
    epochs = [e for e in args.epochs if e in files_map]

    # --- PCA базис по эпохе 0 (или первой доступной) ---
    e0 = epochs[0]
    H0_list = load_H_seq(files_map[e0])
    if args.which_seq == "densest": idx0 = pick_dense_idx(H0_list)
    else:
        try: idx0 = int(args.which_seq)
        except: idx0 = 0
    H0 = downsample_time(H0_list[idx0], args.time_stride)
    pca = PCA(n_components=3, random_state=0).fit(H0)

    rows = []
    png_paths = []

    for e in epochs:
        H_list = load_H_seq(files_map[e])
        idx = pick_dense_idx(H_list) if args.which_seq == "densest" else int(args.which_seq)
        H = downsample_time(H_list[idx], args.time_stride)
        Z3 = pca.transform(H)  # общий базис для сопоставимости

        # символизация в PC1-PC2
        Z2 = Z3[:, :2]
        km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(Z2)
        labels = km.labels_
        P = transition_matrix(labels, args.k)

        # метрики
        LZ = lz_normalized(labels)
        Hm = markov_entropy_rate(P)
        Perm = permutation_entropy(Z2[:,0], m=5, tau=1) / math.log2(math.factorial(5))
        D2 = correlation_dimension_D2(Z2)

        # визуализации
        plot_3d(Z3, f"3D PCA trajectory | epoch {e}", os.path.join(args.out_dir, f"traj3d_epoch_{e:03d}.png"))
        plot_P_heatmap(P, f"Symbol transitions (k={args.k}) | epoch {e}",
                       os.path.join(args.out_dir, f"P_epoch_{e:03d}.png"))
        png_paths.append(os.path.join(args.out_dir, f"traj3d_epoch_{e:03d}.png"))

        rows.append({
            "epoch": e,
            "LZ_norm": LZ,
            "H_markov": Hm,
            "PermEnt_norm": Perm,
            "D2": D2,
            "energy_fro_perT": float(np.linalg.norm(H, ord='fro')) / max(len(H),1),
            "seq_idx": idx,
            "T": int(H.shape[0])
        })

    # metrics.csv
    import csv
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"[ok] saved {csv_path}")

    # GIF (по желанию)
    if args.gif and len(png_paths) > 1:
        gif_path = os.path.join(args.out_dir, "traj3d_epochs.gif")
        imgs = [imageio.imread(p) for p in png_paths]
        imageio.mimsave(gif_path, imgs, duration=0.9)
        print(f"[ok] saved {gif_path}")

if __name__ == "__main__":
    main()
