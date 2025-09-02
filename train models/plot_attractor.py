# plot_attractor.py
# -*- coding: utf-8 -*-
"""
Простая отрисовка аттрактора скрытых состояний по сохранённым states_epoch_*.npz.
Заточено под ваши тренеры (Transformer/RNN), которые пишут H_seq в npz.

Зависимости: numpy, matplotlib, scikit-learn
(без SciPy; апсемплинг — линейный).

Примеры:
  python plot_attractor.py --run_dir runs/lorenz/Transformer --prefer_dense
  python plot_attractor.py --run_dir runs/lorenz/Transformer --epoch 50 --color cluster --k 12
  python plot_attractor.py --run_dir runs/lorenz/Transformer --prefer_dense --plot3d --upsample 4
"""

import os, re, glob, argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ---------- утилиты ----------
def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def latest_states_file(run_dir: str, epoch: int | None) -> str:
    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Не найдено states_epoch_*.npz в {run_dir}")
    if epoch is None:
        return files[-1]
    for f in files:
        if parse_epoch_from_name(f) == epoch:
            return f
    raise FileNotFoundError(f"В {run_dir} нет states_epoch_{epoch:03d}.npz")

def sanitize_seq(H: np.ndarray) -> np.ndarray:
    H = np.asarray(H, float)
    if H.ndim != 2: return np.zeros((0, 0))
    if len(H) == 0: return H
    H = H[np.all(np.isfinite(H), axis=1)]
    return H

def choose_sequence(H_list, which_seq: int | None, prefer_dense: bool) -> np.ndarray:
    # фильтруем пустые/нечисловые
    valid = []
    for h in H_list:
        hh = sanitize_seq(h)
        if len(hh) > 0: valid.append(hh)
    if not valid:
        return np.zeros((0, 0))
    if which_seq is not None and 0 <= which_seq < len(valid):
        return valid[which_seq]
    if prefer_dense:
        idx = int(np.argmax([len(v) for v in valid]))
        return valid[idx]
    # по умолчанию — склейка
    return np.concatenate(valid, axis=0)

def safe_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    X = np.asarray(X, float)
    if X.ndim != 2 or len(X) == 0 or X.shape[1] == 0:
        return np.zeros((0, n_components))
    ncomp = int(min(n_components, X.shape[1], len(X)))
    if ncomp == 0:
        return np.zeros((0, n_components))
    Z = PCA(n_components=ncomp, random_state=0).fit_transform(X)
    # паддинг нулями до нужного числа компонент (если ncomp < n_components)
    if ncomp < n_components:
        Z = np.c_[Z, np.zeros((len(Z), n_components - ncomp))]
    return Z

def upsample_linear(Z: np.ndarray, factor: int) -> np.ndarray:
    """Простой линейный апсемплинг траектории в factor раз (без SciPy)."""
    if factor is None or factor <= 1 or len(Z) < 2:
        return Z
    t = np.arange(len(Z), dtype=float)
    tt = np.linspace(0.0, len(Z) - 1.0, len(Z) * int(factor))
    out = []
    for d in range(Z.shape[1]):
        out.append(np.interp(tt, t, Z[:, d]))
    return np.column_stack(out)

# ---------- основной рендер ----------
def plot_attractor(run_dir: str, epoch: int | None, which_seq: int | None, prefer_dense: bool,
                   color_mode: str, k: int, upsample: int, plot3d: bool,
                   outdir: str):
    path = latest_states_file(run_dir, epoch)
    epoch_found = parse_epoch_from_name(path)
    dat = np.load(path, allow_pickle=True)
    H_list = dat["H_seq"]

    H = choose_sequence(H_list, which_seq, prefer_dense)  # [T, D]
    if len(H) < 2:
        raise RuntimeError("Нет валидных точек для отрисовки (последовательность пуста).")

    # PCA → 2D/3D для визуализации
    Z2 = safe_pca(H, n_components=2)
    title_model = os.path.basename(os.path.normpath(run_dir))
    title = f"{title_model} — epoch {epoch_found}"
    cmap = plt.cm.viridis

    # Цвета
    if color_mode == "time":
        c_vals = np.linspace(0, 1, len(Z2))
        c_kw = dict(c=c_vals, cmap=cmap)
        legend_note = "color = time"
    elif color_mode == "cluster" and k >= 2:
        k_eff = int(min(k, max(2, len(H))))
        labs = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit_predict(H)
        c_vals = labs
        c_kw = dict(c=c_vals, cmap=plt.cm.tab20)
        legend_note = f"clusters k={k_eff}"
    else:
        # fallback
        c_vals = np.linspace(0, 1, len(Z2))
        c_kw = dict(c=c_vals, cmap=cmap)
        legend_note = "color = time"

    # Линия (апсемплинг только для линии)
    Z2_line = upsample_linear(Z2, upsample)

    ensure_dir(outdir)
    # 2D
    fig = plt.figure(figsize=(6, 5.6))
    if len(Z2_line) >= 2:
        plt.plot(Z2_line[:, 0], Z2_line[:, 1], lw=0.9, alpha=0.8)
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], s=10, alpha=0.9, **c_kw)
    plt.title(f"{title}\nPCA 2D ({legend_note})" + (f" | upsample x{upsample}" if upsample and upsample > 1 else ""))
    plt.xlabel("PC1"); plt.ylabel("PC2")
    if color_mode == "time":
        cb = plt.colorbar(sc); cb.set_label("time")
    plt.tight_layout()
    out_png = os.path.join(outdir, f"attractor2d_{title_model}_e{epoch_found:03d}.png")
    plt.savefig(out_png, dpi=170); plt.close(fig)

    # 3D (опционально)
    out_png3d = None
    if plot3d:
        Z3 = safe_pca(H, n_components=3)
        if len(Z3) >= 2:
            Z3_line = upsample_linear(Z3, upsample)
            fig = plt.figure(figsize=(6.4, 6.0))
            ax = fig.add_subplot(111, projection='3d')
            if len(Z3_line) >= 2:
                ax.plot(Z3_line[:, 0], Z3_line[:, 1], Z3_line[:, 2], lw=0.9, alpha=0.85)
            sc3 = ax.scatter(Z3[:, 0], Z3[:, 1], Z3[:, 2], s=8, alpha=0.9, **c_kw)
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.set_title(f"{title}\nPCA 3D ({legend_note})" + (f" | upsample x{upsample}" if upsample and upsample > 1 else ""))
            if color_mode == "time":
                fig.colorbar(sc3, ax=ax, fraction=0.03, pad=0.06, label="time")
            plt.tight_layout()
            out_png3d = os.path.join(outdir, f"attractor3d_{title_model}_e{epoch_found:03d}.png")
            plt.savefig(out_png3d, dpi=160); plt.close(fig)

    print(f"[ok] Saved: {out_png}" + (f"\n[ok] Saved: {out_png3d}" if out_png3d else ""))

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Папка модели с states_epoch_*.npz")
    ap.add_argument("--epoch", type=int, default=None, help="Какой epoch выбрать (по умолчанию последний)")
    ap.add_argument("--which_seq", type=int, default=None, help="Если задан — брать конкретный индекс из H_seq")
    ap.add_argument("--prefer_dense", action="store_true", help="Брать самую длинную последовательность (обычно dense)")
    ap.add_argument("--color", type=str, default="time", choices=["time", "cluster"], help="Окраска: по времени или по кластерам")
    ap.add_argument("--k", type=int, default=8, help="Число кластеров для color=cluster")
    ap.add_argument("--upsample", type=int, default=1, help="Апсемплинг линии (>=1)")
    ap.add_argument("--plot3d", action="store_true", help="Сохранить также 3D-картинку")
    ap.add_argument("--outdir", type=str, default="results/attractor_plots")
    args = ap.parse_args()

    plot_attractor(args.run_dir, args.epoch, args.which_seq, args.prefer_dense,
                   args.color, args.k, args.upsample, args.plot3d, args.outdir)

if __name__ == "__main__":
    main()
