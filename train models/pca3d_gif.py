# pca3d_gif.py
# -*- coding: utf-8 -*-
"""
3D PCA + вращающийся GIF:
- режим 1: --csv <path> и --cols X Y Z U  (берём числовые столбцы, делаем PCA→3D)
- режим 2: --states <path_to_states_epoch_xxx.npz> (конкатенируем H_seq, делаем PCA→3D)
"""

import os, argparse, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (регистрация 3D-проекции)
from sklearn.decomposition import PCA
import imageio.v2 as imageio

def load_from_csv(path, cols, max_points=None):
    df = pd.read_csv(path)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"В CSV нет столбца '{c}'. Доступные: {list(df.columns)}")
    X = df[cols].select_dtypes(include=[np.number]).values.astype("float32")
    if max_points is not None and len(X) > max_points:
        idx = np.linspace(0, len(X) - 1, max_points).astype(int)
        X = X[idx]
    return X

def load_from_states(npz_path, which_seq=None, max_points=None):
    dat = np.load(npz_path, allow_pickle=True)
    H_list = dat["H_seq"]
    if which_seq is not None:
        H = np.asarray(H_list[int(which_seq)], dtype="float32")
    else:
        H = np.concatenate([np.asarray(h, dtype="float32") for h in H_list], axis=0)
    if max_points is not None and len(H) > max_points:
        idx = np.linspace(0, len(H) - 1, max_points).astype(int)
        H = H[idx]
    return H

def pca3d(X):
    X = np.asarray(X, dtype="float32")
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)  # нормировка по признакам
    Z = PCA(n_components=3, random_state=0).fit_transform(X)  # [N,3]
    return Z

def make_rotating_gif(Z, out_gif, frames=90, elev=20, azim_start=0, dpi=110):
    """Z: [N,3]; цвет — градиент по времени"""
    # цвета вдоль времени
    t = np.linspace(0, 1, len(Z))
    colors = cm.viridis(t)  # градиент (можно заменить на любую cmap)

    imgs = []
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # фиксируем лимиты, чтобы масштаб не прыгал
    lim = np.percentile(np.abs(Z), 99)  # устойчиво к выбросам
    lim = max(lim, 1e-3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)

    sc = ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=6, c=colors, depthshade=True)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

    for i in range(frames):
        az = azim_start + (360.0 * i / frames)
        ax.view_init(elev=elev, azim=az)
        fig.tight_layout()
        # отрисуем кадр в буфер и в GIF
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(img.copy())

    imageio.mimsave(out_gif, imgs, duration=0.08, loop=0)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, help="Путь к CSV (например, lorenze_attractor.csv)")
    ap.add_argument("--cols", nargs="+", default=["X","Y","Z","U"], help="Колонки для PCA (из CSV)")
    ap.add_argument("--states", type=str, help="Путь к states_epoch_XXX.npz (из train_*.py)")
    ap.add_argument("--which_seq", type=int, default=None, help="Номер окна из H_seq (если не задано — конкатенируем все)")
    ap.add_argument("--out", type=str, required=True, help="Имя выходного GIF, напр. pca3d.gif")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--elev", type=float, default=20.0)
    ap.add_argument("--azim_start", type=float, default=0.0)
    ap.add_argument("--max_points", type=int, default=8000, help="Ограничить число точек для скорости")
    args = ap.parse_args()

    if (args.csv is None) == (args.states is None):
        raise SystemExit("Укажи ровно один источник: либо --csv, либо --states")

    if args.csv:
        X = load_from_csv(args.csv, args.cols, max_points=args.max_points)
    else:
        X = load_from_states(args.states, which_seq=args.which_seq, max_points=args.max_points)

    Z = pca3d(X)
    os.makedirs(os.path.dirname(os.path.abspath(args.out) or "."), exist_ok=True)
    make_rotating_gif(Z, args.out, frames=args.frames, elev=args.elev, azim_start=args.azim_start)
    print(f"Saved GIF to: {args.out}  | points={len(Z)}")

if __name__ == "__main__":
    main()
