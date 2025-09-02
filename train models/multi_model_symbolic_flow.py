# multi_model_symbolic_flow.py
# -*- coding: utf-8 -*-

import os, re, glob, math, argparse
from typing import List, Tuple, Dict
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ---------- базовые утилиты ----------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def parse_epoch_from_name(path: str) -> int:
    m = re.search(r"states_epoch_(\d+)\.npz$", os.path.basename(path))
    return int(m.group(1)) if m else -1

def latest_states_file(run_dir: str, epoch: int = None) -> str:
    files = sorted(glob.glob(os.path.join(run_dir, "states_epoch_*.npz")), key=parse_epoch_from_name)
    files += sorted(glob.glob(os.path.join(run_dir, "*", "states_epoch_*.npz")), key=parse_epoch_from_name)
    if not files:
        raise FileNotFoundError(f"Нет states_epoch_*.npz в {run_dir}")
    if epoch is None:
        return files[-1]
    for f in files:
        if parse_epoch_from_name(f) == epoch:
            return f
    raise FileNotFoundError(f"В {run_dir} нет epoch={epoch}")

# ---------- символьные метрики (как в analyze_... , короче) ----------

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

# ---------- символизация одного run_dir ----------

def symbolize_hidden(H: np.ndarray, k: int, random_state: int = 0):
    """
    H: [T, D] скрытые состояния одной последовательности (или конкатенация окон).
    Возвращает:
      labels [T], top_states (индексы самых частых кластеров),
      метрики (словарь), а также PC1 (для PermEnt).
    """
    # PCA -> 2D для k-means и диаграмм
    Z2 = PCA(n_components=2, random_state=random_state).fit_transform(H)
    pc1 = Z2[:, 0]

    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(Z2)

    # Символьные метрики
    P = transition_matrix(labels, k)
    metrics = {
        "LZ_norm": lz_normalized(labels),
        "H_markov": markov_entropy_rate(P),
        "PermEnt_norm": permutation_entropy(pc1, m=5, tau=1) / math.log2(math.factorial(5)),
    }

    # Топ-кластеры по частоте (для компактной Sankey)
    from collections import Counter
    cnt = Counter(labels.tolist())
    top_states = [j for j, _ in cnt.most_common(min(6, k))]  # до 6 групп

    return labels, top_states, metrics, Z2

def load_H_from_run(run_dir: str, epoch: int = None, which_seq: int = None) -> np.ndarray:
    """
    Груз H из states_epoch_<epoch>.npz.
    Если which_seq задан, берём одно окно; иначе конкатенируем все окна в T.
    """
    f = latest_states_file(run_dir, epoch=epoch)
    dat = np.load(f, allow_pickle=True)
    H_list = dat['H_seq']
    if which_seq is not None and 0 <= which_seq < len(H_list):
        H = np.asarray(H_list[which_seq], float)
    else:
        H = np.concatenate([np.asarray(h, float) for h in H_list], axis=0)
    return H

# ---------- построение общей Sankey ----------

def build_multimodel_sankey(run_dirs: List[str], model_names: List[str],
                            k: int, epoch: int = None, which_seq: int = None,
                            out_html: str = "symbolic_bridge.html"):

    assert len(run_dirs) == len(model_names)
    # узлы и рёбра
    nodes: List[str] = []
    src: List[int] = []
    tgt: List[int] = []
    val: List[float] = []
    col: List[str] = []

    color_model = ["#5DA5DA", "#60BD68", "#F17CB0", "#B2912F", "#B276B2", "#F15854"]
    color_states = "#A0A0A0"
    color_metrics = {
        "LZ_norm": "#FF7F0E",
        "H_markov": "#1F77B4",
        "PermEnt_norm": "#2CA02C",
        "D2": "#9467BD"  # оставлено на будущее, для D2
    }

    # 0) Input
    idx_input = len(nodes); nodes.append("Input sequence")

    model_root_idx = []
    model_symb_idx = []
    model_metrics_rows = []

    # 1) для каждой модели — загрузка H, символизация, метрики
    for mi, (rd, name) in enumerate(zip(run_dirs, model_names)):
        H = load_H_from_run(rd, epoch=epoch, which_seq=which_seq)  # [T,D]
        T, D = H.shape

        # энергия скрытых состояний — вес рёбер Input->Model
        energy = float(np.linalg.norm(H, ord='fro')) / max(T, 1)

        # узел модели
        idx_model = len(nodes); nodes.append(f"{name}")
        model_root_idx.append(idx_model)

        # ребро Input -> Model
        src.append(idx_input); tgt.append(idx_model); val.append(max(energy, 1e-6)); col.append(color_model[mi % len(color_model)])

        # символизация
        labels, top_states, metrics, Z2 = symbolize_hidden(H, k=k, random_state=0)

        # узел «Symbolization (k=K)» для модели
        idx_symb = len(nodes); nodes.append(f"{name}: Symbolization (k={k})")
        model_symb_idx.append(idx_symb)

        # ребро Model -> Symbolization
        src.append(idx_model); tgt.append(idx_symb); val.append(1.0); col.append(color_model[mi % len(color_model)])

        # узлы состояний (топ)
        state_node_idx: Dict[int, int] = {}
        from collections import Counter
        cnt = Counter(labels.tolist()); total = sum(cnt.values()) + 1e-12

        for j in top_states:
            node_j = len(nodes); nodes.append(f"{name}: S{j}")
            state_node_idx[j] = node_j

            # доля кластера
            w = float(cnt[j] / total)
            src.append(idx_symb); tgt.append(node_j); val.append(max(w, 1e-6))
            col.append(color_states)

        # узлы метрик (общие — но с отдельными рёбрами от каждой модели)
        # чтобы не дублировать узлы, создадал их один раз в конце;
        # здесь запоминаю веса для текущей модели
        model_metrics_rows.append({
            "model": name,
            "idx_symb": idx_symb,
            "metrics": metrics
        })

    # 2) глобальные узлы метрик (одни на всех)
    metric_node_idx = {}
    for mname in ["LZ_norm", "H_markov", "PermEnt_norm"]:
        node_m = len(nodes); nodes.append(f"Metric: {mname}")
        metric_node_idx[mname] = node_m

    # 3) рёбра от каждого «Symbolization (model)» к метрикам
    for row in model_metrics_rows:
        idx_symb = row["idx_symb"]
        metrics = row["metrics"]
        for mname, node_m in metric_node_idx.items():
            if mname in metrics:
                w = float(metrics[mname])
                # нормирую, чтобы толщина была читаемой
                # беру мягкую нормировку в [0.2..1.0]
                w_plot = 0.2 + 0.8 * (w / (1.0 + w))
                src.append(idx_symb); tgt.append(node_m); val.append(w_plot)
                col.append(color_metrics[mname])

    # 4) Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=14, thickness=20,
            label=nodes,
            color=["#444"] * len(nodes)
        ),
        link=dict(
            source=src,
            target=tgt,
            value=val,
            color=col
        )
    )])
    fig.update_layout(title_text="Symbolic Bridge across Models", font_size=11, width=1500, height=750)

    ensure_dir(os.path.dirname(out_html) or ".")
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[ok] Saved: {out_html}")
    return fig


# CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dirs", nargs="+", required=True, help="Папки моделей с states_epoch_*.npz")
    ap.add_argument("--model_names", nargs="+", required=False, help="Подписи моделей (по порядку). Если не заданы — берутся из имён папок.")
    ap.add_argument("--k", type=int, default=16, help="Число кластеров для символизации")
    ap.add_argument("--epoch", type=int, default=None, help="Какой epoch взять (по умолчанию последний)")
    ap.add_argument("--which_seq", type=int, default=None, help="Какое окно H_seq взять (по умолчанию все склеить)")
    ap.add_argument("--out_html", type=str, default="results/symbolic_bridge.html")
    args = ap.parse_args()

    if not args.model_names:
        args.model_names = [os.path.basename(os.path.normpath(p)) for p in args.run_dirs]

    build_multimodel_sankey(args.run_dirs, args.model_names, k=args.k,
                            epoch=args.epoch, which_seq=args.which_seq,
                            out_html=args.out_html)

if __name__ == "__main__":
    main()
