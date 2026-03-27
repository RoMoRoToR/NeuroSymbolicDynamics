#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Device helper
# ----------------------------
def resolve_device(requested: str) -> torch.device:
    """Resolve requested device string to an actually available torch.device.

    Supported: auto, cpu, cuda, mps. Falls back to CPU if requested backend is unavailable.
    """
    req = (requested or "auto").strip().lower()

    # Detect backends safely
    cuda_ok = bool(getattr(torch, "cuda", None)) and bool(torch.cuda.is_available())
    mps_ok = bool(getattr(torch.backends, "mps", None)) and bool(torch.backends.mps.is_available())

    if req == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")

    if req == "cuda":
        if cuda_ok:
            return torch.device("cuda")
        print("[WARN] --device cuda requested, but CUDA is not available in this PyTorch build. Falling back to CPU.")
        return torch.device("cpu")

    if req == "mps":
        if mps_ok:
            return torch.device("mps")
        print("[WARN] --device mps requested, but MPS is not available on this machine/PyTorch. Falling back to CPU.")
        return torch.device("cpu")

    # cpu or anything else -> cpu
    if req not in ("cpu",):
        print(f"[WARN] Unknown --device={requested!r}. Falling back to CPU.")
    return torch.device("cpu")


# ----------------------------
# Reproducibility
# ----------------------------
def set_global_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


# ----------------------------
# Data loader: CSV
# ----------------------------
def load_csv_numeric(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # drop first non-numeric (date) column if present
    if df.shape[1] >= 2 and not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = df.iloc[:, 1:]

    # keep numeric only
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {csv_path}")

    # If only one numeric column -> add time index as a second feature (so models have >1 feature)
    if df.shape[1] == 1:
        yname = df.columns[0]
        df = df.rename(columns={yname: "y"})
        df.insert(0, "t", np.arange(len(df), dtype=np.float32))
    return df


def find_target_col(df: pd.DataFrame) -> int:
    cols = df.columns.tolist()
    low = [c.lower() for c in cols]

    # Preferred targets
    if "ot" in low:
        return low.index("ot")
    if "close" in low:
        return low.index("close")
    if "y" in low:
        return low.index("y")

    # Avoid obvious junk
    bad_names = {"ignore"}
    candidates = []
    for i, c in enumerate(cols):
        if c.lower() in bad_names:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().all():
            continue
        if float(s.std(skipna=True)) < 1e-12:   # constant column
            continue
        candidates.append(i)

    if candidates:
        return candidates[-1]

    # fallback
    return df.shape[1] - 1



def make_splits(T: int, train_frac: float, val_frac: float, test_frac: float, gap: int) -> Tuple[slice, slice, slice]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    tr_e = int(T * train_frac)
    va_e = tr_e + int(T * val_frac)
    te_e = T

    tr_s = 0
    va_s = tr_e + gap
    te_s = va_e + gap

    if va_s >= va_e or te_s >= te_e:
        raise ValueError("Split overflow; reduce split_gap or adjust fractions.")
    return slice(tr_s, tr_e), slice(va_s, va_e), slice(te_s, te_e)


# ----------------------------
# Windowed forecasting dataset
# ----------------------------
class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, target_idx: int, seq_len: int, horizon: int):
        self.X = X.astype(np.float32)
        self.target_idx = int(target_idx)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)

        T = self.X.shape[0]
        self.n = T - self.seq_len - self.horizon + 1
        if self.n <= 0:
            raise ValueError("Not enough rows for given seq_len/horizon")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        x = self.X[i:i + self.seq_len, :]
        y = self.X[i + self.seq_len + self.horizon - 1, self.target_idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ----------------------------
# Models
# ----------------------------
class RNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h, c) = self.rnn(x)
        h_last = h[-1]
        y = self.head(h_last).squeeze(-1)
        return y, h_last


class BiRNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, (h, c) = self.rnn(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        y = self.head(h_cat).squeeze(-1)
        return y, h_cat


class TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        if hidden_dim % nhead != 0:
            for h in [8, 4, 2, 1]:
                if hidden_dim % h == 0:
                    nhead = h
                    break
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        z = self.in_proj(x)
        h = self.enc(z)
        h_pool = h.mean(dim=1)
        y = self.head(h_pool).squeeze(-1)
        return y, h_pool


def build_model(name: str, in_dim: int, hidden_dim: int) -> nn.Module:
    name = name.lower()
    if name == "rnn":
        return RNNRegressor(in_dim, hidden_dim)
    if name == "birnn":
        return BiRNNRegressor(in_dim, hidden_dim)
    if name == "transformer":
        return TransformerRegressor(in_dim, hidden_dim)
    raise ValueError(f"Unknown model: {name}")


# ----------------------------
# Fixed subset utilities
# ----------------------------
def fixed_subset_from_loader(dl: DataLoader, max_batches: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    if max_batches <= 0:
        return out
    for i, (X, y) in enumerate(dl):
        out.append((X.clone(), y.clone()))
        if (i + 1) >= max_batches:
            break
    return out


@torch.no_grad()
def eval_loss_on_subset(model: nn.Module, subset: List[Tuple[torch.Tensor, torch.Tensor]], device: str) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    for X, y in subset:
        X = X.to(device)
        y = y.to(device)
        yhat, _ = model(X)
        loss = loss_fn(yhat, y)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_loss_full(model: nn.Module, dl: DataLoader, device: str) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    for X, y in dl:
        X = X.to(device)
        y = y.to(device)
        yhat, _ = model(X)
        loss = loss_fn(yhat, y)
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def eval_repr_on_subset(model: nn.Module, subset: List[Tuple[torch.Tensor, torch.Tensor]], device: str) -> np.ndarray:
    model.eval()
    reps: List[np.ndarray] = []
    for X, _ in subset:
        X = X.to(device)
        _, h = model(X)
        reps.append(to_np(h))
    if not reps:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(reps, axis=0)


def train_one_epoch(model: nn.Module, dl: DataLoader, opt: torch.optim.Optimizer, device: str) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    for X, y in dl:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        yhat, _ = model(X)
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


# ----------------------------
# Noise injection
# ----------------------------
def apply_noise(X: np.ndarray, sigma: float, seed: int, mode: str = "all", target_idx: Optional[int] = None) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return X.astype(np.float32)
    rng = np.random.default_rng(int(seed))
    eps = rng.standard_normal(size=X.shape).astype(np.float32)
    if mode == "all":
        return (X + sigma * eps).astype(np.float32)
    if mode == "target":
        if target_idx is None:
            raise ValueError("target_idx required for noise_mode=target")
        Y = X.copy().astype(np.float32)
        Y[:, int(target_idx)] = (Y[:, int(target_idx)] + sigma * eps[:, int(target_idx)]).astype(np.float32)
        return Y
    raise ValueError(f"Unknown noise_mode: {mode}")


# ----------------------------
# SVCCA (more forgiving defaults)
# ----------------------------
def _pca_topk(X: np.ndarray, k: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0], U.shape[1])
    return (U[:, :k] * S[:k])


def svcca_similarity(A: np.ndarray, B: np.ndarray, k: int = 20) -> float:
    if A.size == 0 or B.size == 0:
        return 0.0
    k = int(k)
    A2 = _pca_topk(A, k)
    B2 = _pca_topk(B, k)
    A2 = A2 - A2.mean(axis=0, keepdims=True)
    B2 = B2 - B2.mean(axis=0, keepdims=True)
    C = (A2.T @ B2) / max(1, A2.shape[0] - 1)
    s = np.linalg.svd(C, compute_uv=False)
    if s.size == 0:
        return 0.0
    return float(np.clip(np.mean(np.abs(s)), 0.0, 1.0))


def find_stop_svcca(reprs: List[np.ndarray], min_epoch: int, dim: int, sim_thr: float, patience: int) -> int:
    bad = 0
    for e in range(1, len(reprs) + 1):
        if e < max(min_epoch, 2):
            continue
        sim = svcca_similarity(reprs[e - 2], reprs[e - 1], k=dim)
        if sim >= float(sim_thr):
            bad += 1
        else:
            bad = 0
        if bad >= int(patience):
            return e
    return len(reprs)


# ----------------------------
# CDSC (Correlation Dimension Stabilization Criterion) on representations
# ----------------------------
def _pairwise_dists(X: np.ndarray) -> np.ndarray:
    # X: [N, D] -> condensed distance matrix length N*(N-1)/2
    # We'll compute efficiently with dot products.
    X = X.astype(np.float64)
    G = X @ X.T
    diag = np.diag(G)
    D2 = diag[:, None] - 2.0 * G + diag[None, :]
    D2 = np.maximum(D2, 0.0)
    # condensed
    idx = np.triu_indices(D2.shape[0], k=1)
    return np.sqrt(D2[idx])


def estimate_corr_dim(X: np.ndarray, sample_n: int = 256, n_radii: int = 12, q_lo: float = 0.10, q_hi: float = 0.90) -> float:
    """
    Correlation dimension estimate via Grassberger–Procaccia style:
      - sample points
      - compute pairwise distances
      - choose radii as quantiles of distances
      - compute C(r) = P(dist < r)
      - dim ~ slope of log C(r) vs log r in mid-quantile region
    """
    if X.size == 0:
        return float("nan")
    rng = np.random.default_rng(12345)
    N = X.shape[0]
    n = min(int(sample_n), N)
    if n < 30:
        return float("nan")
    idx = rng.choice(N, size=n, replace=False)
    Y = X[idx]

    d = _pairwise_dists(Y)
    d = d[np.isfinite(d)]
    if d.size < 100:
        return float("nan")

    # radii grid from quantiles
    lo = float(np.quantile(d, q_lo))
    hi = float(np.quantile(d, q_hi))
    if not (hi > lo > 0.0):
        return float("nan")

    rs = np.geomspace(lo, hi, num=int(n_radii)).astype(np.float64)
    C = np.array([np.mean(d < r) for r in rs], dtype=np.float64)
    eps = 1e-12
    x = np.log(rs + eps)
    y = np.log(C + eps)

    # linear regression slope
    x0 = x - x.mean()
    denom = float(np.dot(x0, x0) + 1e-12)
    slope = float(np.dot(x0, y - y.mean()) / denom)
    return float(max(slope, 0.0))


def find_stop_cdsc(reprs: List[np.ndarray],
                   min_epoch: int,
                   patience: int,
                   thr: float,
                   sample_n: int,
                   n_radii: int,
                   smooth_win: int = 3) -> int:
    """
    CDSC: stop when correlation dimension stabilizes:
      |dim_t - dim_{t-1}| <= thr for `patience` checks.
    """
    dims: List[float] = []
    for e in range(1, len(reprs) + 1):
        dim = estimate_corr_dim(reprs[e - 1], sample_n=sample_n, n_radii=n_radii)
        dims.append(dim)

    # smooth dims (simple moving avg, ignoring nans)
    def smooth(vals: List[float], w: int) -> List[float]:
        if w <= 1:
            return vals[:]
        out = []
        for i in range(len(vals)):
            s = max(0, i - w + 1)
            seg = [v for v in vals[s:i + 1] if np.isfinite(v)]
            out.append(float(np.mean(seg)) if seg else float("nan"))
        return out

    dims_s = smooth(dims, int(smooth_win))

    bad = 0
    for e in range(1, len(dims_s) + 1):
        if e < max(min_epoch, 2):
            continue
        a = dims_s[e - 2]
        b = dims_s[e - 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            bad = 0
            continue
        if abs(b - a) <= float(thr):
            bad += 1
        else:
            bad = 0
        if bad >= int(patience):
            return e
    return len(dims_s)


# ----------------------------
# Val-based baselines: Patience + Slope
# ----------------------------
def _lin_slope(y: np.ndarray) -> float:
    n = int(y.size)
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x) + 1e-12)
    return float(np.dot(x, y - y.mean()) / denom)


def find_stop_patience(val: List[float], min_epoch: int, patience: int, min_delta: float) -> int:
    best = float("inf")
    bad = 0
    for e in range(1, len(val) + 1):
        v = float(val[e - 1])
        if v < best - float(min_delta):
            best = v
            bad = 0
        else:
            if e >= int(min_epoch):
                bad += 1
        if bad >= int(patience):
            return e
    return len(val)


def find_stop_slope(val: List[float], min_epoch: int, win: int, slope_eps: float, patience: int) -> int:
    bad = 0
    for e in range(1, len(val) + 1):
        if e < max(int(min_epoch), int(win)):
            continue
        y = np.array(val[e - win:e], dtype=np.float64)
        slope = _lin_slope(y)
        if slope >= -float(slope_eps):
            bad += 1
        else:
            bad = 0
        if bad >= int(patience):
            return e
    return len(val)


# ----------------------------
# SES helpers
# ----------------------------
def moving_avg(x: List[float], w: int) -> List[float]:
    if w <= 1:
        return x[:]
    out = []
    for i in range(len(x)):
        s = max(0, i - w + 1)
        out.append(float(np.mean(x[s:i + 1])))
    return out


def robust_zscore_1d(x: List[float]) -> List[float]:
    a = np.asarray(x, dtype=np.float64)
    med = float(np.median(a))
    q1 = float(np.percentile(a, 25))
    q3 = float(np.percentile(a, 75))
    iqr = q3 - q1
    if iqr < 1e-12:
        sd = float(a.std())
        scale = sd if sd > 1e-12 else 1.0
    else:
        scale = iqr
    z = (a - med) / scale
    return [float(v) for v in z]


def rep_change_metrics(prev: np.ndarray, cur: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    mu0 = prev.mean(axis=0)
    mu1 = cur.mean(axis=0)

    d_mu = float(np.linalg.norm(mu1 - mu0) / (np.linalg.norm(mu0) + eps))

    X0 = prev - mu0[None, :]
    X1 = cur - mu1[None, :]

    C0 = (X0.T @ X0) / max(1, prev.shape[0] - 1)
    C1 = (X1.T @ X1) / max(1, cur.shape[0] - 1)

    cov_fro0 = float(np.sqrt(np.sum(C0 * C0)))
    cov_fro_diff = float(np.sqrt(np.sum((C1 - C0) * (C1 - C0))))
    d_cov = float(cov_fro_diff / (cov_fro0 + eps))

    s0 = np.linalg.svd(X0, compute_uv=False, full_matrices=False)
    s1 = np.linalg.svd(X1, compute_uv=False, full_matrices=False)
    k = min(len(s0), len(s1), 32)
    s0 = s0[:k]
    s1 = s1[:k]
    d_s = float(np.linalg.norm(s1 - s0, ord=1) / (np.linalg.norm(s0, ord=1) + eps))

    denom = (np.linalg.norm(mu0) * np.linalg.norm(mu1) + eps)
    cos = float(np.dot(mu0, mu1) / denom)
    d_cos = float(1.0 - np.clip(cos, -1.0, 1.0))
    return {"d_mu": d_mu, "d_cov": d_cov, "d_s": d_s, "d_cos": d_cos}


def ses_ensemble_from_metrics(metric_series: Dict[str, List[float]],
                              rank_top: float,
                              smooth_win: int) -> List[float]:
    """
    Robust-normalize each metric over time -> rank-median top-q smallest -> smooth.
    Lower = more stable.
    """
    keys = sorted(metric_series.keys())
    T = len(metric_series[keys[0]])
    q = float(rank_top)
    q = min(max(q, 0.25), 1.0)
    m = len(keys)
    k_top = max(1, int(math.ceil(q * m)))

    normed = {k: robust_zscore_1d(metric_series[k]) for k in keys}

    raw = []
    for t in range(T):
        vals = [normed[k][t] for k in keys]
        vals_sorted = sorted(vals)
        top = vals_sorted[:k_top]
        raw.append(float(np.median(top)))
    return moving_avg(raw, int(smooth_win))


# ----------------------------
# Mapper discretization (numpy-only)
# ----------------------------
def _pca1_lens(X: np.ndarray) -> np.ndarray:
    """
    First principal component (1D lens) for Mapper.
    Works with X shape (N, D), returns lens shape (N,).
    """
    if X.size == 0:
        return np.zeros((0,), dtype=np.float64)
    Xc = X.astype(np.float64) - X.mean(axis=0, keepdims=True).astype(np.float64)
    # SVD: Xc = U S V^T, first PC direction is V[0]
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        v1 = Vt[0]
    except np.linalg.LinAlgError:
        # fallback: random direction
        rng = np.random.default_rng(0)
        v1 = rng.standard_normal(size=(Xc.shape[1],)).astype(np.float64)
        v1 /= (np.linalg.norm(v1) + 1e-12)
    lens = Xc @ v1
    return lens.astype(np.float64)


def _cover_intervals(lens: np.ndarray, n_bins: int, overlap: float) -> List[Tuple[float, float]]:
    """
    Create an overlapping cover of [min(lens), max(lens)] with n_bins intervals.
    overlap in [0, 0.95] is the fraction overlap between consecutive intervals.
    """
    n_bins = int(max(1, n_bins))
    overlap = float(np.clip(overlap, 0.0, 0.95))
    if lens.size == 0:
        return []
    lo = float(np.min(lens))
    hi = float(np.max(lens))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return [(lo - 0.5, lo + 0.5)]
    base_w = (hi - lo) / float(n_bins)
    # widen intervals to achieve overlap; step stays base_w
    w = base_w / max(1e-6, (1.0 - overlap))
    step = base_w
    intervals = []
    for i in range(n_bins):
        a = lo + i * step
        b = a + w
        intervals.append((a, b))
    return intervals


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    G = X @ X.T
    diag = np.diag(G)
    D2 = diag[:, None] - 2.0 * G + diag[None, :]
    D2[D2 < 0.0] = 0.0
    return D2


def _knn_graph_components(X: np.ndarray, local_k: int) -> List[np.ndarray]:
    """
    Build an undirected kNN graph (symmetrized) and return connected components
    as arrays of local indices into X.
    """
    n = int(X.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [np.array([0], dtype=np.int64)]
    k = int(max(1, min(local_k, n - 1)))

    D2 = _pairwise_sq_dists(X)
    # argsort gives self at position 0
    nn_idx = np.argsort(D2, axis=1)[:, 1:k + 1]

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in nn_idx[i]:
            j = int(j)
            adj[i].append(j)
            adj[j].append(i)

    # BFS/DFS components
    seen = np.zeros((n,), dtype=bool)
    comps = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(np.array(comp, dtype=np.int64))
    return comps


def build_mapper_graph(points: np.ndarray,
                       n_bins: int = 10,
                       overlap: float = 0.30,
                       local_k: int = 10,
                       merge_eps: float = 0.50,
                       max_nodes: int = 800) -> Tuple[np.ndarray, List[np.ndarray], List[Tuple[int, int]]]:
    """
    Minimal Mapper:
      - lens: PCA1
      - cover: overlapping intervals
      - clustering: connected components in kNN graph within each interval
      - edges: nodes share at least one original point index (due to overlap)
      - merge: union nodes with centroid distance <= merge_eps (optional; keeps membership unioned)
    Returns:
      centroids: (K, D)
      members: list of arrays of global indices into `points`
      edges: list of (u, v) undirected edges over node ids [0..K-1]
    """
    X = points.astype(np.float64)
    N, D = X.shape[0], X.shape[1]
    if N == 0:
        return np.zeros((0, D), dtype=np.float64), [], []
    lens = _pca1_lens(X)
    intervals = _cover_intervals(lens, int(n_bins), float(overlap))

    nodes_members: List[np.ndarray] = []
    # Build nodes
    for (a, b) in intervals:
        mask = (lens >= a) & (lens <= b)
        idx = np.nonzero(mask)[0].astype(np.int64)
        if idx.size < 2:
            continue
        Xsub = X[idx]
        comps = _knn_graph_components(Xsub, int(local_k))
        for c in comps:
            if c.size == 0:
                continue
            members = idx[c]
            nodes_members.append(np.unique(members))

            if len(nodes_members) >= int(max_nodes):
                break
        if len(nodes_members) >= int(max_nodes):
            break

    if not nodes_members:
        # fallback: single node with all points
        nodes_members = [np.arange(N, dtype=np.int64)]

    # centroids
    centroids = np.stack([X[m].mean(axis=0) for m in nodes_members], axis=0)

    # build edges by shared membership
    point_to_nodes: Dict[int, List[int]] = {}
    for ni, mem in enumerate(nodes_members):
        for p in mem.tolist():
            point_to_nodes.setdefault(int(p), []).append(int(ni))

    edge_set = set()
    for p, lst in point_to_nodes.items():
        if len(lst) >= 2:
            lst2 = sorted(set(lst))
            for i in range(len(lst2)):
                for j in range(i + 1, len(lst2)):
                    edge_set.add((lst2[i], lst2[j]))

    edges = sorted(edge_set)

    # optional merge by centroid distance (union-find)
    merge_eps = float(merge_eps)
    if merge_eps > 0.0 and centroids.shape[0] > 1:
        K = centroids.shape[0]
        parent = np.arange(K, dtype=np.int64)

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # pairwise centroid dists (cheap if K small; capped by max_nodes)
        C = centroids
        # compute squared distances
        G = C @ C.T
        diag = np.diag(G)
        D2 = diag[:, None] - 2.0 * G + diag[None, :]
        D2[D2 < 0.0] = 0.0
        thr2 = merge_eps * merge_eps
        for i in range(K):
            for j in range(i + 1, K):
                if D2[i, j] <= thr2:
                    union(i, j)

        # rebuild merged nodes
        groups: Dict[int, List[int]] = {}
        for i in range(K):
            r = int(find(i))
            groups.setdefault(r, []).append(i)

        if len(groups) < K:
            new_members = []
            for r, idxs in groups.items():
                mem = np.concatenate([nodes_members[i] for i in idxs], axis=0)
                new_members.append(np.unique(mem))
            nodes_members = new_members
            centroids = np.stack([X[m].mean(axis=0) for m in nodes_members], axis=0)

            # rebuild edges after merge
            point_to_nodes = {}
            for ni, mem in enumerate(nodes_members):
                for p in mem.tolist():
                    point_to_nodes.setdefault(int(p), []).append(int(ni))
            edge_set = set()
            for p, lst in point_to_nodes.items():
                if len(lst) >= 2:
                    lst2 = sorted(set(lst))
                    for i in range(len(lst2)):
                        for j in range(i + 1, len(lst2)):
                            edge_set.add((lst2[i], lst2[j]))
            edges = sorted(edge_set)

    return centroids.astype(np.float64), nodes_members, edges


def mapper_assignments(points: np.ndarray,
                       centroids: np.ndarray,
                       members: List[np.ndarray]) -> np.ndarray:
    """
    Deterministic assignment of each point to exactly one Mapper node:
      - candidates = nodes that contain the point
      - choose the candidate with closest centroid (tie -> smallest node id)
      - if a point has no candidates (shouldn't), assign to nearest centroid globally
    Returns assignments shape (N,), values in [0..K-1]
    """
    X = points.astype(np.float64)
    N = X.shape[0]
    K = centroids.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N,), dtype=np.int64)

    # point -> candidate nodes
    cand = [[] for _ in range(N)]
    for k, mem in enumerate(members):
        for i in mem.tolist():
            if 0 <= int(i) < N:
                cand[int(i)].append(int(k))

    assign = np.zeros((N,), dtype=np.int64)
    for i in range(N):
        cands = cand[i]
        if cands:
            # pick closest centroid among candidates
            xi = X[i]
            best_k = cands[0]
            best_d2 = float(np.sum((xi - centroids[best_k]) ** 2))
            for k in cands[1:]:
                d2 = float(np.sum((xi - centroids[k]) ** 2))
                if d2 < best_d2 - 1e-12 or (abs(d2 - best_d2) <= 1e-12 and k < best_k):
                    best_d2 = d2
                    best_k = k
            assign[i] = int(best_k)
        else:
            # fallback: nearest centroid
            xi = X[i]
            d2 = np.sum((centroids - xi[None, :]) ** 2, axis=1)
            assign[i] = int(np.argmin(d2))
    return assign


def _safe_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p / max(1e-12, float(p.sum()))
    m = p > 0.0
    return float(-np.sum(p[m] * np.log(p[m] + 1e-12)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(1e-12, float(p.sum()))
    q = q / max(1e-12, float(q.sum()))
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p + 1e-12) - np.log(m + 1e-12))))
    kl_qm = float(np.sum(q * (np.log(q + 1e-12) - np.log(m + 1e-12))))
    return float(0.5 * (kl_pm + kl_qm))


def rep_change_metrics_mapper(prev: np.ndarray,
                              cur: np.ndarray,
                              mapper_bins: int,
                              mapper_overlap: float,
                              mapper_local_k: int,
                              mapper_merge_eps: float,
                              mapper_max_nodes: int) -> Dict[str, float]:
    """
    Drift metrics computed in Mapper-symbol space (functional Mapper in SES):
      - Build Mapper graph on concatenated points [prev; cur]
      - Assign each point to exactly one Mapper node
      - Build node histograms p0 and p1
      - Compute drift/divergence + geometry/graph stats deltas
    """
    eps = 1e-8
    if prev.size == 0 or cur.size == 0:
        return {
            "jsd": 1.0,
            "tv": 1.0,
            "d_wmu": 1.0,
            "d_wcov": 1.0,
            "d_ent": 1.0,
            "d_deg": 1.0,
            "d_active": 1.0,
        }

    n0 = int(prev.shape[0])
    X = np.concatenate([prev, cur], axis=0).astype(np.float64)

    centroids, members, edges = build_mapper_graph(
        X,
        n_bins=int(mapper_bins),
        overlap=float(mapper_overlap),
        local_k=int(mapper_local_k),
        merge_eps=float(mapper_merge_eps),
        max_nodes=int(mapper_max_nodes),
    )
    K = int(centroids.shape[0])
    if K <= 0:
        return {
            "jsd": 1.0,
            "tv": 1.0,
            "d_wmu": 1.0,
            "d_wcov": 1.0,
            "d_ent": 1.0,
            "d_deg": 1.0,
            "d_active": 1.0,
        }

    assign = mapper_assignments(X, centroids, members)

    # histograms per epoch
    a0 = assign[:n0]
    a1 = assign[n0:]
    p0 = np.bincount(a0, minlength=K).astype(np.float64)
    p1 = np.bincount(a1, minlength=K).astype(np.float64)
    p0 /= max(1e-12, float(p0.sum()))
    p1 /= max(1e-12, float(p1.sum()))

    # divergences
    jsd = _js_divergence(p0, p1)
    tv = float(0.5 * np.sum(np.abs(p1 - p0)))

    # weighted centroid mean drift
    mu0 = (p0[:, None] * centroids).sum(axis=0)
    mu1 = (p1[:, None] * centroids).sum(axis=0)
    d_wmu = float(np.linalg.norm(mu1 - mu0) / (np.linalg.norm(mu0) + eps))

    # weighted centroid covariance drift (Frobenius)
    C0 = centroids - mu0[None, :]
    C1 = centroids - mu1[None, :]
    Cov0 = (C0.T * p0[None, :]) @ C0
    Cov1 = (C1.T * p1[None, :]) @ C1
    cov_fro0 = float(np.sqrt(np.sum(Cov0 * Cov0)))
    cov_fro_diff = float(np.sqrt(np.sum((Cov1 - Cov0) * (Cov1 - Cov0))))
    d_wcov = float(cov_fro_diff / (cov_fro0 + eps))

    # entropy drift
    d_ent = float(abs(_safe_entropy(p1) - _safe_entropy(p0)))

    # graph degree drift (weighted avg degree)
    deg = np.zeros((K,), dtype=np.float64)
    for u, v in edges:
        if 0 <= int(u) < K and 0 <= int(v) < K and u != v:
            deg[int(u)] += 1.0
            deg[int(v)] += 1.0
    d0 = float(np.dot(p0, deg))
    d1 = float(np.dot(p1, deg))
    d_deg = float(abs(d1 - d0) / (abs(d0) + eps))

    # active nodes drift
    a0n = float(np.sum(p0 > 0.0))
    a1n = float(np.sum(p1 > 0.0))
    d_active = float(abs(a1n - a0n) / max(1.0, float(K)))

    return {
        "jsd": float(jsd),
        "tv": float(tv),
        "d_wmu": float(d_wmu),
        "d_wcov": float(d_wcov),
        "d_ent": float(d_ent),
        "d_deg": float(d_deg),
        "d_active": float(d_active),
    }



# ----------------------------
# SES-Solo v2 (fixed for noise): plateau + rise-from-min trigger
# ----------------------------
def find_stop_ses_solo_v2(train_loss: List[float],
                          reprs: List[np.ndarray],
                          min_epoch: int,
                          patience: int,
                          ahead_win: int,
                          smooth_win: int,
                          rank_top: float,
                          base_win: int,
                          gamma_loss: float,
                          beta_slope: float,
                          rise_kappa: float,
                          rise_patience: int,
                          ses_repr: str = "raw",
                          mapper_bins: int = 10,
                          mapper_overlap: float = 0.30,
                          mapper_local_k: int = 10,
                          mapper_merge_eps: float = 0.50,
                          mapper_max_nodes: int = 800) -> int:
    """
    Standalone SES (NO val), robust on high noise.

    Build ensemble drift score E[e] (lower => stable), robust-normalized.

    Stop when (learning slowed) AND one of:
      A) drift plateau: slope(E) >= -slope_eps
      B) drift rises from its running minimum: E[e] - min(E[:e]) >= rise_thr
    for `patience` checks (rise trigger uses rise_patience).

    Adaptive thresholds:
      - slope_eps from MAD of early slopes
      - rise_thr from MAD of early E-values
      - loss_eps from median early loss drop * gamma_loss
    """
    if len(train_loss) != len(reprs):
        raise ValueError("train_loss and reprs must have same length")
    E = len(train_loss)
    if E <= max(min_epoch + 2, ahead_win + 1):
        return E

    # metric series across epochs
    use_mapper = (str(ses_repr).lower() == "mapper")

    if use_mapper:
        sample_m = rep_change_metrics_mapper(
            reprs[0], reprs[1],
            mapper_bins=mapper_bins,
            mapper_overlap=mapper_overlap,
            mapper_local_k=mapper_local_k,
            mapper_merge_eps=mapper_merge_eps,
            mapper_max_nodes=mapper_max_nodes,
        ) if E >= 2 else {"jsd": 1.0, "tv": 1.0, "d_wmu": 1.0, "d_wcov": 1.0, "d_ent": 1.0, "d_deg": 1.0, "d_active": 1.0}
        keys = sorted(sample_m.keys())
    else:
        sample_m = rep_change_metrics(reprs[0], reprs[1]) if E >= 2 else {"d_mu": 1.0, "d_cov": 1.0, "d_s": 1.0, "d_cos": 1.0}
        keys = sorted(sample_m.keys())

    metric_series = {k: [] for k in keys}
    for e in range(1, E + 1):
        if e == 1:
            for k in keys:
                metric_series[k].append(1.0)
        else:
            if use_mapper:
                m = rep_change_metrics_mapper(
                    reprs[e - 2], reprs[e - 1],
                    mapper_bins=mapper_bins,
                    mapper_overlap=mapper_overlap,
                    mapper_local_k=mapper_local_k,
                    mapper_merge_eps=mapper_merge_eps,
                    mapper_max_nodes=mapper_max_nodes,
                )
            else:
                m = rep_change_metrics(reprs[e - 2], reprs[e - 1])
            for k in keys:
                metric_series[k].append(float(m[k]))

    ens = ses_ensemble_from_metrics(metric_series, rank_top=rank_top, smooth_win=smooth_win)

    # slopes over ahead_win
    slopes = [0.0] * E
    for i in range(E):
        if (i + 1) < ahead_win:
            continue
        y = np.array(ens[i - ahead_win + 1:i + 1], dtype=np.float64)
        slopes[i] = _lin_slope(y)

    b0 = max(int(min_epoch), int(ahead_win))
    b1 = min(E, b0 + int(base_win))

    # adaptive slope_eps
    early_sl = np.asarray([abs(slopes[i]) for i in range(b0, b1)], dtype=np.float64)
    if early_sl.size >= 3:
        med = float(np.median(early_sl))
        mad = float(np.median(np.abs(early_sl - med)) + 1e-12)
        slope_eps = float(max(1e-6, float(beta_slope) * mad))
    else:
        slope_eps = 1e-4

    # adaptive rise_thr from early ensemble variability
    early_ens = np.asarray([ens[i] for i in range(b0, b1)], dtype=np.float64)
    if early_ens.size >= 5:
        med_e = float(np.median(early_ens))
        mad_e = float(np.median(np.abs(early_ens - med_e)) + 1e-12)
        rise_thr = float(max(1e-6, float(rise_kappa) * mad_e))
    else:
        rise_thr = 1e-3

    # training loss "slow" threshold
    drops = [0.0] * E
    for i in range(1, E):
        drops[i] = float(train_loss[i - 1] - train_loss[i])
    early_drops = np.asarray([max(0.0, drops[i]) for i in range(b0, b1)], dtype=np.float64)
    early_drop = float(np.median(early_drops)) if early_drops.size > 0 else 0.0
    loss_eps = float(max(1e-12, float(gamma_loss) * early_drop))

    # running min of ensemble
    run_min = [ens[0]]
    for i in range(1, E):
        run_min.append(float(min(run_min[-1], ens[i])))

    bad_plateau = 0
    bad_rise = 0

    for e in range(1, E + 1):
        i = e - 1
        if e < max(int(min_epoch), int(ahead_win)):
            continue

        # learning slowed: recent median drop on last ahead_win
        j0 = max(1, i - int(ahead_win) + 1)
        recent_drop = float(np.median([max(0.0, drops[j]) for j in range(j0, i + 1)]))
        loss_slow = (recent_drop <= loss_eps)

        if not loss_slow:
            bad_plateau = 0
            bad_rise = 0
            continue

        # A) plateau
        if slopes[i] >= -slope_eps:
            bad_plateau += 1
        else:
            bad_plateau = 0

        # B) rise from minimum (memorization/noise-fitting)
        if (ens[i] - run_min[i]) >= rise_thr:
            bad_rise += 1
        else:
            bad_rise = 0

        if bad_plateau >= int(patience):
            return e
        if bad_rise >= int(rise_patience):
            return e

    return E


# ----------------------------
# Summary tables
# ----------------------------
def _median_iqr_str(x: np.ndarray, fmt: str) -> str:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    return (fmt.format(med) + " [" + fmt.format(iqr) + "]")


def make_summary_tables(df_all: pd.DataFrame, oracle_eps: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["ses", "pat", "slope", "svcca", "cdsc"]
    group_cols = ["dataset", "model", "noise_sigma"]

    df_all = df_all.copy()
    df_all["noise_sigma"] = pd.to_numeric(df_all["noise_sigma"], errors="coerce")

    rows = []
    for (ds, model, sigma), g in df_all.groupby(group_cols, dropna=False):
        sigma = float(sigma) if sigma is not None else float("nan")
        out = {"dataset": ds, "model": model, "noise_sigma": sigma}
        for m in methods:
            e = g[f"{m}_epoch"].to_numpy(dtype=float)
            v = g[f"{m}_val_at_stop"].to_numpy(dtype=float)
            d = g[f"{m}_delta_best"].to_numpy(dtype=float)
            s = g[f"{m}_saved_epochs"].to_numpy(dtype=float)

            out[f"{m}_stop_epoch_med_iqr"] = _median_iqr_str(e, "{:.1f}")
            out[f"{m}_val_at_stop_med_iqr"] = _median_iqr_str(v, "{:.6f}")
            out[f"{m}_delta_best_med_iqr"] = _median_iqr_str(d, "{:.6f}")
            out[f"{m}_epochs_saved_med_iqr"] = _median_iqr_str(s, "{:.1f}")
            out[f"{m}_within_oracle@{oracle_eps:.2f}"] = float(np.mean(d <= float(oracle_eps)))

        rows.append(out)

    df_summary = pd.DataFrame(rows).sort_values(["dataset", "model", "noise_sigma"]).reset_index(drop=True)

    compact_rows = []
    method_label = {"ses": "SES", "pat": "PAT", "slope": "SLOPE", "svcca": "SVCCA", "cdsc": "CDSC"}
    # IMPORTANT: groupby by column name (not list) to avoid tuple key
    for sigma, g in df_all.groupby("noise_sigma", dropna=False):
        sigma = float(sigma) if sigma is not None else float("nan")
        for m in methods:
            compact_rows.append({
                "noise_sigma": sigma,
                "method": method_label[m],
                "loss@stop_median": float(np.median(g[f"{m}_val_at_stop"].to_numpy(dtype=float))),
                "stop_epoch_median": float(np.median(g[f"{m}_epoch"].to_numpy(dtype=float))),
                "epochs_saved_median": float(np.median(g[f"{m}_saved_epochs"].to_numpy(dtype=float))),
            })

    df_compact = pd.DataFrame(compact_rows).sort_values(["noise_sigma", "method"]).reset_index(drop=True)
    return df_summary, df_compact


# ----------------------------
# Args
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ett_dir", type=str, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out_root", type=str, required=True)

    ap.add_argument("--models", nargs="+", default=["rnn", "birnn", "transformer"])
    ap.add_argument("--n_runs", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)

    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--pred_horizon", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])

    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.1)
    ap.add_argument("--split_gap", type=int, default=0)

    # noise sweep
    ap.add_argument("--noise_sigma", type=float, default=0.0)
    ap.add_argument("--noise_min", type=float, default=None)
    ap.add_argument("--noise_max", type=float, default=None)
    ap.add_argument("--noise_step", type=float, default=None)
    ap.add_argument("--noise_mode", type=str, default="all", choices=["all", "target"])

    # eval subsets
    ap.add_argument("--val_eval_batches", type=int, default=0, help="0 => full val, >0 => fixed subset")
    ap.add_argument("--repr_source", type=str, default="train", choices=["train", "val"])
    ap.add_argument("--repr_batches", type=int, default=8)

    # SES mode (we keep only solo v2 here as the main SES)
    ap.add_argument("--ses_rank_top", type=float, default=0.40)
    ap.add_argument("--ses_patience", type=int, default=3)
    ap.add_argument("--ses_ahead_win", type=int, default=6)
    ap.add_argument("--ses_smooth_win", type=int, default=5)
    ap.add_argument("--ses_min_epoch", type=int, default=2)
    ap.add_argument("--ses_base_win", type=int, default=12)
    ap.add_argument("--ses_gamma_loss", type=float, default=0.20)
    ap.add_argument("--ses_beta_slope", type=float, default=1.0)
    ap.add_argument("--ses_rise_kappa", type=float, default=2.0)
    ap.add_argument("--ses_rise_patience", type=int, default=2)

    # SES representation space for drift metrics
    #   raw    : original continuous drift metrics (d_mu, d_cov, d_s, d_cos)
    #   mapper : build Mapper on [repr_{e-1}; repr_e] and compute drift in symbolic Mapper space
    ap.add_argument("--ses_repr", type=str, default="raw", choices=["raw", "mapper"])

    # Mapper params (used when --ses_repr mapper)
    ap.add_argument("--mapper_bins", type=int, default=10)
    ap.add_argument("--mapper_overlap", type=float, default=0.30)
    ap.add_argument("--mapper_local_k", type=int, default=10)
    ap.add_argument("--mapper_merge_eps", type=float, default=0.50)
    ap.add_argument("--mapper_max_nodes", type=int, default=800)

    # Patience (recommended classic)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--pat_min_epoch", type=int, default=2)
    ap.add_argument("--pat_min_delta", type=float, default=1e-4)

    # Slope baseline
    ap.add_argument("--slope_win", type=int, default=8)
    ap.add_argument("--slope_eps", type=float, default=1e-4)
    ap.add_argument("--slope_patience", type=int, default=5)
    ap.add_argument("--slope_min_epoch", type=int, default=2)

    # SVCCA (return to more usable defaults)
    ap.add_argument("--svcca_dim", type=int, default=20)
    ap.add_argument("--svcca_sim_thr", type=float, default=0.99)
    ap.add_argument("--svcca_patience", type=int, default=3)
    ap.add_argument("--svcca_min_epoch", type=int, default=2)

    # CDSC (correlation dimension) - recommended style thr
    ap.add_argument("--cdsc_thr", type=float, default=0.3)
    ap.add_argument("--cdsc_patience", type=int, default=3)
    ap.add_argument("--cdsc_min_epoch", type=int, default=2)
    ap.add_argument("--cdsc_sample_n", type=int, default=256)
    ap.add_argument("--cdsc_n_radii", type=int, default=12)
    ap.add_argument("--cdsc_smooth_win", type=int, default=3)

    ap.add_argument("--oracle_eps", type=float, default=0.01)

    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=123)
    ap.add_argument("--deterministic", action="store_true")
    return ap.parse_args()


# ----------------------------
# One run
# ----------------------------
def run_one(csv_path: str, model_name: str, seed: int, noise_sigma: float, args) -> Dict[str, float]:
    set_global_seed(seed, deterministic=bool(args.deterministic))

    df = load_csv_numeric(csv_path)
    X_raw = df.to_numpy(dtype=np.float32)
    T, F = X_raw.shape
    target_idx = find_target_col(df)

    tr_sl, va_sl, _ = make_splits(T, args.train_frac, args.val_frac, args.test_frac, args.split_gap)

    mu = X_raw[tr_sl].mean(axis=0, keepdims=True)
    sd = X_raw[tr_sl].std(axis=0, keepdims=True) + 1e-6
    X = (X_raw - mu) / sd

    noise_seed = int(seed * 100000 + int(round(float(noise_sigma) * 1000)))
    X = apply_noise(X, sigma=float(noise_sigma), seed=noise_seed, mode=args.noise_mode, target_idx=target_idx)

    X_tr = X[tr_sl]
    X_va = X[va_sl]

    ds_tr = WindowDataset(X_tr, target_idx=target_idx, seq_len=args.seq_len, horizon=args.pred_horizon)
    ds_va = WindowDataset(X_va, target_idx=target_idx, seq_len=args.seq_len, horizon=args.pred_horizon)

    g = torch.Generator()
    g.manual_seed(seed)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g)
    dl_tr_fixed = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=False, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # repr subset for SES/SVCCA/CDSC
    if args.repr_source == "train":
        repr_subset = fixed_subset_from_loader(dl_tr_fixed, max_batches=int(args.repr_batches))
    else:
        repr_subset = fixed_subset_from_loader(dl_va, max_batches=int(args.repr_batches))

    ses_N = int(sum(int(x.shape[0]) for x, _ in repr_subset))

    # val eval subset (optional)
    if int(args.val_eval_batches) > 0:
        val_subset = fixed_subset_from_loader(dl_va, max_batches=int(args.val_eval_batches))
    else:
        val_subset = []

    device = resolve_device(args.device)
    model = build_model(model_name, in_dim=F, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    tr_losses: List[float] = []
    va_losses: List[float] = []
    reprs: List[np.ndarray] = []

    for e in range(1, int(args.epochs) + 1):
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        tr_losses.append(float(tr_loss))

        va_loss = eval_loss_on_subset(model, val_subset, device) if val_subset else eval_loss_full(model, dl_va, device)
        va_losses.append(float(va_loss))

        rep = eval_repr_on_subset(model, repr_subset, device)
        reprs.append(rep)

        if (e % int(args.log_every)) == 0 or e == 1:
            print(f"[{Path(csv_path).stem} | {model_name} | noise={noise_sigma:.2f} | seed={seed}] "
                  f"epoch {e:03d}/{args.epochs} | train {tr_loss:.6f} | val {va_loss:.6f} | reprN {ses_N}")

    oracle_e = int(np.argmin(np.array(va_losses, dtype=np.float64)) + 1)
    oracle_v = float(np.min(np.array(va_losses, dtype=np.float64)))

    # baselines
    pat_stop = find_stop_patience(va_losses, min_epoch=args.pat_min_epoch, patience=args.patience, min_delta=args.pat_min_delta)
    slope_stop = find_stop_slope(va_losses, min_epoch=args.slope_min_epoch, win=args.slope_win, slope_eps=args.slope_eps, patience=args.slope_patience)
    svcca_stop = find_stop_svcca(reprs, min_epoch=args.svcca_min_epoch, dim=args.svcca_dim, sim_thr=args.svcca_sim_thr, patience=args.svcca_patience)
    cdsc_stop = find_stop_cdsc(
        reprs,
        min_epoch=args.cdsc_min_epoch,
        patience=args.cdsc_patience,
        thr=args.cdsc_thr,
        sample_n=args.cdsc_sample_n,
        n_radii=args.cdsc_n_radii,
        smooth_win=args.cdsc_smooth_win,
    )

    # SES (solo v2)
    ses_stop = find_stop_ses_solo_v2(
        train_loss=tr_losses,
        reprs=reprs,
        min_epoch=args.ses_min_epoch,
        patience=args.ses_patience,
        ahead_win=args.ses_ahead_win,
        smooth_win=args.ses_smooth_win,
        rank_top=args.ses_rank_top,
        base_win=args.ses_base_win,
        gamma_loss=args.ses_gamma_loss,
        beta_slope=args.ses_beta_slope,
        rise_kappa=args.ses_rise_kappa,
        rise_patience=args.ses_rise_patience,
        ses_repr=args.ses_repr,
        mapper_bins=args.mapper_bins,
        mapper_overlap=args.mapper_overlap,
        mapper_local_k=args.mapper_local_k,
        mapper_merge_eps=args.mapper_merge_eps,
        mapper_max_nodes=args.mapper_max_nodes,
    )

    def pack(method: str, stop_e: int) -> Dict[str, float]:
        stop_e = int(min(max(1, stop_e), int(args.epochs)))
        v_stop = float(va_losses[stop_e - 1])
        return {
            f"{method}_epoch": float(stop_e),
            f"{method}_val_at_stop": v_stop,
            f"{method}_delta_best": float(v_stop - oracle_v),
            f"{method}_saved_epochs": float(int(args.epochs) - stop_e),
        }

    row: Dict[str, float] = {
        "dataset": Path(csv_path).stem,
        "model": model_name,
        "seed": float(seed),
        "noise_sigma": float(noise_sigma),
        "epochs": float(args.epochs),
        "oracle_epoch": float(oracle_e),
        "oracle_val": float(oracle_v),
        "reprN": float(ses_N),
        "repr_source": args.repr_source,
        "noise_mode": args.noise_mode,
    }
    row.update(pack("ses", ses_stop))
    row.update(pack("pat", pat_stop))
    row.update(pack("slope", slope_stop))
    row.update(pack("svcca", svcca_stop))
    row.update(pack("cdsc", cdsc_stop))
    return row


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    out_root = Path(args.out_root)
    ensure_dir(out_root)

    data_dir = Path(args.ett_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    csv_paths: List[str] = []
    for ds in args.datasets:
        p = data_dir / f"{ds}.csv"
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        csv_paths.append(str(p))

    if args.noise_min is not None and args.noise_max is not None and args.noise_step is not None:
        sigmas = list(np.round(np.arange(float(args.noise_min), float(args.noise_max) + 1e-12, float(args.noise_step)), 10))
    else:
        sigmas = [float(args.noise_sigma)]

    all_rows: List[Dict[str, float]] = []
    for csv_path in csv_paths:
        print(f"[DATASET] {csv_path}")
        for model_name in args.models:
            for sigma in sigmas:
                for i in range(int(args.n_runs)):
                    seed = int(args.base_seed + i)
                    print(f"[RUN] {Path(csv_path).stem} | {model_name} | noise={sigma:.2f} | seed={seed}")
                    row = run_one(csv_path, model_name=model_name, seed=seed, noise_sigma=float(sigma), args=args)
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)

    out_all = out_root / ("es_results_all_noise.csv" if len(sigmas) > 1 else "es_results_all.csv")
    df.to_csv(out_all, index=False)
    print(f"[SAVED] {out_all}")

    df_summary, df_compact = make_summary_tables(df, oracle_eps=float(args.oracle_eps))
    out_summary = out_root / "es_results_summary_by_noise.csv"
    out_compact = out_root / "es_results_compact_table.csv"
    df_summary.to_csv(out_summary, index=False)
    df_compact.to_csv(out_compact, index=False)
    print(f"[SAVED] {out_summary}")
    print(f"[SAVED] {out_compact}")


if __name__ == "__main__":
    main()

# python .\online_es_timeseries_multi_sota_ETT_noise_ses_rankmedian_online_fixed.py `
#   --ett_dir "C:\Users\User\PycharmProjects\NeuroSymbolicDynamics\data" `
#   --datasets lorenz `
#   --out_root out_lorenz_noise_exp3 `
#   --models rnn birnn transformer `
#   --n_runs 10 `
#   --epochs 100 `
#   --seq_len 24 `
#   --pred_horizon 1 `
#   --batch_size 8 `
#   --lr 0.001 `
#   --hidden_dim 64 `
#   --device cuda `
#   --noise_min 0.0 `
#   --noise_max 0.5 `
#   --noise_step 0.1 `
#   --noise_mode all `
#   --repr_source train `
#   --repr_batches 8 `
#   --ses_rank_top 0.40 `
#   --ses_patience 2 `
#   --ses_ahead_win 4 `
#   --ses_smooth_win 3 `
#   --ses_min_epoch 2 `
#   --ses_beta_slope 2.5 `
#   --ses_gamma_loss 0.35 `
#   --ses_base_win 8
