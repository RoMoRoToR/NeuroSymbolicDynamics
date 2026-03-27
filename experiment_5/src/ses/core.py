from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.linalg import svd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA as SKCCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    req = (requested or "auto").strip().lower()
    cuda_ok = bool(getattr(torch, "cuda", None)) and bool(torch.cuda.is_available())
    mps_ok = bool(getattr(torch.backends, "mps", None)) and bool(torch.backends.mps.is_available())
    if req == "auto":
        if cuda_ok:
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if req == "cuda":
        return torch.device("cuda" if cuda_ok else "cpu")
    if req == "mps":
        return torch.device("mps" if mps_ok else "cpu")
    return torch.device("cpu")


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def load_csv_numeric(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.shape[1] >= 2 and not np.issubdtype(df.iloc[:, 0].dtype, np.number):
        df = df.iloc[:, 1:]
    df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    if df.shape[1] == 0:
        raise ValueError(f"No numeric columns found in {csv_path}")
    if df.shape[1] == 1:
        name = df.columns[0]
        df = df.rename(columns={name: "y"})
        df.insert(0, "t", np.arange(len(df), dtype=np.float32))
    return df


def find_target_col(df: pd.DataFrame) -> int:
    cols = df.columns.tolist()
    low = [c.lower() for c in cols]
    if "ot" in low:
        return low.index("ot")
    if "close" in low:
        return low.index("close")
    if "y" in low:
        return low.index("y")
    candidates: List[int] = []
    for i, col in enumerate(cols):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().all():
            continue
        if float(s.std(skipna=True)) < 1e-12:
            continue
        candidates.append(i)
    return candidates[-1] if candidates else df.shape[1] - 1


def make_splits(timesteps: int, train_frac: float, val_frac: float, test_frac: float, gap: int) -> Tuple[slice, slice, slice]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    tr_e = int(timesteps * train_frac)
    va_e = tr_e + int(timesteps * val_frac)
    te_e = timesteps
    tr_s = 0
    va_s = tr_e + gap
    te_s = va_e + gap
    if va_s >= va_e or te_s >= te_e:
        raise ValueError("Split overflow; reduce split_gap or adjust fractions.")
    return slice(tr_s, tr_e), slice(va_s, va_e), slice(te_s, te_e)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, target_idx: int, seq_len: int, horizon: int):
        self.x = x.astype(np.float32)
        self.target_idx = int(target_idx)
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.n = self.x.shape[0] - self.seq_len - self.horizon + 1
        if self.n <= 0:
            raise ValueError("Not enough rows for given seq_len/horizon")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xx = self.x[idx:idx + self.seq_len, :]
        yy = self.x[idx + self.seq_len + self.horizon - 1, self.target_idx]
        return torch.from_numpy(xx), torch.tensor(yy, dtype=torch.float32)


class RNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.rnn(x)
        rep = h[-1]
        return self.head(rep).squeeze(-1), rep


class BiRNNRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.rnn(x)
        rep = torch.cat([h[-2], h[-1]], dim=-1)
        return self.head(rep).squeeze(-1), rep


class TransformerRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, nhead: int = 4, nlayers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        if hidden_dim % nhead != 0:
            for head in [8, 4, 2, 1]:
                if hidden_dim % head == 0:
                    nhead = head
                    break
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=nlayers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.in_proj(x)
        h = self.encoder(z)
        rep = h.mean(dim=1)
        return self.head(rep).squeeze(-1), rep


def build_model(name: str, in_dim: int, hidden_dim: int) -> nn.Module:
    name = name.lower()
    if name == "rnn":
        return RNNRegressor(in_dim, hidden_dim)
    if name == "birnn":
        return BiRNNRegressor(in_dim, hidden_dim)
    if name == "transformer":
        return TransformerRegressor(in_dim, hidden_dim)
    raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model: nn.Module, dl: DataLoader, opt: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        pred, _ = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def collect_validation_outputs(model: nn.Module, dl: DataLoader, device: torch.device, max_batches: int = 0) -> Tuple[float, np.ndarray, List[int]]:
    model.eval()
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    reps: List[np.ndarray] = []
    lengths: List[int] = []
    for batch_idx, (x, y) in enumerate(dl):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        pred, rep = model(x)
        losses.append(float(loss_fn(pred, y).detach().cpu().item()))
        rep_np = to_np(rep)
        reps.append(rep_np)
        lengths.append(int(rep_np.shape[0]))
    if not reps:
        return float("nan"), np.zeros((0, 1), dtype=np.float32), []
    return float(np.mean(losses)), np.concatenate(reps, axis=0).astype(np.float32), lengths


def apply_noise(x: np.ndarray, sigma: float, seed: int, mode: str = "all", target_idx: Optional[int] = None) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        return x.astype(np.float32)
    rng = np.random.default_rng(int(seed))
    eps = rng.standard_normal(size=x.shape).astype(np.float32)
    if mode == "all":
        return (x + sigma * eps).astype(np.float32)
    if mode == "target":
        if target_idx is None:
            raise ValueError("target_idx required for noise_mode=target")
        y = x.copy().astype(np.float32)
        y[:, int(target_idx)] = y[:, int(target_idx)] + sigma * eps[:, int(target_idx)]
        return y
    raise ValueError(f"Unknown noise_mode: {mode}")


class MapperSymbolizer:
    def __init__(self, d_embed: int, n_bins: int, overlap: float, local_k: int, merge_eps: float, seed: int):
        self.d_embed = int(d_embed)
        self.n_bins = int(n_bins)
        self.overlap = float(overlap)
        self.local_k = int(local_k)
        self.merge_eps = float(merge_eps)
        self.seed = int(seed)
        self.pca_embed: Optional[PCA] = None
        self.pca_lens: Optional[PCA] = None
        self.centers: Optional[np.ndarray] = None
        self.boxes: List[Tuple[float, float]] = []

    def _embed(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.d_embed and x.shape[1] > self.d_embed:
            if self.pca_embed is None:
                self.pca_embed = PCA(n_components=self.d_embed, random_state=self.seed).fit(x)
            return self.pca_embed.transform(x)
        return x

    def _lens(self, xr: np.ndarray) -> np.ndarray:
        if self.pca_lens is None:
            self.pca_lens = PCA(n_components=1, random_state=self.seed).fit(xr)
        return self.pca_lens.transform(xr).reshape(-1)

    def fit(self, x: np.ndarray) -> "MapperSymbolizer":
        xr = self._embed(x)
        f = self._lens(xr)
        f_min = float(np.min(f))
        f_max = float(np.max(f))
        width = (f_max - f_min) / max(self.n_bins, 1)
        if width <= 0.0:
            self.centers = xr.mean(axis=0, keepdims=True)
            self.boxes = [(f_min, f_max)]
            return self
        step = width * max(1e-6, 1.0 - self.overlap)
        local: List[np.ndarray] = []
        boxes: List[Tuple[float, float]] = []
        for i in range(self.n_bins):
            lo = f_min + i * step
            hi = lo + width
            mask = (f >= lo) & (f <= hi)
            if mask.sum() == 0:
                continue
            xloc = xr[mask]
            uniq = np.unique(xloc, axis=0)
            k_eff = min(self.local_k, max(1, len(uniq)))
            if k_eff == 1:
                local.append(uniq.mean(axis=0))
                boxes.append((float(lo), float(hi)))
            else:
                km = KMeans(n_clusters=k_eff, n_init=10, random_state=self.seed).fit(uniq)
                for c in range(k_eff):
                    sel = km.labels_ == c
                    if np.any(sel):
                        local.append(uniq[sel].mean(axis=0))
                        boxes.append((float(lo), float(hi)))
        if not local:
            self.centers = xr.mean(axis=0, keepdims=True)
            self.boxes = [(f_min, f_max)]
            return self
        c = np.stack(local, axis=0)
        c_std = np.std(c, axis=0) + 1e-9
        cn = (c - c.mean(axis=0)) / c_std
        keep: List[int] = []
        used = np.zeros((len(c),), dtype=bool)
        for i in range(len(c)):
            if used[i]:
                continue
            used[i] = True
            keep.append(i)
            for j in range(i + 1, len(c)):
                if not used[j] and np.linalg.norm(cn[i] - cn[j]) <= self.merge_eps:
                    used[j] = True
        self.centers = c[keep]
        self.boxes = [boxes[i] for i in keep]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("MapperSymbolizer must be fit before transform().")
        xr = self._embed(x)
        nn = NearestNeighbors(n_neighbors=1).fit(self.centers)
        _, idx = nn.kneighbors(xr)
        return idx.reshape(-1)


def split_symbols_by_lengths(symbols: np.ndarray, lengths: List[int]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    start = 0
    for n in lengths:
        out.append(np.asarray(symbols[start:start + n], dtype=int))
        start += n
    return out


def lz76_phrase_count(sym_seq: np.ndarray) -> int:
    s = np.asarray(sym_seq, dtype=int)
    n = len(s)
    if n <= 1:
        return n
    i = 0
    l = 1
    k = 1
    c = 1
    while True:
        if i + k - 1 >= n or l + k - 1 >= n:
            c += 1
            break
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if l == i + 1:
                c += 1
                i += k
                if i + 1 > n:
                    break
                l = 1
                k = 1
            else:
                l += 1
                if l + k > n:
                    c += 1
                    i += k
                    if i + 1 > n:
                        break
                    l = 1
                    k = 1
    return int(c)


def lz_normalized(sym_seq: np.ndarray, alphabet_size: int) -> float:
    s = np.asarray(sym_seq, dtype=int)
    n = len(s)
    if n <= 1 or alphabet_size <= 1:
        return 0.0
    c = lz76_phrase_count(s)
    return float((c * math.log(max(alphabet_size, 2), 2)) / max(n, 1))


def markov_entropy_rate(sym_seq: np.ndarray) -> float:
    s = np.asarray(sym_seq, dtype=int)
    if len(s) < 2:
        return 0.0
    k = int(s.max()) + 1
    counts = np.zeros((k, k), dtype=np.float64)
    occ = np.zeros((k,), dtype=np.float64)
    for i in range(len(s) - 1):
        occ[s[i]] += 1.0
        counts[s[i], s[i + 1]] += 1.0
    trans = np.zeros_like(counts)
    for i in range(k):
        if occ[i] > 0:
            trans[i] = counts[i] / occ[i]
    pi = occ / max(occ.sum(), 1.0)
    h = 0.0
    for i in range(k):
        p = trans[i]
        p = p[p > 0.0]
        if pi[i] > 0.0 and p.size > 0:
            h += float(pi[i] * (-(p * np.log(p + 1e-12)).sum()))
    return float(h / math.log(2))


def permutation_entropy(series: np.ndarray, m: int = 5, tau: int = 1) -> float:
    x = np.asarray(series, dtype=np.float64)
    if x.size < (m - 1) * tau + 1:
        return float("nan")
    patterns: Dict[Tuple[int, ...], int] = {}
    for i in range(x.size - (m - 1) * tau):
        block = x[i:i + (m * tau):tau]
        key = tuple(np.argsort(block))
        patterns[key] = patterns.get(key, 0) + 1
    p = np.array(list(patterns.values()), dtype=np.float64)
    p /= p.sum()
    h = float(-(p * np.log2(p + 1e-12)).sum())
    return float(h / (math.log2(math.factorial(m)) + 1e-12))


def correlation_dimension_d2(x: np.ndarray, d_emb: int = 8, r_bins: int = 25, sample_cap: int = 8000, seed: int = 0) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 100:
        return float("nan")
    xr = PCA(n_components=min(d_emb, x.shape[1]), random_state=seed).fit_transform(x)
    rng = np.random.default_rng(seed)
    idx = rng.choice(xr.shape[0], size=min(sample_cap, xr.shape[0]), replace=False)
    y = xr[idx]
    d = pdist(y, metric="euclidean")
    if d.size == 0:
        return float("nan")
    lo, hi = np.percentile(d, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= 0.0:
        return float("nan")
    radii = np.geomspace(max(lo, 1e-6), hi, num=r_bins)
    d_sorted = np.sort(d)
    c = np.searchsorted(d_sorted, radii, side="right") / d_sorted.size
    xx = np.log(radii + 1e-12)
    yy = np.log(c + 1e-12)
    mask = (c > 1e-3) & (c < 1 - 1e-3)
    if mask.sum() < max(5, r_bins // 4):
        return float("nan")
    reg = LinearRegression().fit(xx[mask].reshape(-1, 1), yy[mask])
    return float(reg.coef_[0])


def box_counting_dimension_df(x: np.ndarray, d_emb: int = 4, n_scales: int = 6, seed: int = 0) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 100:
        return float("nan")
    xr = PCA(n_components=min(d_emb, x.shape[1]), random_state=seed).fit_transform(x)
    mins = xr.min(axis=0)
    maxs = xr.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    xn = (xr - mins) / span
    eps_values = np.geomspace(0.5, 0.05, num=n_scales)
    xs: List[float] = []
    ys: List[float] = []
    for eps in eps_values:
        bins = np.floor(xn / eps).astype(int)
        count = np.unique(bins, axis=0).shape[0]
        if count > 1:
            xs.append(float(np.log(1.0 / eps)))
            ys.append(float(np.log(count)))
    if len(xs) < 3:
        return float("nan")
    reg = LinearRegression().fit(np.array(xs).reshape(-1, 1), np.array(ys))
    return float(reg.coef_[0])


def ema(x: np.ndarray, alpha: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    y = np.zeros_like(x)
    acc = x[0]
    for i in range(len(x)):
        acc = alpha * x[i] + (1.0 - alpha) * acc
        y[i] = acc
    return y


def prefix_rank01(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    r = np.zeros_like(y)
    for t in range(len(y)):
        pref = y[:t + 1]
        r[t] = (np.sum(pref <= y[t]) - 1) / max(1, len(pref) - 1) if t > 0 else 1.0
    return r


def lin_slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x) + 1e-12)
    return float(np.dot(x, y - y.mean()) / denom)


def compute_epoch_symbolic_metrics(
    embeddings: np.ndarray,
    batch_lengths: List[int],
    cfg: "ExperimentConfig",
    seed: int,
) -> Dict[str, float]:
    if embeddings.size == 0:
        out = {"LZ": float("nan"), "hM": float("nan"), "PermEn": float("nan"), "D2": float("nan")}
        if cfg.ses_include_df:
            out["DF"] = float("nan")
        return out
    mapper = MapperSymbolizer(
        d_embed=cfg.mapper_d_embed,
        n_bins=cfg.mapper_bins,
        overlap=cfg.mapper_overlap,
        local_k=cfg.mapper_local_k,
        merge_eps=cfg.mapper_merge_eps,
        seed=seed,
    ).fit(embeddings)
    symbols = mapper.transform(embeddings)
    sequences = split_symbols_by_lengths(symbols, batch_lengths)
    alphabet_size = int(symbols.max()) + 1 if len(symbols) else 1
    lz_vals = [lz_normalized(seq, alphabet_size) for seq in sequences if len(seq) > 1]
    hm_vals = [markov_entropy_rate(seq) for seq in sequences if len(seq) > 1]
    pe_vals = [permutation_entropy(seq.astype(float), m=cfg.pe_m, tau=cfg.pe_tau) for seq in sequences if len(seq) >= (cfg.pe_m - 1) * cfg.pe_tau + 1]
    metrics: Dict[str, float] = {
        "LZ": float(np.nanmean(lz_vals)) if lz_vals else float("nan"),
        "hM": float(np.nanmean(hm_vals)) if hm_vals else float("nan"),
        "PermEn": float(np.nanmean(pe_vals)) if pe_vals else float("nan"),
        "D2": correlation_dimension_d2(
            embeddings,
            d_emb=cfg.d2_emb,
            r_bins=cfg.d2_rbins,
            sample_cap=cfg.d2_cap,
            seed=seed,
        ),
    }
    if cfg.ses_include_df:
        metrics["DF"] = box_counting_dimension_df(
            embeddings,
            d_emb=cfg.df_emb,
            n_scales=cfg.df_scales,
            seed=seed,
        )
    return metrics


def build_symbolic_score(metric_history: Dict[str, List[float]], cfg: "ExperimentConfig") -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    keys = sorted(metric_history.keys())
    smoothed: Dict[str, np.ndarray] = {}
    ranks: Dict[str, np.ndarray] = {}
    for key in keys:
        vals = np.asarray(metric_history[key], dtype=np.float64)
        smoothed[key] = ema(vals, alpha=cfg.ses_ema_alpha)
        ranks[key] = prefix_rank01(smoothed[key])
    score = np.zeros((len(next(iter(metric_history.values()))),), dtype=np.float64)
    for t in range(len(score)):
        active_vals: List[float] = []
        for key in keys:
            series = smoothed[key]
            lo = max(0, t - cfg.ses_liveness_win + 1)
            window = series[lo:t + 1]
            if np.all(~np.isfinite(window)):
                continue
            abs_rng = float(np.nanmax(window) - np.nanmin(window))
            denom = float(np.nanmedian(np.abs(window)) + 1e-12)
            rel_rng = abs_rng / denom
            alive = (abs_rng >= cfg.ses_liveness_abs) and (rel_rng >= cfg.ses_liveness_rel)
            if alive:
                active_vals.append(float(ranks[key][t]))
        if not active_vals:
            active_vals = [float(ranks[key][t]) for key in keys if np.isfinite(ranks[key][t])]
        active_vals = sorted(active_vals)
        if cfg.ses_agg == "median":
            score[t] = float(np.median(active_vals))
        elif cfg.ses_agg == "topq":
            top_k = max(1, int(math.ceil(cfg.ses_rank_top * len(active_vals))))
            score[t] = float(np.median(active_vals[:top_k]))
        elif cfg.ses_agg == "min":
            score[t] = float(active_vals[0])
        else:
            raise ValueError(f"Unknown ses_agg: {cfg.ses_agg}")
    return score, smoothed, ranks


def validation_guard_ok(val_losses: List[float], idx: int, cfg: "ExperimentConfig") -> bool:
    lo = max(0, idx - cfg.val_guard_win + 1)
    window = np.asarray(val_losses[lo:idx + 1], dtype=np.float64)
    if window.size < 2:
        return False
    improvement = float(window[0] - np.nanmin(window))
    rel = improvement / max(abs(float(window[0])), 1e-12)
    return not ((improvement > cfg.val_guard_abs) and (rel > cfg.val_guard_rel))


def find_stop_symbolic_hybrid(symbolic_score: List[float], val_losses: List[float], cfg: "ExperimentConfig") -> int:
    score = np.asarray(symbolic_score, dtype=np.float64)
    best = float("inf")
    no_improve = 0
    stall = np.zeros((len(score),), dtype=bool)
    plateau = np.zeros((len(score),), dtype=bool)
    guard = np.zeros((len(score),), dtype=bool)
    stall_seen = False
    plateau_seen = False
    for i in range(len(score)):
        epoch = i + 1
        if score[i] < best - cfg.ses_min_delta_sym:
            best = float(score[i])
            no_improve = 0
        elif epoch >= cfg.ses_min_epoch:
            no_improve += 1
        stall[i] = epoch >= cfg.ses_min_epoch and no_improve >= cfg.ses_patience
        if epoch >= max(cfg.ses_min_epoch, cfg.ses_slope_win):
            plateau[i] = abs(lin_slope(score[max(0, i - cfg.ses_slope_win + 1):i + 1])) < cfg.ses_slope_eps
        guard[i] = validation_guard_ok(val_losses, i, cfg)
        stall_seen = stall_seen or bool(stall[i])
        plateau_seen = plateau_seen or bool(plateau[i])
        if stall_seen and plateau_seen and guard[i]:
            return epoch
    return len(score)


def _center_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return z - z.mean(axis=1, keepdims=True)


def _svdproj_feats_by_samples(x: np.ndarray, k: int) -> np.ndarray:
    u, s, vt = svd(x, full_matrices=False)
    k = max(1, min(k, min(x.shape[0], max(1, x.shape[1] - 1))))
    return s[:k, None] * vt[:k, :]


def svcca_score(a: np.ndarray, b: np.ndarray, proj: int = 64) -> float:
    if a is None or b is None:
        return 0.0
    za = _svdproj_feats_by_samples(_center_rows(np.asarray(a, dtype=np.float64).T), proj)
    zb = _svdproj_feats_by_samples(_center_rows(np.asarray(b, dtype=np.float64).T), proj)
    n = min(za.shape[1], zb.shape[1])
    if n < 20:
        return 0.0
    za = _center_rows(za[:, :n]).T
    zb = _center_rows(zb[:, :n]).T
    k = min(za.shape[1], zb.shape[1], proj)
    cca = SKCCA(n_components=k, max_iter=5000)
    cca.fit(za, zb)
    xa, xb = cca.transform(za, zb)
    sims = []
    for i in range(xa.shape[1]):
        num = float(np.dot(xa[:, i], xb[:, i]))
        den = float(np.linalg.norm(xa[:, i]) * np.linalg.norm(xb[:, i]) + 1e-12)
        sims.append(num / den if den > 0.0 else 0.0)
    sims = np.clip(np.asarray(sims, dtype=np.float64), 0.0, 1.0)
    return float(np.mean(sims)) if sims.size else 0.0


def svcca_stop(reps_by_epoch: Dict[int, np.ndarray], epochs: np.ndarray, svcca_thr: float, patience: int, min_epoch: int, proj: int, val_curve: np.ndarray) -> int:
    e = np.asarray(sorted(int(x) for x in epochs))
    idx = np.where(e <= min_epoch)[0]
    ref_idx = int(idx[np.argmin(val_curve[idx])]) if len(idx) else 0
    ref_epoch = int(e[ref_idx])
    if ref_epoch not in reps_by_epoch:
        ref_epoch = min(reps_by_epoch.keys(), key=lambda z: abs(int(z) - ref_epoch))
    ref = reps_by_epoch[ref_epoch]
    ok = 0
    last_ok = int(e[0])
    for epoch in e:
        if epoch < min_epoch or epoch not in reps_by_epoch:
            continue
        cur = reps_by_epoch[epoch]
        n = min(len(ref), len(cur))
        if n < 20:
            continue
        s = svcca_score(ref[:n], cur[:n], proj=proj)
        if s >= svcca_thr:
            ok += 1
            last_ok = int(epoch)
            if ok >= patience:
                return last_ok
        else:
            ok = 0
    return int(e[-1])


def patience_es(val: List[float], epochs: np.ndarray, min_epoch: int, patience: int, min_delta: float) -> int:
    v = np.asarray(val, dtype=np.float64)
    e = np.asarray(epochs, dtype=int)
    best = float("inf")
    best_ep = int(e[0])
    count = 0
    for i in range(len(e)):
        if v[i] < best - min_delta:
            best = float(v[i])
            best_ep = int(e[i])
            count = 0
        else:
            count += 1
        if e[i] >= min_epoch and count >= patience:
            return best_ep
    return int(e[-1])


def slope_es(val: List[float], epochs: np.ndarray, win: int, eps: float, min_epoch: int) -> int:
    v = np.asarray(val, dtype=np.float64)
    e = np.asarray(epochs, dtype=int)
    for i in range(len(e)):
        if e[i] < min_epoch or i + 1 < win:
            continue
        if lin_slope(v[i - win + 1:i + 1]) >= -eps:
            return int(e[i])
    return int(e[-1])


def cdsc_stop(val: List[float], epochs: np.ndarray, win: int, ratio: float, min_epoch: int) -> int:
    v = np.asarray(val, dtype=np.float64)
    e = np.asarray(epochs, dtype=int)
    for i in range(len(e)):
        if e[i] < min_epoch or i < 2 * win:
            continue
        prev = v[i - 2 * win:i - win + 1]
        curr = v[i - win + 1:i + 1]
        if len(prev) < 2 or len(curr) < 2:
            continue
        imp_prev = float(prev[0] - np.min(prev))
        imp_curr = float(curr[0] - np.min(curr))
        if imp_prev <= 0.0:
            continue
        if imp_curr <= imp_prev / max(ratio, 1.0):
            return int(e[i])
    return int(e[-1])


def _median_iqr_str(x: np.ndarray, fmt: str) -> str:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    return fmt.format(med) + " [" + fmt.format(q3 - q1) + "]"


def make_summary_tables(df_all: pd.DataFrame, oracle_eps: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["ses", "pat", "slope", "svcca", "cdsc"]
    rows = []
    for (dataset, model, sigma), group in df_all.groupby(["dataset", "model", "noise_sigma"], dropna=False):
        row = {"dataset": dataset, "model": model, "noise_sigma": float(sigma)}
        for method in methods:
            epochs = group[f"{method}_epoch"].to_numpy(dtype=float)
            vals = group[f"{method}_val_at_stop"].to_numpy(dtype=float)
            deltas = group[f"{method}_delta_best"].to_numpy(dtype=float)
            saved = group[f"{method}_saved_epochs"].to_numpy(dtype=float)
            row[f"{method}_stop_epoch_med_iqr"] = _median_iqr_str(epochs, "{:.1f}")
            row[f"{method}_val_at_stop_med_iqr"] = _median_iqr_str(vals, "{:.6f}")
            row[f"{method}_delta_best_med_iqr"] = _median_iqr_str(deltas, "{:.6f}")
            row[f"{method}_epochs_saved_med_iqr"] = _median_iqr_str(saved, "{:.1f}")
            row[f"{method}_within_oracle@{oracle_eps:.2f}"] = float(np.mean(deltas <= float(oracle_eps)))
        rows.append(row)
    df_summary = pd.DataFrame(rows).sort_values(["dataset", "model", "noise_sigma"]).reset_index(drop=True)
    compact_rows = []
    labels = {"ses": "SES", "pat": "PAT", "slope": "SLOPE", "svcca": "SVCCA", "cdsc": "CDSC"}
    for sigma, group in df_all.groupby("noise_sigma", dropna=False):
        for method in methods:
            compact_rows.append(
                {
                    "noise_sigma": float(sigma),
                    "method": labels[method],
                    "loss@stop_median": float(np.median(group[f"{method}_val_at_stop"].to_numpy(dtype=float))),
                    "stop_epoch_median": float(np.median(group[f"{method}_epoch"].to_numpy(dtype=float))),
                    "epochs_saved_median": float(np.median(group[f"{method}_saved_epochs"].to_numpy(dtype=float))),
                }
            )
    df_compact = pd.DataFrame(compact_rows).sort_values(["noise_sigma", "method"]).reset_index(drop=True)
    return df_summary, df_compact


@dataclass
class ExperimentConfig:
    data_root: str = "data"
    datasets: List[str] = field(default_factory=lambda: ["ETTh1"])
    out_root: str = "results/article_run"
    models: List[str] = field(default_factory=lambda: ["rnn", "birnn", "transformer"])
    n_runs: int = 10
    epochs: int = 100
    seq_len: int = 96
    pred_horizon: int = 1
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 64
    device: str = "auto"
    train_frac: float = 0.7
    val_frac: float = 0.2
    test_frac: float = 0.1
    split_gap: int = 0
    noise_sigma: float = 0.0
    noise_min: Optional[float] = None
    noise_max: Optional[float] = None
    noise_step: Optional[float] = None
    noise_mode: str = "all"
    repr_batches: int = 0
    pe_m: int = 5
    pe_tau: int = 1
    d2_emb: int = 8
    d2_rbins: int = 25
    d2_cap: int = 8000
    df_emb: int = 4
    df_scales: int = 6
    ses_include_df: bool = False
    ses_agg: str = "median"
    ses_rank_top: float = 0.40
    ses_ema_alpha: float = 0.25
    ses_liveness_win: int = 5
    ses_liveness_abs: float = 1e-3
    ses_liveness_rel: float = 0.05
    ses_min_epoch: int = 10
    ses_patience: int = 6
    ses_min_delta_sym: float = 5e-3
    ses_slope_win: int = 5
    ses_slope_eps: float = 3e-3
    val_guard_win: int = 4
    val_guard_abs: float = 1.5e-4
    val_guard_rel: float = 2.5e-3
    mapper_d_embed: int = 8
    mapper_bins: int = 8
    mapper_overlap: float = 0.30
    mapper_local_k: int = 10
    mapper_merge_eps: float = 0.50
    patience: int = 10
    pat_min_epoch: int = 10
    pat_min_delta: float = 0.0
    slope_win: int = 5
    slope_eps: float = 1e-4
    slope_min_epoch: int = 10
    svcca_dim: int = 64
    svcca_sim_thr: float = 0.985
    svcca_patience: int = 2
    svcca_min_epoch: int = 10
    cdsc_win: int = 6
    cdsc_ratio: float = 2.0
    cdsc_min_epoch: int = 10
    oracle_eps: float = 0.01
    log_every: int = 5
    base_seed: int = 123
    deterministic: bool = False
    save_epoch_metrics: bool = True


def resolve_dataset_paths(data_root: Path, datasets: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for dataset in datasets:
        candidate = Path(dataset)
        if candidate.exists():
            resolved.append(candidate.resolve())
            continue
        direct = data_root / f"{dataset}.csv"
        if direct.exists():
            resolved.append(direct.resolve())
            continue
        matches = sorted(data_root.rglob(f"{dataset}.csv"))
        if matches:
            resolved.append(matches[0].resolve())
            continue
        raise FileNotFoundError(f"Could not resolve dataset '{dataset}' under {data_root}")
    return resolved


def noise_schedule(cfg: ExperimentConfig) -> List[float]:
    if cfg.noise_min is not None and cfg.noise_max is not None and cfg.noise_step is not None:
        return list(np.round(np.arange(float(cfg.noise_min), float(cfg.noise_max) + 1e-12, float(cfg.noise_step)), 10))
    return [float(cfg.noise_sigma)]


def run_one(csv_path: Path, model_name: str, seed: int, noise_sigma: float, cfg: ExperimentConfig) -> Tuple[Dict[str, float], pd.DataFrame]:
    set_global_seed(seed, deterministic=bool(cfg.deterministic))
    df = load_csv_numeric(csv_path)
    x_raw = df.to_numpy(dtype=np.float32)
    timesteps, features = x_raw.shape
    target_idx = find_target_col(df)
    tr_sl, va_sl, _ = make_splits(timesteps, cfg.train_frac, cfg.val_frac, cfg.test_frac, cfg.split_gap)
    mu = x_raw[tr_sl].mean(axis=0, keepdims=True)
    sd = x_raw[tr_sl].std(axis=0, keepdims=True) + 1e-6
    x = (x_raw - mu) / sd
    noise_seed = int(seed * 100000 + int(round(float(noise_sigma) * 1000)))
    x = apply_noise(x, sigma=float(noise_sigma), seed=noise_seed, mode=cfg.noise_mode, target_idx=target_idx)

    ds_tr = WindowDataset(x[tr_sl], target_idx=target_idx, seq_len=cfg.seq_len, horizon=cfg.pred_horizon)
    ds_va = WindowDataset(x[va_sl], target_idx=target_idx, seq_len=cfg.seq_len, horizon=cfg.pred_horizon)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=generator)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = resolve_device(cfg.device)
    model = build_model(model_name, in_dim=features, hidden_dim=cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))

    train_losses: List[float] = []
    val_losses: List[float] = []
    reps_by_epoch: Dict[int, np.ndarray] = {}
    metric_history: Dict[str, List[float]] = {"LZ": [], "hM": [], "PermEn": [], "D2": []}
    if cfg.ses_include_df:
        metric_history["DF"] = []

    for epoch in range(1, int(cfg.epochs) + 1):
        tr_loss = train_one_epoch(model, dl_tr, opt, device)
        val_loss, embeddings, batch_lengths = collect_validation_outputs(model, dl_va, device, max_batches=int(cfg.repr_batches))
        metrics = compute_epoch_symbolic_metrics(embeddings, batch_lengths, cfg, seed=seed + epoch)
        train_losses.append(float(tr_loss))
        val_losses.append(float(val_loss))
        reps_by_epoch[epoch] = embeddings
        for key in metric_history:
            metric_history[key].append(float(metrics.get(key, float("nan"))))
        if epoch == 1 or epoch % int(cfg.log_every) == 0:
            print(
                f"[{csv_path.stem} | {model_name} | noise={noise_sigma:.2f} | seed={seed}] "
                f"epoch {epoch:03d}/{cfg.epochs} | train {tr_loss:.6f} | val {val_loss:.6f} | "
                f"LZ {metrics['LZ']:.4f} | hM {metrics['hM']:.4f}"
            )

    epochs = np.arange(1, len(val_losses) + 1, dtype=int)
    sym_score, smoothed, ranks = build_symbolic_score(metric_history, cfg)
    ses_stop = find_stop_symbolic_hybrid(sym_score.tolist(), val_losses, cfg)
    pat_stop = patience_es(val_losses, epochs, min_epoch=cfg.pat_min_epoch, patience=cfg.patience, min_delta=cfg.pat_min_delta)
    slope_stop = slope_es(val_losses, epochs, win=cfg.slope_win, eps=cfg.slope_eps, min_epoch=cfg.slope_min_epoch)
    svcca_epoch = svcca_stop(
        reps_by_epoch,
        epochs,
        svcca_thr=cfg.svcca_sim_thr,
        patience=cfg.svcca_patience,
        min_epoch=cfg.svcca_min_epoch,
        proj=cfg.svcca_dim,
        val_curve=np.asarray(val_losses, dtype=np.float64),
    )
    cdsc_epoch = cdsc_stop(val_losses, epochs, win=cfg.cdsc_win, ratio=cfg.cdsc_ratio, min_epoch=cfg.cdsc_min_epoch)
    oracle_epoch = int(np.argmin(np.asarray(val_losses, dtype=np.float64)) + 1)
    oracle_val = float(np.min(np.asarray(val_losses, dtype=np.float64)))

    def pack(method: str, stop_epoch: int) -> Dict[str, float]:
        stop_epoch = int(min(max(1, stop_epoch), int(cfg.epochs)))
        val_at_stop = float(val_losses[stop_epoch - 1])
        return {
            f"{method}_epoch": float(stop_epoch),
            f"{method}_val_at_stop": val_at_stop,
            f"{method}_delta_best": float(val_at_stop - oracle_val),
            f"{method}_saved_epochs": float(int(cfg.epochs) - stop_epoch),
        }

    row: Dict[str, float] = {
        "dataset": csv_path.stem,
        "model": model_name,
        "seed": float(seed),
        "noise_sigma": float(noise_sigma),
        "epochs": float(cfg.epochs),
        "oracle_epoch": float(oracle_epoch),
        "oracle_val": float(oracle_val),
        "repr_source": "validation",
        "noise_mode": cfg.noise_mode,
    }
    for method, stop_epoch in [("ses", ses_stop), ("pat", pat_stop), ("slope", slope_stop), ("svcca", svcca_epoch), ("cdsc", cdsc_epoch)]:
        row.update(pack(method, stop_epoch))

    metrics_df = pd.DataFrame(
        {
            "epoch": epochs,
            "train_loss": np.asarray(train_losses, dtype=np.float64),
            "val_loss": np.asarray(val_losses, dtype=np.float64),
            "LZ": np.asarray(metric_history["LZ"], dtype=np.float64),
            "hM": np.asarray(metric_history["hM"], dtype=np.float64),
            "PermEn": np.asarray(metric_history["PermEn"], dtype=np.float64),
            "D2": np.asarray(metric_history["D2"], dtype=np.float64),
            "symbolic_score": sym_score,
            "LZ_rank": ranks["LZ"],
            "hM_rank": ranks["hM"],
            "PermEn_rank": ranks["PermEn"],
            "D2_rank": ranks["D2"],
        }
    )
    if cfg.ses_include_df:
        metrics_df["DF"] = np.asarray(metric_history["DF"], dtype=np.float64)
        metrics_df["DF_rank"] = ranks["DF"]
    return row, metrics_df


def run_experiment(cfg: ExperimentConfig) -> Dict[str, str]:
    out_root = Path(cfg.out_root)
    data_root = Path(cfg.data_root)
    ensure_dir(out_root)
    dataset_paths = resolve_dataset_paths(data_root, cfg.datasets)
    histories_dir = out_root / "epoch_metrics"
    if cfg.save_epoch_metrics:
        ensure_dir(histories_dir)
    sigmas = noise_schedule(cfg)
    all_rows: List[Dict[str, float]] = []
    for csv_path in dataset_paths:
        print(f"[DATASET] {csv_path}")
        for model_name in cfg.models:
            for sigma in sigmas:
                for i in range(int(cfg.n_runs)):
                    seed = int(cfg.base_seed + i)
                    print(f"[RUN] {csv_path.stem} | {model_name} | noise={sigma:.2f} | seed={seed}")
                    row, metrics_df = run_one(csv_path=csv_path, model_name=model_name, seed=seed, noise_sigma=float(sigma), cfg=cfg)
                    all_rows.append(row)
                    if cfg.save_epoch_metrics:
                        metrics_path = histories_dir / f"{csv_path.stem}__{model_name}__noise_{sigma:.2f}__seed_{seed}.csv"
                        metrics_df.to_csv(metrics_path, index=False)
    df = pd.DataFrame(all_rows)
    per_run = out_root / "per_run.csv"
    summary = out_root / "summary_by_noise.csv"
    compact = out_root / "compact_table.csv"
    df.to_csv(per_run, index=False)
    df_summary, df_compact = make_summary_tables(df, oracle_eps=float(cfg.oracle_eps))
    df_summary.to_csv(summary, index=False)
    df_compact.to_csv(compact, index=False)
    (out_root / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    return {"per_run": str(per_run), "summary": str(summary), "compact": str(compact), "config": str(out_root / "run_config.json")}
