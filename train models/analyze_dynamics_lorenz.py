# -*- coding: utf-8 -*-
"""
Analyze hidden-state dynamics for Lorenz-trained models (RNN_LSTM, RNN_BiLSTM, Transformer).
Reads NPZ dumps created by train_* scripts, stitches a continuous trajectory from overlapping
windows, then computes: D2 (Grassberger–Procaccia), h_KS from block entropies with
Miller–Madow, Markov entropy rate from the transition matrix (k-means symbols), LZ_norm,
and permutation entropy on PC1. Produces per-epoch metrics CSV and comparison plots.
"""
import os, math, json, argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree, NearestNeighbors

def load_states_npz(path):
    z = np.load(path, allow_pickle=True)
    H_list = list(z['H_seq'])
    return [np.array(h, dtype=float) for h in H_list]

def stitch_hidden(H_list, mode='stitch1'):
    if not H_list:
        return np.empty((0,0))
    if mode == 'last':
        return np.stack([h[-1] for h in H_list], axis=0)
    out = [H_list[0]]
    for h in H_list[1:]:
        out.append(h[-1:])
    return np.concatenate(out, axis=0)

def delayed_mutual_information(x, max_tau=120, bins=64):
    x = np.asarray(x, float).ravel()
    hist_x, edges = np.histogram(x, bins=bins, density=True)
    p_x = hist_x / (hist_x.sum() + 1e-12)
    I = []
    for tau in range(1, max_tau+1):
        x1 = x[:-tau]; x2 = x[tau:]
        H, _, _ = np.histogram2d(x1, x2, bins=bins, density=True)
        p12 = H/(H.sum()+1e-12)
        p2, _ = np.histogram(x2, bins=edges, density=True)
        p2 = p2/(p2.sum()+1e-12)
        I.append(float(np.nansum(p12 * np.log((p12+1e-12)/((p_x[:,None]*p2[None,:])+1e-12)))))
    return np.array(I)

def first_minimum(a):
    for i in range(1, len(a)-1):
        if a[i] < a[i-1] and a[i] <= a[i+1]:
            return i+1
    return int(np.argmin(a)+1)

def false_nearest_neighbors(pc1, tau, m_max=12, Rtol=15.0, Atol=2.0):
    x = np.asarray(pc1, float).ravel()
    N = len(x) - m_max*tau
    if N <= 10: return 3
    def embed(m):
        idx = np.arange(m)[:,None]*tau + np.arange(N)[None,:]
        return x[idx].T
    fracs=[]
    for m in range(1, m_max+1):
        Xm = embed(m)
        nbr = NearestNeighbors(n_neighbors=2).fit(Xm)
        d, idx = nbr.kneighbors(Xm, return_distance=True)
        d1 = d[:,1]+1e-12
        if m < m_max:
            Xm1 = embed(m+1)
            j = idx[:,1]
            num = np.abs(Xm1[:, -1] - Xm1[j, -1])
            R = num/d1
            A = num/np.std(x)
            fnn = np.mean((R>Rtol)|(A>Atol))
        else:
            fnn = fracs[-1] if fracs else 1.0
        fracs.append(fnn)
        if m>=2 and fnn<0.1: return m
    return int(np.argmin(fracs)+1)

def takens(x, m, tau):
    x = np.asarray(x, float).ravel()
    N = len(x) - (m-1)*tau
    idx = np.arange(m)[:,None]*tau + np.arange(N)[None,:]
    return x[idx].T

def correlation_dimension(X, q_lo=0.02, q_hi=0.98, n_eps=24, theiler=10, rng=0):
    X = np.asarray(X, float); N = len(X)
    if N < 200: return float('nan'), None
    tree = KDTree(X)
    rs = np.random.RandomState(rng)
    idx = rs.choice(N, size=min(3000,N), replace=False)
    d2,_ = tree.query(X[idx], k=2); d1 = d2[:,1]
    lo, hi = np.quantile(d1, q_lo), np.quantile(d1, q_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:
        return float('nan'), None
    eps = np.geomspace(lo, hi, n_eps)
    counts=[]
    batch=2048
    for e in eps:
        cnt=0; den=0
        for s in range(0,N,batch):
            t = min(N, s+batch)
            ind = tree.query_radius(X[s:t], r=e, count_only=False, return_distance=False)
            for i_local, neigh in enumerate(ind):
                i = s+i_local
                neigh = neigh[(neigh!=i) & (np.abs(neigh-i) > theiler)]
                cnt += neigh.size
                den += max(0, N-1-2*theiler)
        C = (cnt/max(den,1)) if den>0 else 0.0
        counts.append(max(C,1e-12))
    xlog=np.log(eps); ylog=np.log(np.array(counts))
    a=int(len(xlog)*0.2); b=int(len(xlog)*0.8)
    A=np.vstack([xlog[a:b], np.ones(b-a)]).T
    slope,intercept = np.linalg.lstsq(A, ylog[a:b], rcond=None)[0]
    return float(max(0.0,slope)), (eps, counts, (a,b), (slope,intercept))

def permutation_entropy(x, m=5, tau=1):
    x = np.asarray(x, float).ravel()
    n = len(x)-(m-1)*tau
    if n<=0: return float('nan'), float('nan')
    patterns=[]
    for i in range(n):
        w = x[i:(i+m*tau):tau]
        patterns.append(tuple(np.argsort(w)))
    cnt = Counter(patterns); total=sum(cnt.values())
    p = np.array([v/total for v in cnt.values()], float)
    H = -np.nansum(p*np.log2(p+1e-12))
    Hmax = math.log2(math.factorial(m))
    return float(H), float(H/Hmax)

def transition_matrix(labels, K):
    P = np.zeros((K,K), float)
    for i in range(len(labels)-1):
        P[labels[i], labels[i+1]] += 1.
    row = P.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.divide(P, row, out=np.zeros_like(P), where=row>0)
    P = (P+1e-12); P = P/P.sum(1, keepdims=True)
    return P

def pi_stat(P, tol=1e-10, iters=10000):
    K=P.shape[0]; pi=np.ones(K)/K
    for _ in range(iters):
        new = pi @ P
        if np.linalg.norm(new-pi,1)<tol: return new
        pi=new
    return pi

def markov_entropy_rate(P):
    pi = pi_stat(P)
    h=0.0
    for i in range(P.shape[0]):
        row=P[i]; h += pi[i]*(- (row*np.log2(row+1e-12)).sum())
    return float(h)

def block_entropies_mm(labels, nmax=8, min_samples=800, min_unique=20):
    ns, Hs = [], []
    L=len(labels)
    for n in range(1, nmax+1):
        if L<n: break
        tuples=[tuple(labels[i:i+n]) for i in range(L-n+1)]
        Nn=len(tuples); cnt=Counter(tuples); kn=len(cnt)
        if Nn<min_samples or kn<min_unique: break
        p=np.array([v/Nn for v in cnt.values()], float)
        Hmle=(-p*np.log2(p+1e-12)).sum()
        Hmm=Hmle + (max(kn,1)-1)/(2.0*Nn*math.log(2.0))
        ns.append(n); Hs.append(float(Hmm))
    return ns, Hs

def ks_slope(ns, Hs, tail=3):
    if len(ns)<2: return float('nan')
    Hs = np.maximum.accumulate(np.asarray(Hs,float))
    d = np.diff(Hs); k=max(1, min(tail, len(d)))
    return float(max(0.0, np.mean(d[-k:])))

def one_series_metrics(H, K=10, ami_max_tau=120, fnn_mmax=12, theiler=10):
    if H.ndim != 2 or len(H) < 64:
        return dict(D2=np.nan, hKS=np.nan, Hmarkov=np.nan, LZ=np.nan,
                    PermEnt=np.nan, PermEntNorm=np.nan, tau=np.nan, m=np.nan, N=len(H))
    pc1 = PCA(n_components=2, random_state=0).fit_transform(H)[:,0]
    I = delayed_mutual_information(pc1, max_tau=ami_max_tau, bins=64)
    tau = first_minimum(I)
    m = false_nearest_neighbors(pc1, tau=tau, m_max=fnn_mmax)
    X = takens(pc1, m=m, tau=tau)
    Z = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=K, n_init=20, random_state=0).fit_predict(Z)
    P = transition_matrix(labels, K)
    Hmarkov = markov_entropy_rate(P)
    ns, Hs = block_entropies_mm(labels, nmax=8, min_samples=800, min_unique=20)
    hKS = ks_slope(ns, Hs, tail=3)
    def lz_complexity_int(seq):
        s = list(map(int, seq)); n=len(s)
        if n==0: return 0
        c,i,k=1,0,1
        while True:
            if i+k>n: break
            sub=s[i:i+k]
            found=any(s[j:j+k]==sub for j in range(0,i))
            if found:
                k+=1
                if i+k>n: c+=1; break
            else:
                c+=1; i+=k; k=1
                if i+1>n: break
        return c
    LZ = np.nan if len(labels)<2 else lz_complexity_int(labels)/(len(labels)/math.log(len(labels)))
    Hp, HpN = permutation_entropy(pc1, m=5, tau=tau)
    D2, _ = correlation_dimension(X, q_lo=0.02, q_hi=0.98, n_eps=24, theiler=max(2*tau, theiler))
    return dict(D2=D2, hKS=hKS, Hmarkov=Hmarkov, LZ=LZ, PermEnt=Hp, PermEntNorm=HpN, tau=tau, m=m, N=len(H))

def analyze(root_dirs, epochs, outdir, stitch_mode='stitch1', K=10):
    os.makedirs(outdir, exist_ok=True)
    models = [(name, rd) for name, rd in root_dirs]
    by_model = {name: [] for name,_ in models}
    for name, rd in models:
        for ep in epochs:
            npz = os.path.join(rd, f"states_epoch_{ep:03d}.npz")
            if not os.path.exists(npz):
                by_model[name].append((ep, None)); continue
            H_list = load_states_npz(npz)
            Htraj = stitch_hidden(H_list, mode=stitch_mode)
            m = one_series_metrics(Htraj, K=K)
            by_model[name].append((ep, m))
    # CSV
    import csv
    csv_path = os.path.join(outdir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model","epoch","N","tau","m","D2","h_KS","H_markov","LZ_norm","PermEnt","PermEnt_norm"])
        for name, rows in by_model.items():
            for ep, m in rows:
                if m is None:
                    w.writerow([name, ep, "", "", "", "", "", "", "", "", ""])
                else:
                    w.writerow([name, ep, m["N"], m["tau"], m["m"], m["D2"], m["hKS"], m["Hmarkov"], m["LZ"], m["PermEnt"], m["PermEntNorm"]])
    # plots
    def plot_metric(metric_key, title, ylabel, fname):
        plt.figure(figsize=(10,5))
        for name, rows in by_model.items():
            xs=[]; ys=[]
            for ep, m in rows:
                if m is None or (m.get(metric_key) is None) or (isinstance(m.get(metric_key), float) and np.isnan(m[metric_key])):
                    continue
                xs.append(ep); ys.append(m[metric_key])
            if xs:
                xs, ys = zip(*sorted(zip(xs,ys)))
                plt.plot(xs, ys, marker='o', label=name)
        plt.xlabel("epoch"); plt.ylabel(ylabel); plt.title(title); plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=150); plt.close()

    plot_metric("D2", "Correlation dimension D2", "Correlation dimension D2", "compare_D2.png")
    plot_metric("hKS", "h_KS (slope of H(n))", "h_KS (slope of H(n))", "compare_hKS.png")
    plot_metric("Hmarkov", "Entropy rate (bits/step)", "Entropy rate (bits/step)", "compare_Hmarkov.png")
    plot_metric("LZ", "LZ (norm)", "LZ (norm)", "compare_LZ_norm.png")
    plot_metric("PermEntNorm", "Permutation entropy (norm)", "Permutation entropy (norm)", "compare_PermEnt_norm.png")
    return {
        "csv": csv_path,
        "plots": [os.path.join(outdir, f) for f in [
            "compare_D2.png","compare_hKS.png","compare_Hmarkov.png","compare_LZ_norm.png","compare_PermEnt_norm.png"
        ]]
    }

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnn_dir", type=str, required=True)
    ap.add_argument("--bilstm_dir", type=str, required=True)
    ap.add_argument("--trf_dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[0,1,5,10,25,50,100,150,200])
    ap.add_argument("--outdir", type=str, default="lorenz_compare")
    ap.add_argument("--stitch_mode", type=str, default="stitch1", choices=["last","stitch1"])
    ap.add_argument("--K", type=int, default=10)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    res = analyze(
        root_dirs=[("RNN_LSTM", args.rnn_dir), ("RNN_BiLSTM", args.bilstm_dir), ("Transformer", args.trf_dir)],
        epochs=args.epochs, outdir=args.outdir, stitch_mode=args.stitch_mode, K=args.K
    )
    print(json.dumps(res, indent=2))
