#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FAMS = ["CKA","LZ","hM","Perm","D2","hKS"]

def ensure_out(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_summary(path):
    df = pd.read_csv(path)
    # унифицируем имена колонок, на всякий случай
    cols = {c.lower(): c for c in df.columns}
    def pick(names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols: return cols[n.lower()]
        return None

    # family из predictor
    fam = []
    for s in df.get(pick(["predictor","metric","name"]), pd.Series([""]*len(df))).astype(str):
        sl = s.lower()
        if "cka" in sl: fam.append("CKA")
        elif "lz" in sl: fam.append("LZ")
        elif "hmarkov" in sl or sl=="hm" or "entropy_rate" in sl: fam.append("hM")
        elif "perm" in sl: fam.append("Perm")
        elif "d2" in sl: fam.append("D2")
        elif "hks" in sl: fam.append("hKS")
        else: fam.append(s)
    df["family"] = fam

    # нормализуем ключевые поля
    for need in ["dataset","family","E0","EarlyR2","ETT","AUC_dyn","AUC(1-CKA)"]:
        if need not in df.columns:
            df[need] = np.nan
    # coerces
    for num in ["E0","EarlyR2","ETT","AUC_dyn","AUC(1-CKA)"]:
        df[num] = pd.to_numeric(df[num], errors="coerce")
    return df

def heatmap_earlyr2(df, E0, outpng):
    sub = df[(df["E0"] == E0) & df["EarlyR2"].notna()]
    if sub.empty:
        print(f"[warn] нет EarlyR² для E0={E0} — пропускаю heatmap")
        return
    g = sub.groupby(["dataset","family"])["EarlyR2"].mean().reset_index()
    datasets = sorted(g["dataset"].unique())
    M = np.full((len(datasets), len(FAMS)), np.nan)
    for i, d in enumerate(datasets):
        row = g[g["dataset"]==d].set_index("family")["EarlyR2"]
        for j, f in enumerate(FAMS):
            if f in row.index:
                M[i, j] = row.loc[f]
    plt.figure(figsize=(1.6+1.2*len(FAMS), 1.2+0.8*max(1,len(datasets))))
    im = plt.imshow(M, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="EarlyR²")
    plt.xticks(range(len(FAMS)), FAMS)
    plt.yticks(range(len(datasets)), datasets)
    plt.title(f"EarlyR² heatmap (E₀={E0})")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i,j]
            if np.isfinite(v):
                plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v>0.5 else "k")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def leaderboard_earlyr2(df, E0, outpng, topk=5):
    sub = df[(df["E0"]==E0) & df["EarlyR2"].notna()]
    if sub.empty:
        print(f"[warn] нет EarlyR² для E0={E0} — пропускаю leaderboard")
        return
    items = []
    for d in sorted(sub["dataset"].unique()):
        g = sub[sub["dataset"]==d].groupby("family")["EarlyR2"].mean().reset_index()
        g = g.sort_values("EarlyR2", ascending=False).head(topk)
        for _, r in g.iterrows():
            items.append((d, r["family"], r["EarlyR2"]))
    data = pd.DataFrame(items, columns=["dataset","family","EarlyR2"])
    if data.empty:
        print(f"[warn] leaderboard(E0={E0}): пусто")
        return
    rows = []
    for d in sorted(data["dataset"].unique()):
        block = data[data["dataset"]==d].sort_values("EarlyR2", ascending=True)
        for _, r in block.iterrows():
            rows.append((f"{d} · {r['family']}", r["EarlyR2"]))
    lab = [r[0] for r in rows]
    val = [r[1] for r in rows]
    y = np.arange(len(val))
    plt.figure(figsize=(10, 0.6+0.5*len(val)))
    plt.barh(y, val)
    plt.yticks(y, lab)
    plt.xlabel("EarlyR²"); plt.xlim(0,1.0)
    plt.title(f"Лидерборд предикторов (E₀={E0})")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def bars_ett_auc(df, outpng):
    # агрегируем
    datasets = sorted(df["dataset"].dropna().unique())
    if not datasets:
        print("[warn] нет датасетов в summary для ETT/AUC")
        return
    n = len(datasets)
    fig_h = 2.4*n
    plt.figure(figsize=(12, fig_h))
    plot_idx = 1
    for d in datasets:
        sub = df[df["dataset"]==d].groupby("family")[["ETT","AUC_dyn","AUC(1-CKA)"]].mean()
        # панель ETT
        ax = plt.subplot(n, 2, plot_idx); plot_idx += 1
        if sub["ETT"].notna().any():
            x = np.arange(len(FAMS))
            vals = [sub["ETT"].get(f, np.nan) for f in FAMS]
            ax.bar(x, np.nan_to_num(vals, nan=0.0))
            ax.set_xticks(x); ax.set_xticklabels(FAMS)
            ax.set_ylabel("эпохи")
            ax.set_title(f"{d}: ETT (меньше — лучше)")
            ax.grid(True, axis="y", alpha=0.3)
        else:
            ax.axis("off")
            ax.text(0.02, 0.5, f"{d}: ETT — нет данных", transform=ax.transAxes)
        # панель AUC
        ax = plt.subplot(n, 2, plot_idx); plot_idx += 1
        has_dyn = sub["AUC_dyn"].notna().any()
        has_cka = sub["AUC(1-CKA)"].notna().any()
        x = np.arange(len(FAMS))
        if has_dyn or has_cka:
            width = 0.35
            if has_dyn:
                vals_dyn = [sub["AUC_dyn"].get(f, np.nan) for f in FAMS]
                ax.bar(x - width/2, np.nan_to_num(vals_dyn, nan=0.0), width, label="AUC_dyn (наши)")
            if has_cka:
                vals_cka = [sub["AUC(1-CKA)"].get(f, np.nan) for f in FAMS]
                ax.bar(x + width/2, np.nan_to_num(vals_cka, nan=0.0), width, label="AUC(1-CKA)")
            ax.set_xticks(x); ax.set_xticklabels(FAMS)
            ax.set_title(f"{d}: площадь нестабильности (меньше — лучше)")
            ax.grid(True, axis="y", alpha=0.3); ax.legend()
        else:
            ax.axis("off")
            ax.text(0.02, 0.5, f"{d}: AUC — нет данных", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def dumbbell_best_vs_cka(df, E0, outpng):
    sub = df[(df["E0"]==E0) & df["EarlyR2"].notna()]
    if sub.empty:
        print(f"[warn] нет EarlyR² для E0={E0} — пропускаю dumbbell")
        return
    rows = []
    for d in sorted(sub["dataset"].unique()):
        g = sub[sub["dataset"]==d].groupby("family")["EarlyR2"].mean()
        cka = g.get("CKA", np.nan)
        ours = g.drop(labels=["CKA"], errors="ignore")
        if ours.empty: continue
        best_name = ours.sort_values(ascending=False).index[0]
        best_val  = ours.max()
        rows.append((d, cka, best_name, best_val))
    if not rows:
        print(f"[warn] dumbbell(E0={E0}): нечего рисовать")
        return
    dt = pd.DataFrame(rows, columns=["dataset","CKA","best_ours_name","best_ours_val"])
    plt.figure(figsize=(8, 0.8 + 0.8*len(dt)))
    y = np.arange(len(dt))
    for i, r in dt.iterrows():
        x1 = r["CKA"] if np.isfinite(r["CKA"]) else 0.0
        x2 = r["best_ours_val"] if np.isfinite(r["best_ours_val"]) else 0.0
        plt.plot([x1, x2], [i, i], linewidth=2)
        plt.scatter([x1, x2], [i, i], s=40)
        plt.text(1.02, i, f"{r['dataset']} ({r['best_ours_name']})", va="center")
    plt.yticks([], [])
    plt.xlim(0,1.05)
    plt.xlabel("EarlyR²")
    plt.title(f"CKA vs лучший из наших (E₀={E0}) → вправо лучше")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/_comparative/summary.csv")
    ap.add_argument("--outdir",  default="results/_comparative/viz")
    args = ap.parse_args()

    ensure_out(args.outdir)
    df = load_summary(args.summary)

    # Какие E0 реально есть
    e0_vals = sorted([int(e) for e in df["E0"].dropna().unique()])
    if not e0_vals:
        print("[warn] В summary.csv нет столбца/значений E0 — нарисую только AUC/ETT панели.")
    else:
        for e0 in e0_vals:
            heatmap_earlyr2(df, e0, os.path.join(args.outdir, f"earlyr2_heatmap_E0_{e0}.png"))
            leaderboard_earlyr2(df, e0, os.path.join(args.outdir, f"leaderboard_E0_{e0}.png"))
            dumbbell_best_vs_cka(df, e0, os.path.join(args.outdir, f"dumbbell_best_vs_cka_E0_{e0}.png"))

    bars_ett_auc(df, os.path.join(args.outdir, "ett_auc_panels.png"))
    print("[ok] визуализации готовы в", args.outdir)

if __name__ == "__main__":
    main()
