#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED_FAMILIES = ["CKA","LZ","hM","Perm","D2","hKS"]

def load_summary(path):
    df = pd.read_csv(path)
    # приведём предикторы к семействам (CKA, LZ, hM, Perm, D2, hKS)
    fam = []
    for s in df["predictor"].astype(str).fillna(""):
        s_low = s.lower()
        if "cka" in s_low:
            fam.append("CKA")
        elif "lz" in s_low:
            fam.append("LZ")
        elif "hm" in s_low or "hmarkov" in s_low:
            fam.append("hM")
        elif "perment" in s_low or "perm" in s_low:
            fam.append("Perm")
        elif "d2" in s_low:
            fam.append("D2")
        elif "hks" in s_low:
            fam.append("hKS")
        else:
            fam.append(s)
    df["family"] = fam
    return df

def ensure_out(dirpath):
    Path(dirpath).mkdir(parents=True, exist_ok=True)

def heatmap_earlyr2(df, E0, outpng):
    sub = df[(df["EarlyR2"].notna()) & (df["E0"] == E0)]
    # усредним по target, если их несколько (берём mean)
    g = sub.groupby(["dataset","family"])["EarlyR2"].mean().reset_index()
    datasets = sorted(g["dataset"].unique())
    fams = PRED_FAMILIES
    M = np.full((len(datasets), len(fams)), np.nan)
    for i,d in enumerate(datasets):
        row = g[g["dataset"]==d].set_index("family")["EarlyR2"]
        for j,f in enumerate(fams):
            if f in row.index:
                M[i,j] = row.loc[f]
    plt.figure(figsize=(1.2+1.4*len(fams), 1.2+0.8*len(datasets)))
    im = plt.imshow(M, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="EarlyR²")
    plt.xticks(range(len(fams)), fams, rotation=0)
    plt.yticks(range(len(datasets)), datasets)
    plt.title(f"EarlyR² heatmap (E₀={E0})")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i,j]
            if np.isfinite(val):
                txt = f"{val:.2f}"
                plt.text(j, i, txt, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def leaderboard_earlyr2(df, E0, outpng, topk=5):
    sub = df[(df["EarlyR2"].notna()) & (df["E0"] == E0)]
    # Для каждого датасета берём лучший предиктор
    items = []
    for d in sorted(sub["dataset"].unique()):
        g = sub[sub["dataset"]==d].groupby("family")["EarlyR2"].mean().reset_index()
        g = g.sort_values("EarlyR2", ascending=False).head(topk)
        for _,r in g.iterrows():
            items.append((d, r["family"], r["EarlyR2"]))
    if not items:
        return
    # рисуем сгруппированные бары: по датасетам, внутри — предикторы
    data = pd.DataFrame(items, columns=["dataset","family","EarlyR2"])
    datasets = sorted(data["dataset"].unique())
    fams = []
    for d in datasets:
        fams += list(data[data["dataset"]==d].sort_values("EarlyR2", ascending=False)["family"].values)
    # построим плоский барчарт в порядке (dataset blocks)
    plt.figure(figsize=(10, 0.5 + 0.6*len(items)))
    ylabels, yvals = [], []
    for d in datasets:
        block = data[data["dataset"]==d].sort_values("EarlyR2", ascending=True)
        for _,r in block.iterrows():
            ylabels.append(f"{d} · {r['family']}")
            yvals.append(r["EarlyR2"])
    y = np.arange(len(yvals))
    plt.barh(y, yvals)
    plt.yticks(y, ylabels)
    plt.xlabel("EarlyR²")
    plt.title(f"Лидерборд предикторов (E₀={E0})")
    plt.xlim(0,1.0)
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def bars_ett_auc(df, outpng):
    # возьмём средние по моделям/таргетам
    sub_ett = df[df["ETT"].notna()].copy()
    sub_auc1 = df[df["AUC(1-CKA)"].notna()].copy()
    sub_aucd = df[df["AUC_dyn"].notna()].copy()
    datasets = sorted(set(sub_ett["dataset"]) | set(sub_auc1["dataset"]) | set(sub_aucd["dataset"]))
    fams = PRED_FAMILIES

    rows = []
    for d in datasets:
        for f in fams:
            ett = sub_ett[(sub_ett["dataset"]==d) & (sub_ett["family"]==f)]["ETT"].mean()
            auc1 = sub_auc1[(sub_auc1["dataset"]==d) & (sub_auc1["family"]==f)]["AUC(1-CKA)"].mean()
            aucd = sub_aucd[(sub_aucd["dataset"]==d) & (sub_aucd["family"]==f)]["AUC_dyn"].mean()
            rows.append((d,f,ett,auc1,aucd))
    tab = pd.DataFrame(rows, columns=["dataset","family","ETT","AUC1minusCKA","AUCdyn"])

    # рисуем по каждому датасету 2 графика: ETT и AUC (динамический/CKA)
    n = len(datasets)
    plt.figure(figsize=(12, 3*n))
    for i,d in enumerate(datasets):
        sub = tab[tab["dataset"]==d]
        x = np.arange(len(fams))
        # ETT
        ax = plt.subplot(n, 2, 2*i+1)
        ax.bar(x, sub["ETT"].values)
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=0)
        ax.set_title(f"{d}: ETT (меньше — лучше)"); ax.set_ylabel("эпохи")
        ax.grid(True, axis="y", alpha=0.3)
        # AUC
        ax = plt.subplot(n, 2, 2*i+2)
        # сравним AUC_dyn (наши панели) и AUC(1-CKA) (только для CKA)
        width = 0.35
        auc_dyn = sub["AUCdyn"].values
        auc_cka = sub["AUC1minusCKA"].values
        ax.bar(x - width/2, auc_dyn, width, label="AUC_dyn (наши)")
        ax.bar(x + width/2, auc_cka, width, label="AUC(1-CKA)")
        ax.set_xticks(x); ax.set_xticklabels(fams, rotation=0)
        ax.set_title(f"{d}: площадь нестабильности (меньше — лучше)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def dumbbell_best_vs_cka(df, E0, outpng):
    # сравним CKA vs лучший из наших по EarlyR² для каждого датасета
    sub = df[(df["EarlyR2"].notna()) & (df["E0"]==E0)]
    rows = []
    for d in sorted(sub["dataset"].unique()):
        g = sub[sub["dataset"]==d].groupby("family")["EarlyR2"].mean()
        cka = g.get("CKA", np.nan)
        # лучший из наших (не CKA)
        ours = g.drop(labels=["CKA"], errors="ignore")
        if ours.empty:
            best_name, best_val = "(none)", np.nan
        else:
            best_name = ours.sort_values(ascending=False).index[0]
            best_val = ours.max()
        rows.append((d, cka, best_name, best_val))
    dt = pd.DataFrame(rows, columns=["dataset","CKA","best_ours_name","best_ours_val"])
    # рисуем dumbbell
    plt.figure(figsize=(8, 0.8 + 0.8*len(dt)))
    y = np.arange(len(dt))
    for i, r in dt.iterrows():
        x1 = 0 if not np.isfinite(r["CKA"]) else r["CKA"]
        x2 = 0 if not np.isfinite(r["best_ours_val"]) else r["best_ours_val"]
        plt.plot([x1, x2], [i, i], linewidth=2)
        plt.scatter([x1, x2], [i, i], s=40)
        label = f"{r['dataset']}  ({r['best_ours_name']})"
        plt.text(1.02, i, label, va="center")  # подпись справа
    plt.yticks([], [])
    plt.xlim(0,1.05)
    plt.xlabel("EarlyR²")
    plt.title(f"CKA vs лучший из наших (E₀={E0}) → вправо = лучше")
    plt.tight_layout()
    plt.savefig(outpng, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/_comparative/summary.csv")
    ap.add_argument("--outdir", default="results/_comparative/viz")
    args = ap.parse_args()

    ensure_out(args.outdir)
    df = load_summary(args.summary)

    # 1) теплокарты EarlyR²
    heatmap_earlyr2(df, 10, os.path.join(args.outdir,"earlyr2_heatmap_E0_10.png"))
    heatmap_earlyr2(df, 25, os.path.join(args.outdir,"earlyr2_heatmap_E0_25.png"))

    # 2) лидерборды
    leaderboard_earlyr2(df, 10, os.path.join(args.outdir,"leaderboard_E0_10.png"))
    leaderboard_earlyr2(df, 25, os.path.join(args.outdir,"leaderboard_E0_25.png"))

    # 3) ETT/AUC
    bars_ett_auc(df, os.path.join(args.outdir,"ett_auc_panels.png"))

    # 4) гантели CKA vs лучший из наших
    dumbbell_best_vs_cka(df, 10, os.path.join(args.outdir,"dumbbell_best_vs_cka_E0_10.png"))
    dumbbell_best_vs_cka(df, 25, os.path.join(args.outdir,"dumbbell_best_vs_cka_E0_25.png"))

    print("[ok] визуализации готовы в", args.outdir)

if __name__ == "__main__":
    main()
