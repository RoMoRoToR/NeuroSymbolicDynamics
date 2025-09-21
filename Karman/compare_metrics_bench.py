#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение наших метрик vs CKA/SVCCA/проч. на уже посчитанных результатах.
Ожидаемые входы по датасетам (созданы analyze_dynamics_sota_new_2.py):
  results/<ds>/metrics.csv             # наши метрики по эпохам и моделям
  results/<ds>/cka_vs_epoch0.csv       # CKA по эпохам (колонки = модели)
  results/<ds>/val_scores.csv          # финальные качественные метрики (если есть)
  results/karman/physics.csv           # для вихревого: St, CL_rms, CD_mean (опц.)

Выход:
  results/_comparative/<ds>/*.csv, *.png  # таблицы и диаграммы
  results/_comparative/summary.csv        # общий свод
"""

import os, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

# ------------- конфиг (под свои имена моделей при необходимости) -------------
DATASETS = ["lorenz","air","bitcoin","karman"]
MODELS = ["Transformer","RNN","BiLSTM"]     # должны совпадать с колонками/именами в csv

EARLY_EPOCHS = [10, 25]
CKA_THR = 0.95
STABLE_Z = 0.5
STABLE_K = 5

def read_csv_safe(path):
    return pd.read_csv(path) if Path(path).exists() else None

def load_metrics(ds):
    base = Path(f"results/{ds}")
    m = read_csv_safe(base / "metrics.csv")
    cka = read_csv_safe(base / "cka_vs_epoch0.csv")
    val = read_csv_safe(base / "val_scores.csv")     # опц.: финальные качества
    phys = read_csv_safe(base / "physics.csv")       # только karman
    return m, cka, val, phys

def first_stable_epoch(series, zthr=STABLE_Z, K=STABLE_K):
    if series is None or series.empty: return None
    x = (series - series.mean()) / (series.std() + 1e-8)
    # ищем первый отрезок длиной K, где |z|<zthr
    for i in range(0, len(x)-K+1):
        seg = x.iloc[i:i+K].abs()
        if (seg < zthr).all():
            return int(series.index[i])
    return None

def auc_instability(series):
    if series is None or series.empty: return None
    z = (series - series.mean()) / (series.std() + 1e-8)
    pos = np.clip(z, 0, None)
    return float(np.trapz(pos, x=series.index))

def early_r2(early_values, target):
    """
    Robust EarlyR^2: squared Pearson correlation with intercept.
    No LAPACK calls, ignores NaN/Inf, handles constant vectors.
    """
    x = np.array(early_values, dtype=float)
    y = np.array(target, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2:
        return None
    sx = x.std()
    sy = y.std()
    if sx == 0 or sy == 0:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(r):
        return None
    return float(r * r)


def kendall_tau_rank(order_a, order_b):
    try:
        # order_* — Series со значениями "качества" по моделям
        a = order_a.rank()
        b = order_b.rank()
        tau, _ = kendalltau(a, b)
        return float(tau)
    except Exception:
        return None

def cosine_drift_stub(metrics_df, model):
    # прокси для "smoothness": возьмём hMarkov как меру неожиданности перехода;
    # в идеале сюда подставить реальную косинусную дистанцию между h_t и h_{t+1}.
    col = f"{model}_hmarkov" if f"{model}_hmarkov" in metrics_df.columns else None
    return metrics_df[["epoch", col]].set_index("epoch")[col] if col else None

def collect_targets(ds, val, phys):
    # что предсказываем? одно число на модель/ран
    # приоритет: физика для karman, иначе финальная валидационная метрика (если есть)
    targets = {}
    if ds == "karman" and phys is not None:
        # пример: минимизируем CL_rms и/или CD_mean; высота пика PSD(CL) — тоже цель
        if "CL_rms" in phys.columns:
            targets["CL_rms"] = float(phys["CL_rms"].iloc[-1])
        if "CD_mean" in phys.columns:
            targets["CD_mean"] = float(phys["CD_mean"].iloc[-1])
        if "PSDpeak" in phys.columns:
            targets["PSDpeak"] = float(phys["PSDpeak"].iloc[-1])
    elif val is not None:
        # ожидаем столбцы вида: model, final_val_loss (или аналог)
        # соберём по моделям среднее
        if {"model","final"}.issubset(set(map(str.lower, val.columns))):
            # если заранее приведено
            pass
        targets["val_proxy"] = val.groupby("model").agg({"final_metric":"mean"}).squeeze() \
            if "final_metric" in val.columns else None
    return targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--outdir", default="results/_comparative")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for ds in args.datasets:
        m, cka, val, phys = load_metrics(ds)
        if m is None:
            print(f"[warn] {ds}: нет metrics.csv — пропускаю")
            continue

        # нормализуем колонки: epoch + наши метрики по моделям
        if "epoch" not in m.columns:
            # пытаемся угадать
            if "Epoch" in m.columns:
                m.rename(columns={"Epoch":"epoch"}, inplace=True)
        m = m.copy()
        if "epoch" in m.columns: m["epoch"] = m["epoch"].astype(int)
        m = m.sort_values("epoch")

        # приготовим сериальные ряды по моделям
        m_by = {model:{} for model in MODELS}
        idx = m["epoch"].values
        for model in MODELS:
            for key in ["LZ_norm","Hmarkov","PermEnt_norm","D2","hKS"]:
                col = f"{model}_{key}"
                if col in m.columns:
                    m_by[model][key] = pd.Series(m[col].values, index=idx)

        # CKA
        cka_by = {}
        if cka is not None and "epoch" in cka.columns:
            cka_ = cka.set_index("epoch")
            for model in MODELS:
                # ищем колонку, содержащую имя модели
                col = [c for c in cka_.columns if model.lower() in c.lower()]
                if col:
                    cka_by[model] = pd.Series(cka_[col[0]].values, index=cka_.index)

        # smoothness proxy (косинусный дрейф) — здесь заглушка через hMarkov
        sm_by = {model: cosine_drift_stub(m, model) for model in MODELS}

        # цели (что предсказываем)
        targets = collect_targets(ds, val, phys)
        # если пусто — сделаем суррогат: возьмём финальный средний Hmarkov (меньше — «лучше упорядоченность»)
        if not targets:
            targets = {"dyn_order": pd.Series({
                model: m_by[model]["Hmarkov"].iloc[-1] if "Hmarkov" in m_by[model] else np.nan
                for model in MODELS
            })}

        # считаем агрегаты
        rows = []
        for name, target in targets.items():
            # target может быть scalar (karman) или Series по моделям
            if np.isscalar(target):
                # одна цель на весь датасет — будем смотреть корреляции по эпохам (мульти-подход)
                pass
            else:
                # Series по моделям
                tgt = target.dropna()
                # EarlyR2 для каждого предиктора на ранней эпохе
                for E0 in EARLY_EPOCHS:
                    # наши метрики
                    for model in MODELS:
                        if model not in tgt.index: continue
                    # формируем вектор X по моделям: среднее наших метрик на E0
                    def x_from(metric_key):
                        vals = []
                        idxs = []
                        for model in MODELS:
                            if model not in tgt.index: continue
                            s = m_by[model].get(metric_key)
                            if s is None or E0 not in s.index: continue
                            vals.append(float(s.loc[E0]))
                            idxs.append(model)
                        return pd.Series(vals, index=idxs)

                    Xs = {
                        "LZ": x_from("LZ_norm"),
                        "hM": x_from("Hmarkov"),
                        "Perm": x_from("PermEnt_norm"),
                        "D2": x_from("D2"),
                        "hKS": x_from("hKS")
                    }
                    # CKA и smoothness
                    def x_from_cka():
                        vals, idxs = [], []
                        for model in MODELS:
                            s = cka_by.get(model)
                            if s is None or E0 not in s.index: continue
                            vals.append(float(s.loc[E0]))
                            idxs.append(model)
                        return pd.Series(vals, index=idxs)
                    def x_from_sm():
                        vals, idxs = [], []
                        for model in MODELS:
                            s = sm_by.get(model)
                            if s is None or E0 not in s.index: continue
                            vals.append(float(s.loc[E0]))
                            idxs.append(model)
                        return pd.Series(vals, index=idxs)

                    Xs["CKA"] = x_from_cka()
                    Xs["Smooth"] = x_from_sm()

                    # EarlyR2
                    for pred_name, X in Xs.items():
                        common = tgt.index.intersection(X.index)
                        if len(common) >= 2:
                            r2 = early_r2(X.loc[common].values, tgt.loc[common].values)
                            rows.append({"dataset": ds, "target": name, "E0": E0,
                                         "predictor": pred_name, "EarlyR2": r2})

                # ETT, AUC и Kendall-τ (ранжирование)
                # строим «качество» по моделям на финале
                final_quality = tgt  # чем меньше — тем лучше (при необходимости инвертируй знак)
                # ранжирование моделей нашими метриками на финале
                rank_pred = {}
                for pred_name, key in [("LZ","LZ_norm"),("hM","Hmarkov"),("Perm","PermEnt_norm"),
                                       ("D2","D2"),("hKS","hKS")]:
                    vals = {}
                    for model in MODELS:
                        s = m_by[model].get(key)
                        if s is not None and not s.empty:
                            vals[model] = s.iloc[-1]
                    rank_pred[pred_name] = pd.Series(vals)
                # Kendall-τ
                for pred_name, sr in rank_pred.items():
                    common = final_quality.index.intersection(sr.index)
                    if len(common) >= 2:
                        tau = kendall_tau_rank(final_quality.loc[common], sr.loc[common])
                        rows.append({"dataset": ds, "target": name, "E0": "final",
                                     "predictor": pred_name, "KendallTau": tau})

        # ETT и AUC (модельно по сериям)
        for model in MODELS:
            # CKA
            if model in cka_by:
                s = cka_by[model].dropna()
                ett = next((int(e) for e,v in s.items() if v>=CKA_THR), None)
                auc = float(np.trapz(1.0 - s.values, x=s.index)) if len(s)>1 else None
                rows.append({"dataset": ds, "target": "stability", "E0": "—",
                             "predictor": f"CKA@{model}", "ETT": ett, "AUC(1-CKA)": auc})
            # наши
            for key, lab in [("LZ_norm","LZ"),("Hmarkov","hM"),("D2","D2")]:
                s = m_by[model].get(key)
                if s is not None and not s.empty:
                    ett = first_stable_epoch(s)
                    auc = auc_instability(s)
                    rows.append({"dataset": ds, "target": "stability", "E0": "—",
                                 "predictor": f"{lab}@{model}", "ETT": ett, "AUC_dyn": auc})

        out_ds = Path(args.outdir) / ds
        out_ds.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(out_ds / "comparative_table.csv", index=False)
        summary_rows += rows
        print(f"[ok] {ds}: {len(rows)} записей")

    pd.DataFrame(summary_rows).to_csv(Path(args.outdir)/"summary.csv", index=False)
    print(f"[ok] summary -> {Path(args.outdir)/'summary.csv'}")

if __name__ == "__main__":
    main()
