#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Собирает единый отчёт (report.md и report.html) по результатам:
 - results/<dataset>/*.png (compare_LZ_norm.png, compare_Hmarkov.png, compare_PermEnt_norm.png, compare_D2.png, compare_hKS.png, CKA_vs_epoch0.png, ...)
 - results/<dataset>/symbolic_bridge.html (ссылка)
 - (опц.) CSV c CKA (CKA_vs_epoch0.csv) — чтобы посчитать AUC(1-CKA) и "эпоху стабилизации" (CKA>=0.95)

Предполагается, что ты уже запустил analyze_dynamics_sota_new_2.py и multi_model_symbolic_flow.py
для датасетов lorenz/air/bitcoin и сохранил результаты в results/<dataset>/.

Если структура и имена немного отличаются — скрипт максимально терпим:
он ищет файлы по паттернам и пропускает отсутствующие.
"""

import os, io, sys, glob, base64, csv, math
from pathlib import Path
from collections import defaultdict, OrderedDict

# -------- настрой: где лежат результаты --------
DATASETS = OrderedDict({
    "Lorenz":   "results/lorenz",
    "Air":      "results/air",
    "Bitcoin":  "results/bitcoin",
})

# какие картинки пытаться собирать в мини-галерею (в нужном порядке)
GALLERY_PATTERNS = [
    "compare_LZ_norm*.png",
    "compare_Hmarkov*.png",
    "compare_PermEnt_norm*.png",
    "compare_D2*.png",
    "compare_hKS*.png",
    "CKA_vs_epoch0*.png",
]

# имена CSV для CKA (любая из них, если найдётся)
CKA_CSV_CANDIDATES = [
    "CKA_vs_epoch0.csv", "cka_vs_epoch0.csv",
    "CKA_vs_epoch0*.csv", "cka_vs_epoch0*.csv",
]

# порог стабилизации CKA
CKA_STAB_THRESHOLD = 0.95

# -------- утиль --------
def find_one(path_glob_list):
    """вернуть первый существующий путь по списку/паттернам, иначе None"""
    for patt in path_glob_list:
        for p in sorted(glob.glob(patt)):
            return p
    return None

def find_all(outdir, patterns):
    files = []
    for patt in patterns:
        files.extend(sorted(glob.glob(os.path.join(outdir, patt))))
    # deduplicate сохраняя порядок
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def safe_rel(a, b):
    """относительный путь a относительно b (если не получается — вернуть a)"""
    try:
        return os.path.relpath(a, start=b)
    except Exception:
        return a

def parse_cka_csv(cka_csv_path):
    """
    читает CSV с CKA по эпохам. Ожидаемые схемы:
      epoch, RNN, BiLSTM, Transformer
      или epoch,<model1>,<model2>,...
    возвращает: dict(model -> list[(epoch, value)])
    """
    series = defaultdict(list)
    if not cka_csv_path or not os.path.exists(cka_csv_path):
        return series

    with open(cka_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in reader.fieldnames or []]
        # найдём столбец эпохи
        epoch_col = None
        for cand in ["epoch", "Epoch", "ep", "EPOCH"]:
            if cand in fieldnames:
                epoch_col = cand; break
        if epoch_col is None:
            # возьмём первую колонку как эпоху
            epoch_col = fieldnames[0] if fieldnames else None
        metric_cols = [c for c in fieldnames if c != epoch_col]

        for row in reader:
            try:
                ep = float(row[epoch_col]) if epoch_col else None
            except Exception:
                continue
            for m in metric_cols:
                try:
                    v = float(str(row[m]).replace(",", "."))
                except Exception:
                    continue
                series[m].append((ep, v))
    # отсортируем по эпохам
    for m in list(series.keys()):
        series[m] = sorted(series[m], key=lambda t: t[0])
    return series

def auc_trapz(xs, ys):
    """площадь по трапециям; xs возрастают"""
    if len(xs) < 2: return float("nan")
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        area += 0.5 * dx * (ys[i] + ys[i-1])
    return area

def compute_cka_stats(cka_series, threshold=0.95):
    """
    вход: dict(model -> list[(epoch, cka)])
    выход: dict(model -> {"stab_epoch": e*, "auc_1mCKA": AUC})
    """
    stats = {}
    for model, pairs in cka_series.items():
        if not pairs:
            stats[model] = {"stab_epoch": None, "auc_1mCKA": None}
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        # стабилизация: минимальная эпоха с CKA>=threshold
        stab = None
        for e, v in pairs:
            if v >= threshold:
                stab = e; break
        # AUC по (1-CKA) — меньше = быстрее стабилизация
        ys_1m = [max(0.0, 1.0 - y) for y in ys]
        auc = auc_trapz(xs, ys_1m)
        stats[model] = {"stab_epoch": stab, "auc_1mCKA": auc}
    return stats

def html_escape(s):
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# -------- сборка отчёта --------
def build_markdown(datasets, outdir_root, md_path, html_rel_links=True):
    """
    Собирает report.md с галереями (ссылки на картинки) и ссылками на Sankey.
    """
    lines = []
    lines.append(f"# Отчёт по онлайн-мониторингу (RNN/BiLSTM/Transformer)\n")
    lines.append("")
    lines.append(f"_Автогенерация из `{__file__}`. Датасеты: {', '.join(datasets.keys())}_\n")

    # сводка по CKA
    lines.append("\n## Сводная таблица CKA (эпоха стабилизации, AUC(1-CKA))\n")
    lines.append("| Датасет | Модель | Эпоха стабилизации (CKA≥0.95) | AUC(1-CKA) |")
    lines.append("|---|---|---:|---:|")

    cka_stats_per_ds = {}

    for ds_name, outdir in datasets.items():
        outdir = str(outdir)
        # найдём CSV для CKA
        cka_csv = find_one([os.path.join(outdir, patt) for patt in CKA_CSV_CANDIDATES])
        series = parse_cka_csv(cka_csv) if cka_csv else {}
        stats = compute_cka_stats(series, threshold=CKA_STAB_THRESHOLD)
        cka_stats_per_ds[ds_name] = stats or {}

        # если нет метрик — вставим пустые строки
        if not stats:
            lines.append(f"| {ds_name} | — | — | — |")
        else:
            for model, st in stats.items():
                se = "—" if (st.get("stab_epoch") is None) else f"{st['stab_epoch']:.0f}"
                auc = st.get("auc_1mCKA")
                aucs = "—" if (auc is None or math.isnan(auc)) else f"{auc:.4g}"
                lines.append(f"| {ds_name} | {model} | {se} | {aucs} |")

    # секции по датасетам
    for ds_name, outdir in datasets.items():
        outdir = str(outdir)
        lines.append(f"\n## {ds_name}\n")
        sankey_html = os.path.join(outdir, "symbolic_bridge.html")
        if os.path.exists(sankey_html):
            rel = safe_rel(sankey_html, os.path.dirname(md_path)) if html_rel_links else sankey_html
            lines.append(f"[Интерактивная Sankey–диаграмма]({rel})\n")

        # мини-галерея: 2 строки по 3 картинки (что найдём)
        pics = find_all(outdir, GALLERY_PATTERNS)
        if not pics:
            lines.append("_Картинки не найдены._\n")
            continue

        # выводим по 3 в ряд
        row = []
        for i, p in enumerate(pics, 1):
            rel = safe_rel(p, os.path.dirname(md_path)) if html_rel_links else p
            row.append(f"![]({rel})")
            if (i % 3 == 0) or (i == len(pics)):
                lines.append(" ".join(row))
                row = []

    md = "\n".join(lines)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    return md

def build_html(datasets, outdir_root, html_path):
    """
    Собирает report.html с base64-встроенными мини-галереями и ссылками на Sankey.
    """
    parts = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'>")
    parts.append("<title>Отчёт по онлайн-мониторингу</title>")
    parts.append("<style>body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;line-height:1.35;padding:24px;} h1,h2{margin-top:1.2em;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;} .card{border:1px solid #eee;border-radius:12px;padding:10px;} img{max-width:100%;height:auto;border-radius:6px;}</style>")
    parts.append("</head><body>")
    parts.append("<h1>Отчёт по онлайн-мониторингу (RNN / BiLSTM / Transformer)</h1>")
    parts.append(f"<p><em>Автогенерация из {html_escape(os.path.basename(__file__))}. Датасеты: {', '.join(datasets.keys())}</em></p>")

    # сводка CKA
    parts.append("<h2>Сводная таблица CKA</h2>")
    parts.append("<table cellspacing='0' cellpadding='6' border='0' style='border-collapse:collapse;border:1px solid #ddd'>")
    parts.append("<tr style='background:#fafafa'><th>Датасет</th><th>Модель</th><th>Эпоха стабилизации (CKA≥0.95)</th><th>AUC(1-CKA)</th></tr>")
    for ds_name, outdir in datasets.items():
        outdir = str(outdir)
        cka_csv = find_one([os.path.join(outdir, patt) for patt in CKA_CSV_CANDIDATES])
        series = parse_cka_csv(cka_csv) if cka_csv else {}
        stats = compute_cka_stats(series, threshold=CKA_STAB_THRESHOLD)
        if not stats:
            parts.append(f"<tr><td>{html_escape(ds_name)}</td><td>—</td><td>—</td><td>—</td></tr>")
        else:
            for model, st in stats.items():
                se = "—" if (st.get('stab_epoch') is None) else f"{st['stab_epoch']:.0f}"
                auc = st.get('auc_1mCKA')
                aucs = "—" if (auc is None or math.isnan(auc)) else f"{auc:.4g}"
                parts.append(f"<tr><td>{html_escape(ds_name)}</td><td>{html_escape(model)}</td><td style='text-align:right'>{se}</td><td style='text-align:right'>{aucs}</td></tr>")
    parts.append("</table>")

    # секции по датасетам
    for ds_name, outdir in datasets.items():
        outdir = str(outdir)
        parts.append(f"<h2>{html_escape(ds_name)}</h2>")
        sankey_html = os.path.join(outdir, "symbolic_bridge.html")
        if os.path.exists(sankey_html):
            rel = os.path.relpath(sankey_html, os.path.dirname(html_path))
            parts.append(f"<p><a href='{rel}'>Интерактивная Sankey–диаграмма</a></p>")

        pics = find_all(outdir, GALLERY_PATTERNS)
        if not pics:
            parts.append("<p><em>Картинки не найдены.</em></p>")
        else:
            parts.append("<div class='grid'>")
            for p in pics:
                try:
                    b64 = img_to_base64(p)
                    parts.append(f"<div class='card'><img src='data:image/png;base64,{b64}' alt='{html_escape(os.path.basename(p))}'><div style='font-size:12px;color:#666;margin-top:6px'>{html_escape(os.path.basename(p))}</div></div>")
                except Exception as e:
                    rel = os.path.relpath(p, os.path.dirname(html_path))
                    parts.append(f"<div class='card'><a href='{rel}'>{html_escape(os.path.basename(p))}</a></div>")
            parts.append("</div>")

    parts.append("</body></html>")
    html = "\n".join(parts)
    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results", help="корневая папка с результатами")
    ap.add_argument("--report_dir",   default="results/_report", help="куда положить итоговые отчёты")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    help="какие датасеты включать (по ключам словаря DATASETS)")
    args = ap.parse_args()

    # пост-обработка словаря DATASETS по args.datasets
    selected = OrderedDict()
    for k in args.datasets:
        if k not in DATASETS:
            raise SystemExit(f"Неизвестный датасет '{k}'. Доступны: {', '.join(DATASETS.keys())}")
        selected[k] = DATASETS[k]

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    md_path   = str(report_dir / "report.md")
    html_path = str(report_dir / "report.html")

    md = build_markdown(selected, args.results_root, md_path, html_rel_links=True)
    html = build_html(selected, args.results_root, html_path)
    print(f"[ok] Markdown: {md_path}")
    print(f"[ok] HTML:     {html_path}")

if __name__ == "__main__":
    main()
