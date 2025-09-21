# visit_viz.py
# -*- coding: utf-8 -*-
"""
VISIT-style поток вкладов через один TransformerEncoderLayer.
Поддерживается PyTorch nn.TransformerEncoderLayer (pre-norm).
Рисуем sankey: input token -> heads -> top keys -> top W_o neurons -> top FF neurons -> block output.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import plotly.graph_objects as go


# -------------------------
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# -------------------------
def _as_batch_first(x: torch.Tensor) -> torch.Tensor:
    """Приводим к [B, T, D]. nn.Transformer_old обычно [T, B, D]."""
    if x.dim() != 3:
        raise ValueError("Ожидаем 3D тензор [T,B,D] или [B,T,D]")
    if x.shape[0] < x.shape[1]:  # эвристика: скорее всего [T,B,D]
        return x.transpose(0, 1).contiguous()
    return x


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    x: [B, T, D] -> [B, H, T, d] где d = D//H
    """
    B, T, D = x.shape
    d = D // num_heads
    return x.view(B, T, num_heads, d).transpose(1, 2).contiguous()


def _combine_heads(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, H, T, d] -> [B, T, H*d]
    """
    B, H, T, d = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, H * d)


def compute_qkv_from_mha(mha: nn.MultiheadAttention, x_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Воссоздаём Q,K,V как внутри MultiheadAttention.
    x_norm: [B, T, D] после LayerNorm перед вниманием.
    Возвращает q, k, v формы [B, H, T, d].
    """
    assert mha.batch_first, "Для простоты включите batch_first=True в MHA (или адаптируйте приведение размерностей)."
    B, T, D = x_norm.shape
    H = mha.num_heads
    W = mha.in_proj_weight    # [3D, D]
    b = mha.in_proj_bias      # [3D]
    Wq, Wk, Wv = W[:D], W[D:2*D], W[2*D:]
    bq, bk, bv = b[:D], b[D:2*D], b[2*D:]

    q = F.linear(x_norm, Wq, bq)
    k = F.linear(x_norm, Wk, bk)
    v = F.linear(x_norm, Wv, bv)

    q = _split_heads(q, H)  # [B,H,T,d]
    k = _split_heads(k, H)
    v = _split_heads(v, H)
    return q, k, v


def scaled_dot_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q,k,v: [B,H,T,d]
    returns:
      context: [B,H,T,d], attn: [B,H,T,T] (веса softmax по последней T)
    """
    B, H, T, d = q.shape
    scale = 1.0 / (d ** 0.5)
    attn_logits = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,T,T]
    attn = torch.softmax(attn_logits, dim=-1)
    ctx = torch.matmul(attn, v)  # [B,H,T,d]
    return ctx, attn


def head_contributions(mha: nn.MultiheadAttention, ctx: torch.Tensor) -> torch.Tensor:
    """
    Оценка вклада голов после out_proj (W_o).
    ctx: [B,H,T,d] — контекст по головам для каждого токена.
    Возвращает contrib [B,T,H] = ||W_o_chunk * head_vec||_1 (сумма абсолютов по выходным фичам).
    """
    Wo: torch.Tensor = mha.out_proj.weight  # [D, D]
    B, H, T, d = ctx.shape
    D = H * d
    assert Wo.shape == (D, D), "Ожидаем квадратный out_proj для concat голов."

    ctx_cat = _combine_heads(ctx)  # [B,T,D]

    # Разрезаем W_o на блоки по головам по столбцам исходного concat
    contrib = []
    for h in range(H):
        sl = slice(h * d, (h + 1) * d)
        # вклад головы h — проекция только её части через соответствующие столбцы W_o
        part = torch.matmul(ctx_cat[:, :, sl], Wo[:, sl].transpose(0, 1))  # [B,T,D]
        contrib.append(part.abs().sum(dim=-1))  # [B,T]
    contrib = torch.stack(contrib, dim=-1)  # [B,T,H]
    return contrib


def ff_activations(layer: nn.TransformerEncoderLayer, x_ff_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Возвращает:
      a1 = linear1(x), g = act(a1), y = linear2(g)
    Все формы [B,T,D] (a1/g приведены обратно к D через псевдо-проекцию 0-паддингом),
    а для топ-нейронов linear1/linear2 отдельно вернём исходные значения.
    """
    lin1: nn.Linear = layer.linear1
    lin2: nn.Linear = layer.linear2
    act = layer.activation

    a1 = lin1(x_ff_in)           # [B,T,D_ff]
    g = act(a1)                  # [B,T,D_ff]
    y = lin2(g)                  # [B,T,D]

    return a1, g, y


def topk_idx(values: torch.Tensor, k: int) -> List[int]:
    v = values.detach().float().cpu().numpy().ravel()
    k = min(k, len(v))
    if k <= 0: return []
    idx = np.argpartition(-np.abs(v), kth=k-1)[:k]
    idx = idx[np.argsort(-np.abs(v[idx]))]
    return idx.tolist()


# -------------------------
# ОСНОВНАЯ ФУНКЦИЯ ВИЗУАЛИЗАЦИИ
# -------------------------
@torch.no_grad()
def visualize_block_flow(
    layer: nn.TransformerEncoderLayer,
    x_in: torch.Tensor,
    pos: int,
    token_texts: Optional[List[str]] = None,
    topk_heads: int = 6,
    topk_keys_per_head: int = 8,
    topk_Wo_neurons: int = 10,
    topk_FF_neurons: int = 10,
    title: str = "Block flow",
    save_html: Optional[str] = "visit_sankey.html",
    save_png: Optional[str] = None,
) -> go.Figure:
    """
    Рисует VISIT-подобную sankey для одного блока и одной позиции pos.
    Предполагается pre-norm слой (как в PyTorch по умолчанию).
    """
    layer.eval()
    assert isinstance(layer.self_attn, nn.MultiheadAttention), "Нужен MultiheadAttention внутри слоя"
    assert layer.self_attn.batch_first, "mha.batch_first=True обязательно"

    # --------- проход через блок вручную (без дропаута)
    x = x_in  # [B,T,D] ожидаем
    B, T, D = x.shape
    assert pos >= 0 and pos < T, "pos вне диапазона"

    # LN перед вниманием
    x1n = layer.norm1(x)

    # Q,K,V, внимание
    q, k, v = compute_qkv_from_mha(layer.self_attn, x1n)  # [B,H,T,d] each
    ctx, attn = scaled_dot_attn(q, k, v)                  # ctx [B,H,T,d], attn [B,H,T,T]

    # Выход внимания до residual
    ctx_cat = _combine_heads(ctx)                          # [B,T,D]
    y_attn = layer.self_attn.out_proj(ctx_cat)             # [B,T,D]

    # Residual + LN
    x2 = x + y_attn
    x2n = layer.norm2(x2)

    # FF
    a1, g, y_ff = ff_activations(layer, x2n)               # a1 [B,T,D_ff], g [B,T,D_ff], y_ff [B,T,D]
    out = x2 + y_ff                                        # выход блока

    # --------- метрики вкладов для выбранного токена pos
    # 1) головы: суммарный вклад через W_o
    contrib_heads = head_contributions(layer.self_attn, ctx)  # [B,T,H]
    heads_score = contrib_heads[0, pos]                        # [H]

    H = layer.self_attn.num_heads
    d = D // H

    # топ головы
    top_heads = topk_idx(heads_score, topk_heads)

    # 2) ключевые токены (по head) — берём веса внимания для [pos]
    # attn: [B,H,T,T]; query=pos → распределение по j
    attn_pos = attn[0, :, pos, :]  # [H, T]
    top_keys_by_head: Dict[int, List[int]] = {}
    for h in top_heads:
        w = attn_pos[h]  # [T]
        topj = topk_idx(w, topk_keys_per_head)
        top_keys_by_head[h] = topj

    # 3) топ-«нейроны» W_o (по столбцам concat голов до проекции)
    # Берём локальный вектор concat голов для pos:
    concat_vec = ctx_cat[0, pos]  # [D]
    Wo: torch.Tensor = layer.self_attn.out_proj.weight  # [D,D]
    # вклад столбцов Wo пропорционален |concat_vec| (амплитуда источника)
    top_Wo_cols = topk_idx(concat_vec, topk_Wo_neurons)

    # 4) топ-нейроны FF (по a1/g активностям)
    a1_pos = a1[0, pos]  # [D_ff]
    g_pos  = g[0, pos]   # [D_ff]
    top_ff_neurons = topk_idx(g_pos, topk_FF_neurons)

    # --------- Готовим узлы и ссылки Sankey
    def tok_label(j: int) -> str:
        if token_texts and 0 <= j < len(token_texts):
            return token_texts[j]
        return f"t{j}"

    nodes: List[str] = []
    links_src: List[int] = []
    links_tgt: List[int] = []
    links_val: List[float] = []
    links_color: List[str] = []

    # Узлы-столбцы
    # 0) input token
    idx_input = len(nodes); nodes.append(f"input: {tok_label(pos)}")

    # 1) heads
    head_idx_map = {}
    for h in top_heads:
        head_idx_map[h] = len(nodes)
        nodes.append(f"head {h}")

        # link: input -> head (value = вклад головы)
        links_src.append(idx_input)
        links_tgt.append(head_idx_map[h])
        links_val.append(float(abs(heads_score[h].item())))
        links_color.append("rgba(100,180,255,0.6)")

    # 2) keys per head
    key_node_map: Dict[Tuple[int,int], int] = {}
    for h in top_heads:
        keys = top_keys_by_head[h]
        for j in keys:
            node_id = len(nodes)
            key_node_map[(h, j)] = node_id
            nodes.append(f"key: {tok_label(j)}")

            links_src.append(head_idx_map[h])
            links_tgt.append(node_id)
            links_val.append(float(attn_pos[h, j].item()))
            links_color.append("rgba(160,160,160,0.5)")

    # 3) W_o columns (нейроны проекции внимания)
    wo_node_map = {}
    for c in top_Wo_cols:
        nid = len(nodes)
        wo_node_map[c] = nid
        h_id = c // d
        nodes.append(f"W_o col {c} (h{h_id})")

        # источник — соответствующая голова (flow от головы к W_o колонке),
        # вес — |concat_vec[c]|
        links_src.append(head_idx_map.get(h_id, idx_input))
        links_tgt.append(nid)
        links_val.append(float(abs(concat_vec[c].item())))
        links_color.append("rgba(255,120,120,0.5)")

    # 4) FF neurons (linear1 hidden)
    ff_node_map = {}
    for u in top_ff_neurons:
        nid = len(nodes)
        ff_node_map[u] = nid
        nodes.append(f"FF n{u}")

        # источник — «attention output» узел; для простоты соединим из всех W_o топ-колонок
        src_sum = sum(abs(concat_vec[c].item()) for c in top_Wo_cols) + 1e-8
        weight = float(abs(g_pos[u].item()))
        # распылим ссылку от каждого wo-узла, чтобы видно «переток»
        for c in top_Wo_cols:
            links_src.append(wo_node_map[c])
            links_tgt.append(nid)
            # пропорционально доле этой колонки
            links_val.append(weight * (abs(concat_vec[c].item()) / src_sum))
            links_color.append("rgba(120,200,120,0.5)")

    # 5) output token (после блока)
    idx_output = len(nodes); nodes.append(f"output: {tok_label(pos)}")

    # финальные ссылки: FF -> output
    ff_total = sum(abs(g_pos[u].item()) for u in top_ff_neurons) + 1e-8
    for u in top_ff_neurons:
        links_src.append(ff_node_map[u])
        links_tgt.append(idx_output)
        links_val.append(float(abs(g_pos[u].item()) / ff_total))
        links_color.append("rgba(90,160,90,0.6)")

    # Sankey
    fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        node=dict(
            pad=12, thickness=18,
            label=nodes,
            color=["#444"] * len(nodes)
        ),
        link=dict(
            source=links_src,
            target=links_tgt,
            value=links_val,
            color=links_color
        )
    )])
    fig.update_layout(title_text=title, font_size=11, width=1400, height=600)

    if save_html:
        fig.write_html(save_html, include_plotlyjs="cdn")
    if save_png:
        try:
            fig.write_image(save_png, scale=2)  # требует 'kaleido'
        except Exception as e:
            print(f"[warn] PNG не сохранён (нужен kaleido): {e}")

    return fig


# -------------------------
# ПРИМЕР ИСПОЛЬЗОВАНИЯ С nn.TransformerEncoder
# -------------------------
@torch.no_grad()
def demo_with_encoder_layer(layer: nn.TransformerEncoderLayer, x: torch.Tensor, pos: int, tokens: Optional[List[str]] = None):
    """
    layer: один блок encoder
    x: вход в блок, [B,T,D] (batch_first=True) или [T,B,D]
    pos: индекс токена (0..T-1)
    """
    if not getattr(layer.self_attn, "batch_first", False):
        # оборачиваем в batch_first без пересоздания слоя — прокинем вход как [B,T,D]
        x = _as_batch_first(x)
        layer.self_attn.batch_first = True
    else:
        x = _as_batch_first(x)

    return visualize_block_flow(
        layer=layer, x_in=x, pos=pos, token_texts=tokens,
        title=f"VISIT-style flow — layer (pos {pos})",
        save_html="visit_sankey.html", save_png=None
    )
