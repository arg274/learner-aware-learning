from __future__ import annotations

"""Column detection and text grouping helpers."""

from typing import Any, Dict, List

from .types import Block
from .utils import _avg_char_width, _should_insert_space, _passes_conf, _norm_bullet


def _x_center(b: Block) -> float:
    """Return horizontal center of a block bbox."""
    return 0.5 * (b.bbox[0] + b.bbox[2])


def _kmeans_1d(xs, k=2, iters=20):
    if len(xs) < k:
        return [0] * len(xs), [sum(xs) / max(1, len(xs))]
    c = [min(xs), max(xs)] if k == 2 else [sum(xs) / len(xs)]
    for _ in range(iters):
        a = [0 if abs(x - c[0]) <= abs(x - c[-1]) else 1 for x in xs] if k == 2 else [0] * len(xs)
        new_c = []
        for ki in range(k):
            pts = [x for x, ai in zip(xs, a) if ai == ki]
            new_c.append(sum(pts) / len(pts) if pts else c[ki])
        if max(abs(n - o) for n, o in zip(new_c, c)) < 1e-6:
            break
        c = new_c
    return a, c


def _detect_columns(tokens: List[Block], page_w: int, min_gutter_ratio=0.12, min_share=0.18):
    """Estimate up to two column bounds from token centers."""
    if len(tokens) < 8:
        return [(0.0, float(page_w))], [0] * len(tokens)
    xs = [_x_center(t) for t in tokens]
    assign, cents = _kmeans_1d(xs, k=2)
    order = sorted(range(2), key=lambda i: cents[i])
    remap = {order[0]: 0, order[1]: 1}
    assign = [remap[a] for a in assign]
    cents = [cents[i] for i in order]
    gap = abs(cents[1] - cents[0])
    left_share = assign.count(0) / len(assign)
    right_share = assign.count(1) / len(assign)
    if gap < min_gutter_ratio * page_w or min(left_share, right_share) < min_share:
        return [(0.0, float(page_w))], [0] * len(tokens)
    left_xs = [t.bbox[0] for t, a in zip(tokens, assign) if a == 0] + [t.bbox[2] for t, a in zip(tokens, assign) if a == 0]
    right_xs = [t.bbox[0] for t, a in zip(tokens, assign) if a == 1] + [t.bbox[2] for t, a in zip(tokens, assign) if a == 1]
    col0 = (min(left_xs), max(left_xs))
    col1 = (min(right_xs), max(right_xs))
    return [col0, col1], assign


def merge_blocks_into_lines_column_aware(
    blocks: List[Block], page_w: int, page_h: int, min_conf: float = 0.7, y_merge_ratio: float = 0.02, max_columns: int = 2
) -> List[Dict[str, any]]:
    """Group tokens into line dicts, inserting spaces when needed."""
    toks: List[Block] = []
    for b in blocks:
        if b.source.startswith("ocr_"):
            if _passes_conf(b, min_conf):
                toks.append(b)
        elif b.source == "pdf_span":
            toks.append(b)
    if not toks:
        return []
    if max_columns >= 2:
        cols, assign = _detect_columns(toks, page_w)
    else:
        cols, assign = [(0.0, float(page_w))], [0] * len(toks)

    def yc(b: Block):
        return 0.5 * (b.bbox[1] + b.bbox[3])

    merged: List[Dict[str, any]] = []
    for cid in range(len(cols)):
        col_toks = [t for t, a in zip(toks, assign) if a == cid]
        if not col_toks:
            continue
        col_toks.sort(key=lambda b: (yc(b), b.bbox[0]))
        y_thresh = max(6.0, y_merge_ratio * float(page_h))
        lines: List[List[Block]] = []
        cur = [col_toks[0]]
        cur_y = yc(col_toks[0])
        for t in col_toks[1:]:
            c = yc(t)
            if abs(c - cur_y) <= y_thresh:
                cur.append(t)
                cur_y = (cur_y * (len(cur) - 1) + c) / len(cur)
            else:
                lines.append(cur)
                cur = [t]
                cur_y = c
        if cur:
            lines.append(cur)

        lid_base = len(merged) * 1000
        for j, group in enumerate(lines):
            group.sort(key=lambda b: b.bbox[0])
            boxes = [b.bbox for b in group]
            avg_cw = _avg_char_width(boxes)
            parts: List[str] = []
            last_box = None
            last_text = None
            for b in group:
                t = (b.text or "").strip()
                if not t:
                    continue
                if _should_insert_space(last_text, t, last_box, b.bbox, avg_cw):
                    parts.append(" ")
                parts.append(t.replace("|", ""))
                last_box = b.bbox
                last_text = t
            text = _norm_bullet("".join(parts)).strip()
            if not text:
                continue
            x0 = min(b.bbox[0] for b in group)
            y0 = min(b.bbox[1] for b in group)
            x1 = max(b.bbox[2] for b in group)
            y1 = max(b.bbox[3] for b in group)
            merged.append({"id": f"L{lid_base + j:04d}", "bbox": (x0, y0, x1, y1), "text": text, "col_id": cid})

    merged.sort(key=lambda L: (L.get("col_id", 0), 0.5 * (L["bbox"][1] + L["bbox"][3]), L["bbox"][0]))
    return merged


def group_lines_into_paragraphs(lines: List[Dict[str, Any]], indent_tol: float = 18.0, y_gap_tol: float = 22.0):
    """Join lines into paragraphs using indent/y-gap heuristics."""

    if not lines:
        return []
    out = []
    for cid in sorted(set(L.get("col_id", 0) for L in lines)):
        Ls = [L for L in lines if L.get("col_id", 0) == cid]
        Ls = sorted(Ls, key=lambda L: (0.5 * (L["bbox"][1] + L["bbox"][3]), L["bbox"][0]))
        paras: List[List[Dict[str, Any]]] = []
        cur = [Ls[0]]
        def left(b):
            return b["bbox"][0]
        for L in Ls[1:]:
            prev = cur[-1]
            same_left = abs(left(L) - left(prev)) <= indent_tol or L["text"].startswith("•")
            y_gap = L["bbox"][1] - prev["bbox"][3]
            if same_left and y_gap <= y_gap_tol:
                cur.append(L)
            else:
                paras.append(cur)
                cur = [L]
        if cur:
            paras.append(cur)
        for i, p in enumerate(paras):
            text = "\n".join(L["text"] for L in p)
            x0 = min(L["bbox"][0] for L in p)
            y0 = min(L["bbox"][1] for L in p)
            x1 = max(L["bbox"][2] for L in p)
            y1 = max(L["bbox"][3] for L in p)
            out.append({"id": f"P{cid}_{i:04d}", "bbox": (x0, y0, x1, y1), "text": text, "col_id": cid})
    out.sort(key=lambda P: (P.get("col_id", 0), P["bbox"][1], P["bbox"][0]))
    return out
