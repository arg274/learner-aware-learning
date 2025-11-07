from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .constants import MATH_CHARS, MATH_TOKENS, EMAIL_RE, URL_RE, BULLETS
from .types import Block, PageResult, BBox


def math_likeliness(s: str) -> float:
    """Heuristic mathiness score used for optional math OCR."""

    s_norm = (s or "").strip()
    if not s_norm:
        return 0.0
    mc = sum(1 for ch in s_norm if ch in MATH_CHARS)
    ascii_mc = sum(1 for ch in s_norm if ch in "^_{}[]()=+-*/<>|")
    token_mc = sum(1 for t in MATH_TOKENS if t in s_norm)
    score = (mc * 1.0 + ascii_mc * 0.3 + token_mc * 2.0) / max(len(s_norm), 1)
    return float(min(score, 1.0))


def _iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    aarea = max(0.0, (ax1 - ax0)) * max(0.0, (ay1 - ay0))
    barea = max(0.0, (bx1 - bx0)) * max(0.0, (by1 - by0))
    union = aarea + barea - inter
    return inter / max(union, 1e-9)


def _norm_text_for_dupe(s: str) -> str:
    """Normalize text for matching OCR spans to vector spans."""

    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.lstrip("".join(BULLETS) + " ")


def _inter_area(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return iw * ih


def dedupe_blocks_prefer_vector(
    blocks: List[Block], iou_thresh: float = 0.6, cover_thresh: float = 0.65
) -> List[Block]:
    kept: List[Block] = []
    vec_by_text: Dict[str, List[Block]] = {}
    vec_spans: List[Block] = []
    for b in blocks:
        if b.source == "pdf_span" and (b.text or "").strip():
            key = _norm_text_for_dupe(b.text)
            vec_by_text.setdefault(key, []).append(b)
            vec_spans.append(b)

    def _coverage_with_vec(ocr_bbox: BBox) -> float:
        o_area = _bbox_area(ocr_bbox)
        if o_area <= 0:
            return 0.0
        inter_sum = 0.0
        ox0, oy0, ox1, oy1 = ocr_bbox
        for v in vec_spans:
            vx0, vy0, vx1, vy1 = v.bbox
            if vx1 <= ox0 or vx0 >= ox1 or vy1 <= oy0 or vy0 >= oy1:
                continue
            inter_sum += _inter_area(ocr_bbox, v.bbox)
        return float(min(inter_sum, o_area) / o_area)

    for b in blocks:
        if (b.text or "").strip() == "":
            continue
        if b.source.startswith("ocr_"):
            key = _norm_text_for_dupe(b.text)
            vec_candidates = vec_by_text.get(key, [])
            if any(_iou(b.bbox, v.bbox) >= iou_thresh for v in vec_candidates):
                continue
            if _coverage_with_vec(b.bbox) >= cover_thresh:
                continue
            kept.append(b)
        else:
            kept.append(b)

    seen_exact = set()
    deduped: List[Block] = []
    for b in kept:
        sig = (b.source, _norm_text_for_dupe(b.text), tuple(round(x, 2) for x in b.bbox))
        if sig in seen_exact:
            continue
        seen_exact.add(sig)
        deduped.append(b)

    final: List[Block] = []
    by_text: Dict[str, List[Block]] = {}
    for b in deduped:
        tnorm = _norm_text_for_dupe(b.text)
        if any(_iou(b.bbox, prev.bbox) >= 0.85 for prev in by_text.get(tnorm, [])):
            continue
        final.append(b)
        by_text.setdefault(tnorm, []).append(b)
    return final


def _norm_bullet(t: str) -> str:
    """Normalize bullet characters to a consistent leading glyph."""

    t = t.strip()
    if not t:
        return t
    if t and t[0] in BULLETS:
        return "• " + t.lstrip("".join(BULLETS)).strip()
    return t


def _bbox_norm(b) -> BBox:
    x0, y0, x1, y1 = b
    return float(x0), float(y0), float(x1), float(y1)


def _bbox_area(b: BBox) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))


def _union_bbox(boxes: List[BBox]) -> BBox:
    xs0 = [b[0] for b in boxes]
    ys0 = [b[1] for b in boxes]
    xs1 = [b[2] for b in boxes]
    ys1 = [b[3] for b in boxes]
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _x_gap_space(prev_box: BBox, cur_box: BBox, avg_char_w: float, factor: float = 0.50) -> bool:
    px1 = prev_box[2]
    cx0 = cur_box[0]
    gap = max(0.0, cx0 - px1)
    return gap >= (avg_char_w * factor)


def _should_insert_space(prev_text: str | None, cur_text: str | None, prev_box: BBox | None, cur_box: BBox, avg_char_w: float) -> bool:
    if prev_box is not None and _x_gap_space(prev_box, cur_box, avg_char_w, factor=0.50):
        return True
    if not prev_text or not cur_text:
        return False
    if prev_text[-1] == '-':
        return False
    return prev_text[-1].isalnum() and cur_text[0].isalnum()


def _avg_char_width(line_boxes: List[BBox]) -> float:
    widths = [(b[2] - b[0]) for b in line_boxes if (b[2] - b[0]) > 0]
    if not widths:
        return 8.0
    approx = sorted(w / 6.0 for w in widths)
    mid = len(approx) // 2
    if len(approx) % 2:
        return float(approx[mid])
    return float(0.5 * (approx[mid - 1] + approx[mid]))


def _passes_conf(b: Block, min_conf: float) -> bool:
    if b.conf is None:
        return True
    c = float(b.conf)
    if c <= 1.0:
        c *= 100.0
    thresh = (min_conf * 100.0) if min_conf <= 1.0 else float(min_conf)
    return c >= thresh


def _is_low_contrast(pil_image: Image.Image, std_thresh: float = 5.0) -> bool:
    arr = np.asarray(pil_image.convert("L"), dtype=np.float32)
    if arr.size == 0:
        return True
    return float(arr.std()) < std_thresh


def _strip_repeaters(pages: List[PageResult], pct_threshold: float = 0.6, margin_ratio: float = 0.14) -> None:
    """Drop repeating headers/footers/meta content across pages."""

    if not pages:
        return
    avgH = sum(p.height for p in pages) / len(pages)
    top_limit = avgH * margin_ratio
    bot_limit = avgH * (1.0 - margin_ratio)
    counts: Dict[str, int] = {}
    for p in pages:
        for b in p.blocks:
            t = (b.text or "").strip()
            if not t:
                continue
            y_center = 0.5 * (b.bbox[1] + b.bbox[3])
            near_margin = (y_center <= top_limit) or (y_center >= bot_limit)
            looks_meta = bool(EMAIL_RE.search(t) or URL_RE.search(t))
            looks_page_no = bool(re.fullmatch(r"(page\s*)?\d+\s*(/\s*\d+)?", t, re.I))
            if near_margin or looks_meta or looks_page_no:
                key = t.lower()
                counts[key] = counts.get(key, 0) + 1
    n_pages = len(pages)
    repeaters = {t for t, c in counts.items() if c / n_pages >= pct_threshold}
    if not repeaters:
        return

    def should_drop(text: str, bbox: BBox) -> bool:
        s = (text or "").strip()
        if not s:
            return False
        if s.lower() in repeaters:
            return True
        y_center = 0.5 * (bbox[1] + bbox[3])
        near_margin = (y_center <= top_limit) or (y_center >= bot_limit)
        looks_meta = bool(EMAIL_RE.search(s) or URL_RE.search(s))
        looks_page_no = bool(re.fullmatch(r"(page\s*)?\d+\s*(/\s*\d+)?", s, re.I))
        return near_margin and (looks_meta or looks_page_no)

    for p in pages:
        p.blocks = [b for b in p.blocks if not should_drop(b.text, b.bbox)]
        if p.merged_lines:
            p.merged_lines = [L for L in p.merged_lines if not should_drop(L.get("text", ""), L.get("bbox", (0, 0, 0, 0)))]
        if p.paragraphs:
            p.paragraphs = [Q for Q in p.paragraphs if not should_drop(Q.get("text", ""), Q.get("bbox", (0, 0, 0, 0)))]


def _paragraph_texts(p: PageResult) -> List[str]:
    """Return the best available paragraph strings for a page."""

    if p.paragraphs:
        return [q.get("text", "") for q in p.paragraphs if (q.get("text") or "").strip()]
    if p.merged_lines:
        return [ln.get("text", "") for ln in p.merged_lines if (ln.get("text") or "").strip()]
    return [b.text for b in p.blocks if (b.text or "").strip()]


def _build_slide_summary(p: PageResult) -> Dict[str, Any]:
    """Construct the minimal {title,text[],slide_number} payload."""

    para_texts = _paragraph_texts(p)
    title = ""
    body: List[str] = []
    for para in para_texts:
        lines = [ln.rstrip() for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        if not title:
            title = lines[0]
            body.extend(lines[1:])
        else:
            body.extend(lines)
    if not title:
        if body:
            title = body.pop(0)
        else:
            title = f"Slide {p.page}"
    return {"title": title, "text": body, "slide_number": p.page}
