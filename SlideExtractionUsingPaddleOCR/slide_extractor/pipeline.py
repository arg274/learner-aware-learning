"""Extraction pipeline orchestrating vector text, OCR, math OCR, and layout."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import fitz

from .layout import merge_blocks_into_lines_column_aware, group_lines_into_paragraphs
from .ocr import PaddleBackend, MathOCR, page_text_image_stats, ocr_page_to_pdf_space
from .types import Block, PageResult
from .utils import (
    dedupe_blocks_prefer_vector,
    _strip_repeaters,
    _build_slide_summary,
    math_likeliness,
    _union_bbox,
    _bbox_area,
    _is_low_contrast,
)


def extract(
    pdf_path: str,
    ocr_policy: str = "hybrid",
    force_ocr: bool = False,
    min_vec_chars: int = 20,
    min_vec_lines: int = 3,
    min_vec_area_ratio: float = 0.02,
    min_image_area_ratio: float = 0.30,
    paddle_device: str = "auto",
    ocr_zoom: float = 2.8,
    ocr_max_side: Optional[int] = None,
    math_ocr: str = "auto",
    math_threshold: float = 0.35,
    math_min_area_ratio: float = 0.003,
    math_crop_pad: int = 6,
    min_conf: float = 0.7,
    y_merge_ratio: float = 0.02,
    max_columns: int = 2,
    minimal: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run the full extraction pipeline for a single PDF."""
    paddle = PaddleBackend(device_pref=paddle_device)
    math_backend = None
    if math_ocr in ("auto", "always"):
        try:
            math_backend = MathOCR()
        except Exception:
            math_backend = None

    doc = fitz.open(pdf_path)
    pages: List[PageResult] = []

    for pno in range(len(doc)):
        page = doc.load_page(pno)
        vec_lines, W, H, img_ratio, vec_span_tokens = page_text_image_stats(page)
        vec_char_count = sum(len(b.text) for b in vec_lines)
        vec_lines_n = len(vec_lines)
        vec_area = sum(_bbox_area(b.bbox) for b in vec_lines)
        page_area = float(W * H) if W and H else 1.0
        vec_area_ratio = vec_area / page_area

        if ocr_policy == "always":
            use_ocr = True
        elif ocr_policy == "never":
            use_ocr = False
        elif ocr_policy == "auto":
            use_ocr = force_ocr or (vec_char_count < min_vec_chars)
        else:
            use_ocr = (
                force_ocr
                or vec_char_count < min_vec_chars
                or vec_lines_n <= min_vec_lines
                or vec_area_ratio < min_vec_area_ratio
                or img_ratio >= min_image_area_ratio
            )
        if debug:
            print(
                f"[p{pno+1}] vec_chars={vec_char_count} lines={vec_lines_n} vec_area={vec_area_ratio:.3f}"
                f" img_area={img_ratio:.3f} -> use_ocr={use_ocr}"
            )

        all_blocks: List[Block] = []
        all_blocks.extend(vec_span_tokens)
        use_handwriting_preproc = img_ratio >= 0.30

        if use_ocr:
            ocr_blocks = ocr_page_to_pdf_space(
                page, paddle, zoom=ocr_zoom, max_side=ocr_max_side, use_handwriting_preproc=use_handwriting_preproc
            )
            all_blocks.extend(ocr_blocks)

        if (math_ocr != "none") and (math_backend is not None) and (use_ocr or math_ocr == "always"):
            def y_center(b: Block) -> float:
                return 0.5 * (b.bbox[1] + b.bbox[3])

            cand_blocks = [b for b in all_blocks if (b.text or "").strip()]
            cand_blocks.sort(key=lambda b: (y_center(b), b.bbox[0]))
            y_thresh = max(6.0, 0.02 * float(H))
            groups: List[List[Block]] = []
            cur: List[Block] = []
            last_y = None
            for b in cand_blocks:
                yc = y_center(b)
                if last_y is None or abs(yc - last_y) <= y_thresh:
                    cur.append(b)
                    last_y = yc if last_y is None else (last_y * 0.7 + yc * 0.3)
                else:
                    groups.append(cur)
                    cur = [b]
                    last_y = yc
            if cur:
                groups.append(cur)

            pm = None
            pil_page = None
            sx = sy = 1.0
            for gi, grp in enumerate(groups):
                text_join = " ".join((g.text or "").strip() for g in sorted(grp, key=lambda x: x.bbox[0]))
                ml = math_likeliness(text_join)
                if (math_ocr == "always") or (ml >= math_threshold):
                    box = _union_bbox([g.bbox for g in grp])
                    if (_bbox_area(box) / float(W * H)) < math_min_area_ratio:
                        continue
                    if pm is None:
                        from .ocr import rasterize_page, page_image_to_pil

                        pm = rasterize_page(page, zoom=ocr_zoom, max_side=ocr_max_side)
                        pil_page = page_image_to_pil(pm)
                        sx = pm.width / float(W)
                        sy = pm.height / float(H)
                    x0, y0, x1, y1 = map(int, box)
                    x0 = max(0, x0 - math_crop_pad)
                    y0 = max(0, y0 - math_crop_pad)
                    x1 = min(int(W), x1 + math_crop_pad)
                    y1 = min(int(H), y1 + math_crop_pad)
                    crop = pil_page.crop((int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)))
                    if _is_low_contrast(crop, std_thresh=5.0):
                        continue
                    latex = math_backend.infer(crop)
                    if latex:
                        all_blocks.append(
                            Block(
                                id=f"m{pno}_{gi}",
                                bbox=(float(x0), float(y0), float(x1), float(y1)),
                                text=latex,
                                type="math",
                                source="math_pix2tex",
                                conf=None,
                                math_likeliness=1.0,
                            )
                        )

        all_blocks = dedupe_blocks_prefer_vector(all_blocks, iou_thresh=0.6)

        merged_lines = merge_blocks_into_lines_column_aware(
            all_blocks, W, H, min_conf=min_conf, y_merge_ratio=y_merge_ratio, max_columns=max_columns
        )
        paragraphs = group_lines_into_paragraphs(merged_lines)

        pages.append(
            PageResult(
                page=pno + 1,
                width=W,
                height=H,
                blocks=all_blocks,
                merged_lines=merged_lines,
                paragraphs=paragraphs,
            )
        )

    _strip_repeaters(pages, pct_threshold=0.6, margin_ratio=0.14)

    out_pages = []
    for p in pages:
        raw_blocks = [asdict(b) for b in p.blocks if (b.text or '').strip()]
        if minimal:
            page_obj = _build_slide_summary(p)
        else:
            page_obj = {
                "page": p.page,
                "width": p.width,
                "height": p.height,
                "blocks": raw_blocks,
            }
            if p.merged_lines:
                page_obj["merged_lines"] = p.merged_lines
            if p.paragraphs:
                page_obj["paragraphs"] = p.paragraphs
        out_pages.append(page_obj)

    return {
        "source": os.path.basename(pdf_path),
        "engine": f"pymupdf+ocr:paddle" + ("+math" if math_backend else ""),
        "num_pages": len(pages),
        "pages": out_pages,
    }
