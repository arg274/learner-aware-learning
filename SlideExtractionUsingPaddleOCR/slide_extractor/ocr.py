"""OCR / rasterization helpers (Paddle + optional Pix2Tex)."""

from __future__ import annotations

import os
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from paddleocr import PaddleOCR

from .types import Block, BBox
from .utils import math_likeliness, _bbox_norm, _bbox_area


class PaddleBackend:
    """Version-safe PaddleOCR wrapper with simple device/lang defaults."""

    def __init__(self, device_pref: str = "auto"):
        import inspect

        sig = inspect.signature(PaddleOCR.__init__)

        def supports(name: str) -> bool:
            return name in sig.parameters

        def _has_cuda() -> bool:
            try:
                import paddle  # type: ignore

                fn = getattr(paddle, "is_compiled_with_cuda", None)
                return bool(fn() if callable(fn) else False)
            except Exception:
                return False

        if device_pref == "gpu":
            device = "gpu"
        elif device_pref == "cpu":
            device = "cpu"
        else:
            device = "gpu" if _has_cuda() else "cpu"

        kwargs = {}
        if supports("device"):
            kwargs["device"] = device
        if supports("use_textline_orientation"):
            kwargs["use_textline_orientation"] = True
        if supports("use_angle_cls"):
            kwargs["use_angle_cls"] = True
        if supports("use_gpu"):
            kwargs["use_gpu"] = (device == "gpu")
        if supports("lang"):
            kwargs["lang"] = "en"

        self.paddle = PaddleOCR(**kwargs)

    def ocr_image(self, img: Image.Image):
        """Run OCR and return boxes/text/confidences in image pixels."""

        arr = np.array(img)
        out = []
        if hasattr(self.paddle, "predict"):
            preds = self.paddle.predict(arr)
            for p in preds:
                polys = p.get("dt_polys") or []
                texts = p.get("rec_texts") or []
                scores = p.get("rec_scores") or []
                for poly, txt, sc in zip(polys, texts, scores):
                    if not txt:
                        continue
                    xs = [pt[0] for pt in poly]
                    ys = [pt[1] for pt in poly]
                    out.append({"text": txt, "bbox": (min(xs), min(ys), max(xs), max(ys)), "conf": float(sc)})
            return out
        res = self.paddle.ocr(arr[:, :, ::-1])
        for det in (res[0] if res else []):
            box, (text, conf) = det
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            out.append({"text": text, "bbox": (min(xs), min(ys), max(xs), max(ys)), "conf": float(conf)})
        return out


class MathOCR:
    """Thin Pix2Tex wrapper for math-only crops."""

    def __init__(self):
        try:
            from pix2tex.cli import LatexOCR  # type: ignore
        except Exception:
            from latex_ocr import LatexOCR  # type: ignore

        self.model = LatexOCR()

    def infer(self, pil_image: Image.Image) -> str:
        try:
            return (self.model(pil_image) or "").strip()
        except Exception:
            return ""


def rasterize_page(page: fitz.Page, zoom: float = 2.8, max_side: Optional[int] = None):
    """Rasterize page with optional max long-side clamp."""

    if max_side is not None and max_side > 0:
        w0 = float(page.rect.width) * zoom
        h0 = float(page.rect.height) * zoom
        long0 = max(w0, h0)
        if long0 > max_side:
            zoom *= max_side / long0
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)


def page_image_to_pil(pm: fitz.Pixmap):
    """Convert a PyMuPDF Pixmap into a PIL RGB image."""

    return Image.frombytes("RGB", (pm.width, pm.height), pm.samples)


def preprocess_for_handwriting(img: Image.Image) -> Image.Image:
    g = img.convert("L")
    g = ImageOps.autocontrast(g, cutoff=1)
    g = g.filter(ImageFilter.UnsharpMask(radius=1.2, percent=140, threshold=3))
    g = ImageEnhance.Contrast(g).enhance(1.08)
    g = ImageEnhance.Brightness(g).enhance(1.02)
    return Image.merge("RGB", (g, g, g))


def page_text_image_stats(page: fitz.Page):
    w, h = int(round(page.rect.width)), int(round(page.rect.height))
    raw = page.get_text("dict")
    vec_line_blocks, vec_span_tokens = [], []
    image_area = 0.0
    bid_line = 0
    bid_span = 0
    for block in raw.get("blocks", []):
        if block.get("type") == 0:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                parts = [s.get("text", "") for s in spans if (s.get("text", "") or "").strip()]
                if parts:
                    xs0 = [s["bbox"][0] for s in spans]
                    ys0 = [s["bbox"][1] for s in spans]
                    xs1 = [s["bbox"][2] for s in spans]
                    ys1 = [s["bbox"][3] for s in spans]
                    bboxL = (min(xs0), min(ys0), max(xs1), max(ys1))
                    textL = " ".join(p.strip() for p in parts if p.strip())
                    vec_line_blocks.append(Block(
                        id=f"v{page.number}_{bid_line}",
                        bbox=_bbox_norm(bboxL),
                        text=textL,
                        type="text",
                        source="pdf_text",
                        conf=None,
                        math_likeliness=math_likeliness(textL),
                    ))
                    bid_line += 1
                for s in spans:
                    t = (s.get("text", "") or "").strip()
                    if not t:
                        continue
                    b = s.get("bbox", [0, 0, 0, 0])
                    vec_span_tokens.append(Block(
                        id=f"vs{page.number}_{bid_span}",
                        bbox=_bbox_norm(b),
                        text=t,
                        type="text",
                        source="pdf_span",
                        conf=None,
                        math_likeliness=math_likeliness(t),
                    ))
                    bid_span += 1
        elif block.get("type") == 1:
            bbox = block.get("bbox", [0, 0, 0, 0])
            image_area += _bbox_area(_bbox_norm(bbox))
    page_area = float(w * h) if w and h else 1.0
    image_area_ratio = min(1.0, image_area / page_area)
    return vec_line_blocks, w, h, image_area_ratio, vec_span_tokens


def ocr_page_to_pdf_space(
    page: fitz.Page,
    backend: PaddleBackend,
    zoom: float = 2.8,
    max_side: Optional[int] = None,
    use_handwriting_preproc: bool = False,
) -> List[Block]:
    pm = rasterize_page(page, zoom=zoom, max_side=max_side)
    img = page_image_to_pil(pm)
    if use_handwriting_preproc:
        img = preprocess_for_handwriting(img)

    pdf_w = float(page.rect.width)
    pdf_h = float(page.rect.height)
    sx = pdf_w / float(pm.width)
    sy = pdf_h / float(pm.height)

    results = backend.ocr_image(img)
    blocks: List[Block] = []
    for i, r in enumerate(results):
        t = (r.get("text", "") or "").strip()
        if not t:
            continue
        x0, y0, x1, y1 = r.get("bbox", (0, 0, 0, 0))
        bbox = (float(x0) * sx, float(y0) * sy, float(x1) * sx, float(y1) * sy)
        blocks.append(Block(
            id=f"o{page.number}_{i}",
            bbox=bbox,
            text=t,
            type="text",
            source="ocr_paddle",
            conf=float(r.get("conf")) if r.get("conf") is not None else None,
            math_likeliness=math_likeliness(t),
        ))
    return blocks
