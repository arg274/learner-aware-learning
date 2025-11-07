from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

BBox = Tuple[float, float, float, float]


@dataclass
class Block:
    """Primitive text/maths unit with bbox and provenance."""

    id: str
    bbox: BBox
    text: str
    type: str            # "text" | "math"
    source: str          # "pdf_text" | "pdf_span" | "ocr_paddle" | "math_pix2tex"
    conf: Optional[float] = None
    math_likeliness: Optional[float] = None


@dataclass
class PageResult:
    """Holds raw/merged results for a single page."""

    page: int
    width: int
    height: int
    blocks: List[Block]
    merged_lines: Optional[List[Dict[str, Any]]] = None
    paragraphs: Optional[List[Dict[str, Any]]] = None
