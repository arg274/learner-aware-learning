"""Writers for the various extractor output formats."""

from __future__ import annotations

import json
import os
from typing import Any, Dict


def write_monolith_json(doc: Dict[str, Any], out_dir: str, minimal: bool = False):
    payload: Any = doc.get("pages", []) if minimal else doc
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "extraction.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_pages_json(doc: Dict[str, Any], out_dir: str):
    pages_dir = os.path.join(out_dir, "pages")
    os.makedirs(pages_dir, exist_ok=True)
    for p in doc.get("pages", []):
        pno = p.get("page")
        with open(os.path.join(pages_dir, f"page_{pno:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(p, f, ensure_ascii=False, indent=2)


def write_blocks_ndjson(doc: Dict[str, Any], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for p in doc.get("pages", []):
            pno = p.get("page")
            for b in p.get("blocks", []):
                row = {
                    "page": pno,
                    "id": b.get("id"),
                    "bbox": b.get("bbox"),
                    "type": b.get("type"),
                    "source": b.get("source"),
                    "conf": b.get("conf"),
                    "math_likeliness": b.get("math_likeliness"),
                    "text": b.get("text"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(doc: Dict[str, Any], out_dir: str, math_threshold: float = 0.35):
    os.makedirs(out_dir, exist_ok=True)
    md_dir = os.path.join(out_dir, "markdown")
    os.makedirs(md_dir, exist_ok=True)
    for p in doc.get("pages", []):
        pno = p.get("page")
        paras = p.get("paragraphs") or []
        cols = sorted(set([q.get("col_id", 0) for q in paras]))
        if len(cols) == 2 and paras:
            left = [q["text"] for q in paras if q.get("col_id", 0) == 0]
            right = [q["text"] for q in paras if q.get("col_id", 0) == 1]
            sep = "\n\n"
            html = (
                f"# Page {pno}\n\n"
                "<table><tr><td style=\"vertical-align:top; padding-right:24px;\">\n"
                f"{sep.join(left)}\n"
                "</td><td style=\"vertical-align:top;\">\n"
                f"{sep.join(right)}\n"
                "</td></tr></table>\n"
            )
            with open(os.path.join(md_dir, f"page_{pno:03d}.md"), "w", encoding="utf-8") as f:
                f.write(html)
            continue
        body = []
        if paras:
            body = [q["text"] for q in paras]
        elif p.get("merged_lines"):
            body = [L["text"] for L in p["merged_lines"]]
        else:
            for b in p.get("blocks", []):
                t = (b.get("text") or "").strip()
                if not t:
                    continue
                if b.get("type") == "math":
                    body.append(f"$$\n{t}\n$$")
                else:
                    ml = float(b.get("math_likeliness") or 0.0)
                    body.append(f"$$\n{t}\n$$" if ml >= math_threshold else t)
        md = f"# Page {pno}\n\n" + ("\n\n".join(body) if body else "_(no text)_")
        with open(os.path.join(md_dir, f"page_{pno:03d}.md"), "w", encoding="utf-8") as f:
            f.write(md)
