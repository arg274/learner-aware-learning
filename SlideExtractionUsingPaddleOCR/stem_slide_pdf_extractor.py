#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from slide_extractor import (
    extract,
    write_blocks_ndjson,
    write_markdown,
    write_monolith_json,
    write_pages_json,
)


def run_single(pdf_path: str, out_dir: str, args: argparse.Namespace):
    minimal_effective = (not getattr(args, "full_json", False)) and (
        getattr(args, "minimal", False) or getattr(args, "text_only", False)
    )

    doc = extract(
        pdf_path=pdf_path,
        ocr_policy=args.ocr_policy,
        force_ocr=args.force_ocr,
        min_vec_chars=args.min_vec_chars,
        min_vec_lines=args.min_vec_lines,
        min_vec_area_ratio=args.min_vec_area_ratio,
        min_image_area_ratio=args.min_image_area_ratio,
        paddle_device=args.paddle_device,
        ocr_zoom=args.ocr_zoom,
        ocr_max_side=args.ocr_max_side,
        math_ocr=args.math_ocr,
        math_threshold=args.math_threshold,
        math_min_area_ratio=args.math_min_area_ratio,
        math_crop_pad=args.math_crop_pad,
        min_conf=args.min_conf,
        y_merge_ratio=args.y_merge_ratio,
        max_columns=args.max_columns,
        minimal=minimal_effective,
        debug=args.debug,
    )

    os.makedirs(out_dir, exist_ok=True)
    write_monolith_json(doc, out_dir, minimal=minimal_effective)

    if not minimal_effective:
        if not args.no_pages:
            write_pages_json(doc, out_dir)
        if args.blocks_ndjson:
            write_blocks_ndjson(doc, os.path.join(out_dir, "blocks.ndjson"))
        if args.markdown:
            write_markdown(doc, out_dir)

    print(f"Source: {doc.get('source')} | Engine: {doc.get('engine')} | Pages: {doc.get('num_pages')}")
    print(f"Wrote output to: {os.path.abspath(out_dir)}")


def run_batch(input_dir: str, output_root: str, args: argparse.Namespace):
    os.makedirs(output_root, exist_ok=True)
    pdfs = [os.path.join(input_dir, n) for n in os.listdir(input_dir) if n.lower().endswith(".pdf")]
    pdfs.sort()
    if not pdfs:
        print(f"[batch] No PDFs found under: {input_dir}")
        return
    print(f"[batch] Found {len(pdfs)} PDFs under: {input_dir}")
    for pdf_path in pdfs:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        out_dir = os.path.join(output_root, stem)
        print(f"[batch] Processing: {pdf_path} -> {out_dir}")
        run_single(pdf_path, out_dir, args)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="STEM slide PDF extractor (PaddleOCR-only, math OCR optional)")

    ap.add_argument("--minimal", action="store_true", help="Emit slide summaries instead of full JSON.")
    ap.add_argument("--text-only", action="store_true", help="Alias of --minimal: emit only slide summaries.")
    ap.add_argument("--full-json", action="store_true", help="Keep full JSON (overrides --minimal/--text-only).")
    ap.set_defaults(text_only=True)

    ap.add_argument("pdf", nargs="?", help="Path to a single PDF")
    ap.add_argument("--out", help="Output directory for a single run (default: ./outputs/<pdf_stem>)")
    ap.add_argument("--batch", action="store_true", help="Process all PDFs under --input-dir")
    ap.add_argument("--input-dir", default="/data", help="Batch input directory")
    ap.add_argument("--output-root", default="/outputs", help="Batch output root directory")

    ap.add_argument("--ocr-policy", choices=["hybrid", "auto", "always", "never"], default="hybrid")
    ap.add_argument("--force-ocr", action="store_true", help="Force OCR alongside vector spans")
    ap.add_argument("--min-vec-chars", type=int, default=20)
    ap.add_argument("--min-vec-lines", type=int, default=3)
    ap.add_argument("--min-vec-area-ratio", type=float, default=0.02)
    ap.add_argument("--min-image-area-ratio", type=float, default=0.30)
    ap.add_argument("--paddle-device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--ocr-zoom", type=float, default=2.8)
    ap.add_argument("--ocr-max-side", type=int, default=3800)

    ap.add_argument("--math-ocr", choices=["auto", "always", "none"], default="auto")
    ap.add_argument("--math-threshold", type=float, default=0.35)
    ap.add_argument("--math-min-area-ratio", type=float, default=0.003)
    ap.add_argument("--math-crop-pad", type=int, default=6)

    ap.add_argument("--min-conf", type=float, default=0.7)
    ap.add_argument("--y-merge-ratio", type=float, default=0.02)
    ap.add_argument("--max-columns", type=int, choices=[1, 2], default=2)

    ap.add_argument("--no-pages", action="store_true", help="Skip per-page JSON output")
    ap.add_argument("--blocks-ndjson", action="store_true", help="Write blocks.ndjson (full JSON only)")
    ap.add_argument("--markdown", action="store_true", help="Write per-page Markdown (full JSON only)")
    ap.add_argument("--debug", action="store_true")
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()
    if args.batch:
        run_batch(args.input_dir, args.output_root, args)
        return
    if not args.pdf:
        ap.error("Please provide a PDF path or use --batch.")
    pdf_path = args.pdf
    if not os.path.isfile(pdf_path):
        ap.error(f"PDF does not exist: {pdf_path}")
    out_dir = args.out or os.path.join("./outputs", os.path.splitext(os.path.basename(pdf_path))[0])
    run_single(pdf_path, out_dir, args)


if __name__ == "__main__":
    main()
