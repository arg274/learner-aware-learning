# STEM Slide PDF Extractor

## Quick Start
```bash
python stem_slide_pdf_extractor_refactored.py ./data/intro_to_fintech.pdf \
  --out ./outputs/intro_to_fintech
```
By default the extractor emits text-only slide summaries (`[{"title","text","slide_number"}, …]`).

## Installation
1. Create/activate a Python 3.9+ virtual environment.
2. Install dependencies (PyMuPDF, PaddleOCR, PIL, numpy, pix2tex optional).

## Common Flags
| Flag | Description |
|------|-------------|
| `--out DIR` | Output directory (default `./outputs/<pdf_stem>`). |
| `--ocr-policy {hybrid,auto,always,never}` | When to run OCR (default `hybrid`). |
| `--force-ocr` | Force OCR alongside vector spans. |
| `--paddle-device {auto,cpu,gpu}` | PaddleOCR device preference. |
| `--ocr-zoom FLOAT` | Raster zoom for OCR (default 2.8). |
| `--ocr-max-side INT` | Clamp long side of raster (default 3800 px). |
| `--math-ocr {auto,always,none}` | Enable Pix2Tex math recognition (default auto). |
| `--frac math-threshold FLOAT` | Likelihood threshold for math auto mode. |
| `--math-min-area-ratio FLOAT` | Skip tiny math crops. |
| `--minimal` / `--text-only` | Emit only slide summaries (`title`, `text[]`, `slide_number`). |
| `--full-json` | Emit full detail (blocks, merged_lines, paragraphs). Overrides minimal. |
| `--markdown` | Write per-page Markdown (only meaningful with full JSON). |
| `--blocks-ndjson` | Write flattened block stream for auditing (full JSON only). |
| `--batch` | Process all PDFs under `--input-dir` into `--output-root/<stem>`. |
| `--max-columns {1,2}` | Column detection limit per page (default 2). |
| `--min-conf` | Minimum OCR confidence (0–1 or 0–100). |
| `--debug` | Print heuristics per page. |

## Use-case Recipes
### 1. Text-heavy Slides (default)
```bash
python stem_slide_pdf_extractor_refactored.py ./data/deck.pdf --out ./outputs/deck
```
Creates `outputs/deck/extraction.json` as an array of:
```json
{
  "title": "Slide title",
  "text": ["bullet 1", "bullet 2"],
  "slide_number": 1
}
```

### 2. Graph/Diagram Heavy Slides (needs layout + NDJSON)
```bash
python stem_slide_pdf_extractor_refactored.py ./data/deck.pdf \
  --out ./outputs/deck_full \
  --full-json --markdown --blocks-ndjson
```
Outputs:
- `extraction.json` (full object)
- `pages/page_XXX.json`
- `blocks.ndjson` (inspect OCR vs vector on chart labels)
- `markdown/page_XXX.md` (preview multi-column pages)

### 3. Chart-heavy But Vector-based (skip OCR)
```bash
python stem_slide_pdf_extractor_refactored.py ./data/deck.pdf \
  --out ./outputs/deck_vector \
  --ocr-policy never --math-ocr none --text-only
```
Skips OCR and math OCR for well-formed vector PDFs.

### 4. Data Tables / Mixed Media (batch lots of decks)
```bash
python stem_slide_pdf_extractor_refactored.py --batch \
  --input-dir ./data --output-root ./outputs
```
Processes every PDF in `./data/` into `./outputs/<pdf_stem>/`.

## Minimal vs Full Outputs
- **Minimal / Text-only**: best for RAG pipelines, quick QA, meeting notes. `extraction.json` is a list of slide summaries.
- **Full JSON**: includes `blocks` (with `source` and `bbox`), column-aware `merged_lines`, and `paragraphs`. Required for layout overlays and NDJSON/Markdown export.

## Tips
- Raise `--min-conf` (e.g., 0.85) to filter noisy OCR.
- Disable math OCR (`--math-ocr none`) for non-math decks to speed up extraction.
- For math-heavy decks, set `--math-ocr always` and ensure Pix2Tex dependencies are installed.
- Use `--debug` during tuning to see when OCR triggers (vector thresholds, image coverage, etc.).
