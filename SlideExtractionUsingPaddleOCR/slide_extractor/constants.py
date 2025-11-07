from __future__ import annotations

import re

MATH_CHARS = set("=±×÷∓≈≠≤≥∑∏∫∂∇√→←↔↦⇒⇔⊂⊆⊇⊃∈∉∪∩⊕⊗∧∨¬∀∃∴∵∅≅≡⊥∥∠°′″≔≙ˆˇ˜·•∘⁄")
MATH_TOKENS = [
    "\\frac", "\\sum", "\\int", "\\lim", "\\sqrt",
    "\\alpha", "\\beta", "\\gamma", "\\theta", "\\lambda", "\\pi",
    "\\cdot", "\\times"
]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
BULLETS = {"•", "‣", "◦", "∙", "·", "●", "○", "▪", "▫", "–", "—", "-", "*"}
