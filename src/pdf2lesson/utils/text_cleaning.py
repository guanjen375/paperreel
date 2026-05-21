"""Text helpers: CJK char count, normalisation, verbatim-overlap detection."""
from __future__ import annotations

import re
import unicodedata

_CJK_RANGES = (
    (0x3400, 0x4DBF),    # CJK Ext A
    (0x4E00, 0x9FFF),    # CJK Unified
    (0x20000, 0x2A6DF),  # CJK Ext B
    (0x2A700, 0x2EBEF),  # CJK Ext C/D/E/F
    (0x30000, 0x3134F),  # CJK Ext G
    (0xF900, 0xFAFF),    # CJK Compat
)

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINE_RE = re.compile(r"\n{3,}")


def is_cjk_char(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in _CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def cjk_char_count(text: str) -> int:
    return sum(1 for c in text if is_cjk_char(c))


def normalise_text(text: str) -> str:
    """NFC + collapse whitespace + clamp blank-line runs."""
    text = unicodedata.normalize("NFC", text)
    text = _WS_RE.sub(" ", text)
    text = _BLANK_LINE_RE.sub("\n\n", text)
    return text.strip()


def looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 60:
        return False
    # Common heading patterns: "第X章", "Chapter X", "1.", "1.2", or single short CJK title line.
    if re.match(r"^第[一二三四五六七八九十百千零〇\d]+[章節篇課]", s):
        return True
    if re.match(r"^(Chapter|Section|Part)\s+\d+", s, flags=re.IGNORECASE):
        return True
    if re.match(r"^\d+(\.\d+){0,3}\s+\S", s):
        return True
    return False


def extract_headings(text: str, *, max_lines: int = 200) -> list[str]:
    out: list[str] = []
    for line in text.splitlines()[:max_lines]:
        if looks_like_heading(line):
            out.append(line.strip())
    return out


def verbatim_overlap_ratio(
    candidate: str,
    source: str,
    *,
    window: int = 30,
) -> float:
    """Rough fraction of `candidate` covered by length-`window` runs that also
    appear verbatim in `source`. Used to flag講稿 over-copying PDF 原文."""
    if not candidate or len(candidate) < window:
        return 0.0
    src = source
    hit = 0
    i = 0
    n = len(candidate)
    while i <= n - window:
        if candidate[i:i + window] in src:
            hit += window
            i += window
        else:
            i += 1
    return min(1.0, hit / n)
