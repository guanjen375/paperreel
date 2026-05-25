"""Conservative Traditional Chinese narration normalization for TTS only."""
from __future__ import annotations

import re


_DIGITS = "零一二三四五六七八九"
_ACRONYMS = {"MSC", "FIT"}


def _strip_commas(raw: str) -> int:
    return int(raw.replace(",", ""))


def _number_zh(n: int) -> str:
    if n < 0:
        return "負" + _number_zh(abs(n))
    if n < 10:
        return _DIGITS[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        head = "" if tens == 1 else _DIGITS[tens]
        return head + "十" + ("" if ones == 0 else _DIGITS[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return _DIGITS[hundreds] + "百"
        sep = "零" if rest < 10 else ""
        return _DIGITS[hundreds] + "百" + sep + _number_zh(rest)
    if n < 10000:
        thousands, rest = divmod(n, 1000)
        if rest == 0:
            return _DIGITS[thousands] + "千"
        sep = "零" if rest < 100 else ""
        return _DIGITS[thousands] + "千" + sep + _number_zh(rest)
    if n < 100000000:
        high, rest = divmod(n, 10000)
        if rest == 0:
            return _number_zh(high) + "萬"
        sep = "零" if rest < 1000 else ""
        return _number_zh(high) + "萬" + sep + _number_zh(rest)
    return "".join(_DIGITS[int(ch)] for ch in str(n))


def _year_digits(raw: str) -> str:
    return "".join(_DIGITS[int(ch)] for ch in raw)


def normalize_zh_tw_for_tts(text: str) -> str:
    """Normalize common PDF narration tokens before TTS synthesis.

    This is deliberately narrow and should not be used for visual text,
    source quotes, or evidence matching.
    """
    out = text

    out = re.sub(
        r"(?:NT\$|NTD\s*)\s*([0-9][0-9,]*)\s*(?:元)?",
        lambda m: f"新臺幣{_number_zh(_strip_commas(m.group(1)))}元",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"([0-9][0-9,]*)\s*元",
        lambda m: f"{_number_zh(_strip_commas(m.group(1)))}元",
        out,
    )
    out = re.sub(
        r"(\d{1,3})\s*[%％]",
        lambda m: f"百分之{_number_zh(int(m.group(1)))}",
        out,
    )
    out = re.sub(
        r"(\d{1,3})\s*[-~～－–—]\s*(\d{1,3})\s*天",
        lambda m: f"{_number_zh(int(m.group(1)))}到{_number_zh(int(m.group(2)))}天",
        out,
    )
    out = re.sub(
        r"(\d{1,3})\s*天",
        lambda m: f"{_number_zh(int(m.group(1)))}天",
        out,
    )
    out = re.sub(
        r"\b((?:19|20)\d{2})\b",
        lambda m: _year_digits(m.group(1)),
        out,
    )
    for acronym in _ACRONYMS:
        out = re.sub(
            rf"\b{acronym}\b",
            " ".join(acronym),
            out,
        )
    return out

