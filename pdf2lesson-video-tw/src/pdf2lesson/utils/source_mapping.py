"""Helpers for tracking which PDF pages contributed to a scene."""
from __future__ import annotations

from typing import Iterable


def format_source_ref(pdf_name: str, pages: Iterable[int]) -> str:
    ps = sorted(set(int(p) for p in pages))
    if not ps:
        return f"{pdf_name}"
    runs: list[str] = []
    start = prev = ps[0]
    for p in ps[1:]:
        if p == prev + 1:
            prev = p
            continue
        runs.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = p
    runs.append(f"{start}" if start == prev else f"{start}-{prev}")
    return f"{pdf_name} p.{','.join(runs)}"


def pages_for_chunk(start: int, end: int) -> list[int]:
    return list(range(int(start), int(end) + 1))
