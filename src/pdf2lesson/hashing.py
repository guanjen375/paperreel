"""Stable hashing helpers — input_hash drives skip / resume decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    """Deterministic JSON dump used for hashing dict-like inputs."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hash_inputs(*parts: Any) -> str:
    """Hash a list of arbitrary JSON-serialisable parts into one digest."""
    return sha256_text("␟".join(stable_json(p) for p in parts))
