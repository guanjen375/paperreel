"""Atomic file writes + JSON helpers.

Every artefact in the pipeline lands on disk through `atomic_write_*` so a
crashed/cancelled run never leaves a half-written file that a resume would
mistake for valid output.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@contextmanager
def atomic_write(path: str | Path, mode: str = "wb") -> Iterator[Any]:
    """Yield a file handle that, on successful close, atomically replaces `path`.

    Writes to `path + ".tmp.<pid>"` in the same directory, fsyncs, then renames.
    Same-directory rename is atomic on POSIX and best-effort atomic on NTFS via
    os.replace().
    """
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    encoding = None
    binary = "b" in mode
    if not binary:
        encoding = "utf-8"
    try:
        with open(tmp, mode, encoding=encoding) as f:
            yield f
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # fsync may fail on some FS (e.g. tmpfs); the rename is still safer
                # than a partial write.
                pass
        os.replace(tmp, p)
    except BaseException:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    with atomic_write(path, "w") as f:
        f.write(text)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    with atomic_write(path, "wb") as f:
        f.write(data)


def atomic_write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=indent, sort_keys=False)
    atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_copy(src: str | Path, dst: str | Path) -> None:
    ensure_dir(Path(dst).parent)
    shutil.copy2(src, dst)
