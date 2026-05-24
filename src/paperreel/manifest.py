"""Per-artifact manifest sidecars.

Why this exists: the original resume logic was "file exists ⇒ skip".
That works for the happy path but quietly serves stale output when any
input that fed the artefact changes — a new TTS voice, an edited
narration, a different SDXL prompt, a swapped speaker_wav reference.
The sidecar manifest pins the exact inputs each artefact was built from,
so resume can decide rebuild-vs-skip on contents, not on filesystem
presence.

Layout: every artefact at ``foo/bar.wav`` has a sibling
``foo/bar.wav.manifest.json`` containing the stage name, scene_id,
input_hash, and the inputs dict that produced the hash (kept for human
inspection / debugging — only ``input_hash`` is load-bearing).

Cheap, atomic (uses :func:`atomic_write_json`), and survives across
processes. Stages that haven't been migrated to manifests keep working;
``manifest_matches`` just returns False if no manifest is present, and
the stage falls back to its existing behaviour.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .io_utils import atomic_write_json

MANIFEST_SUFFIX = ".manifest.json"


def manifest_path_for(artifact_path: str | Path) -> Path:
    """Return the sidecar manifest path for ``artifact_path``.

    We deliberately *append* ``.manifest.json`` rather than swap the
    extension, so e.g. ``foo.wav`` → ``foo.wav.manifest.json``. That
    way the manifest path stays adjacent to the artefact even when an
    artefact's extension changes (e.g. wav → m4a) without colliding
    with the artefact itself.
    """
    p = Path(artifact_path)
    return p.with_name(p.name + MANIFEST_SUFFIX)


def write_manifest(
    artifact_path: str | Path,
    *,
    stage: str,
    scene_id: str | None,
    input_hash: str,
    inputs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write the sidecar manifest atomically. Returns the manifest path."""
    path = manifest_path_for(artifact_path)
    payload: dict[str, Any] = {
        "stage": stage,
        "scene_id": scene_id,
        "artifact": Path(artifact_path).name,
        "input_hash": input_hash,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": inputs,
    }
    if extra:
        # Caller-supplied fields take precedence over the defaults above —
        # use that for stage-specific metadata.
        payload.update(extra)
    atomic_write_json(path, payload)
    return path


def read_manifest(artifact_path: str | Path) -> dict[str, Any] | None:
    """Return the manifest dict for ``artifact_path``, or None if absent
    / unreadable. A corrupt manifest is treated as "no manifest" so the
    caller rebuilds rather than crashing on resume."""
    path = manifest_path_for(artifact_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def manifest_matches(artifact_path: str | Path, expected_hash: str) -> bool:
    """True iff (a) the artefact exists, (b) it has a sidecar manifest,
    and (c) the manifest's ``input_hash`` matches ``expected_hash``.

    Any mismatch returns False so the caller rebuilds. The caller does
    NOT need to also call :func:`Path.exists` — this does it.
    """
    p = Path(artifact_path)
    if not p.exists():
        return False
    m = read_manifest(p)
    if not m:
        return False
    return m.get("input_hash") == expected_hash


def sha256_of(path: str | Path | None) -> str | None:
    """SHA-256 of a file by path; None if path is empty or missing.

    Used to fingerprint reference inputs (speaker_wav, source figure
    crops) so manifest-based resume picks up changes to the file's
    *contents*, not just its path."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return sha256_file(p)
