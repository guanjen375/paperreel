"""Config loading — merges default.yaml + optional overlay + project overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return data


def load_config(overlay: str | Path | None = None, *, extra: dict | None = None) -> dict[str, Any]:
    """Load configs/default.yaml then merge an optional overlay + extra dict."""
    cfg = load_yaml(CONFIG_DIR / "default.yaml")
    if overlay is not None:
        ov_path = Path(overlay)
        if not ov_path.exists():
            candidate = CONFIG_DIR / overlay
            if candidate.exists():
                ov_path = candidate
            else:
                raise FileNotFoundError(f"config overlay not found: {overlay}")
        cfg = _deep_merge(cfg, load_yaml(ov_path))
    if extra:
        cfg = _deep_merge(cfg, extra)
    return cfg
