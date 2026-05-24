"""Config loading — merges default.yaml + optional overlay + project overrides.

Default + bundled overlays ship inside the installed package
(``paperreel.configs``) and are loaded via :mod:`importlib.resources`,
so ``pip install paperreel`` followed by ``paperreel init …`` works
without the repo checkout being present on disk. Before this, the
default config path was computed relative to ``__file__``'s great-
grandparent ``configs/`` directory — fine in editable installs, broken
once you built a wheel.
"""
from __future__ import annotations

import copy
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml


PACKAGE_CONFIG_PKG = "paperreel.configs"


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


def _read_packaged_yaml(name: str) -> dict:
    """Load a bundled YAML by its package-relative filename.

    Uses :func:`importlib.resources.as_file` so the file works whether
    the package is installed as source, as a wheel, or zip-imported.
    """
    resource = files(PACKAGE_CONFIG_PKG).joinpath(name)
    with as_file(resource) as p:
        return load_yaml(p)


def list_packaged_overlays() -> list[str]:
    """Yaml filenames bundled with the package (default.yaml excluded
    since it's always loaded as the base). Useful for ``paperreel
    init --config <name>`` to suggest options."""
    out: list[str] = []
    for entry in files(PACKAGE_CONFIG_PKG).iterdir():
        if entry.is_file() and entry.name.endswith(".yaml") and entry.name != "default.yaml":
            out.append(entry.name)
    return sorted(out)


def load_config(overlay: str | Path | None = None, *, extra: dict | None = None) -> dict[str, Any]:
    """Load packaged default.yaml then merge an optional overlay + extra dict.

    ``overlay`` resolution order: literal filesystem path → packaged
    overlay by name (``bigvram`` resolves to the bundled
    ``bigvram.yaml``) → ``<name>.yaml`` variant of the above. Raises
    ``FileNotFoundError`` if nothing matches so a typo doesn't silently
    fall back to defaults.
    """
    cfg = _read_packaged_yaml("default.yaml")
    if overlay is not None:
        ov_path = Path(overlay)
        if ov_path.exists():
            cfg = _deep_merge(cfg, load_yaml(ov_path))
        else:
            # Try packaged overlay: support both "bigvram" and
            # "bigvram.yaml" for ergonomics.
            candidates = (
                str(overlay) if str(overlay).endswith(".yaml")
                else f"{overlay}.yaml",
            )
            loaded = False
            for cand in candidates:
                resource = files(PACKAGE_CONFIG_PKG).joinpath(cand)
                if resource.is_file():
                    cfg = _deep_merge(cfg, _read_packaged_yaml(cand))
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f"config overlay not found: {overlay}")
    if extra:
        cfg = _deep_merge(cfg, extra)
    return cfg
