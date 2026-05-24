"""Config files ship inside the package so wheel installs work.

The original layout pointed config.py at ``__file__.parents[2] / "configs"``
— fine in an editable install, broken in a built wheel because the
``configs/`` directory lived at the repo root, not inside the package.
These tests pin the new behaviour:

1. ``load_config()`` works using only :mod:`importlib.resources`, with
   no filesystem assumptions beyond what pip puts in site-packages.
2. Bundled overlays resolve by short name (``rtx5090`` →
   ``rtx5090.yaml``).
3. Filesystem-path overlays still work for user overrides.
4. A typo'd overlay raises ``FileNotFoundError`` instead of silently
   falling back to defaults.
5. The package's ``configs/`` directory is enumerated correctly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from paperreel.config import (PACKAGE_CONFIG_PKG, list_packaged_overlays,
                                load_config)


def test_default_config_loads_without_repo_layout(tmp_path: Path,
                                                    monkeypatch) -> None:
    """Even when cwd is somewhere unrelated to the source tree (the
    typical wheel-install case), default config still loads."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert "llm" in cfg
    assert "tts" in cfg
    assert cfg["llm"]["provider"] == "ollama"


def test_packaged_overlay_resolves_by_short_name() -> None:
    cfg = load_config("rtx5090")
    assert cfg["llm"]["provider"] == "ollama"
    # rtx5090 overlay bumps the LLM model; whatever the specific value
    # is, it must differ from the default.
    assert cfg["llm"]["model"] != load_config()["llm"]["model"]


def test_packaged_overlay_accepts_yaml_extension() -> None:
    """``--config rtx5090`` and ``--config rtx5090.yaml`` should give
    the same result so users don't trip over the extension."""
    a = load_config("rtx5090")
    b = load_config("rtx5090.yaml")
    assert a == b


def test_filesystem_path_overlay_still_works(tmp_path: Path) -> None:
    overlay = tmp_path / "custom.yaml"
    overlay.write_text("renderer:\n  background_color: '#123456'\n",
                        encoding="utf-8")
    cfg = load_config(overlay)
    assert cfg["renderer"]["background_color"] == "#123456"


def test_missing_overlay_raises_rather_than_silently_falling_back(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)  # so a path that doesn't exist truly doesn't
    with pytest.raises(FileNotFoundError):
        load_config("no_such_overlay_xyz")


def test_list_packaged_overlays_excludes_default() -> None:
    overlays = list_packaged_overlays()
    assert "default.yaml" not in overlays
    # At least the rtx5090 overlay should be there.
    assert "rtx5090.yaml" in overlays


def test_load_config_returns_independent_objects() -> None:
    """Mutating the returned dict must not bleed into the cached
    default. Each load_config() call should return a fresh deep copy
    so callers can freely overlay ad-hoc overrides."""
    a = load_config()
    a["llm"]["model"] = "mutated"
    b = load_config()
    assert b["llm"]["model"] != "mutated"


def test_config_path_uses_importlib_resources(tmp_path: Path) -> None:
    """White-box-ish: the resource the loader queries must live inside
    the installed package, not at repo root. This is the regression
    check — the old `Path(__file__).parents[2] / "configs"` only
    worked in editable installs."""
    from importlib.resources import files
    pkg = files(PACKAGE_CONFIG_PKG)
    yaml_files = [e.name for e in pkg.iterdir() if e.is_file()]
    assert "default.yaml" in yaml_files, (
        "default.yaml must be a package resource, not a sibling of the "
        f"package dir. Found in {PACKAGE_CONFIG_PKG}: {yaml_files}"
    )
