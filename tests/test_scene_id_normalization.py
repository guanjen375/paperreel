"""Scene IDs are forced to ``ch_NNN_sc_NNN`` regardless of LLM output.

Background: the script stage prompts the LLM independently per chapter
and asks it to label scenes ``<chapter_id>_sc_001`` etc. Two failure
modes used to slip through:

1. The LLM happily reused ``_sc_001`` across chapters — every chapter's
   first scene collided on disk (same .wav / .png / .mp4 filename), so
   later chapters overwrote earlier ones.
2. Outline chapter ids could be non-canonical (Chinese title, missing
   prefix) which then cascaded into garbage scene_ids.

These tests pin the normalisation contract so a regression there can't
quietly corrupt artefacts again.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from paperreel.providers.llm_base import LLMProvider
from paperreel.stages import (build_outline, build_scene_graph, ingest_pdf,
                               write_script)
from paperreel.state import StateDB
from paperreel.stages.write_script import (_canonical_chapter_ids,
                                            _validate_unique_scene_ids)
from paperreel.models import ScriptScene, VisualType


class _StubLLM(LLMProvider):
    """LLM stub that returns chapter scripts with caller-controlled
    scene_ids — used to prove the normalisation overrides them."""

    name = "stub"

    def __init__(self, cfg=None, *, raw_scene_ids: list[str] | None = None):
        self.cfg = cfg or {}
        self.raw_scene_ids = raw_scene_ids or ["_sc_001", "_sc_002"]

    def chunk_summarize(self, chunk_text, *, page_range, target_chars):
        return {
            "summary": "stub summary",
            "key_points": ["重點"],
            "headings": [],
            "page_range": list(page_range),
        }

    def build_outline(self, chunk_summaries, *, target_minutes, project):
        # Returned via build_outline.run; we mostly use the script stage
        # below, so this is irrelevant in tests that monkey-patch only
        # write_script's provider factory.
        return {
            "project": project,
            "language": "zh-TW",
            "target_minutes": float(target_minutes),
            "rationale": "stub",
            "chapters": [],
        }

    def write_chapter_script(self, chapter, source_pages_text,
                              *, chars_per_scene, forbid_verbatim=True):
        scenes: list[dict] = []
        for idx, sid in enumerate(self.raw_scene_ids, start=1):
            scenes.append({
                "scene_id": sid,                       # deliberately bad / colliding
                "chapter_id": chapter.get("chapter_id", "ch_001"),
                "title": f"stub scene {idx}",
                "source_pages": chapter.get("source_pages") or [1],
                "source_refs": [f"p.{p}" for p in (chapter.get("source_pages") or [1])],
                "narration_text_zh_tw": (
                    f"這是第 {idx} 個 stub 旁白，"
                    "用來驗證 scene_id 正規化邏輯。"
                ),
                "on_screen_text": "重點",
                "visual_hint": "簡報式重點卡",
                "visual_type": "bullet_card",
                "estimated_duration_sec": 25.0,
            })
        return scenes


# --- _canonical_chapter_ids helper -----------------------------------------

def test_canonical_chapter_ids_preserves_already_canonical(monkeypatch):
    from types import SimpleNamespace
    chapters = [SimpleNamespace(chapter_id=f"ch_{i:03d}") for i in range(1, 4)]
    assert _canonical_chapter_ids(chapters) == ["ch_001", "ch_002", "ch_003"]


def test_canonical_chapter_ids_replaces_non_canonical_with_position(monkeypatch):
    """Non-canonical chapter_ids get position-based replacements; canonical
    ones (even out-of-order) stay. The result still has to be unique."""
    from types import SimpleNamespace
    chapters = [SimpleNamespace(chapter_id="第一章"),
                SimpleNamespace(chapter_id="ch_005"),  # canonical but out-of-order
                SimpleNamespace(chapter_id="")]        # missing
    out = _canonical_chapter_ids(chapters)
    assert out == ["ch_001", "ch_005", "ch_003"]
    assert len(set(out)) == len(out)


def test_canonical_chapter_ids_renumbers_when_position_replacement_collides(monkeypatch):
    """If a non-canonical chapter at position N would replace to ch_NNN but
    another chapter already has that canonical id, fall back to dense
    renumber so uniqueness is preserved."""
    from types import SimpleNamespace
    chapters = [SimpleNamespace(chapter_id="第一章"),  # would propose ch_001
                SimpleNamespace(chapter_id="ch_001")]  # already canonical ch_001
    out = _canonical_chapter_ids(chapters)
    assert out == ["ch_001", "ch_002"]


def test_canonical_chapter_ids_collisions_force_position_based(monkeypatch):
    from types import SimpleNamespace
    chapters = [SimpleNamespace(chapter_id="ch_001"),
                SimpleNamespace(chapter_id="ch_001"),  # duplicate from LLM
                SimpleNamespace(chapter_id="ch_002")]
    out = _canonical_chapter_ids(chapters)
    # Duplicate forces full renumber so caller's loop indices align.
    assert out == ["ch_001", "ch_002", "ch_003"]


# --- _validate_unique_scene_ids helper -------------------------------------

def _ss(scene_id: str, chapter_id: str = "ch_001") -> ScriptScene:
    return ScriptScene(
        scene_id=scene_id,
        chapter_id=chapter_id,
        title="t",
        source_pages=[1],
        narration_text_zh_tw="x",
        visual_type=VisualType.bullet_card,
        estimated_duration_sec=20.0,
    )


def test_validate_unique_scene_ids_passes_when_distinct():
    _validate_unique_scene_ids([_ss("ch_001_sc_001"), _ss("ch_001_sc_002")])


def test_validate_unique_scene_ids_raises_on_duplicates():
    with pytest.raises(ValueError, match="duplicate scene_id"):
        _validate_unique_scene_ids([_ss("ch_001_sc_001"),
                                    _ss("ch_001_sc_001", "ch_002")])


# --- end-to-end via write_script -------------------------------------------

def _drive_to_script(project_dir: Path, pdf: Path, cfg: dict,
                     stub: _StubLLM) -> Path:
    """Run ingest + outline + script with `stub` swapped in for write_script.

    The conftest autouse fixture already replaces the LLM with a fake;
    we override it once more here so the test owns the data the LLM
    returns. The autouse fake is still used for build_outline (we only
    care about what write_script does)."""
    import paperreel.stages.write_script as _ws

    original = _ws.make_llm_provider
    try:
        _ws.make_llm_provider = lambda _cfg: stub  # type: ignore[assignment]
        db = StateDB(project_dir / "state.sqlite")
        ingest_pdf.run(pdf_path=pdf, project_root=project_dir, db=db, config=cfg)
        build_outline.run(project_root=project_dir, project_name=project_dir.name,
                          db=db, config=cfg, target_minutes="auto")
        write_script.run(project_root=project_dir, db=db, config=cfg)
        db.close()
    finally:
        _ws.make_llm_provider = original  # type: ignore[assignment]

    return project_dir / "intermediate" / "script.json"


def test_write_script_overrides_colliding_llm_scene_ids(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """LLM returns identical scene_ids for every chapter → write_script
    must rewrite them so all ids are globally unique."""
    stub = _StubLLM(raw_scene_ids=["_sc_001", "_sc_002"])  # would collide across chapters
    script_path = _drive_to_script(project_dir, tiny_pdf, test_cfg, stub)
    data = json.loads(script_path.read_text(encoding="utf-8"))
    ids = [sc["scene_id"] for sc in data["scenes"]]
    assert len(ids) == len(set(ids)), f"duplicate scene_ids slipped through: {ids}"
    # And every id must match the canonical pattern.
    import re
    pat = re.compile(r"^ch_\d{3,}_sc_\d{3,}$")
    assert all(pat.match(i) for i in ids), f"non-canonical scene_ids: {ids}"


def test_write_script_scene_ids_match_chapter_ids(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """Each scene_id must start with its chapter_id, so files for one
    chapter can never end up under another chapter's prefix."""
    stub = _StubLLM(raw_scene_ids=["whatever_999", "garbage"])
    script_path = _drive_to_script(project_dir, tiny_pdf, test_cfg, stub)
    data = json.loads(script_path.read_text(encoding="utf-8"))
    for sc in data["scenes"]:
        assert sc["scene_id"].startswith(sc["chapter_id"] + "_sc_"), (
            f"scene_id {sc['scene_id']} doesn't live under chapter {sc['chapter_id']}"
        )


def test_scene_graph_round_trip_uses_normalized_ids(
    project_dir: Path, tiny_pdf: Path, test_cfg: dict
) -> None:
    """build_scene_graph consumes script.json and must preserve the
    normalised ids (plus generate distinct recap-scene ids)."""
    stub = _StubLLM(raw_scene_ids=["_sc_001"])
    _drive_to_script(project_dir, tiny_pdf, test_cfg, stub)
    db = StateDB(project_dir / "state.sqlite")
    g = build_scene_graph.run(
        project_root=project_dir, project_name=project_dir.name,
        pdf_name=tiny_pdf.name, db=db, config=test_cfg,
    )
    db.close()
    ids = [s.scene_id for s in g.scenes]
    assert len(ids) == len(set(ids)), f"scene graph has duplicate ids: {ids}"
    for s in g.scenes:
        # recap scenes are derived from a parent scene_id + "_recap"; the
        # parent itself must still be canonical.
        head = s.scene_id.removesuffix("_recap")
        assert head.startswith(s.chapter_id + "_sc_"), (
            f"scene {s.scene_id} not under chapter {s.chapter_id}"
        )
