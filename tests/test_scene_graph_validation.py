"""End-to-end through stage 4 (build_scene_graph) using the fake LLM.

The autouse fixture in conftest.py swaps the production Ollama provider
for a deterministic fake — this is the lightest realistic exercise of
the early pipeline (ingest, plan, script, scenes) and asserts the
resulting scene graph survives JSON round-trip and matches the contract
every later stage relies on.
"""
from __future__ import annotations

from pathlib import Path

from paperreel.io_utils import read_json
from paperreel.models import SceneGraph, VisualType
from paperreel.stages import (build_outline, build_scene_graph, ingest_pdf,
                               write_script)
from paperreel.state import StateDB


def _drive_to_scenegraph(project_dir: Path, pdf: Path, cfg: dict) -> SceneGraph:
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=pdf, project_root=project_dir, db=db, config=cfg)
    build_outline.run(project_root=project_dir, project_name=project_dir.name,
                      db=db, config=cfg, target_minutes="auto")
    write_script.run(project_root=project_dir, db=db, config=cfg)
    g = build_scene_graph.run(project_root=project_dir, project_name=project_dir.name,
                              pdf_name=pdf.name, db=db, config=cfg)
    db.close()
    return g


def test_scene_graph_has_required_fields(project_dir: Path, tiny_pdf: Path,
                                         test_cfg: dict) -> None:
    g = _drive_to_scenegraph(project_dir, tiny_pdf, test_cfg)
    assert len(g.scenes) >= 1
    for sc in g.scenes:
        assert sc.scene_id
        assert sc.chapter_id
        assert sc.input_hash
        assert sc.source_pages, "every scene must trace back to source pages"
        assert sc.narration_text_zh_tw
        assert isinstance(sc.visual_type, VisualType)


def test_scene_graph_json_is_valid_pydantic(project_dir: Path, tiny_pdf: Path,
                                            test_cfg: dict) -> None:
    g = _drive_to_scenegraph(project_dir, tiny_pdf, test_cfg)
    path = project_dir / "intermediate" / "scene_graph.json"
    assert path.exists()
    data = read_json(path)
    g2 = SceneGraph.model_validate(data)
    assert [s.scene_id for s in g2.scenes] == [s.scene_id for s in g.scenes]


def test_each_scene_id_unique(project_dir: Path, tiny_pdf: Path,
                              test_cfg: dict) -> None:
    g = _drive_to_scenegraph(project_dir, tiny_pdf, test_cfg)
    ids = [s.scene_id for s in g.scenes]
    assert len(ids) == len(set(ids))
