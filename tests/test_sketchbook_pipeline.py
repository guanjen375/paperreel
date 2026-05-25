"""End-to-end sketchbook mode through stage 4 (build_scene_graph), then
through stage 6 (render_visuals), using fake LLM + fake TTS.

These tests are heavier than the unit tests above — they exercise the
new branches in build_outline, write_script (sketchbook builder),
build_scene_graph, render_visuals, and the SketchbookRenderer dispatch.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import fitz  # PyMuPDF
import pytest

from paperreel.config import load_config
from paperreel.io_utils import read_json
from paperreel.models import (DocKind, DocProfile, ScriptScene, SceneGraph,
                                VisualType)
from paperreel.stages import (build_outline, build_scene_graph, ingest_pdf,
                                match_pdf_visuals, render_visuals,
                                review as review_stage, write_script)
from paperreel.state import StateDB


REFERENCE_ROOT = Path("dev_samples/reference")
LEGACY_REFERENCE_ROOT = Path("dev_examples/reference")
REFERENCE_SAMPLE_PDF = REFERENCE_ROOT / "sample.pdf"
LEGACY_REFERENCE_SAMPLE_PDF = LEGACY_REFERENCE_ROOT / "sample.pdf"
REFERENCE_SHORT_VIDEO = REFERENCE_ROOT / "notebooklm_short.mp4"
REFERENCE_LONG_VIDEO = REFERENCE_ROOT / "notebooklm_long.mp4"
REFERENCE_FRAMES_DIR = REFERENCE_ROOT / "frames"


def _optional_reference_sample_pdf() -> Path | None:
    if REFERENCE_SAMPLE_PDF.exists():
        return REFERENCE_SAMPLE_PDF
    if LEGACY_REFERENCE_SAMPLE_PDF.exists():
        return LEGACY_REFERENCE_SAMPLE_PDF
    return None


def _fact_group_present(blob: str, groups: list[tuple[str, ...]]) -> bool:
    return all(any(token in blob for token in group) for group in groups)


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@pytest.fixture
def contract_pdf(tmp_path: Path) -> Path:
    """Synthetic contract-ish PDF — exercises the contract storyboard."""
    pdf_path = tmp_path / "contract.pdf"
    doc = fitz.open()
    pages = [
        "第一條 簽約\n"
        "本合約由甲方與乙方雙方簽署。雙方應於 45 天內完成付款，"
        "違約金為合約金額的 30%。請提供護照影本與身分證影本。",

        "第二條 取消與退費\n"
        "若於 30 天前取消，退費 80%；於 14 天前取消，退費 50%；"
        "於 7 天前取消，退費 20%。逾期不予退款。",

        "第三條 風險與責任\n"
        "請特別注意天候風險。雙方不得擅自轉讓本契約。"
        "天候不佳所造成損失，乙方不予賠償。"
        "若延遲提供資料超過 14 天，視同違約。",
    ]
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 80), text, fontsize=12, fontname="china-s",
                          color=(0, 0, 0))
    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture
def sketchbook_cfg() -> dict:
    """sketchbook overlay merged with test-only renderer overrides."""
    cfg = load_config("sketchbook")
    overrides = {
        "tts": {"sample_rate_hz": 24000},
        "renderer": {"resolution": [1280, 720], "fps": 24},
        "runtime": {"max_hours": 0.5, "parallelism": 1},
        # Auto-classifier should pick contract; we don't ask the fake
        # LLM to refine — saves the cost.
        "doc_explainer": {
            "classify": {"use_llm_refinement": False},
            "grounding": {"quote_match_min_ratio": 0.45},
        },
    }
    return _deep_merge(cfg, overrides)


def _drive_through_script(project_dir: Path, pdf: Path,
                           cfg: dict, *,
                           target_minutes: str = "auto") -> tuple[StateDB, dict]:
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=pdf, project_root=project_dir, db=db, config=cfg)
    build_outline.run(project_root=project_dir, project_name=project_dir.name,
                       db=db, config=cfg, target_minutes=target_minutes)
    script = write_script.run(project_root=project_dir, db=db, config=cfg)
    return db, script.model_dump(mode="json")


def test_normal_default_config_uses_explainer_without_style(
    project_dir: Path, contract_pdf: Path,
) -> None:
    cfg = load_config()
    cfg = _deep_merge(cfg, {
        "tts": {"sample_rate_hz": 24000},
        "renderer": {"resolution": [1280, 720], "fps": 24},
        "doc_explainer": {"grounding": {"quote_match_min_ratio": 0.45}},
    })
    _drive_through_script(project_dir, contract_pdf, cfg, target_minutes="2")
    blob = read_json(project_dir / "intermediate" / "script.json")
    assert blob["scenes"]
    assert any(sc.get("scene_kind") for sc in blob["scenes"])
    assert all(sc["visual_type"] != "generated_image" for sc in blob["scenes"])


def test_sketchbook_classifies_contract(project_dir: Path, contract_pdf: Path,
                                          sketchbook_cfg: dict) -> None:
    _drive_through_script(project_dir, contract_pdf, sketchbook_cfg)
    profile_path = project_dir / "intermediate" / "doc_profile.json"
    assert profile_path.exists()
    profile = DocProfile.model_validate(read_json(profile_path))
    assert profile.doc_kind == DocKind.contract
    assert "deadline_timeline" in profile.suggested_storyboard


def test_sketchbook_script_has_scene_kinds_and_evidence(
    project_dir: Path, contract_pdf: Path, sketchbook_cfg: dict,
) -> None:
    _drive_through_script(project_dir, contract_pdf, sketchbook_cfg)
    script_path = project_dir / "intermediate" / "script.json"
    blob = read_json(script_path)
    scene_kinds = {sc.get("scene_kind") for sc in blob["scenes"]}
    assert "cover" in scene_kinds
    assert "recap_card" in scene_kinds
    # Contract storyboard should produce at least one factual card.
    factual = scene_kinds & {"deadline_timeline", "penalty_table",
                              "checklist", "risk_warning"}
    assert factual, f"sketchbook contract run had no factual scenes: {scene_kinds}"
    # Every factual scene must carry at least one evidence span on disk.
    for sc in blob["scenes"]:
        if sc.get("scene_kind") in {"deadline_timeline", "penalty_table",
                                      "checklist", "risk_warning",
                                      "do_dont", "key_number", "source_crop"}:
            assert sc["evidence_spans"], (
                f"scene {sc['scene_id']} ({sc['scene_kind']}) "
                "has no evidence_spans"
            )


def test_sketchbook_default_forbids_generated_image(
    project_dir: Path, contract_pdf: Path, sketchbook_cfg: dict,
) -> None:
    _drive_through_script(project_dir, contract_pdf, sketchbook_cfg)
    script_path = project_dir / "intermediate" / "script.json"
    blob = read_json(script_path)
    for sc in blob["scenes"]:
        assert sc["visual_type"] != "generated_image", (
            f"sketchbook mode produced visual_type=generated_image "
            f"on scene {sc['scene_id']}"
        )


def test_sketchbook_renders_visuals_via_sketchbook_renderer(
    project_dir: Path, contract_pdf: Path, sketchbook_cfg: dict,
) -> None:
    db, _ = _drive_through_script(project_dir, contract_pdf, sketchbook_cfg)
    build_scene_graph.run(
        project_root=project_dir, project_name=project_dir.name,
        pdf_name=contract_pdf.name, db=db, config=sketchbook_cfg,
    )
    # match_visuals must be a no-op in sketchbook (off by default).
    match_pdf_visuals.run(project_root=project_dir, db=db, config=sketchbook_cfg)
    # We skip audio + subtitles for this test; render_visuals doesn't
    # actually need audio paths populated.
    graph = render_visuals.run(project_root=project_dir, db=db,
                                config=sketchbook_cfg, resume=False)
    # Every rendered card lives under assets/visuals.
    visuals_dir = project_dir / "assets" / "visuals"
    assert visuals_dir.exists()
    rendered = list(visuals_dir.glob("*.png"))
    assert rendered, "render_visuals did not produce any cards"
    # No SDXL-generated PNGs (sketchbook mode forbids them).
    generated_dir = project_dir / "assets" / "generated"
    assert not generated_dir.exists() or not list(generated_dir.glob("*.png"))


def test_legacy_default_mode_still_works(project_dir: Path, tiny_pdf: Path,
                                         test_cfg: dict) -> None:
    """Regression: explicit style=default keeps the legacy storyboard."""
    legacy_cfg = _deep_merge(test_cfg, {"project": {"style": "default"}})
    db = StateDB(project_dir / "state.sqlite")
    ingest_pdf.run(pdf_path=tiny_pdf, project_root=project_dir, db=db,
                    config=legacy_cfg)
    build_outline.run(project_root=project_dir, project_name=project_dir.name,
                       db=db, config=legacy_cfg, target_minutes="auto")
    script = write_script.run(project_root=project_dir, db=db, config=legacy_cfg)
    assert script.scenes
    # No scene_kind tags in default mode.
    for sc in script.scenes:
        assert sc.scene_kind is None
    db.close()


def test_target_minutes_inserts_grounded_expansion_scenes(
    tmp_path: Path, contract_pdf: Path, sketchbook_cfg: dict,
) -> None:
    project = tmp_path / "target5"
    project.mkdir()
    _db, script_blob = _drive_through_script(
        project, contract_pdf, sketchbook_cfg, target_minutes="5",
    )
    expansions = [
        sc for sc in script_blob["scenes"]
        if sc.get("layout_payload", {}).get("expansion_of")
    ]
    assert expansions, "under-budget target did not insert expansion scenes"
    assert all(sc["evidence_spans"] for sc in expansions)
    assert not all(sc["scene_kind"] == "paragraph_card" for sc in expansions)

    penalty_expansions = [
        sc for sc in expansions if sc.get("scene_kind") == "penalty_table"
    ]
    assert penalty_expansions, "penalty_table was not expanded by row"
    assert all(
        len(sc.get("layout_payload", {}).get("rows") or []) == 1
        for sc in penalty_expansions
    )
    payload_keys = {
        json.dumps(sc.get("layout_payload", {}), ensure_ascii=False, sort_keys=True)
        for sc in expansions
    }
    assert len(payload_keys) == len(expansions), "duplicate expansion card payload"

    plan = read_json(project / "intermediate" / "sketchbook_plan.json")
    assert plan["expansion_scene_count"] > 0
    assert plan["actual_estimated_seconds"] >= plan["min_seconds"]


def test_target_minutes_changes_scene_budget(
    tmp_path: Path, contract_pdf: Path, sketchbook_cfg: dict,
) -> None:
    p2 = tmp_path / "target2"
    p5 = tmp_path / "target5"
    p2.mkdir(); p5.mkdir()
    _db2, script2 = _drive_through_script(
        p2, contract_pdf, sketchbook_cfg, target_minutes="2",
    )
    _db5, script5 = _drive_through_script(
        p5, contract_pdf, sketchbook_cfg, target_minutes="5",
    )
    assert len(script5["scenes"]) > len(script2["scenes"])
    assert script5["total_estimated_minutes"] > script2["total_estimated_minutes"]


def test_reference_sample_path_prefers_dev_samples() -> None:
    chosen = _optional_reference_sample_pdf()
    if REFERENCE_SAMPLE_PDF.exists():
        assert chosen == REFERENCE_SAMPLE_PDF
    elif LEGACY_REFERENCE_SAMPLE_PDF.exists():
        assert chosen == LEGACY_REFERENCE_SAMPLE_PDF
    else:
        assert chosen is None
    assert REFERENCE_SHORT_VIDEO == Path("dev_samples/reference/notebooklm_short.mp4")
    assert REFERENCE_LONG_VIDEO == Path("dev_samples/reference/notebooklm_long.mp4")
    assert REFERENCE_FRAMES_DIR == Path("dev_samples/reference/frames")


def test_reference_sample_contract_tiers_smoke(
    tmp_path: Path, sketchbook_cfg: dict,
) -> None:
    sample = _optional_reference_sample_pdf()
    if sample is None:
        pytest.skip("optional reference sample PDF not present")
    project = tmp_path / "sample_contract"
    project.mkdir()
    _db, script_blob = _drive_through_script(
        project, sample, sketchbook_cfg, target_minutes="5",
    )
    text_blob = json.dumps(script_blob, ensure_ascii=False)
    checks = {
        "45-day payment/data deadline": [
            ("出發前45", "45 天"),
            ("全額費用", "繳付", "正確名單", "資料"),
        ],
        "cancellation tiers": [
            ("全額訂",), ("30%",), ("50%",), ("75%",), ("100%",),
        ],
        "NT$3,000 name/cabin change fee": [
            ("3,000", "3000"),
            ("改名", "更改名單", "艙房"),
        ],
        "within 30 days no name/cabin changes": [
            ("出發前30", "30 天內"),
            ("無法更動", "恕無法"),
        ],
        "passport/visa/boarding responsibility": [
            ("護照",), ("簽證",), ("登船",),
        ],
        "rejected boarding / no refund / extra costs": [
            ("拒絕登船", "登船"),
            ("恕不退還", "不予退款", "不退費"),
            ("額外費", "額外費用", "交通安排", "住宿"),
        ],
        "insurance/health reminders": [
            ("保險",),
            ("健康", "疾病", "醫療", "醫師", "適航"),
        ],
        "force majeure / itinerary change / possible no refund": [
            ("不可抗力",),
            ("行程", "變更"),
            ("無退費義務", "不予退款", "退款"),
        ],
    }
    missing = [
        name for name, groups in checks.items()
        if not _fact_group_present(text_blob, groups)
    ]
    assert not missing, "missing sample facts: " + ", ".join(missing)


def test_review_stage_produces_artefacts(
    project_dir: Path, contract_pdf: Path, sketchbook_cfg: dict,
) -> None:
    db, _ = _drive_through_script(project_dir, contract_pdf, sketchbook_cfg)
    build_scene_graph.run(
        project_root=project_dir, project_name=project_dir.name,
        pdf_name=contract_pdf.name, db=db, config=sketchbook_cfg,
    )
    match_pdf_visuals.run(project_root=project_dir, db=db, config=sketchbook_cfg)
    render_visuals.run(project_root=project_dir, db=db, config=sketchbook_cfg,
                        resume=False)
    summary = review_stage.run(project_root=project_dir, db=db,
                                 config=sketchbook_cfg)
    assert summary["scene_count"] > 0
    review_dir = project_dir / "outputs" / "review"
    assert (review_dir / "semantic_quality.json").exists()
    assert (review_dir / "storyboard.html").exists()
    # Contact sheet should exist because visuals were rendered.
    assert (review_dir / "contact_sheet.jpg").exists()
