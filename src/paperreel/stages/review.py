"""Review / quality stage for sketchbook + default modes.

Emits three artefacts under ``outputs/review/``:

- ``contact_sheet.jpg`` — a thumbnail grid of every rendered card so
  you can eyeball pacing + layout in one glance.
- ``storyboard.html`` — a static HTML report with each scene's title,
  narration, evidence, and rendered card.
- ``semantic_quality.json`` — machine-readable quality signal: missing
  evidence, dense tables, oversized text, generated-image leakage,
  unreadable crops, source-coverage stats, duration compliance.

The review runs entirely on CPU. Optional local-VLM scoring lives in
``vlm_review`` and is disabled unless ``doc_explainer.vlm_review.model``
is set; absence of a VLM must NOT fail the review.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from ..io_utils import atomic_write_json, atomic_write_text, ensure_dir, read_json
from ..models import (ChunkedSources, DocProfile, SceneGraph, VisualType)
from ..state import StateDB
from ..utils import grounding


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "chunked": root / "intermediate" / "chunked_sources.json",
        "doc_profile": root / "intermediate" / "doc_profile.json",
        "plan_report": root / "intermediate" / "sketchbook_plan.json",
        "duration": root / "intermediate" / "duration_plan.json",
        "review_dir": root / "outputs" / "review",
        "contact_sheet": root / "outputs" / "review" / "contact_sheet.jpg",
        "storyboard": root / "outputs" / "review" / "storyboard.html",
        "semantic_quality": root / "outputs" / "review" / "semantic_quality.json",
    }


def run(*, project_root: str | Path, db: StateDB, config: dict
        ) -> dict[str, Any]:
    p = paths_for(project_root)
    ensure_dir(p["review_dir"])
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    sources = ChunkedSources.model_validate(read_json(p["chunked"]))

    duration_plan: dict[str, Any] = {}
    if p["duration"].exists():
        duration_plan = read_json(p["duration"])
    profile: DocProfile | None = None
    if p["doc_profile"].exists():
        try:
            profile = DocProfile.model_validate(read_json(p["doc_profile"]))
        except Exception:
            profile = None
    plan_report: dict[str, Any] = {}
    if p["plan_report"].exists():
        try:
            plan_report = read_json(p["plan_report"])
        except Exception:
            plan_report = {}

    sketchbook = (
        str((config.get("project") or {}).get("style") or "default").lower()
        in ("sketchbook", "document_explainer")
    )
    de_cfg = config.get("doc_explainer", {}) or {}
    grounding_cfg = de_cfg.get("grounding", {}) or {}
    cards_cfg = de_cfg.get("cards", {}) or {}
    min_quote_ratio = float(grounding_cfg.get("quote_match_min_ratio", 0.55))
    allow_generated = bool(de_cfg.get("allow_generated_images", False))

    findings: list[dict] = []
    coverage_pages = set()
    for sc in graph.scenes:
        coverage_pages.update(sc.source_pages)
        if sketchbook and sc.visual_type == VisualType.generated_image and not allow_generated:
            findings.append(_finding("generated_image_leak", "error",
                                     "scene asks for generated_image while "
                                     "doc_explainer.allow_generated_images=false",
                                     sc.scene_id))
        if sc.visual_asset_paths:
            card_path = Path(sc.visual_asset_paths[0])
            if not card_path.exists():
                findings.append(_finding("missing_visual", "error",
                                         f"visual asset missing: {card_path}",
                                         sc.scene_id))
            else:
                size = _safe_image_size(card_path)
                if size and (size[0] < 800 or size[1] < 450):
                    findings.append(_finding(
                        "low_res_card", "warning",
                        f"card resolution {size[0]}x{size[1]} too small "
                        "for readable text",
                        sc.scene_id,
                    ))
        else:
            findings.append(_finding("no_visual_path", "error",
                                     "scene has no visual_asset_paths",
                                     sc.scene_id))
        # Pull text-overflow signal from layout payload sizes.
        payload = sc.layout_payload or {}
        if (sc.scene_kind or "") == "penalty_table":
            rows = payload.get("rows") or []
            if len(rows) > int(cards_cfg.get("max_table_rows", 6)):
                findings.append(_finding(
                    "table_too_dense", "warning",
                    f"penalty_table has {len(rows)} rows — over "
                    f"{cards_cfg.get('max_table_rows', 6)} risks unreadable text",
                    sc.scene_id,
                ))
        if (sc.scene_kind or "") == "checklist":
            items = payload.get("items") or []
            if len(items) > int(cards_cfg.get("max_checklist_items", 8)):
                findings.append(_finding(
                    "checklist_too_long", "warning",
                    f"checklist has {len(items)} items — over the "
                    "configured max", sc.scene_id,
                ))
        # Sketchbook-only: enforce evidence on factual scenes.
        prompt_text = " ".join([sc.title, sc.visual_prompt or "", sc.on_screen_text or ""]).lower()
        if any(w in prompt_text for w in ("stamp", "seal", "logo", "印章", "圖章", "標誌")):
            if sc.visual_type in (VisualType.generated_image, VisualType.pdf_image) and not sc.evidence_spans:
                findings.append(_finding(
                    "decorative_visual_risk", "warning",
                    "stamp/logo/seal-like visual is not backed by evidence",
                    sc.scene_id,
                ))
        if sketchbook and (sc.scene_kind or "") in {
            "deadline_timeline", "penalty_table", "checklist",
            "risk_warning", "do_dont", "key_number", "source_crop",
        }:
            if not sc.evidence_spans:
                findings.append(_finding(
                    "missing_evidence", "error",
                    f"factual scene_kind={sc.scene_kind!r} has no evidence_spans",
                    sc.scene_id,
                ))
            else:
                page_text = {pg.page: pg.text for pg in sources.pages}
                bad = grounding.validate_scene(
                    _as_script_scene(sc),
                    page_text=page_text,
                    min_quote_ratio=min_quote_ratio,
                    require_evidence_for_facts=True,
                )
                for issue in bad:
                    findings.append(_finding(
                        "evidence_mismatch", "warning",
                        issue.message, sc.scene_id,
                    ))

    # PDF coverage report — pages that contributed zero scenes show up
    # so reviewers can decide whether to extend the chapter list.
    total_pages = sources.page_count or 1
    coverage_pct = 100.0 * len(coverage_pages) / total_pages
    if coverage_pct < 40.0:
        findings.append(_finding(
            "low_source_coverage", "warning",
            f"only {len(coverage_pages)}/{total_pages} PDF pages "
            f"({coverage_pct:.0f}%) are referenced by any scene",
        ))

    duration_status = _duration_status(duration_plan, graph)
    if duration_status.get("violates_window"):
        findings.append(_finding(
            "duration_off_target", "warning",
            duration_status["message"],
        ))

    expected_fact_summary = _expected_fact_summary(profile, graph)
    findings.extend(expected_fact_summary["findings"])

    # Optional local-VLM scoring. Disabled unless model is set.
    vlm_cfg = de_cfg.get("vlm_review", {}) or {}
    vlm_results: list[dict] = []
    if vlm_cfg.get("enabled") and vlm_cfg.get("model"):
        try:
            vlm_results = _run_vlm_review(graph, vlm_cfg)
        except Exception as e:
            findings.append(_finding(
                "vlm_unavailable", "info",
                f"local VLM review skipped: {e!r}",
            ))

    contact_path = _build_contact_sheet(
        graph, p["contact_sheet"],
    )
    storyboard_path = _build_storyboard(
        graph, sources, profile, plan_report, findings, p["storyboard"],
        sketchbook=sketchbook,
    )

    summary = {
        "schema": "semantic_quality_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "style": "sketchbook" if sketchbook else "default",
        "doc_kind": (profile.doc_kind.value if profile else None),
        "scene_count": len(graph.scenes),
        "scenes_with_evidence": sum(
            1 for sc in graph.scenes if sc.evidence_spans
        ),
        "factual_scene_count": sum(
            1 for sc in graph.scenes
            if (sc.scene_kind or "") in {
                "deadline_timeline", "penalty_table", "checklist",
                "risk_warning", "do_dont", "key_number", "source_crop",
            }
        ),
        "coverage_pct": round(coverage_pct, 1),
        "covered_pages": sorted(coverage_pages),
        "duration": duration_status,
        "generated_image_count": sum(
            1 for sc in graph.scenes if sc.visual_type == VisualType.generated_image
        ),
        "expected_fact_types": expected_fact_summary["types"],
        "findings": findings,
        "vlm_results": vlm_results,
        "artefacts": {
            "contact_sheet": str(contact_path) if contact_path else None,
            "storyboard": str(storyboard_path),
        },
    }
    atomic_write_json(p["semantic_quality"], summary)
    db.register_artifact(p["semantic_quality"], stage="quality",
                         media_type="application/json")
    if contact_path:
        db.register_artifact(contact_path, stage="quality",
                             media_type="image/jpeg")
    db.register_artifact(storyboard_path, stage="quality",
                         media_type="text/html")
    return summary


# ---------- helpers ----------

def _finding(code: str, severity: str, message: str,
             scene_id: str | None = None) -> dict:
    return {"code": code, "severity": severity, "message": message,
            "scene_id": scene_id}


def _safe_image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def _as_script_scene(scene):
    """Adapt a Scene into a ScriptScene-shaped duck so grounding
    validator can re-use its logic. Cheap shim — we only need the
    fields the validator actually reads."""
    from types import SimpleNamespace
    return SimpleNamespace(
        scene_id=scene.scene_id,
        scene_kind=scene.scene_kind,
        source_pages=list(scene.source_pages),
        evidence_spans=list(scene.evidence_spans),
        facts=list(scene.facts),
    )


def _expected_fact_summary(profile: DocProfile | None, graph: SceneGraph) -> dict[str, Any]:
    if profile is None:
        return {"types": [], "findings": []}
    kind = profile.doc_kind.value
    expected = {
        "contract": {
            "deadlines": ("期限", "出發前", "天內", "天以上", "deadline"),
            "penalty_tiers": ("全額訂", "30%", "50%", "75%", "100%"),
            "fees_or_money": ("NT$", "新台幣", "新臺幣", "金額", "3,000"),
            "required_documents": ("護照", "簽證", "資料", "名單"),
            "no_refund_or_boarding_risk": ("拒絕登船", "拒絕入境", "不予退款", "恕不退還", "概不負責"),
            "insurance_or_health": ("保險", "疾病", "適航證明", "醫師"),
            "force_majeure_or_itinerary_change": ("不可抗力", "行程", "變更", "無退費義務"),
            "personal_data": ("個人資料", "蒐集", "處理", "利用", "傳輸"),
        },
        "form": {
            "required_fields": ("姓名", "身分證", "護照", "簽名", "日期"),
            "required_documents": ("附件", "資料", "證明", "護照", "簽證"),
            "deadlines": ("期限", "前", "內"),
        },
        "policy": {
            "scope": ("適用", "範圍", "對象"),
            "obligations": ("應", "須", "不得", "必須"),
            "penalties_or_risks": ("違反", "罰", "風險", "責任"),
        },
        "paper": {
            "problem_or_goal": ("problem", "目標", "問題", "研究"),
            "method": ("method", "方法", "模型", "實驗"),
            "results": ("result", "結果", "%", "提升"),
            "limitations": ("limit", "限制", "未來"),
        },
        "manual": {
            "prerequisites": ("前置", "準備", "需求"),
            "steps": ("步驟", "操作", "安裝", "設定"),
            "warnings": ("警告", "注意", "請勿", "不得"),
            "troubleshooting": ("故障", "排除", "問題"),
        },
        "report": {
            "summary": ("摘要", "summary", "重點"),
            "metrics": ("%", "金額", "指標", "成長", "下降"),
            "risks": ("風險", "risk"),
            "recommendations": ("建議", "recommend"),
        },
        "slides": {
            "sections": ("章節", "section", "重點"),
            "takeaways": ("takeaway", "重點", "回顧"),
        },
    }.get(kind, {})
    if not expected:
        return {"types": [], "findings": []}

    blob = _graph_text_blob(graph)
    types: list[dict[str, Any]] = []
    findings: list[dict] = []
    for type_name, needles in expected.items():
        present = any(n.lower() in blob for n in needles)
        types.append({
            "type": type_name,
            "present": present,
            "needles": list(needles),
        })
        if not present:
            findings.append(_finding(
                "expected_fact_type_missing", "warning",
                f"doc_kind={kind} expected fact type missing: {type_name}",
            ))
    return {"types": types, "findings": findings}


def _graph_text_blob(graph: SceneGraph) -> str:
    parts: list[str] = []
    for sc in graph.scenes:
        parts.extend([sc.title, sc.narration_text_zh_tw, sc.on_screen_text or ""])
        for fact in sc.facts:
            parts.extend([fact.label, fact.value])
        for ev in sc.evidence_spans:
            parts.extend([ev.label or "", ev.value or "", ev.quote or ""])
        try:
            parts.append(json.dumps(sc.layout_payload, ensure_ascii=False))
        except TypeError:
            parts.append(str(sc.layout_payload))
    return "\n".join(parts).lower()


def _duration_status(duration_plan: dict, graph: SceneGraph) -> dict:
    target_minutes = duration_plan.get("target_minutes") or graph.target_minutes
    target_seconds = float(target_minutes) * 60.0
    actual_seconds = sum(
        (sc.actual_duration_sec or sc.estimated_duration_sec) or 0
        for sc in graph.scenes
    )
    if target_seconds <= 0:
        return {"target_seconds": target_seconds,
                "actual_seconds": round(actual_seconds, 1),
                "violates_window": False,
                "message": "no target set"}
    delta = actual_seconds - target_seconds
    pct = abs(delta) / target_seconds * 100.0
    violates = pct > 12.0
    return {
        "target_seconds": round(target_seconds, 1),
        "actual_seconds": round(actual_seconds, 1),
        "delta_pct": round(pct, 1),
        "violates_window": violates,
        "message": (
            f"target {target_seconds/60:.1f} min vs actual "
            f"{actual_seconds/60:.1f} min ({pct:.1f}% off)"
        ),
    }


def _build_contact_sheet(graph: SceneGraph, out_path: Path
                         ) -> Path | None:
    cards = [sc.visual_asset_paths[0] for sc in graph.scenes
             if sc.visual_asset_paths and Path(sc.visual_asset_paths[0]).exists()]
    if not cards:
        return None
    cols = 3
    rows = (len(cards) + cols - 1) // cols
    thumb_w, thumb_h = 480, 270
    margin = 16
    sheet_w = cols * thumb_w + (cols + 1) * margin
    sheet_h = rows * thumb_h + (rows + 1) * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), (255, 255, 255))
    for idx, card_path in enumerate(cards):
        try:
            im = Image.open(card_path).convert("RGB")
        except Exception:
            continue
        im.thumbnail((thumb_w, thumb_h))
        col = idx % cols
        row = idx // cols
        x = margin + col * (thumb_w + margin) + (thumb_w - im.width) // 2
        y = margin + row * (thumb_h + margin) + (thumb_h - im.height) // 2
        sheet.paste(im, (x, y))
    ensure_dir(out_path.parent)
    sheet.save(out_path, format="JPEG", quality=85)
    return out_path


def _build_storyboard(graph: SceneGraph, sources: ChunkedSources,
                       profile: DocProfile | None, plan_report: dict,
                       findings: list[dict], out_path: Path,
                       *, sketchbook: bool) -> Path:
    ensure_dir(out_path.parent)
    rows: list[str] = []
    for idx, sc in enumerate(graph.scenes, start=1):
        card_src = (sc.visual_asset_paths[0]
                    if sc.visual_asset_paths else None)
        card_html = ""
        if card_src and Path(card_src).exists():
            rel = Path(card_src).resolve()
            card_html = f'<img src="file://{rel}" loading="lazy" />'
        else:
            card_html = '<div class="missing">card missing</div>'
        evidence_html = "".join(
            f"<li><b>p.{ev.page}</b>："
            f"{html.escape((ev.quote or '')[:160])}"
            f"</li>"
            for ev in sc.evidence_spans
        )
        facts_html = "".join(
            f"<li>{html.escape(f.label)} = "
            f"<b>{html.escape(f.value)}</b></li>"
            for f in sc.facts
        )
        rows.append(f"""
<section class=\"scene\">
  <h3>#{idx} · {html.escape(sc.title)} <small>({html.escape(sc.scene_kind or sc.visual_type.value)})</small></h3>
  <div class=\"grid\">
    <div class=\"card\">{card_html}</div>
    <div class=\"meta\">
      <p class=\"narration\">{html.escape(sc.narration_text_zh_tw)}</p>
      <p class=\"src\">來源頁: {', '.join(f'p.{p}' for p in sc.source_pages)}</p>
      {('<h4>事實</h4><ul>' + facts_html + '</ul>') if facts_html else ''}
      {('<h4>來源摘錄</h4><ul>' + evidence_html + '</ul>') if evidence_html else ''}
    </div>
  </div>
</section>""")
    findings_html = "".join(
        f"<li class=\"sev-{html.escape(f['severity'])}\">"
        f"[{html.escape(f['severity'])}] "
        f"{html.escape(f['code'])}: {html.escape(f['message'])}"
        f"{' — ' + html.escape(f['scene_id']) if f.get('scene_id') else ''}"
        f"</li>"
        for f in findings
    ) or "<li>無發現</li>"
    profile_block = ""
    if profile is not None:
        profile_block = (
            f"<p><b>doc_kind</b>: {html.escape(profile.doc_kind.value)} "
            f"(信心 {profile.confidence:.2f})</p>"
            f"<p><small>{html.escape(profile.rationale)}</small></p>"
        )
    plan_block = ""
    if plan_report:
        plan_block = (
            "<pre>"
            + html.escape(json.dumps(plan_report, ensure_ascii=False,
                                      indent=2))
            + "</pre>"
        )
    doc_html = f"""<!doctype html>
<html lang=\"zh-TW\"><head><meta charset=\"utf-8\" />
<title>paperreel · storyboard 預覽</title>
<style>
body {{ font-family: 'Noto Sans CJK TC', sans-serif; max-width: 1200px;
        margin: 24px auto; padding: 0 16px; color: #0f172a; }}
header {{ border-bottom: 4px solid #d97706; padding-bottom: 12px; margin-bottom: 24px; }}
.scene {{ border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px;
          margin-bottom: 24px; background: #fafafa; }}
.grid {{ display: grid; grid-template-columns: 480px 1fr; gap: 24px; align-items: start; }}
.card img {{ width: 100%; border-radius: 8px; border: 1px solid #cbd5e1; }}
.missing {{ width: 100%; aspect-ratio: 16/9; background: #fee2e2;
            border-radius: 8px; display: flex; align-items: center;
            justify-content: center; color: #991b1b; }}
.meta p.narration {{ font-size: 18px; line-height: 1.7; }}
.meta p.src {{ color: #64748b; font-size: 13px; }}
.sev-error {{ color: #b91c1c; }}
.sev-warning {{ color: #b45309; }}
.sev-info {{ color: #1d4ed8; }}
ul {{ padding-left: 20px; }}
pre {{ background: #1e293b; color: #e2e8f0; padding: 16px;
       border-radius: 8px; overflow-x: auto; }}
</style></head>
<body>
<header>
  <h1>{html.escape(graph.project)} · 影片預覽 (storyboard)</h1>
  <p>style: <code>{'sketchbook' if sketchbook else 'default'}</code> ·
     scene 數: {len(graph.scenes)} ·
     目標分鐘: {graph.target_minutes:.1f}</p>
  {profile_block}
  {plan_block}
</header>
<h2>檢查結果</h2>
<ul>{findings_html}</ul>
<h2>逐 scene 預覽</h2>
{''.join(rows)}
</body></html>
"""
    atomic_write_text(out_path, doc_html)
    return out_path


def _run_vlm_review(graph: SceneGraph, vlm_cfg: dict) -> list[dict]:
    """Optional local-VLM scoring via Ollama. No-op if the package
    isn't importable or the model can't be reached. Each scene gets
    one short prompt + a brief judgement string."""
    try:
        import ollama  # type: ignore
    except ImportError:
        raise RuntimeError("ollama package not installed; cannot run VLM review")
    model = vlm_cfg.get("model")
    base_url = vlm_cfg.get("base_url", "http://localhost:11434")
    client = ollama.Client(host=base_url, timeout=120)
    out: list[dict] = []
    for sc in graph.scenes:
        if not sc.visual_asset_paths:
            continue
        prompt = (
            "請用繁體中文,30 字以內,評估這張教學卡片是否"
            "清楚、字夠大、且資訊與所附旁白一致。\n"
            f"旁白:{sc.narration_text_zh_tw[:120]}"
        )
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt,
                            "images": [sc.visual_asset_paths[0]]}],
                options={"temperature": 0.2, "num_predict": 80},
            )
            verdict = (resp.get("message") or {}).get("content", "").strip()
        except Exception as e:
            verdict = f"vlm error: {e!r}"
        out.append({"scene_id": sc.scene_id, "verdict": verdict[:300]})
    return out
