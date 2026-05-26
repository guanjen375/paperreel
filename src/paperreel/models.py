"""Pydantic v2 data models — these are the contracts between stages.

Every intermediate artefact (chunked_sources.json, lesson_outline.json,
script.json, scene_graph.json, render_plan.json) is one of these models
serialised with `.model_dump_json()`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- enums ----------

class VisualType(str, Enum):
    title_card = "title_card"
    bullet_card = "bullet_card"
    diagram = "diagram"
    pdf_image = "pdf_image"
    generated_image = "generated_image"
    recap = "recap"
    quiz = "quiz"
    # Sketchbook / document_explainer kinds. Renderer dispatch happens
    # via Scene.scene_kind when style=sketchbook; we keep this enum
    # populated so existing scene_graph.json files that store the value
    # as visual_type still validate.
    sketchbook_card = "sketchbook_card"


# Document type tags — produced by the heuristic classifier in
# utils/doc_classify.py and stored in intermediate/doc_profile.json.
# Storyboard composition depends on this label; "unknown" falls back to
# the generic paper-style outline.
class DocKind(str, Enum):
    contract = "contract"
    form = "form"
    paper = "paper"
    slides = "slides"
    manual = "manual"
    report = "report"
    policy = "policy"
    unknown = "unknown"


# Sub-types of sketchbook scenes. Stored on Scene.scene_kind alongside
# the legacy visual_type so the existing render_visuals path still
# works for default-style runs.
SCENE_KINDS = (
    "cover", "section_intro", "deadline_timeline", "penalty_table",
    "checklist", "risk_warning", "do_dont", "recap_card",
    "paragraph_card", "source_crop", "key_number",
    "source_visual_explainer", "comparison_visual_card",
    "process_visual_card", "figure_explainer",
    "source_table_explainer", "source_screenshot_explainer",
)


class Importance(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class SceneStatus(str, Enum):
    pending = "pending"
    audio_done = "audio_done"
    visual_done = "visual_done"
    rendered = "rendered"
    failed = "failed"
    skipped = "skipped"


class StageName(str, Enum):
    ingest = "ingest"
    plan = "plan"
    script = "script"
    scenes = "scenes"
    match_visuals = "match_visuals"
    audio = "audio"
    visuals = "visuals"
    subtitles = "subtitles"
    segments = "segments"
    concat = "concat"
    quality = "quality"


# ---------- ingest ----------

class PdfImage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    image_id: str
    page: int
    path: str
    width: int
    height: int
    pixel_count: int
    sha256: str
    caption_hint: str | None = None
    # bbox = (x0, y0, x1, y1) in PDF points. Optional because some
    # embedded images can't be located on the page (rare; mostly forms
    # and inline icons via SMask). Used by match_pdf_visuals for
    # caption discovery and figure prioritisation.
    bbox: tuple[float, float, float, float] | None = None


PageTextSource = Literal["text", "ocr", "empty"]


class PdfPage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    page: int                       # 1-indexed
    text: str
    cjk_char_count: int
    headings: list[str] = []
    # How `text` was obtained — drives the quality report so a deck of
    # scanned pages doesn't masquerade as a clean digital PDF, and lets
    # downstream stages prefer figures over OCR'd text where appropriate.
    text_source: PageTextSource = "text"
    width: float | None = None
    height: float | None = None
    text_area_ratio: float | None = None
    image_area_ratio: float | None = None


class PdfChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chunk_id: str
    start_page: int
    end_page: int
    text: str
    cjk_char_count: int
    headings: list[str] = []


class VisualCandidate(BaseModel):
    """One source visual candidate discovered during ingest.

    Candidates are deliberately source-only: extracted PDF images,
    page renders, or crops derived from the PDF. Generated images never
    enter this inventory.
    """
    model_config = ConfigDict(extra="ignore")
    candidate_id: str
    page: int
    image_path: str | None = None
    page_render_path: str | None = None
    crop_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    nearby_heading: str | None = None
    nearby_caption: str | None = None
    nearby_text: str | None = None
    image_width: int | None = None
    image_height: int | None = None
    image_size: tuple[int, int] | None = None
    page_area_ratio: float | None = None
    visual_role: str = "unknown"
    salience_score: float = 0.0
    is_decorative: bool = False
    likely_useful: bool = False
    repeated: bool = False
    source_image_id: str | None = None
    source_quote: str | None = None


class ChunkedSources(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source_pdf: str
    pdf_sha256: str
    page_count: int
    cjk_char_count: int
    image_count: int
    heading_count: int
    estimated_density: float            # cjk chars per page
    pages: list[PdfPage]
    chunks: list[PdfChunk]
    images: list[PdfImage] = []
    visual_inventory: list[VisualCandidate] = []


# ---------- document profile ----------

class DocProfile(BaseModel):
    """Heuristic + optional LLM document classification.

    Written by the document classifier stage; consumed by the script
    writer to bias storyboard composition (contract -> timeline +
    penalty table, paper -> problem/method/results, etc.).
    """
    model_config = ConfigDict(extra="ignore")
    doc_kind: DocKind = DocKind.unknown
    confidence: float = 0.0
    rationale: str = ""
    keyword_hits: dict[str, int] = {}
    # structural_hits is a mix of densities (floats like
    # avg_cjk_per_page) and integer counts; store as float so neither
    # side coerces away precision.
    structural_hits: dict[str, float] = {}
    suggested_storyboard: list[str] = []
    document_visual_rich: bool = False
    visual_tutorial: bool = False
    visual_rich_score: float = 0.0
    source_visuals_available: int = 0


class VisualAnchor(BaseModel):
    model_config = ConfigDict(extra="ignore")
    page: int
    image_path: str | None = None
    page_render_path: str | None = None
    crop_path: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    visual_role: str | None = None
    caption: str | None = None
    source_quote: str | None = None
    nearby_heading: str | None = None
    why_this_visual: str | None = None


class ScreenPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    headline: str | None = None
    callouts: list[str] = []
    labels: list[str] = []
    highlight_regions: list[dict[str, Any]] = []
    max_screen_text: int | None = None
    layout_hint: str | None = None


# ---------- outline / plan ----------

class ChapterPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chapter_id: str
    title: str
    source_pages: list[int]
    target_minutes: float
    key_points: list[str] = []
    recap: bool = False


class LessonOutline(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project: str
    language: str = "zh-TW"
    target_minutes: float
    rationale: str
    chapters: list[ChapterPlan]


# ---------- evidence / facts ----------

class EvidenceSpan(BaseModel):
    """One source-grounded quote pulled from a specific PDF page.

    Used by the sketchbook validator to ensure every factual claim
    has on-disk provenance: page is required and must exist in the
    ingested ChunkedSources, and quote must appear (normalised) in
    that page's extracted text.
    """
    model_config = ConfigDict(extra="ignore")
    page: int
    quote: str
    label: str | None = None
    value: str | None = None
    importance: str | None = None


class Fact(BaseModel):
    """A structured fact extracted from the source.

    The label/value pair is what the deterministic renderer puts on
    screen. Numbers / dates / fees / percentages must never be invented
    — they are pulled by the heuristic extractor or quoted by the LLM
    from evidence spans.
    """
    model_config = ConfigDict(extra="ignore")
    label: str
    value: str
    importance: str | None = None
    evidence_index: int | None = None  # index into Scene.evidence_spans


# ---------- script ----------

class ScriptScene(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene_id: str
    chapter_id: str
    title: str
    source_pages: list[int]
    source_refs: list[str] = []
    narration_text_zh_tw: str
    on_screen_text: str | None = None
    visual_hint: str | None = None
    visual_type: VisualType = VisualType.bullet_card
    estimated_duration_sec: float
    # Sketchbook / document_explainer additions — optional so default
    # mode keeps working unchanged. When style=sketchbook, the script
    # writer populates these and the validator enforces them.
    scene_kind: str | None = None
    facts: list[Fact] = []
    evidence_spans: list[EvidenceSpan] = []
    layout_payload: dict[str, Any] = {}
    importance: str | None = None
    visual_anchor: VisualAnchor | None = None
    screen_plan: ScreenPlan | None = None


class Script(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project: str
    total_estimated_minutes: float
    scenes: list[ScriptScene]


# ---------- scene graph ----------

class Scene(BaseModel):
    """Core scene model — survives across TTS / renderer / subtitle changes."""
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    scene_id: str
    chapter_id: str
    title: str
    source_pages: list[int]
    source_refs: list[str] = []
    narration_text_zh_tw: str
    visual_prompt: str = ""
    visual_type: VisualType
    on_screen_text: str | None = None
    estimated_duration_sec: float
    actual_duration_sec: float | None = None
    audio_path: str | None = None
    subtitle_path: str | None = None
    # `visual_source_paths` is the *input* the renderer consumes: an
    # extracted PDF figure crop, a freshly generated SDXL image, or a
    # manually-attached asset. `visual_asset_paths` is the renderer's
    # *output*: the final card that segments/quality consume.
    # Keeping them separate lets `match_visuals` + `render_visuals`
    # be idempotent across resumes — otherwise a second render reads
    # its own previous output and embeds the card inside a new card.
    visual_source_paths: list[str] = []
    visual_asset_paths: list[str] = []
    rendered_video_path: str | None = None
    status: SceneStatus = SceneStatus.pending
    input_hash: str
    retry_count: int = 0
    last_error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Sketchbook / document_explainer additions — optional so default
    # scene_graph.json files (no scene_kind) keep validating. Renderer
    # dispatches on scene_kind first when style=sketchbook, falling
    # back to visual_type otherwise.
    scene_kind: str | None = None
    facts: list[Fact] = []
    evidence_spans: list[EvidenceSpan] = []
    layout_payload: dict[str, Any] = {}
    importance: str | None = None
    visual_anchor: VisualAnchor | None = None
    screen_plan: ScreenPlan | None = None


class SceneGraph(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project: str
    target_minutes: float
    scenes: list[Scene]


# ---------- render plan ----------

class RenderPlanEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    scene_id: str
    visual_type: VisualType
    visual_asset_paths: list[str]
    audio_path: str
    subtitle_path: str | None
    output_path: str
    duration_sec: float


class RenderPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project: str
    fps: int
    resolution: tuple[int, int]
    final_output: str
    entries: list[RenderPlanEntry]


# ---------- quality report ----------

class QualityIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    scene_id: str | None = None


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    project: str
    target_minutes: float
    final_duration_sec: float
    final_video_path: str | None
    scene_count: int
    rendered_scene_count: int
    failed_scene_ids: list[str] = []
    issues: list[QualityIssue] = []
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------- projects ----------

class ProjectMeta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    root: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_pdf: str | None = None
    config_overlay: str | None = None
