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


class PdfPage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    page: int                       # 1-indexed
    text: str
    cjk_char_count: int
    headings: list[str] = []


class PdfChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")
    chunk_id: str
    start_page: int
    end_page: int
    text: str
    cjk_char_count: int
    headings: list[str] = []


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
    visual_asset_paths: list[str] = []
    rendered_video_path: str | None = None
    status: SceneStatus = SceneStatus.pending
    input_hash: str
    retry_count: int = 0
    last_error: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
