"""Stage 7 — generate SRT (+ ASS) per scene + a single concatenated full SRT.

We don't yet do word-level alignment; cues are evenly distributed across the
scene's actual audio duration with line breaks at the configured width.
"""
from __future__ import annotations

import math
from pathlib import Path

from ..hashing import hash_inputs
from ..io_utils import atomic_write_text, read_json
from ..models import Scene, SceneGraph, SceneStatus
from ..state import StateDB


def paths_for(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    return {
        "scene_graph": root / "intermediate" / "scene_graph.json",
        "subs_dir": root / "assets" / "subtitles",
        "full_srt": root / "outputs" / "subtitles.srt",
    }


def _split_for_cues(text: str, *, max_chars: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for ch in text:
        cur += ch
        if ch in "。．.!?！？\n" or len(cur) >= max_chars:
            cur = cur.strip()
            if cur:
                out.append(cur)
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def _srt_ts(t: float) -> str:
    if t < 0:
        t = 0.0
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int(round((t - math.floor(t)) * 1000))
    if ms == 1000:
        s += 1; ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_ts(t: float) -> str:
    if t < 0: t = 0.0
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def _write_srt(cues: list[tuple[float, float, str]], path: Path) -> None:
    lines: list[str] = []
    for idx, (start, end, txt) in enumerate(cues, 1):
        lines.append(str(idx))
        lines.append(f"{_srt_ts(start)} --> {_srt_ts(end)}")
        lines.append(txt)
        lines.append("")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _write_ass(cues: list[tuple[float, float, str]], path: Path) -> None:
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Noto Sans CJK TC,46,&H00FFFFFF,&H000000FF,&H00101820,-1,0,0,0,100,100,0,0,"
        "1,2,1,2,80,80,60,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    events = [
        f"Dialogue: 0,{_ass_ts(s)},{_ass_ts(e)},Default,,0,0,0,,{t.replace(chr(10), '\\N')}"
        for s, e, t in cues
    ]
    atomic_write_text(path, header + "\n".join(events) + "\n")


def run(*, project_root: str | Path, db: StateDB, config: dict) -> SceneGraph:
    p = paths_for(project_root)
    graph = SceneGraph.model_validate(read_json(p["scene_graph"]))
    sub_cfg = config.get("subtitles", {})
    formats = [s.lower() for s in sub_cfg.get("formats", ["srt", "ass"])]
    max_chars = int(sub_cfg.get("max_chars_per_line", 22))

    input_hash = hash_inputs("subs_v1", formats, max_chars,
                             [s.scene_id for s in graph.scenes])
    db.start_stage("subtitles", input_hash)

    full_cues: list[tuple[float, float, str]] = []
    global_offset = 0.0
    new_scenes: list[Scene] = []
    for sc in graph.scenes:
        if sc.status == SceneStatus.failed or not sc.audio_path:
            new_scenes.append(sc)
            continue
        duration = sc.actual_duration_sec or sc.estimated_duration_sec
        cues_text = _split_for_cues(sc.narration_text_zh_tw, max_chars=max_chars)
        cues_text = cues_text or [sc.narration_text_zh_tw]
        slot = duration / max(1, len(cues_text))
        local_cues: list[tuple[float, float, str]] = []
        for i, t in enumerate(cues_text):
            s = i * slot
            e = (i + 1) * slot
            local_cues.append((s, e, t))
            full_cues.append((global_offset + s, global_offset + e, t))
        global_offset += duration

        srt_path = p["subs_dir"] / f"{sc.scene_id}.srt"
        if "srt" in formats:
            _write_srt(local_cues, srt_path)
            db.register_artifact(srt_path, stage="subtitles",
                                 scene_id=sc.scene_id, media_type="text/srt")
        if "ass" in formats:
            ass_path = p["subs_dir"] / f"{sc.scene_id}.ass"
            _write_ass(local_cues, ass_path)
            db.register_artifact(ass_path, stage="subtitles",
                                 scene_id=sc.scene_id, media_type="text/ass")
        sc = sc.model_copy(update={"subtitle_path": str(srt_path) if "srt" in formats else None})
        db.upsert_scene(sc.scene_id, sc.chapter_id, sc.status.value,
                        sc.input_hash, sc.model_dump(mode="json"))
        new_scenes.append(sc)

    if "srt" in formats:
        _write_srt(full_cues, p["full_srt"])
        db.register_artifact(p["full_srt"], stage="subtitles", media_type="text/srt")

    graph = graph.model_copy(update={"scenes": new_scenes})
    from ..io_utils import atomic_write_json
    atomic_write_json(p["scene_graph"], graph.model_dump(mode="json"))
    db.finish_stage("subtitles", [str(p["full_srt"])])
    return graph
