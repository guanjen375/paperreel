"""Evidence / source-grounding validator.

Used by sketchbook / document_explainer mode to verify that every
factual scene actually traces back to the ingested PDF text. Failures
get one repair attempt by the script writer; if that still doesn't pass
we raise a ``GroundingError`` rather than silently emit a video full of
fabricated numbers.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ..models import Fact, ScriptScene


class GroundingError(ValueError):
    """Raised when sketchbook validation finds unfixable evidence
    problems. Carries a list of individual issues so the caller can
    log a single clear error message."""

    def __init__(self, issues: list["GroundingIssue"]):
        self.issues = issues
        super().__init__(
            f"{len(issues)} grounding issue(s); first: {issues[0].message}"
            if issues else "grounding failed (no issues attached)"
        )


@dataclass
class GroundingIssue:
    scene_id: str
    code: str
    message: str
    page: int | None = None


_KINDS_REQUIRING_EVIDENCE = {
    "deadline_timeline", "penalty_table", "checklist", "risk_warning",
    "do_dont", "key_number", "source_crop",
}
_FACTUAL_LABEL_KEYWORDS = ("期限", "百分比", "金額", "Deadline", "日期", "數量")


def _normalize(text: str) -> str:
    """NFC + strip whitespace/full-width punctuation noise so quote
    matching survives the lossy round-trip through the LLM."""
    text = unicodedata.normalize("NFC", text)
    # Drop all whitespace (including newlines) — the extracted PDF text
    # often inserts line breaks mid-sentence and the LLM smooths them.
    text = re.sub(r"\s+", "", text)
    # Normalize lookalike punctuation that the LLM tends to swap.
    swap = str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "，": ",", "。": ".", "！": "!", "？": "?",
        "：": ":", "；": ";", "／": "/", "（": "(", "）": ")",
    })
    return text.translate(swap)


def quote_match_ratio(quote: str, page_text: str) -> float:
    """Return the largest contiguous fraction of ``quote`` that appears
    verbatim (post-normalization) in ``page_text``.

    Slides a window down from the full quote length until we either
    find a match or fall below 6 chars. Cheap, deterministic, and
    forgiving of single-char OCR slips because the window doesn't have
    to start at the same position.
    """
    if not quote or not page_text:
        return 0.0
    q = _normalize(quote)
    p = _normalize(page_text)
    if not q:
        return 0.0
    if q in p:
        return 1.0
    # Try progressively shorter windows so an LLM rewrite that drops a
    # filler char still scores high. We only need the *largest* win, so
    # binary-search-ish stepping is enough.
    n = len(q)
    for size in (n, int(n * 0.9), int(n * 0.75), int(n * 0.6),
                 int(n * 0.45), int(n * 0.3)):
        if size < 6:
            break
        for i in range(0, n - size + 1, max(1, size // 4)):
            if q[i: i + size] in p:
                return size / n
    return 0.0


def fact_is_factual(fact: Fact) -> bool:
    """True for facts the validator should treat as load-bearing."""
    if not fact.label:
        return False
    if any(kw in fact.label for kw in _FACTUAL_LABEL_KEYWORDS):
        return True
    if fact.importance in ("high", "medium"):
        return True
    return False


def validate_scene(scene: ScriptScene, *,
                   page_text: dict[int, str],
                   min_quote_ratio: float,
                   require_evidence_for_facts: bool) -> list[GroundingIssue]:
    """Validate one scene's evidence + facts. Returns an empty list
    when the scene is OK."""
    issues: list[GroundingIssue] = []
    kind = (scene.scene_kind or "").lower()

    # 1) Every non-cover sketchbook scene must trace back to a source page,
    #    and cited source pages must exist in the ingested PDF.
    if kind and kind not in ("cover", "recap_card") and not scene.source_pages:
        issues.append(GroundingIssue(
            scene_id=scene.scene_id,
            code="no_source_pages",
            message=f"scene {scene.scene_id!r} has no source_pages",
        ))
    if kind not in ("cover", "recap_card"):
        for page in scene.source_pages or []:
            if page not in page_text:
                issues.append(GroundingIssue(
                    scene_id=scene.scene_id,
                    code="bad_source_page",
                    message=f"scene cites source page {page} which is not in the ingested PDF",
                    page=page,
                ))

    # 2) Factual scene_kinds must carry at least one evidence span.
    needs_evidence = kind in _KINDS_REQUIRING_EVIDENCE
    if needs_evidence and not scene.evidence_spans:
        issues.append(GroundingIssue(
            scene_id=scene.scene_id,
            code="missing_evidence",
            message=(
                f"scene {scene.scene_id!r} (kind={kind}) requires at "
                "least one evidence_span"
            ),
        ))

    # 3) Each evidence quote must point at a real page AND substantially
    #    match that page's text (we forgive minor OCR / paraphrase
    #    slippage via the ratio threshold).
    for idx, span in enumerate(scene.evidence_spans):
        if span.page not in page_text:
            issues.append(GroundingIssue(
                scene_id=scene.scene_id,
                code="bad_page",
                message=(
                    f"evidence[{idx}] cites page {span.page} which is "
                    "not in the ingested PDF"
                ),
                page=span.page,
            ))
            continue
        ratio = quote_match_ratio(span.quote, page_text[span.page])
        if ratio < min_quote_ratio:
            issues.append(GroundingIssue(
                scene_id=scene.scene_id,
                code="quote_mismatch",
                message=(
                    f"evidence[{idx}] quote does not match page "
                    f"{span.page} (ratio={ratio:.2f} < {min_quote_ratio:.2f}): "
                    f"{span.quote[:60]!r}"
                ),
                page=span.page,
            ))

    # 4) Any *factual* Fact must reference an evidence span by index.
    if require_evidence_for_facts:
        for f_idx, fact in enumerate(scene.facts):
            if not fact_is_factual(fact):
                continue
            if fact.evidence_index is None:
                # Implicit pass when ANY span exists — useful for facts
                # the LLM forgot to link explicitly but at least came
                # from a verified source.
                if scene.evidence_spans:
                    continue
                issues.append(GroundingIssue(
                    scene_id=scene.scene_id,
                    code="fact_without_evidence",
                    message=(
                        f"factual fact[{f_idx}] (label={fact.label}, "
                        f"value={fact.value}) has no evidence_index "
                        "and no evidence_spans"
                    ),
                ))
                continue
            if not (0 <= fact.evidence_index < len(scene.evidence_spans)):
                issues.append(GroundingIssue(
                    scene_id=scene.scene_id,
                    code="evidence_index_out_of_range",
                    message=(
                        f"fact[{f_idx}].evidence_index="
                        f"{fact.evidence_index} but only "
                        f"{len(scene.evidence_spans)} spans exist"
                    ),
                ))

    return issues


def validate_scenes(scenes: list[ScriptScene], *,
                    page_text: dict[int, str],
                    min_quote_ratio: float = 0.55,
                    require_evidence_for_facts: bool = True
                    ) -> list[GroundingIssue]:
    out: list[GroundingIssue] = []
    for sc in scenes:
        out.extend(validate_scene(
            sc, page_text=page_text,
            min_quote_ratio=min_quote_ratio,
            require_evidence_for_facts=require_evidence_for_facts,
        ))
    return out
