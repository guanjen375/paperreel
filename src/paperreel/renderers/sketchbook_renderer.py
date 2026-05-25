"""Deterministic Pillow renderer for sketchbook / document_explainer mode.

Every layout here is generated from structured Scene.layout_payload data
— no LLM call at render time, no SDXL, no random generated images.
Cards are designed for 1080p / 720p and prefer large, readable
Traditional Chinese text with a Noto CJK fallback chain.

Card kinds:
  cover, section_intro, deadline_timeline, penalty_table, checklist,
  risk_warning, do_dont, recap_card, paragraph_card, source_crop,
  key_number.

The renderer is intentionally simple — fade/zoom motion is handled by
ffmpeg later. We just need readable static frames.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from ..io_utils import ensure_dir
from ..models import Scene


_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Microsoft/PMingLiU.ttf",
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
]


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(_FONT_CANDIDATES)
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def _wrap(text: str, *, max_chars: int) -> list[str]:
    """Character-based wrap that works for CJK and ASCII alike."""
    if not text:
        return []
    return textwrap.wrap(
        text, width=max_chars,
        break_long_words=True, break_on_hyphens=False,
    ) or [text]


def _text_width(draw: ImageDraw.ImageDraw, text: str,
                font: ImageFont.ImageFont) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _darker(c: tuple[int, int, int], k: float = 0.55
            ) -> tuple[int, int, int]:
    return (max(0, int(c[0] * k)), max(0, int(c[1] * k)), max(0, int(c[2] * k)))


def _lighter(c: tuple[int, int, int], k: float = 0.18
             ) -> tuple[int, int, int]:
    return (
        min(255, int(c[0] + (255 - c[0]) * k)),
        min(255, int(c[1] + (255 - c[1]) * k)),
        min(255, int(c[2] + (255 - c[2]) * k)),
    )


class SketchbookRenderer:
    """Pillow-based sketchbook card renderer.

    All public layout entry points take a :class:`Scene` and return the
    output path. Layout data is expected on ``scene.layout_payload``.
    """

    def __init__(self, *,
                 resolution: tuple[int, int] = (1920, 1080),
                 background: str = "#F8FAFB",
                 foreground: str = "#0F172A",
                 accent: str = "#D97706",
                 font_path: str | None = None,
                 cards_cfg: dict | None = None):
        self.w, self.h = resolution
        self.bg = _hex_to_rgb(background)
        self.fg = _hex_to_rgb(foreground)
        self.accent = _hex_to_rgb(accent)
        self.font_path = font_path
        self.cards = {
            "max_chars_title": 18,
            "max_chars_subtitle": 28,
            "max_chars_paragraph": 90,
            "max_chars_per_bullet": 24,
            "max_bullets": 6,
            "max_timeline_events": 6,
            "max_table_rows": 6,
            "max_checklist_items": 6,
        }
        if cards_cfg:
            self.cards.update({k: v for k, v in cards_cfg.items() if v is not None})

    # ---------- public ----------

    def render(self, scene: Scene, out_path: str | Path) -> str:
        out = Path(out_path)
        ensure_dir(out.parent)
        kind = (scene.scene_kind or "").lower() or "paragraph_card"
        payload = scene.layout_payload or {}
        try:
            img = self._dispatch(kind, scene, payload)
        except Exception:
            # Last-line-of-defence fallback: a paragraph card should
            # always render even if a particular kind is broken on
            # this input.
            img = self._paragraph_card(scene.title, payload, scene)
        self._draw_footer(img, self._footer(scene))
        img.save(out)
        return str(out)

    def _dispatch(self, kind: str, scene: Scene, payload: dict) -> Image.Image:
        if kind == "cover":
            return self._cover(scene.title, payload, scene)
        if kind == "section_intro":
            return self._section_intro(scene.title, payload, scene)
        if kind == "deadline_timeline":
            return self._deadline_timeline(scene.title, payload)
        if kind == "penalty_table":
            return self._penalty_table(scene.title, payload)
        if kind == "checklist":
            return self._checklist(scene.title, payload)
        if kind == "risk_warning":
            return self._risk_warning(scene.title, payload)
        if kind == "do_dont":
            return self._do_dont(scene.title, payload)
        if kind == "recap_card":
            return self._recap_card(scene.title, payload, scene)
        if kind == "source_crop":
            return self._source_crop(scene.title, payload, scene)
        if kind == "key_number":
            return self._key_number(scene.title, payload, scene)
        return self._paragraph_card(scene.title, payload, scene)

    # ---------- card kinds ----------

    def _base(self) -> Image.Image:
        return Image.new("RGB", (self.w, self.h), self.bg)

    def _cover(self, title: str, payload: dict, scene: Scene) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        # Soft top band so the cover doesn't feel like a blank page.
        band = _lighter(self.accent, 0.78)
        draw.rectangle([(0, 0), (self.w, int(self.h * 0.32))], fill=band)

        eyebrow = payload.get("eyebrow") or payload.get("doc_kind") or "說明影片"
        subtitle = payload.get("subtitle") or scene.on_screen_text or ""
        eyebrow_font = _load_font(self.font_path, max(28, self.h // 30))
        title_font = _load_font(self.font_path, max(72, self.h // 9))
        subtitle_font = _load_font(self.font_path, max(36, self.h // 22))

        margin = 96
        y = int(self.h * 0.14)
        draw.text((margin, y), str(eyebrow)[: self.cards["max_chars_subtitle"]],
                  fill=_darker(self.accent), font=eyebrow_font)
        y += eyebrow_font.size + 28

        wrapped_title = _wrap(title, max_chars=self.cards["max_chars_title"])
        for line in wrapped_title[:3]:
            draw.text((margin, y), line, fill=self.fg, font=title_font)
            y += title_font.size + 22

        # Accent underline.
        ul_y = y + 14
        draw.rectangle(
            [(margin, ul_y), (margin + 180, ul_y + 10)],
            fill=self.accent,
        )
        y = ul_y + 44

        if subtitle:
            for line in _wrap(subtitle, max_chars=self.cards["max_chars_subtitle"])[:3]:
                draw.text((margin, y), line, fill=_darker(self.fg, 0.7),
                          font=subtitle_font)
                y += subtitle_font.size + 18

        return img

    def _section_intro(self, title: str, payload: dict, scene: Scene
                       ) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        margin = 96

        number = payload.get("number") or payload.get("section_no") or ""
        body = (payload.get("body") or scene.on_screen_text
                or scene.narration_text_zh_tw or "")

        eyebrow_font = _load_font(self.font_path, max(28, self.h // 28))
        big_no_font = _load_font(self.font_path, max(160, self.h // 5))
        title_font = _load_font(self.font_path, max(56, self.h // 12))
        body_font = _load_font(self.font_path, max(30, self.h // 26))

        if number:
            draw.text((margin, 90), "章節 · SECTION",
                      fill=_darker(self.accent), font=eyebrow_font)
            draw.text((margin, 140), str(number),
                      fill=self.accent, font=big_no_font)
            y = 140 + big_no_font.size + 20
        else:
            draw.text((margin, 90), "章節 · SECTION",
                      fill=_darker(self.accent), font=eyebrow_font)
            y = 90 + eyebrow_font.size + 20

        for line in _wrap(title, max_chars=self.cards["max_chars_title"])[:2]:
            draw.text((margin, y), line, fill=self.fg, font=title_font)
            y += title_font.size + 14
        y += 18

        if body:
            for line in _wrap(body, max_chars=self.cards["max_chars_subtitle"])[:4]:
                draw.text((margin, y), line, fill=_darker(self.fg, 0.7),
                          font=body_font)
                y += body_font.size + 14
        return img

    def _deadline_timeline(self, title: str, payload: dict) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        events = payload.get("events") or []
        events = list(events)[: self.cards["max_timeline_events"]]

        title_font = _load_font(self.font_path, max(48, self.h // 16))
        value_font = _load_font(self.font_path, max(40, self.h // 20))
        label_font = _load_font(self.font_path, max(26, self.h // 30))
        margin = 96

        draw.text((margin, 80), title, fill=self.fg, font=title_font)
        draw.line(
            [(margin, 80 + title_font.size + 14),
             (margin + 260, 80 + title_font.size + 14)],
            fill=self.accent, width=8,
        )

        if not events:
            self._placeholder(draw, "（沒有偵測到時程資料）",
                              start_y=200, margin=margin)
            return img

        track_y = int(self.h * 0.62)
        usable_w = self.w - margin * 2
        gap = usable_w // max(1, len(events))
        x = margin + gap // 2

        # Horizontal timeline track.
        draw.line([(margin, track_y), (self.w - margin, track_y)],
                  fill=_darker(self.accent), width=6)

        for ev in events:
            value = str(ev.get("value", ""))[:16]
            label = str(ev.get("label", ""))[:40]
            page = ev.get("page")
            # Node circle.
            r = 22
            draw.ellipse(
                [(x - r, track_y - r), (x + r, track_y + r)],
                fill=self.accent, outline=_darker(self.accent), width=4,
            )
            # Value above the node.
            vw = _text_width(draw, value, value_font)
            draw.text(
                (x - vw // 2, track_y - r - value_font.size - 18),
                value, fill=self.fg, font=value_font,
            )
            # Label below the node (wrap to two lines if needed).
            for li, line in enumerate(_wrap(label, max_chars=12)[:3]):
                lw = _text_width(draw, line, label_font)
                draw.text(
                    (x - lw // 2,
                     track_y + r + 14 + li * (label_font.size + 8)),
                    line, fill=_darker(self.fg, 0.65), font=label_font,
                )
            if page is not None:
                p_text = f"p.{page}"
                pw = _text_width(draw, p_text, label_font)
                draw.text(
                    (x - pw // 2,
                     track_y + r + 14 + 3 * (label_font.size + 8) + 6),
                    p_text, fill=_darker(self.accent), font=label_font,
                )
            x += gap
        return img

    def _penalty_table(self, title: str, payload: dict) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        rows = list(payload.get("rows") or [])[: self.cards["max_table_rows"]]
        title_font = _load_font(self.font_path, max(48, self.h // 16))
        header_font = _load_font(self.font_path, max(30, self.h // 28))
        cell_font = _load_font(self.font_path, max(28, self.h // 30))
        margin = 96

        draw.text((margin, 80), title, fill=self.fg, font=title_font)
        draw.line(
            [(margin, 80 + title_font.size + 14),
             (margin + 260, 80 + title_font.size + 14)],
            fill=self.accent, width=8,
        )

        if not rows:
            self._placeholder(draw, "（沒有偵測到罰則／費用資料）",
                              start_y=200, margin=margin)
            return img

        col_value_w = int((self.w - margin * 2) * 0.28)
        col_cond_w = (self.w - margin * 2) - col_value_w
        y = 220

        # Header strip.
        draw.rectangle(
            [(margin, y), (self.w - margin, y + header_font.size + 32)],
            fill=_lighter(self.accent, 0.85),
        )
        draw.text((margin + 24, y + 16), "情境 / 條件",
                  fill=self.fg, font=header_font)
        draw.text((margin + col_cond_w + 24, y + 16), "費用 / 比例",
                  fill=_darker(self.accent), font=header_font)
        y += header_font.size + 36

        for i, row in enumerate(rows):
            zebra = _lighter(self.bg, 0.04) if i % 2 == 0 else self.bg
            row_h = cell_font.size * 3 + 24
            draw.rectangle(
                [(margin, y), (self.w - margin, y + row_h)],
                fill=zebra,
            )
            cond_text = str(row.get("condition", ""))[:60]
            for li, line in enumerate(
                _wrap(cond_text, max_chars=24)[:2]
            ):
                draw.text(
                    (margin + 24, y + 12 + li * (cell_font.size + 4)),
                    line, fill=self.fg, font=cell_font,
                )
            value_text = str(row.get("value", ""))[:18]
            draw.text(
                (margin + col_cond_w + 24, y + 12),
                value_text, fill=_darker(self.accent), font=cell_font,
            )
            page = row.get("page")
            if page is not None:
                draw.text(
                    (margin + col_cond_w + 24, y + 12 + cell_font.size + 6),
                    f"p.{page}", fill=_darker(self.fg, 0.55),
                    font=cell_font,
                )
            y += row_h + 6
        return img

    def _checklist(self, title: str, payload: dict) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        items = list(payload.get("items") or [])[: self.cards["max_checklist_items"]]
        title_font = _load_font(self.font_path, max(48, self.h // 16))
        item_font = _load_font(self.font_path, max(32, self.h // 26))
        margin = 96

        draw.text((margin, 80), title, fill=self.fg, font=title_font)
        draw.line(
            [(margin, 80 + title_font.size + 14),
             (margin + 260, 80 + title_font.size + 14)],
            fill=self.accent, width=8,
        )

        if not items:
            self._placeholder(draw, "（沒有偵測到應辦事項）",
                              start_y=200, margin=margin)
            return img

        y = 220
        for item in items:
            text = str(item.get("text", "")) if isinstance(item, dict) else str(item)
            page = item.get("page") if isinstance(item, dict) else None
            box_size = item_font.size + 12
            draw.rectangle(
                [(margin, y), (margin + box_size, y + box_size)],
                outline=self.accent, width=5,
            )
            draw.line([(margin + 10, y + box_size // 2 + 4),
                       (margin + box_size // 2, y + box_size - 8),
                       (margin + box_size - 8, y + 10)],
                      fill=self.accent, width=6)
            x = margin + box_size + 26
            lines = _wrap(text, max_chars=self.cards["max_chars_per_bullet"])[:2]
            for li, line in enumerate(lines):
                draw.text((x, y + li * (item_font.size + 6)),
                          line, fill=self.fg, font=item_font)
            if page is not None:
                draw.text(
                    (self.w - margin - 120, y),
                    f"p.{page}", fill=_darker(self.fg, 0.55), font=item_font,
                )
            y += box_size + 24
            if y > self.h - 140:
                break
        return img

    def _risk_warning(self, title: str, payload: dict) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        items = list(payload.get("items") or [])[: self.cards["max_bullets"]]
        title_font = _load_font(self.font_path, max(48, self.h // 16))
        item_font = _load_font(self.font_path, max(30, self.h // 28))
        margin = 96

        # Left red stripe + warning eyebrow.
        warn = (200, 60, 60)
        draw.rectangle([(0, 0), (24, self.h)], fill=warn)
        eyebrow_font = _load_font(self.font_path, max(28, self.h // 30))
        draw.text((margin, 64), "⚠ 風險 · 注意",
                  fill=warn, font=eyebrow_font)
        draw.text((margin, 110), title, fill=self.fg, font=title_font)
        y = 110 + title_font.size + 40

        if not items:
            self._placeholder(draw, "（沒有偵測到風險條款）",
                              start_y=y, margin=margin)
            return img

        for it in items:
            text = it.get("text", "") if isinstance(it, dict) else str(it)
            page = it.get("page") if isinstance(it, dict) else None
            draw.rectangle(
                [(margin - 12, y - 4),
                 (self.w - margin, y + item_font.size * 2 + 12)],
                fill=_lighter((255, 220, 220), 0.05),
                outline=warn, width=3,
            )
            lines = _wrap(text, max_chars=32)[:2]
            for li, line in enumerate(lines):
                draw.text(
                    (margin + 8, y + li * (item_font.size + 6)),
                    line, fill=self.fg, font=item_font,
                )
            if page is not None:
                draw.text(
                    (self.w - margin - 120, y),
                    f"p.{page}", fill=warn, font=item_font,
                )
            y += item_font.size * 2 + 36
            if y > self.h - 140:
                break
        return img

    def _do_dont(self, title: str, payload: dict) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(48, self.h // 16))
        col_title_font = _load_font(self.font_path, max(40, self.h // 22))
        item_font = _load_font(self.font_path, max(28, self.h // 30))
        margin = 96

        draw.text((margin, 80), title, fill=self.fg, font=title_font)
        draw.line(
            [(margin, 80 + title_font.size + 14),
             (margin + 260, 80 + title_font.size + 14)],
            fill=self.accent, width=8,
        )

        do_items = list(payload.get("do") or [])[: self.cards["max_bullets"]]
        dont_items = list(payload.get("dont") or [])[: self.cards["max_bullets"]]

        col_w = (self.w - margin * 3) // 2
        top_y = 220
        do_color = (32, 150, 96)
        dont_color = (200, 60, 60)

        # DO column.
        draw.rectangle(
            [(margin, top_y), (margin + col_w, self.h - 120)],
            outline=do_color, width=4,
        )
        draw.text((margin + 24, top_y + 18), "✓ 應該做",
                  fill=do_color, font=col_title_font)
        y = top_y + col_title_font.size + 50
        for it in do_items:
            text = it.get("text", "") if isinstance(it, dict) else str(it)
            for li, line in enumerate(_wrap(text, max_chars=18)[:2]):
                draw.text(
                    (margin + 24, y + li * (item_font.size + 4)),
                    "• " + line if li == 0 else "  " + line,
                    fill=self.fg, font=item_font,
                )
            y += item_font.size * 2 + 22
            if y > self.h - 160:
                break

        # DON'T column.
        dx = margin * 2 + col_w
        draw.rectangle(
            [(dx, top_y), (dx + col_w, self.h - 120)],
            outline=dont_color, width=4,
        )
        draw.text((dx + 24, top_y + 18), "✗ 不要做",
                  fill=dont_color, font=col_title_font)
        y = top_y + col_title_font.size + 50
        for it in dont_items:
            text = it.get("text", "") if isinstance(it, dict) else str(it)
            for li, line in enumerate(_wrap(text, max_chars=18)[:2]):
                draw.text(
                    (dx + 24, y + li * (item_font.size + 4)),
                    "× " + line if li == 0 else "  " + line,
                    fill=self.fg, font=item_font,
                )
            y += item_font.size * 2 + 22
            if y > self.h - 160:
                break
        return img

    def _recap_card(self, title: str, payload: dict, scene: Scene
                    ) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        margin = 96
        eyebrow_font = _load_font(self.font_path, max(28, self.h // 30))
        title_font = _load_font(self.font_path, max(56, self.h // 14))
        item_font = _load_font(self.font_path, max(32, self.h // 26))

        draw.text((margin, 80), "重點回顧 · RECAP",
                  fill=_darker(self.accent), font=eyebrow_font)
        y = 80 + eyebrow_font.size + 14
        for line in _wrap(title, max_chars=self.cards["max_chars_title"])[:2]:
            draw.text((margin, y), line, fill=self.fg, font=title_font)
            y += title_font.size + 12
        y += 26

        items = payload.get("items") or _split_narration(scene.narration_text_zh_tw)
        for i, item in enumerate(list(items)[: self.cards["max_bullets"]], start=1):
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            bullet = f"{i}."
            draw.text((margin, y), bullet, fill=self.accent, font=item_font)
            for li, line in enumerate(
                _wrap(text, max_chars=self.cards["max_chars_per_bullet"])[:2]
            ):
                draw.text(
                    (margin + 80, y + li * (item_font.size + 6)),
                    line, fill=self.fg, font=item_font,
                )
            y += item_font.size * 2 + 18
            if y > self.h - 140:
                break
        return img

    def _paragraph_card(self, title: str, payload: dict, scene: Scene
                        ) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        margin = 96

        title_font = _load_font(self.font_path, max(48, self.h // 16))
        body_font = _load_font(self.font_path, max(34, self.h // 22))

        draw.text((margin, 90), title, fill=self.fg, font=title_font)
        draw.line(
            [(margin, 90 + title_font.size + 14),
             (margin + 260, 90 + title_font.size + 14)],
            fill=self.accent, width=8,
        )

        text = (payload.get("body") or scene.on_screen_text
                or scene.narration_text_zh_tw or "")[: self.cards["max_chars_paragraph"] * 4]
        y = 90 + title_font.size + 60
        for line in _wrap(text, max_chars=self.cards["max_chars_subtitle"])[:8]:
            draw.text((margin, y), line, fill=self.fg, font=body_font)
            y += body_font.size + 16
            if y > self.h - 140:
                break
        return img

    def _key_number(self, title: str, payload: dict, scene: Scene
                    ) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        margin = 96
        title_font = _load_font(self.font_path, max(40, self.h // 22))
        number_font = _load_font(self.font_path, max(220, self.h // 4))
        label_font = _load_font(self.font_path, max(36, self.h // 22))
        context_font = _load_font(self.font_path, max(28, self.h // 32))

        items = list(payload.get("items") or [])
        if not items:
            return self._paragraph_card(title, payload, scene)
        primary = items[0]

        draw.text((margin, 80), title, fill=_darker(self.fg, 0.7),
                  font=title_font)
        # Centered big number.
        value = str(primary.get("value", ""))[:18]
        vw = _text_width(draw, value, number_font)
        draw.text(((self.w - vw) // 2, int(self.h * 0.30)),
                  value, fill=self.accent, font=number_font)
        # Label below.
        label = str(primary.get("label", ""))[:16]
        lw = _text_width(draw, label, label_font)
        draw.text(((self.w - lw) // 2,
                   int(self.h * 0.30) + number_font.size + 20),
                  label, fill=self.fg, font=label_font)
        # Context quote.
        ctx = str(primary.get("context", ""))[:80]
        y = int(self.h * 0.30) + number_font.size + label_font.size + 70
        for line in _wrap(ctx, max_chars=self.cards["max_chars_subtitle"])[:3]:
            cw = _text_width(draw, line, context_font)
            draw.text(((self.w - cw) // 2, y), line,
                      fill=_darker(self.fg, 0.6), font=context_font)
            y += context_font.size + 10
        return img

    def _source_crop(self, title: str, payload: dict, scene: Scene
                     ) -> Image.Image:
        """Show a PDF crop with a large explanation on the right.

        ``payload`` may contain:
            crop_path: str          — path to the cropped PDF image
            caption: str            — large explanation text (one or two sentences)
            quote: str              — verbatim quote from the source
        Falls back to a paragraph card if the crop file is unreadable.
        """
        img = self._base()
        draw = ImageDraw.Draw(img)
        margin = 72
        title_font = _load_font(self.font_path, max(40, self.h // 22))
        caption_font = _load_font(self.font_path, max(32, self.h // 26))
        quote_font = _load_font(self.font_path, max(28, self.h // 32))

        draw.text((margin, 60), title, fill=self.fg, font=title_font)
        draw.line(
            [(margin, 60 + title_font.size + 12),
             (margin + 200, 60 + title_font.size + 12)],
            fill=self.accent, width=6,
        )

        crop_path = payload.get("crop_path") or (
            scene.visual_source_paths[0] if scene.visual_source_paths else None
        )
        crop_left_w = int(self.w * 0.52)
        crop_top = 140
        crop_box = (margin, crop_top,
                    margin + crop_left_w, self.h - 120)
        try:
            inset = Image.open(crop_path).convert("RGB") if crop_path else None
        except Exception:
            inset = None
        if inset is not None:
            inset.thumbnail((crop_box[2] - crop_box[0],
                             crop_box[3] - crop_box[1]))
            ox = crop_box[0] + ((crop_box[2] - crop_box[0]) - inset.width) // 2
            oy = crop_box[1] + ((crop_box[3] - crop_box[1]) - inset.height) // 2
            # Light border to make the crop pop on a light card.
            draw.rectangle(
                [(ox - 6, oy - 6), (ox + inset.width + 6, oy + inset.height + 6)],
                outline=_darker(self.fg, 0.4), width=3,
            )
            img.paste(inset, (ox, oy))
        else:
            self._placeholder(draw, "（裁切失敗，請改看下方說明）",
                              start_y=crop_top + 20, margin=margin)

        # Right panel: caption + quote.
        right_x = margin + crop_left_w + 48
        right_w = self.w - margin - right_x
        y = crop_top
        caption = payload.get("caption") or scene.on_screen_text or ""
        if caption:
            for line in _wrap(caption, max_chars=22)[:6]:
                draw.text((right_x, y), line, fill=self.fg, font=caption_font)
                y += caption_font.size + 12
            y += 26
        quote = payload.get("quote")
        if quote:
            draw.text((right_x, y), "原文摘錄", fill=_darker(self.accent),
                      font=quote_font)
            y += quote_font.size + 12
            for line in _wrap(quote, max_chars=24)[:8]:
                draw.text((right_x, y), "「" + line + "」",
                          fill=_darker(self.fg, 0.55), font=quote_font)
                y += quote_font.size + 8
        return img

    # ---------- helpers ----------

    def _placeholder(self, draw: ImageDraw.ImageDraw, text: str,
                     *, start_y: int, margin: int) -> None:
        font = _load_font(self.font_path, max(28, self.h // 28))
        draw.text((margin, start_y), text,
                  fill=_darker(self.fg, 0.5), font=font)

    def _draw_footer(self, img: Image.Image, text: str) -> None:
        draw = ImageDraw.Draw(img)
        font = _load_font(self.font_path, max(18, self.h // 56))
        bar_color = _lighter(self.bg, -0.05)
        # _lighter with negative -> we want darker; fall back to manual:
        bar_color = (
            max(0, self.bg[0] - 18),
            max(0, self.bg[1] - 18),
            max(0, self.bg[2] - 18),
        )
        draw.rectangle([(0, self.h - 48), (self.w, self.h)], fill=bar_color)
        draw.text((36, self.h - 38), text,
                  fill=_darker(self.fg, 0.55), font=font)

    def _footer(self, scene: Scene) -> str:
        pages = ",".join(str(p) for p in (scene.source_pages or [])[:6])
        if len(scene.source_pages or []) > 6:
            pages += "…"
        bits = [scene.scene_id]
        if pages:
            bits.append(f"來源：p.{pages}")
        if scene.scene_kind:
            bits.append(scene.scene_kind)
        return "  ·  ".join(bits)


def _split_narration(text: str, *, max_items: int = 5) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace("。", "。\n").splitlines()
             if p.strip()]
    return parts[:max_items]
