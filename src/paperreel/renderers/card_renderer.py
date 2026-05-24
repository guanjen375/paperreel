"""Pillow-based slide / card renderer.

Produces 1080p (or whatever resolution is passed) PNGs for each scene's
visual layer. Templates are simple, deterministic, and need no GPU.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from ..io_utils import ensure_dir
from ..models import Scene, VisualType


_DEFAULT_FONT_CANDIDATES = [
    # macOS / Linux common
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Microsoft/PMingLiU.ttf",
    # Windows common
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/mingliu.ttc",
    "C:/Windows/Fonts/arial.ttf",
]


def _load_font(font_path: str | None, size: int) -> ImageFont.ImageFont:
    candidates: list[str] = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(_DEFAULT_FONT_CANDIDATES)
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


def _wrap_cjk(text: str, max_chars: int) -> list[str]:
    """Naive char-based wrap that works for both CJK and ASCII."""
    if not text:
        return []
    return textwrap.wrap(
        text, width=max_chars,
        break_long_words=True, break_on_hyphens=False,
    ) or [text]


class CardRenderer:
    def __init__(self, *,
                 resolution: tuple[int, int] = (1920, 1080),
                 background: str = "#0F172A",
                 foreground: str = "#F8FAFC",
                 accent: str = "#22D3EE",
                 font_path: str | None = None):
        self.w, self.h = resolution
        self.bg = _hex_to_rgb(background)
        self.fg = _hex_to_rgb(foreground)
        self.accent = _hex_to_rgb(accent)
        self.font_path = font_path

    # ---------- public ----------

    def render_scene(self, scene: Scene, out_path: str | Path,
                     *, footer: str | None = None) -> str:
        out = Path(out_path)
        ensure_dir(out.parent)
        vt = scene.visual_type
        if vt == VisualType.title_card:
            img = self._title_card(scene.title, scene.on_screen_text or "")
        elif vt == VisualType.recap:
            img = self._recap_card(scene.title, scene.on_screen_text or scene.narration_text_zh_tw)
        elif vt == VisualType.quiz:
            img = self._quiz_card(scene.title, scene.on_screen_text or scene.narration_text_zh_tw)
        elif vt == VisualType.bullet_card:
            bullets = self._derive_bullets(scene)
            img = self._bullet_card(scene.title, bullets)
        elif vt == VisualType.diagram:
            img = self._teaching_card(scene.title, scene.on_screen_text or scene.narration_text_zh_tw[:120])
        elif vt == VisualType.pdf_image and scene.visual_asset_paths:
            img = self._pdf_image_card(
                scene.title, scene.visual_asset_paths[0],
                caption=scene.on_screen_text,
            )
        elif vt == VisualType.generated_image and scene.visual_asset_paths:
            img = self._pdf_image_card(
                scene.title, scene.visual_asset_paths[0],
                caption=scene.on_screen_text,
            )
        else:
            img = self._teaching_card(scene.title, scene.on_screen_text or scene.narration_text_zh_tw[:120])

        # Footer: source pages + scene id, always shown for provenance.
        footer_text = footer or self._default_footer(scene)
        self._draw_footer(img, footer_text)
        img.save(out)
        return str(out)

    # ---------- card variants ----------

    def _base(self) -> Image.Image:
        return Image.new("RGB", (self.w, self.h), self.bg)

    def _title_card(self, title: str, subtitle: str) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(48, self.h // 12))
        sub_font = _load_font(self.font_path, max(28, self.h // 28))
        # accent bar
        draw.rectangle([(80, self.h // 2 - 130), (80 + 12, self.h // 2 + 130)], fill=self.accent)
        draw.text((130, self.h // 2 - 110), title, fill=self.fg, font=title_font)
        if subtitle:
            draw.text((130, self.h // 2 + 30), subtitle, fill=self.accent, font=sub_font)
        return img

    def _bullet_card(self, title: str, bullets: Sequence[str]) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(40, self.h // 18))
        body_font = _load_font(self.font_path, max(28, self.h // 28))
        draw.text((80, 80), title, fill=self.fg, font=title_font)
        draw.line([(80, 80 + title_font.size + 18), (self.w - 80, 80 + title_font.size + 18)],
                  fill=self.accent, width=4)
        y = 80 + title_font.size + 60
        max_chars = max(18, self.w // (body_font.size or 20) // 2)
        for b in bullets[:5]:
            for i, line in enumerate(_wrap_cjk(b, max_chars)):
                prefix = "•  " if i == 0 else "    "
                draw.text((80, y), prefix + line, fill=self.fg, font=body_font)
                y += body_font.size + 14
            y += 12
            if y > self.h - 140:
                break
        return img

    def _recap_card(self, title: str, body: str) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(44, self.h // 16))
        body_font = _load_font(self.font_path, max(28, self.h // 30))
        draw.text((80, 80), "RECAP · 重點回顧", fill=self.accent, font=body_font)
        draw.text((80, 80 + body_font.size + 12), title, fill=self.fg, font=title_font)
        y = 80 + body_font.size + 12 + title_font.size + 60
        max_chars = max(20, self.w // (body_font.size or 20) // 2)
        for line in _wrap_cjk(body, max_chars)[:8]:
            draw.text((80, y), line, fill=self.fg, font=body_font)
            y += body_font.size + 14
        return img

    def _quiz_card(self, title: str, body: str) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(40, self.h // 18))
        body_font = _load_font(self.font_path, max(28, self.h // 28))
        draw.text((80, 80), "小測驗 · QUIZ", fill=self.accent, font=body_font)
        draw.text((80, 80 + body_font.size + 12), title, fill=self.fg, font=title_font)
        y = 80 + body_font.size + 12 + title_font.size + 60
        max_chars = max(20, self.w // (body_font.size or 20) // 2)
        for line in _wrap_cjk(body, max_chars)[:8]:
            draw.text((80, y), line, fill=self.fg, font=body_font)
            y += body_font.size + 14
        return img

    def _teaching_card(self, title: str, body: str) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(40, self.h // 18))
        body_font = _load_font(self.font_path, max(28, self.h // 28))
        draw.text((80, 80), title, fill=self.fg, font=title_font)
        draw.line([(80, 80 + title_font.size + 18), (80 + 240, 80 + title_font.size + 22)],
                  fill=self.accent, width=6)
        y = 80 + title_font.size + 60
        max_chars = max(18, self.w // (body_font.size or 20) // 2)
        for line in _wrap_cjk(body, max_chars)[:10]:
            draw.text((80, y), line, fill=self.fg, font=body_font)
            y += body_font.size + 14
        return img

    def _pdf_image_card(self, title: str, image_path: str,
                         *, caption: str | None = None) -> Image.Image:
        img = self._base()
        draw = ImageDraw.Draw(img)
        title_font = _load_font(self.font_path, max(36, self.h // 22))
        draw.text((80, 60), title, fill=self.fg, font=title_font)

        # Reserve room for a caption strip when on_screen_text is present
        # so the inset image doesn't bleed into it. Footer bar (56 px)
        # is drawn last by _draw_footer.
        caption_lines: list[str] = []
        caption_font = None
        caption_block_h = 0
        if caption:
            caption_font = _load_font(self.font_path, max(26, self.h // 32))
            max_chars = max(20, self.w // (caption_font.size or 20) // 2)
            caption_lines = _wrap_cjk(caption, max_chars)[:3]
            caption_block_h = (caption_font.size + 14) * len(caption_lines) + 24
        try:
            inset = Image.open(image_path).convert("RGB")
            max_w = self.w - 160
            max_h = self.h - 260 - caption_block_h
            inset.thumbnail((max_w, max_h))
            ox = (self.w - inset.width) // 2
            oy = 140 + max(0, (max_h - inset.height) // 2)
            img.paste(inset, (ox, oy))
        except Exception:
            draw.text((80, 200), f"[image load failed: {image_path}]",
                      fill=(255, 120, 120),
                      font=_load_font(self.font_path, 28))

        if caption_lines and caption_font is not None:
            # Caption sits above the footer bar so it's never clipped.
            y = self.h - 56 - caption_block_h + 12
            for line in caption_lines:
                draw.text((80, y), line, fill=self.accent, font=caption_font)
                y += caption_font.size + 14
        return img

    # ---------- helpers ----------

    def _draw_footer(self, img: Image.Image, text: str) -> None:
        draw = ImageDraw.Draw(img)
        font = _load_font(self.font_path, max(18, self.h // 48))
        draw.rectangle([(0, self.h - 56), (self.w, self.h)], fill=(0, 0, 0))
        draw.text((40, self.h - 44), text, fill=(180, 180, 180), font=font)

    def _default_footer(self, scene: Scene) -> str:
        pages = ",".join(str(p) for p in scene.source_pages[:6])
        if len(scene.source_pages) > 6:
            pages += "…"
        return f"{scene.scene_id}  ·  source pages: {pages}"

    def _derive_bullets(self, scene: Scene) -> list[str]:
        # on_screen_text wins if present; else split narration by sentence.
        if scene.on_screen_text:
            return [s.strip() for s in scene.on_screen_text.split("\n") if s.strip()][:5] or [scene.title]
        narration = scene.narration_text_zh_tw
        parts = [s.strip() for s in narration.replace("。", "。\n").splitlines() if s.strip()]
        return parts[:5] or [scene.title]
