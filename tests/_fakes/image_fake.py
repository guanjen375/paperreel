"""Test-only fake image generator — writes a deterministic gradient PNG."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from paperreel.hashing import sha256_text
from paperreel.io_utils import ensure_dir
from paperreel.providers.image_base import ImageProvider


class FakeImage(ImageProvider):
    name = "fake"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def generate(self, prompt: str, out_path: str | Path, *,
                 width: int = 1280, height: int = 720) -> str:
        out = Path(out_path)
        ensure_dir(out.parent)
        digest = sha256_text(prompt)
        r = int(digest[0:2], 16); g = int(digest[2:4], 16); b = int(digest[4:6], 16)
        top = (max(20, r // 2), max(20, g // 2), max(40, b // 2))
        bot = (min(255, r // 2 + 80), min(255, g // 2 + 80), min(255, b // 2 + 120))
        img = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(img)
        for y in range(height):
            t = y / max(1, height - 1)
            col = (
                int(top[0] * (1 - t) + bot[0] * t),
                int(top[1] * (1 - t) + bot[1] * t),
                int(top[2] * (1 - t) + bot[2] * t),
            )
            draw.line([(0, y), (width, y)], fill=col)
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except Exception:
            font = ImageFont.load_default()
        label = (prompt[:60] + ("…" if len(prompt) > 60 else "")) or "fake image"
        draw.text((40, height - 80), f"[fake] {label}",
                  fill=(245, 245, 245), font=font)
        img.save(out)
        return str(out)
