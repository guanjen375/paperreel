"""Image-generation provider interface.

Local-only build: the **only** supported backend is `sdxl`
(Stable Diffusion XL via diffusers).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ImageProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, out_path: str | Path, *,
                 width: int = 1280, height: int = 720) -> str:
        """Return the path of the generated image (str)."""


def make_image_provider(cfg: dict) -> ImageProvider:
    name = (cfg or {}).get("provider", "sdxl").lower()
    if name == "sdxl":
        from .image_sdxl import SdxlImage
        return SdxlImage(cfg)
    raise ValueError(
        f"unknown image provider: {name!r} — local build only supports 'sdxl'"
    )
