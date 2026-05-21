"""Image-generation provider interface (kept minimal for MVP)."""
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
    name = (cfg or {}).get("provider", "mock").lower()
    if name == "mock":
        from .image_mock import MockImage
        return MockImage(cfg)
    raise ValueError(f"unknown image provider: {name}")
