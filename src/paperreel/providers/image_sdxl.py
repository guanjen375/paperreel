"""Stable Diffusion XL image provider via 🤗 diffusers.

Local-only, GPU-accelerated. No mock fallback: if torch/diffusers/the
checkpoint are missing or the GPU OOMs, this raises ``SdxlUnavailable``
and the calling stage decides whether to fall back to a card.

Default model: ``stabilityai/stable-diffusion-xl-base-1.0`` (~6.7 GB).
The first call downloads weights to ``~/.cache/huggingface``.

Install: ``pip install paperreel[sdxl]``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from ..hashing import sha256_text
from ..io_utils import ensure_dir
from .image_base import ImageProvider


DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_NEGATIVE = (
    "low quality, blurry, watermark, text, signature, deformed, ugly, "
    "duplicate, jpeg artifacts"
)

_LOGGING_QUIETED = False


def _silence_diffusers_logging() -> None:
    """Stop diffusers/HF Hub from drowning paperreel's per-stage progress lines.

    Idempotent — safe to call from every ``_ensure_pipe``. Targets:
      * HF Hub "Loading pipeline components: 100%|...|" bar on ``from_pretrained``
      * diffusers INFO-level chatter
      * recurring ``FutureWarning`` from the bundled SDXL pipeline
        (``upcast_vae`` deprecation fires once per inference call)
    The per-call inference tqdm bar is disabled on the pipeline itself
    (see ``set_progress_bar_config(disable=True)`` below) — that flag is
    per-instance, not global, so it lives at the call site.
    """
    global _LOGGING_QUIETED
    if _LOGGING_QUIETED:
        return
    import os
    # Belt-and-suspenders for any HF-side tqdm that reads env at first use.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from diffusers.utils import logging as _diff_logging  # type: ignore
        _diff_logging.set_verbosity_error()
        # Kills the "Loading pipeline components: ..." bar fired by
        # ``from_pretrained``. This is diffusers' own knob (singular) —
        # the HF Hub one below covers other transfer bars.
        _diff_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as _hf_logging  # type: ignore
        _hf_logging.disable_progress_bars()
    except Exception:
        pass
    warnings.filterwarnings(
        "ignore", category=FutureWarning, module=r"diffusers(\..*)?",
    )
    _LOGGING_QUIETED = True


class SdxlUnavailable(RuntimeError):
    """Raised when SDXL cannot be loaded or used."""


class SdxlImage(ImageProvider):
    name = "sdxl"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}
        self.model_id = str(self.cfg.get("model", DEFAULT_MODEL))
        self.steps = int(self.cfg.get("num_inference_steps", 30))
        self.guidance = float(self.cfg.get("guidance_scale", 6.5))
        self.negative_prompt = str(self.cfg.get("negative_prompt", DEFAULT_NEGATIVE))
        self.device = str(self.cfg.get("device", "auto"))  # auto|cuda|cpu
        self.dtype = str(self.cfg.get("dtype", "float16"))
        self._pipe: Any | None = None

    # ---------- model loading (lazy) ----------

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # type: ignore
        except ImportError as e:
            raise SdxlUnavailable(
                "torch not installed — run: pip install -e \".[sdxl]\""
            ) from e
        if not torch.cuda.is_available():
            raise SdxlUnavailable(
                "no CUDA GPU detected — SDXL on CPU is impractical (minutes/image)."
                " Either install a CUDA build of torch or set image.provider to "
                "something else."
            )
        return "cuda"

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        _silence_diffusers_logging()
        try:
            import torch  # type: ignore
            from diffusers import StableDiffusionXLPipeline  # type: ignore
        except ImportError as e:
            raise SdxlUnavailable(
                "diffusers not installed — run: pip install -e \".[sdxl]\""
            ) from e
        device = self._resolve_device()
        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(self.dtype, torch.float16)
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                variant="fp16" if torch_dtype == torch.float16 else None,
            )
            pipe = pipe.to(device)
            # Suppress the per-call inference tqdm bar — one per scene × ~30
            # steps drowns paperreel's stage progress lines.
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            # Saves ~30% VRAM on large UNet without hurting quality much.
            # Use the vae-level API; the pipeline-level shortcut is deprecated
            # and fires a FutureWarning on every load.
            pipe.vae.enable_slicing()
        except Exception as e:
            raise SdxlUnavailable(
                f"failed to load {self.model_id} on {device}: {e!r}"
            ) from e
        self._pipe = pipe
        return pipe

    # ---------- ImageProvider interface ----------

    def generate(self, prompt: str, out_path: str | Path, *,
                 width: int = 1280, height: int = 720) -> str:
        if not prompt or not prompt.strip():
            raise SdxlUnavailable("sdxl: empty prompt")

        out = Path(out_path)
        ensure_dir(out.parent)
        pipe = self._ensure_pipe()
        # SDXL prefers multiples of 8 (or 64 for best fit). Round down.
        w = max(512, (int(width) // 8) * 8)
        h = max(512, (int(height) // 8) * 8)

        # Deterministic seed per prompt so resumes don't shuffle visuals.
        seed = int(sha256_text(prompt)[:8], 16)
        try:
            import torch  # type: ignore
            generator = torch.Generator(device=self._pipe.device).manual_seed(seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance,
                width=w,
                height=h,
                generator=generator,
            )
        except Exception as e:
            raise SdxlUnavailable(f"sdxl inference failed: {e!r}") from e

        img = result.images[0]
        img.save(out)
        return str(out)
