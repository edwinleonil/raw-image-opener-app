"""OCR for stamped/engraved (DPM-style) alphanumeric codes.

Uses PaddleOCR's full pipeline (text detection + text-line orientation +
recognition), not a recognition-only shortcut: real dot-peen/stamped marks
are often multiple lines at an arbitrary angle within one selected box (the
part can be photographed at any rotation), and only the full pipeline
finds and deskews each line on its own - a recognition-only model expects
one already-horizontal line and fails outright on multi-line, rotated
input. Preprocessing just upscales small crops (recognition accuracy on
tiny crops was the main limiting factor in testing); it does not binarize
or enhance contrast, since that testing found plain upscaling more
reliable than adding CLAHE contrast enhancement.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Everything this needs (detection/recognition/orientation models) is
# downloaded once to ~/.paddlex/official_models on first use and cached
# locally after that - disable PaddleX's per-run "phone home" connectivity
# check and force huggingface_hub to use that local cache only, so normal
# use never depends on network access.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

_TARGET_MIN_DIM_PX = 600
_MAX_UPSCALE = 6.0

_engine = None

# The pip-distributed paddlepaddle-gpu wheel doesn't bundle cuDNN/cuBLAS/NVRTC
# on Windows - it expects them on PATH. We depend on the matching NVIDIA pip
# wheels instead (nvidia-cudnn-cu12 etc, see pyproject.toml) so no manual
# driver-site download is needed, but Windows still won't find their DLLs
# unless we register each wheel's bin/ directory before paddle loads them.
_CUDA_DLL_PACKAGES = ("nvidia.cublas", "nvidia.cuda_nvrtc", "nvidia.cudnn")


def _register_cuda_dll_directories() -> None:
    if sys.platform != "win32":
        return
    for package_name in _CUDA_DLL_PACKAGES:
        try:
            module = importlib.import_module(package_name)
        except ImportError:
            continue
        # These are PEP 420 namespace packages (no __init__.py), so they have
        # no __file__ - locate them via __path__ instead.
        for search_path in module.__path__:
            bin_dir = Path(search_path) / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Upscale a small crop so the detector/recognizer has enough pixels to work with."""
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)

    height, width = crop.shape[:2]
    scale = min(max(1.0, _TARGET_MIN_DIM_PX / min(height, width)), _MAX_UPSCALE)
    if scale <= 1.0:
        return crop
    return cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def get_ocr_engine():
    """Lazily construct and cache the OCR engine (slow to load)."""
    global _engine
    if _engine is None:
        _register_cuda_dll_directories()
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(use_textline_orientation=True, lang="en")
    return _engine


def run_ocr(crop: np.ndarray) -> OcrResult:
    """Recognize text in a selected region. Returns "" text if none found.

    The region may contain more than one line (e.g. a multi-line stamped
    code) - all detected lines are joined with newlines in reading order,
    and the confidence is the average across them.
    """
    if crop.size == 0:
        return OcrResult("", 0.0)

    processed = preprocess_for_ocr(crop)
    engine = get_ocr_engine()
    results = list(engine.predict(processed))
    if not results:
        return OcrResult("", 0.0)

    texts: list[str] = []
    scores: list[float] = []
    for result in results:
        texts.extend(result["rec_texts"])
        scores.extend(result["rec_scores"])

    if not texts:
        return OcrResult("", 0.0)

    return OcrResult("\n".join(texts), sum(scores) / len(scores))
