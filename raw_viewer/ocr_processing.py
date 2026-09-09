"""OCR for stamped/engraved (DPM-style) alphanumeric codes.

Uses PaddleOCR's standalone `TextRecognition` model (recognition-only,
no text detection): the caller has already drawn a tight bounding box
around the code, so there is nothing to detect - the whole crop is one
line to recognize. Preprocessing enhances contrast and scale, and converts
back to 3 channels since the recognizer requires an (H, W, 3) input, but
does not binarize, since the recognizer is trained on natural-contrast
images.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_MIN_CROP_HEIGHT_PX = 48

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
    """Upscale + contrast-enhance a crop for OCR recognition."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop

    height = gray.shape[0]
    if 0 < height < _MIN_CROP_HEIGHT_PX:
        scale = _MIN_CROP_HEIGHT_PX / height
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def get_ocr_engine():
    """Lazily construct and cache the OCR engine (slow to load)."""
    global _engine
    if _engine is None:
        _register_cuda_dll_directories()
        from paddleocr import TextRecognition

        _engine = TextRecognition()
    return _engine


def run_ocr(crop: np.ndarray) -> OcrResult:
    """Recognize text in a pre-selected crop. Returns "" text if none found."""
    if crop.size == 0:
        return OcrResult("", 0.0)

    processed = preprocess_for_ocr(crop)
    engine = get_ocr_engine()
    results = list(engine.predict(processed))
    if not results:
        return OcrResult("", 0.0)

    result = results[0]
    return OcrResult(result["rec_text"], result["rec_score"])
