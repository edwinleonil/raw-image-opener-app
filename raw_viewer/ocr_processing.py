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

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["HF_HUB_OFFLINE"] = "1"

_TARGET_MIN_DIM_PX = 600
_MAX_UPSCALE = 6.0
_MAX_DIM_PX = 4000

_engine = None
_MODEL_NAMES = {
    "doc_orientation_classify": "PP-LCNet_x1_0_doc_ori",
    "doc_unwarping": "UVDoc",
    "text_detection": "PP-OCRv5_server_det",
    "textline_orientation": "PP-LCNet_x1_0_textline_ori",
    "text_recognition": "en_PP-OCRv5_mobile_rec",
}

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
    """Convert uint8 grayscale/RGB to bounded, contiguous BGR input for PaddleOCR."""
    if crop.dtype != np.uint8 or not (
        crop.ndim == 2 or (crop.ndim == 3 and crop.shape[2] == 3)
    ):
        raise ValueError("OCR expects an 8-bit grayscale or RGB image")
    if crop.size == 0:
        raise ValueError("OCR crop must not be empty")
    height, width = crop.shape[:2]
    scale = min(
        max(1.0, _TARGET_MIN_DIM_PX / min(height, width)),
        _MAX_UPSCALE,
        _MAX_DIM_PX / max(height, width),
    )
    if scale != 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        crop = cv2.resize(
            crop, size, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        )
    conversion = cv2.COLOR_GRAY2BGR if crop.ndim == 2 else cv2.COLOR_RGB2BGR
    return cv2.cvtColor(crop, conversion)


def _local_model_options() -> dict[str, str]:
    cache = Path(os.environ.get("PADDLE_PDX_CACHE_HOME", str(Path.home() / ".paddlex")))
    model_root = cache.expanduser().resolve() / "official_models"
    options = {}
    missing = []
    for component, name in _MODEL_NAMES.items():
        model_dir = model_root / name
        required_files = ("inference.json", "inference.pdiparams", "inference.yml")
        if not all((model_dir / filename).is_file() for filename in required_files):
            missing.append(name)
        options[f"{component}_model_name"] = name
        options[f"{component}_model_dir"] = str(model_dir)
    if missing:
        raise RuntimeError(
            f"OCR is offline; missing or incomplete models in {model_root}: "
            f"{', '.join(missing)}. Provision the models as described in README.md, then retry."
        )
    return options


def get_ocr_engine():
    """Lazily construct and cache the OCR engine (slow to load)."""
    global _engine
    if _engine is None:
        options = _local_model_options()
        _register_cuda_dll_directories()
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            **options,
        )
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
