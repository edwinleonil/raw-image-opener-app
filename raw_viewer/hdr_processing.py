"""Pure logic for HDR exposure fusion / burst stacking of .raw images.

No Qt imports here - this module is independent of the rest of the app and
only depends on raw_loader for decoding individual .raw files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .raw_loader import load_sidecar_metadata, process_raw_file

MAX_STACK_SIZE = 5
MIN_STACK_SIZE = 2


@dataclass
class StackImage:
    path: Path
    image: np.ndarray
    exposure_us: float
    gain_db: float


def load_stack_image(path: Path) -> StackImage:
    """Decode a .raw file for stacking, using its required <name>.json sidecar.

    Raises ValueError if there's no valid sidecar (width/height/exposure_us
    are required - there's nothing else to infer them from) or RawFormatError
    if the file doesn't match the sidecar's declared dimensions.
    """
    sidecar = load_sidecar_metadata(path)
    if sidecar is None:
        raise ValueError(f"{path.name}: no matching <name>.json sidecar found")

    json_path = path.with_suffix(".json")
    try:
        raw_meta = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name}: could not read {json_path.name} ({exc})") from exc

    exposure_us = raw_meta.get("exposure_us")
    if not isinstance(exposure_us, (int, float)):
        raise ValueError(f"{path.name}: sidecar is missing a numeric 'exposure_us'")
    gain_db = raw_meta.get("gain_db", 0.0)
    if not isinstance(gain_db, (int, float)):
        gain_db = 0.0

    # Fixed (not "Auto min/max") normalization, matched to the sidecar's
    # dtype - per-image auto-normalization would equalize brightness across
    # exposures and defeat the point of exposure fusion.
    norm_mode = "16-bit (0-65535)" if sidecar["bytes_per_pixel"] == 2 else "8-bit (0-255)"

    image8, _ = process_raw_file(
        path,
        sidecar["width"],
        sidecar["height"],
        sidecar["bytes_per_pixel"],
        sidecar["big_endian"],
        sidecar["pattern"],
        norm_mode,
    )

    return StackImage(path=path, image=image8, exposure_us=float(exposure_us), gain_db=float(gain_db))


def merge_exposure_fusion(images: list[np.ndarray], align: bool = True) -> np.ndarray:
    """Merge 2-5 same-shape 8-bit images into one exposure-fused result.

    Raises ValueError if there are too few/many images or their shapes
    don't match.
    """
    if not (MIN_STACK_SIZE <= len(images) <= MAX_STACK_SIZE):
        raise ValueError(
            f"Need {MIN_STACK_SIZE}-{MAX_STACK_SIZE} images to merge, got {len(images)}"
        )

    shapes = {img.shape[:2] for img in images}
    if len(shapes) > 1:
        raise ValueError(f"All images must be the same size to merge, got: {sorted(shapes)}")

    import cv2

    # Promote any mono (2-D) arrays to 3-channel so every image has a
    # uniform shape for the alignment/merge steps below.
    prepared = [np.stack([img] * 3, axis=-1) if img.ndim == 2 else img for img in images]
    prepared = [np.ascontiguousarray(img) for img in prepared]

    if align:
        prepared = _align_translational(prepared)

    merger = cv2.createMergeMertens()
    fused = merger.process(prepared)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def _align_translational(images: list[np.ndarray]) -> list[np.ndarray]:
    """Align each image to the first via phase-correlation-based translation.

    This targets the same small camera-shake correction as OpenCV's AlignMTB,
    but AlignMTB.process()/shiftMat() aren't usable here: the installed
    opencv-python build raises a "!fixedSize()" cv2.error from its
    vector<Mat> output bindings before doing any real work. Phase correlation
    is a standard translational-registration technique and, as a bonus, is
    inherently robust to the brightness differences between exposures since
    it only uses the phase (not magnitude) of the cross-power spectrum.
    """
    import cv2

    def to_gray_f32(img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        return gray.astype(np.float32)

    height, width = images[0].shape[:2]
    reference = to_gray_f32(images[0])
    aligned = [images[0]]
    for img in images[1:]:
        (dx, dy), _ = cv2.phaseCorrelate(reference, to_gray_f32(img))
        translation = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
        aligned.append(cv2.warpAffine(img, translation, (width, height)))
    return aligned
