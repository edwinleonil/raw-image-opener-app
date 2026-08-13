"""Interpret headerless .raw sensor dumps as displayable images.

These files are a flat dump of pixel values with no header at all, so the
caller must supply width/height/bit depth/Bayer pattern - there is nothing
in the file itself to detect this from.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BAYER_PATTERNS = ["None (Mono)", "RGGB", "BGGR", "GRBG", "GBRG"]
NORMALIZATION_MODES = [
    "Auto (min/max)",
    "8-bit (0-255)",
    "10-bit (0-1023)",
    "12-bit (0-4095)",
    "16-bit (0-65535)",
]

_FIXED_MAX = {
    "8-bit (0-255)": 255,
    "10-bit (0-1023)": 1023,
    "12-bit (0-4095)": 4095,
    "16-bit (0-65535)": 65535,
}

_BAYER_CV_CODES = None  # populated lazily on first debayer call


class RawFormatError(ValueError):
    """The chosen width/height/bit depth doesn't fit the file."""


_SIDECAR_BAYER_CODES = {"rg": "RGGB", "bg": "BGGR", "gr": "GRBG", "gb": "GBRG"}


def load_sidecar_metadata(raw_path: Path) -> dict | None:
    """Read a `<stem>.json` sidecar next to a .raw file, if one exists.

    Some capture tools (e.g. this repo's companion Basler capture app) write
    a JSON file alongside each .raw with its exact width/height/pixel
    format, which removes all the manual guesswork below.
    """
    json_path = raw_path.with_suffix(".json")
    if not json_path.is_file():
        return None
    try:
        data = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    width, height = data.get("width"), data.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return None

    dtype = str(data.get("dtype", "uint8")).lower()
    bytes_per_pixel = 2 if "16" in dtype else 1

    fmt = str(data.get("format", "")).lower()
    pattern = "None (Mono)"
    if fmt.startswith("bayer"):
        code = fmt[len("bayer") : len("bayer") + 2]
        pattern = _SIDECAR_BAYER_CODES.get(code, "None (Mono)")

    return {
        "width": width,
        "height": height,
        "bytes_per_pixel": bytes_per_pixel,
        "big_endian": False,
        "pattern": pattern,
        "source": json_path.name,
    }


def process_raw_file(
    path: Path,
    width: int,
    height: int | None,
    bytes_per_pixel: int,
    big_endian: bool,
    pattern: str,
    norm_mode: str,
) -> tuple[np.ndarray, int]:
    """Read a .raw file and return an 8-bit array ready for display.

    Returns (image, resolved_height); resolved_height is the height that was
    used (computed from the file size when `height` is None).
    """
    file_size = path.stat().st_size
    row_bytes = width * bytes_per_pixel
    if row_bytes <= 0:
        raise RawFormatError("Width must be greater than zero.")

    if height is None:
        resolved_height = file_size // row_bytes
        if resolved_height <= 0:
            raise RawFormatError(
                f"File is only {file_size} bytes - too small for width {width}."
            )
    else:
        resolved_height = height

    needed = row_bytes * resolved_height
    if needed > file_size:
        raise RawFormatError(
            f"{width}x{resolved_height} at {bytes_per_pixel} byte(s)/pixel needs "
            f"{needed} bytes, but the file is only {file_size} bytes."
        )

    with open(path, "rb") as f:
        data = f.read(needed)

    dtype = np.dtype("u1") if bytes_per_pixel == 1 else np.dtype(">u2" if big_endian else "<u2")
    array = np.frombuffer(data, dtype=dtype).reshape(resolved_height, width)

    if pattern != "None (Mono)":
        array = _debayer(array, pattern)

    image8 = _normalize_to_8bit(array, norm_mode)
    return image8, resolved_height


def _debayer(bayer: np.ndarray, pattern: str) -> np.ndarray:
    global _BAYER_CV_CODES
    import cv2

    if _BAYER_CV_CODES is None:
        # OpenCV names its Bayer codes after the pixel pattern starting at
        # the *second* row/column, not the top-left pixel like camera
        # vendors (and our pattern names / the JSON sidecar) do - so the
        # vendor "RGGB" pattern needs OpenCV's "BG" code, and so on.
        _BAYER_CV_CODES = {
            "RGGB": cv2.COLOR_BAYER_BG2RGB,
            "BGGR": cv2.COLOR_BAYER_RG2RGB,
            "GRBG": cv2.COLOR_BAYER_GB2RGB,
            "GBRG": cv2.COLOR_BAYER_GR2RGB,
        }

    # cv2's demosaic needs an even-sized array.
    h, w = bayer.shape
    bayer = np.ascontiguousarray(bayer[: h - (h % 2), : w - (w % 2)])
    return cv2.cvtColor(bayer, _BAYER_CV_CODES[pattern])


def _normalize_to_8bit(array: np.ndarray, norm_mode: str) -> np.ndarray:
    if array.dtype == np.uint8 and norm_mode == "8-bit (0-255)":
        return array
    values = array.astype(np.float32)
    if norm_mode == "Auto (min/max)":
        lo, hi = float(values.min()), float(values.max())
    else:
        lo, hi = 0.0, float(_FIXED_MAX[norm_mode])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.uint8)
    scaled = (values - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)
