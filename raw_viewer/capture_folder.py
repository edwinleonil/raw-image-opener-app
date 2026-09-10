"""Parse capture settings encoded in a trial folder's name.

The rig names each capture subfolder `wd<mm>f<f-number>_Capture_<date>_<time>`
(e.g. `wd600f4_Capture_2026-09-10_14-08-56`). The working distance and lens
aperture are recorded there and nowhere else - the .json sidecars next to
each .raw carry exposure/gain/light settings but not these two.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Strict shape: the name must START with wd<number>f<number> followed by
# `_Capture_`. No separator is allowed between the two numbers, and the
# prefix is not matched anywhere else in the name. Both numbers may be
# fractional - the rig really does produce `wd600f2.5_Capture_...`.
_CAPTURE_NAME_RE = re.compile(
    r"^wd(?P<wd_mm>\d+(?:\.\d+)?)f(?P<f_number>\d+(?:\.\d+)?)_Capture_",
    re.IGNORECASE,
)

# `<capture>/FullSize_RAW_Images/img.raw` puts the capture folder 2 levels up
# from the file; allow a little headroom for deeper layouts.
_MAX_ANCESTOR_DEPTH = 3

MISSING = "—"


@dataclass(frozen=True)
class CaptureFolderInfo:
    folder_name: str
    working_distance_mm: float
    f_number: float


def parse_capture_folder_name(name: str) -> CaptureFolderInfo | None:
    """Return the settings encoded in `name`, or None if it doesn't match."""
    match = _CAPTURE_NAME_RE.match(name)
    if match is None:
        return None
    return CaptureFolderInfo(
        folder_name=name,
        working_distance_mm=float(match["wd_mm"]),
        f_number=float(match["f_number"]),
    )


def find_capture_folder_info(path: Path) -> CaptureFolderInfo | None:
    """Walk up from `path` looking for a capture-named ancestor folder.

    `path` is the .raw file being displayed; its parent is usually
    `FullSize_RAW_Images`, so the capture folder is a grandparent. Purely
    string-based - it never touches the filesystem.
    """
    for ancestor in path.parents[:_MAX_ANCESTOR_DEPTH]:
        info = parse_capture_folder_name(ancestor.name)
        if info is not None:
            return info
    return None


def format_number(value: float | None) -> str:
    """Render 600.0 as "600" but keep 2.5 as "2.5"; None as an em dash."""
    return MISSING if value is None else f"{value:g}"


def format_working_distance(info: CaptureFolderInfo | None) -> str:
    return MISSING if info is None else f"{format_number(info.working_distance_mm)} mm"


def format_f_number(info: CaptureFolderInfo | None) -> str:
    return MISSING if info is None else f"f/{format_number(info.f_number)}"
