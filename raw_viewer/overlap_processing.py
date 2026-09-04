"""Pure logic for checkerboard-based overlap measurement between two images.

No Qt imports here - this module is independent of the rest of the app and
only depends on raw_loader for decoding individual .raw files. The two input
images are assumed to show different, possibly non-overlapping-in-content
portions of the same physical checkerboard (10x7 squares, 9x6 inner corners,
20x20mm squares) - sometimes the *only* content the two images share, e.g. a
target placed in the gap between two camera views. Alignment always starts
from a human-supplied approximate translation (a manual drag in the UI),
refined via local pixel correlation - see align_with_manual_guess. The
checkerboard is also used to establish a physical mm scale and to annotate
the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .raw_loader import load_sidecar_metadata, process_raw_file

CHESSBOARD_MAX_COLS = 9
CHESSBOARD_MAX_ROWS = 6
CHESSBOARD_MIN_COLS = 3
CHESSBOARD_MIN_ROWS = 3
SQUARE_SIZE_MM = 25.0

MAX_CANVAS_PIXELS = 60_000_000

# Horizontal dimension-line annotation (_draw_horizontal_dimension_line).
DIMENSION_LINE_COLOR = (0, 255, 255)  # cyan, distinct from the green/orange outlines
DIMENSION_TICK_HALF_LEN_PX = 10

# Manual-alignment fallback (align_with_manual_guess): a human-supplied
# approximate translation, refined via local pixel correlation. These bound
# the template/search-window sizes so the refinement cost stays small
# regardless of source resolution, rather than scanning the whole image like
# detect_checkerboard does - manual align only ever looks at a small region
# the user has already visually placed. Starting points, not fixed: retune
# from real usage if users routinely miss by more than the search radius.
MANUAL_ALIGN_MAX_TEMPLATE_DIM = 400  # px/side cap on the NCC template crop
MANUAL_ALIGN_SEARCH_RADIUS_PX = 50  # px expansion of the search window around the guess
MANUAL_ALIGN_MIN_OVERLAP_DIM_PX = 20  # px minimum guess-implied overlap width/height
MANUAL_ALIGN_LOW_CONFIDENCE_THRESHOLD = 0.5  # advisory only, used for UI phrasing

_SUBPIX_CRITERIA_ITERS = 30
_SUBPIX_CRITERIA_EPS = 0.001


class OverlapCancelled(Exception):
    """Raised to unwind out of merge_and_measure_overlap when the caller cancels it."""


@dataclass
class OverlapImage:
    path: Path
    image: np.ndarray


@dataclass
class CheckerboardDetection:
    pattern_size: tuple[int, int]  # (cols, rows)
    corners: np.ndarray  # (N, 1, 2) float32 pixel coordinates
    object_points_mm: np.ndarray  # (N, 2) float32 relative mm coordinates


@dataclass
class OverlapMeasurement:
    width_mm: float
    height_mm: float
    area_mm2: float


@dataclass
class OverlapResult:
    merged_image: np.ndarray
    mm: OverlapMeasurement | None
    overlap_area_px: int
    percent_of_image1: float
    percent_of_image2: float
    corners_detected_1: int
    corners_detected_2: int
    manual_confidence: float | None  # NCC peak (0..1) from align_with_manual_guess


def load_overlap_image(path: Path) -> OverlapImage:
    """Decode a .raw file for overlap measurement, using its <name>.json sidecar."""
    sidecar = load_sidecar_metadata(path)
    if sidecar is None:
        raise ValueError(f"{path.name}: no matching <name>.json sidecar found")

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
    return OverlapImage(path=path, image=image8)


def merge_and_measure_overlap(
    image1: np.ndarray,
    image2: np.ndarray,
    manual_guess: tuple[float, float],
    cancel_event=None,
) -> OverlapResult:
    """Align image2 onto image1, blend them, and measure their overlap in mm.

    `manual_guess` is a user-supplied approximate (dx, dy) translation (see
    align_with_manual_guess), refined via local pixel correlation.

    `cancel_event`, if given, is anything with an `is_set()` method (e.g. a
    threading.Event); it's checked between the expensive steps below so a
    caller running this on a background thread can abort it early by raising
    OverlapCancelled, instead of waiting for a slow/stuck detection to finish.

    Raises ValueError if `manual_guess` implies negligible overlap.
    """
    import cv2

    def _check_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise OverlapCancelled()

    gray1 = _to_gray(image1)
    gray2 = _to_gray(image2)

    _check_cancelled()
    detection1 = detect_checkerboard(gray1, cancel_event)
    _check_cancelled()
    detection2 = detect_checkerboard(gray2, cancel_event)

    _check_cancelled()
    H2to1, manual_confidence = align_with_manual_guess(gray1, gray2, manual_guess, cancel_event)
    _check_cancelled()

    (canvas_w, canvas_h), H1_shift, H2_shift = _layout_canvas(gray1.shape, gray2.shape, H2to1)

    color1 = _to_color(image1)
    color2 = _to_color(image2)

    warped1 = cv2.warpPerspective(color1, H1_shift, (canvas_w, canvas_h))
    warped2 = cv2.warpPerspective(color2, H2_shift, (canvas_w, canvas_h))
    mask1 = cv2.warpPerspective(
        np.full(gray1.shape, 255, np.uint8), H1_shift, (canvas_w, canvas_h)
    )
    mask2 = cv2.warpPerspective(
        np.full(gray2.shape, 255, np.uint8), H2_shift, (canvas_w, canvas_h)
    )
    overlap_mask = cv2.bitwise_and(mask1, mask2)

    merged = _alpha_blend(warped1, warped2, mask1, mask2, overlap_mask)

    # The checkerboard's known square size gives a mm-per-pixel scale from
    # whichever image has a usable detection (image 1 preferred); that scale
    # is applied across the whole overlap region, not just the board's own
    # detected extent - the board is assumed reasonably coplanar with the
    # rest of the scene.
    mm_result = None
    scale_detection = detection1 if detection1 is not None else detection2
    scale_H_shift = H1_shift if detection1 is not None else H2_shift
    if scale_detection is not None:
        H_mm_to_px = compute_scale_homography(scale_detection)
        if H_mm_to_px is not None:
            canvas_to_mm = np.linalg.inv(H_mm_to_px) @ np.linalg.inv(scale_H_shift)
            mm_result = _mask_to_mm_measurement(overlap_mask, canvas_to_mm)

    _draw_annotations(merged, detection1, H1_shift, detection2, H2_shift, overlap_mask, mm_result)

    overlap_px = int(np.count_nonzero(overlap_mask))
    area1_px = int(np.count_nonzero(mask1))
    area2_px = int(np.count_nonzero(mask2))

    return OverlapResult(
        merged_image=merged,
        mm=mm_result,
        overlap_area_px=overlap_px,
        percent_of_image1=100.0 * overlap_px / area1_px if area1_px else 0.0,
        percent_of_image2=100.0 * overlap_px / area2_px if area2_px else 0.0,
        corners_detected_1=0 if detection1 is None else len(detection1.corners),
        corners_detected_2=0 if detection2 is None else len(detection2.corners),
        manual_confidence=manual_confidence,
    )


# ---------- checkerboard detection ----------
def _candidate_pattern_sizes() -> list[tuple[int, int]]:
    sizes = set()
    for cols in range(CHESSBOARD_MAX_COLS, CHESSBOARD_MIN_COLS - 1, -1):
        for rows in range(CHESSBOARD_MAX_ROWS, CHESSBOARD_MIN_ROWS - 1, -1):
            sizes.add((cols, rows))
    for cols in range(CHESSBOARD_MAX_ROWS, CHESSBOARD_MIN_ROWS - 1, -1):
        for rows in range(CHESSBOARD_MAX_COLS, CHESSBOARD_MIN_COLS - 1, -1):
            sizes.add((cols, rows))
    return sorted(sizes, key=lambda s: s[0] * s[1], reverse=True)


DETECTION_MAX_DIM = 1000


def _downscale_for_detection(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Shrink `gray` so its longest side is DETECTION_MAX_DIM, for fast detection.

    On a real full-resolution capture, trying up to ~70 candidate pattern
    sizes at full size can take minutes per image (each cv2 detector call
    scales with pixel count); at ~1000px a checkerboard square is still
    dozens of pixels across, plenty for reliable detection, while each call
    drops to a fraction of a second - which also makes cancellation (checked
    between candidates) actually responsive.
    """
    import cv2

    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= DETECTION_MAX_DIM:
        return gray, 1.0
    scale = DETECTION_MAX_DIM / float(longest)
    small = cv2.resize(
        gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    return small, scale


def detect_checkerboard(gray: np.ndarray, cancel_event=None) -> CheckerboardDetection | None:
    """Find the largest visible sub-grid of the checkerboard in `gray`.

    Only a portion of the full 9x6-corner board may be visible, so this tries
    decreasing candidate pattern sizes (in both orientations) and returns the
    first (largest) one that's found - object points are relative mm
    coordinates only, since the absolute position within the full board is
    unknown from a single partial view. This loop is the slowest part of the
    whole computation (up to ~70 detector calls), so `cancel_event` (anything
    with an `is_set()` method) is checked between attempts, raising
    OverlapCancelled if set. Detection itself runs on a downscaled copy (see
    _downscale_for_detection) to keep each attempt - and so cancellation -
    fast even on full-resolution captures.
    """
    small, scale = _downscale_for_detection(gray)
    for cols, rows in _candidate_pattern_sizes():
        if cancel_event is not None and cancel_event.is_set():
            raise OverlapCancelled()
        corners = _find_corners(small, (cols, rows))
        if corners is None:
            continue
        if scale != 1.0:
            corners = corners / scale
            corners = _refine_corners_full_res(gray, corners)
        object_points = np.array(
            [[c * SQUARE_SIZE_MM, r * SQUARE_SIZE_MM] for r in range(rows) for c in range(cols)],
            dtype=np.float32,
        )
        return CheckerboardDetection((cols, rows), corners.astype(np.float32), object_points)
    return None


def _refine_corners_full_res(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Sub-pixel refine corners (found on a downscaled image) against the original.

    The downscaled search only locates the pattern approximately - rescaling
    its corners back up carries that downscaled pixel's worth of error times
    the scale factor. Corner positions now directly drive alignment (not just
    a rough mm scale), so re-run cornerSubPix on the full-resolution image
    around each rescaled position to recover real sub-pixel accuracy.
    """
    import cv2

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        _SUBPIX_CRITERIA_ITERS,
        _SUBPIX_CRITERIA_EPS,
    )
    refined = corners.astype(np.float32).copy()
    cv2.cornerSubPix(gray, refined, (21, 21), (-1, -1), criteria)
    return refined


def _find_corners(gray: np.ndarray, pattern_size: tuple[int, int]) -> np.ndarray | None:
    import cv2

    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if found:
            return corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        _SUBPIX_CRITERIA_ITERS,
        _SUBPIX_CRITERIA_EPS,
    )
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners


def compute_scale_homography(detection: CheckerboardDetection) -> np.ndarray | None:
    """Homography mapping relative board mm coordinates to pixel coordinates."""
    import cv2

    H, _ = cv2.findHomography(detection.object_points_mm, detection.corners.reshape(-1, 2))
    return H


# ---------- manual alignment ----------
def _manual_guess_overlap_rects(
    shape1: tuple[int, int], shape2: tuple[int, int], guess: tuple[float, float]
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Intersect image2's guess-translated box with image1's box.

    Returns ((x0, y0, x1, y1) in image1's own frame, the same rectangle
    re-expressed in image2's own frame). Raises ValueError if `guess` implies
    negligible overlap - a human-dragged position that's simply wrong,
    rather than something a search window could recover from.
    """
    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    dx, dy = guess

    x0, y0 = max(0.0, dx), max(0.0, dy)
    x1, y1 = min(float(w1), dx + w2), min(float(h1), dy + h2)

    if x1 - x0 < MANUAL_ALIGN_MIN_OVERLAP_DIM_PX or y1 - y0 < MANUAL_ALIGN_MIN_OVERLAP_DIM_PX:
        raise ValueError(
            "The dragged position has little or no overlap between the two images - "
            "drag image 2 closer to its true position before refining."
        )

    rect1 = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    rect2 = (
        int(round(x0 - dx)), int(round(y0 - dy)), int(round(x1 - dx)), int(round(y1 - dy))
    )
    return rect1, rect2


def _crop_manual_align_template(
    gray: np.ndarray, rect: tuple[int, int, int, int], max_dim: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop a <=max_dim-per-side window centered in `rect`, for align_with_manual_guess.

    Returns (crop, (x0, y0)) where (x0, y0) is the crop's origin in `gray`.
    """
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0

    if w > max_dim:
        cx = x0 + w // 2
        x0, x1 = cx - max_dim // 2, cx - max_dim // 2 + max_dim
    if h > max_dim:
        cy = y0 + h // 2
        y0, y1 = cy - max_dim // 2, cy - max_dim // 2 + max_dim

    return gray[y0:y1, x0:x1], (x0, y0)


def align_with_manual_guess(
    gray1: np.ndarray,
    gray2: np.ndarray,
    guess: tuple[float, float],
    cancel_event=None,
) -> tuple[np.ndarray, float]:
    """Refine a user-supplied approximate translation via local pixel correlation.

    `guess = (dx, dy)` is the translation mapping image2's pixel coordinates
    into image1's frame, as roughly placed by the user (e.g. by dragging in
    the UI). Needs no detectable features at all - a human can align the
    images by eye even when there's too little distinctive structure for
    feature-based methods.

    Crops a bounded template from image2's guess-implied overlap region and
    searches a correspondingly bounded window in image1 via
    cv2.matchTemplate (TM_CCOEFF_NORMED), converting the best match location
    back into a refined (dx, dy). Returns (translation_matrix, confidence)
    where confidence is the NCC peak clamped to [0, 1] - always returned
    rather than used to block: the user already visually confirmed the
    images look aligned, so a low score more likely reflects genuinely
    low-texture content (which they can still judge by eye) than a wrong
    placement.

    This is one bounded, fast operation with no natural mid-operation
    checkpoint (unlike detect_checkerboard's per-candidate loop), so
    cancellation is only checked once, at entry.

    Raises ValueError if `guess` implies negligible overlap.
    """
    import cv2

    if cancel_event is not None and cancel_event.is_set():
        raise OverlapCancelled()

    rect1, rect2 = _manual_guess_overlap_rects(gray1.shape, gray2.shape, guess)
    template, (tx0, ty0) = _crop_manual_align_template(
        gray2, rect2, MANUAL_ALIGN_MAX_TEMPLATE_DIM
    )
    th, tw = template.shape[:2]

    dx, dy = guess
    expected_x, expected_y = tx0 + dx, ty0 + dy
    h1, w1 = gray1.shape[:2]
    sx0 = max(0, int(np.floor(expected_x - MANUAL_ALIGN_SEARCH_RADIUS_PX)))
    sy0 = max(0, int(np.floor(expected_y - MANUAL_ALIGN_SEARCH_RADIUS_PX)))
    sx1 = min(w1, int(np.ceil(expected_x + tw + MANUAL_ALIGN_SEARCH_RADIUS_PX)))
    sy1 = min(h1, int(np.ceil(expected_y + th + MANUAL_ALIGN_SEARCH_RADIUS_PX)))

    # Near image1's border the clipped search window can shrink below the
    # template; shrink the template symmetrically to fit rather than erroring.
    if sx1 - sx0 < tw or sy1 - sy0 < th:
        tw, th = min(tw, sx1 - sx0), min(th, sy1 - sy0)
        template = template[:th, :tw]

    search = gray1[sy0:sy1, sx0:sx1]
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    confidence = float(max(0.0, min(1.0, max_val)))

    refined_dx = (sx0 + max_loc[0]) - tx0
    refined_dy = (sy0 + max_loc[1]) - ty0
    H = np.array(
        [[1.0, 0.0, float(refined_dx)], [0.0, 1.0, float(refined_dy)], [0.0, 0.0, 1.0]]
    )
    return H, confidence


# ---------- compositing ----------
def _layout_canvas(
    shape1: tuple[int, int], shape2: tuple[int, int], H2to1: np.ndarray
) -> tuple[tuple[int, int], np.ndarray, np.ndarray]:
    import cv2

    h1, w1 = shape1[:2]
    h2, w2 = shape2[:2]
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    corners2_in_1 = cv2.perspectiveTransform(corners2, H2to1)
    all_corners = np.concatenate([corners1, corners2_in_1], axis=0).reshape(-1, 2)

    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    offset_x, offset_y = max(0, -x_min), max(0, -y_min)
    canvas_w = int(x_max + offset_x)
    canvas_h = int(y_max + offset_y)

    if canvas_w <= 0 or canvas_h <= 0 or canvas_w * canvas_h > MAX_CANVAS_PIXELS:
        raise ValueError("Alignment produced an implausible transform - check the input images.")

    T = np.array([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y], [0.0, 0.0, 1.0]])
    H1_shift = T
    H2_shift = T @ H2to1
    return (canvas_w, canvas_h), H1_shift, H2_shift


def _alpha_blend(
    warped1: np.ndarray, warped2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray, overlap_mask: np.ndarray
) -> np.ndarray:
    import cv2

    merged = np.zeros_like(warped1)
    only1 = cv2.bitwise_and(mask1, cv2.bitwise_not(mask2))
    only2 = cv2.bitwise_and(mask2, cv2.bitwise_not(mask1))
    merged[only1 > 0] = warped1[only1 > 0]
    merged[only2 > 0] = warped2[only2 > 0]

    ov = overlap_mask > 0
    merged[ov] = (
        warped1[ov].astype(np.float32) * 0.5 + warped2[ov].astype(np.float32) * 0.5
    ).astype(np.uint8)
    return merged


def _mask_to_mm_measurement(mask: np.ndarray, px_to_mm: np.ndarray) -> OverlapMeasurement | None:
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None

    pts_px = contour.reshape(-1, 1, 2).astype(np.float32)
    pts_mm = cv2.perspectiveTransform(pts_px, px_to_mm.astype(np.float32)).reshape(-1, 2)

    (_, _), (w_mm, h_mm), _ = cv2.minAreaRect(pts_mm.astype(np.float32))
    area_mm2 = abs(cv2.contourArea(pts_mm.astype(np.float32)))
    return OverlapMeasurement(width_mm=float(w_mm), height_mm=float(h_mm), area_mm2=float(area_mm2))


# ---------- annotation ----------
def _draw_annotations(
    canvas: np.ndarray,
    detection1: CheckerboardDetection | None,
    H1_shift: np.ndarray,
    detection2: CheckerboardDetection | None,
    H2_shift: np.ndarray,
    overlap_mask: np.ndarray,
    mm_result: OverlapMeasurement | None,
) -> None:
    import cv2

    if detection1 is not None:
        pts = cv2.perspectiveTransform(detection1.corners.reshape(-1, 1, 2), H1_shift)
        _draw_corners(canvas, pts, (0, 200, 255))
    if detection2 is not None:
        pts = cv2.perspectiveTransform(detection2.corners.reshape(-1, 1, 2), H2_shift)
        _draw_corners(canvas, pts, (255, 120, 0))

    contours, _ = cv2.findContours(overlap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea) if contours else None
    if largest is not None:
        cv2.drawContours(canvas, [largest], -1, (0, 255, 0), 2)
        if mm_result is not None:
            _draw_horizontal_dimension_line(canvas, overlap_mask, mm_result)


def _draw_corners(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    import cv2

    for p in pts.reshape(-1, 2):
        cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 4, color, -1, lineType=cv2.LINE_AA)


def _draw_horizontal_dimension_line(
    canvas: np.ndarray, mask: np.ndarray, mm_result: OverlapMeasurement
) -> None:
    """Draw a horizontal tick/dimension line through `mask`'s vertical center.

    Spans the mask's actual left-right extent at that row (not just its
    bounding-rect extent, which could overstate the true width if the
    region isn't a perfect axis-aligned rectangle), labeled with the
    measured width in mm.
    """
    import cv2

    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return
    y_center = int(round((int(ys.min()) + int(ys.max())) / 2))

    row = np.nonzero(mask[y_center] > 0)[0]
    if len(row) == 0:
        row = xs  # fallback: the exact center row happens to be empty
    x_left, x_right = int(row.min()), int(row.max())

    cv2.line(canvas, (x_left, y_center), (x_right, y_center), DIMENSION_LINE_COLOR, 2, cv2.LINE_AA)
    for x in (x_left, x_right):
        cv2.line(
            canvas,
            (x, y_center - DIMENSION_TICK_HALF_LEN_PX),
            (x, y_center + DIMENSION_TICK_HALF_LEN_PX),
            DIMENSION_LINE_COLOR,
            2,
            cv2.LINE_AA,
        )

    mid_x = (x_left + x_right) // 2
    _draw_label(canvas, f"{mm_result.width_mm:.1f} mm", (mid_x, y_center - DIMENSION_TICK_HALF_LEN_PX - 6))


def _draw_label(canvas: np.ndarray, text: str, anchor: tuple[int, int]) -> None:
    import cv2

    cx, cy = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = max(4, cx - text_w // 2)
    y = max(text_h + 8, cy)
    cv2.rectangle(canvas, (x - 6, y - text_h - 8), (x + text_w + 6, y + baseline + 4), (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ---------- helpers ----------
def _to_gray(image: np.ndarray) -> np.ndarray:
    import cv2

    if image.ndim == 2:
        return np.ascontiguousarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def _to_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.ascontiguousarray(np.stack([image] * 3, axis=-1))
    return np.ascontiguousarray(image)
