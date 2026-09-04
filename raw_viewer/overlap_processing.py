"""Pure logic for checkerboard-based overlap measurement between two images.

No Qt imports here - this module is independent of the rest of the app and
only depends on raw_loader for decoding individual .raw files. The two input
images are assumed to show different, possibly non-overlapping-in-content
portions of the same physical checkerboard (10x7 squares, 9x6 inner corners,
20x20mm squares), so alignment is done with general-purpose feature matching
rather than by matching checkerboard corners between the images. The
checkerboard is only used to establish a physical mm scale and to annotate
the result.

Alignment is modeled as a pure 2D translation (dx, dy), not a general
homography: real rig motion between two views isn't always purely
horizontal, but a full projective homography has enough degrees of freedom
to "explain" a handful of scattered, inconsistent feature matches as
consistent even when they aren't a real match set - see align_images.
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
SQUARE_SIZE_MM = 20.0

MIN_GOOD_MATCHES = 4
MIN_TRANSLATION_INLIERS = 3
TRANSLATION_TOLERANCE_PX = 8.0
MAX_CANVAS_PIXELS = 60_000_000

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
    inlier_matches: int


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
    image1: np.ndarray, image2: np.ndarray, cancel_event=None
) -> OverlapResult:
    """Align image2 onto image1, blend them, and measure their overlap in mm.

    `cancel_event`, if given, is anything with an `is_set()` method (e.g. a
    threading.Event); it's checked between the expensive steps below so a
    caller running this on a background thread can abort it early by raising
    OverlapCancelled, instead of waiting for a slow/stuck detection to finish.

    Raises ValueError if the images can't be reliably aligned.
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
    H2to1, inliers = align_images(gray1, gray2, detection1, detection2)
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

    mm_result = None
    measurable_mask = None
    if detection1 is not None:
        H_mm_to_px1 = compute_scale_homography(detection1)
        if H_mm_to_px1 is not None:
            H_px_to_mm1 = np.linalg.inv(H_mm_to_px1)
            canvas_to_mm = H_px_to_mm1 @ np.linalg.inv(H1_shift)
            # A homography fit from the checkerboard's own (possibly small,
            # partial) detected extent is only valid there - extrapolating it
            # across the whole overlap region is meaningless once the scene
            # isn't coplanar with the board (e.g. cables/hardware at other
            # depths), so restrict the mm measurement to where the board
            # itself was actually detected, inside the overlap.
            board_pts = cv2.perspectiveTransform(detection1.corners.reshape(-1, 1, 2), H1_shift)
            board_mask = _hull_mask(overlap_mask.shape, board_pts)
            measurable_mask = cv2.bitwise_and(overlap_mask, board_mask)
            mm_result = _mask_to_mm_measurement(measurable_mask, canvas_to_mm)

    _draw_annotations(
        merged, detection1, H1_shift, detection2, H2_shift, overlap_mask, measurable_mask, mm_result
    )

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
        inlier_matches=inliers,
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
        object_points = np.array(
            [[c * SQUARE_SIZE_MM, r * SQUARE_SIZE_MM] for r in range(rows) for c in range(cols)],
            dtype=np.float32,
        )
        return CheckerboardDetection((cols, rows), corners.astype(np.float32), object_points)
    return None


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


def _estimate_square_px(detection: CheckerboardDetection) -> float:
    """Mean pixel spacing between adjacent detected corners, for sizing masks."""
    cols, rows = detection.pattern_size
    pts = detection.corners.reshape(rows, cols, 2)
    spacings = []
    if cols > 1:
        spacings.append(np.linalg.norm(pts[:, 1:] - pts[:, :-1], axis=-1).mean())
    if rows > 1:
        spacings.append(np.linalg.norm(pts[1:, :] - pts[:-1, :], axis=-1).mean())
    return float(np.mean(spacings)) if spacings else 20.0


def _keypoint_exclusion_mask(
    shape: tuple[int, int], detection: CheckerboardDetection | None
) -> np.ndarray | None:
    """A feature-detection mask that blocks out the checkerboard's region.

    Every inner corner of a checkerboard is locally identical, so generic
    keypoint matching there is fundamentally ambiguous. Returns None (no
    mask) when no checkerboard was detected.
    """
    import cv2

    if detection is None:
        return None

    mask = np.full(shape, 255, dtype=np.uint8)
    hull = cv2.convexHull(detection.corners.reshape(-1, 2).astype(np.float32))
    hull_mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull.astype(np.int32), 255)

    margin = max(1, int(round(_estimate_square_px(detection) * 0.6)))
    kernel = np.ones((margin * 2 + 1, margin * 2 + 1), np.uint8)
    hull_mask = cv2.dilate(hull_mask, kernel)

    mask[hull_mask > 0] = 0
    return mask


# ---------- alignment ----------
def align_images(
    gray1: np.ndarray,
    gray2: np.ndarray,
    detection1: CheckerboardDetection | None = None,
    detection2: CheckerboardDetection | None = None,
) -> tuple[np.ndarray, int]:
    """Estimate the 2D translation placing image2's content onto image1's frame.

    Uses general-purpose ORB feature matching on the *non-checkerboard* scene
    content, not the checkerboard corners themselves: every inner corner of a
    checkerboard looks locally identical, which fools descriptor matching
    into a self-consistent but wrong "shifted by N squares" alignment. When a
    checkerboard was detected, its region (plus a margin) is masked out of
    keypoint detection so matches only come from genuinely distinctive scene
    content; the checkerboard is used elsewhere purely for physical scale.

    The fitted transform is a pure translation (dx, dy), not a general
    homography: a full homography has 8 degrees of freedom, which is enough
    to "fit" a handful of scattered, mutually-inconsistent matches as
    consistent even when they aren't a real correspondence set (verified in
    practice - see _ransac_translation). A translation model makes RANSAC's
    inlier test meaningful: a candidate pair is only an inlier if it implies
    (nearly) the same shift as others, which is a much stronger and more
    appropriate constraint for two views of a roughly-planar scene.

    Raises ValueError if there aren't enough reliable matches.
    """
    import cv2

    mask1 = _keypoint_exclusion_mask(gray1.shape, detection1)
    mask2 = _keypoint_exclusion_mask(gray2.shape, detection2)

    orb = cv2.ORB_create(nfeatures=4000)
    kp1, des1 = orb.detectAndCompute(gray1, mask1)
    kp2, des2 = orb.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        raise ValueError(
            "Not enough non-checkerboard scene content to reliably align the images "
            "(the checkerboard's repeating pattern can't be used for alignment by itself)."
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]
    if len(good) < MIN_GOOD_MATCHES:
        raise ValueError(
            f"Not enough matching features between images ({len(good)} found, "
            f"need {MIN_GOOD_MATCHES})."
        )

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    (dx, dy), inliers = _ransac_translation(pts1, pts2)
    if inliers < MIN_TRANSLATION_INLIERS:
        raise ValueError(
            f"Alignment too unreliable ({inliers} consistent matches, "
            f"need {MIN_TRANSLATION_INLIERS})."
        )

    H = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])
    return H, inliers


def _ransac_translation(pts1: np.ndarray, pts2: np.ndarray) -> tuple[tuple[float, float], int]:
    """Robustly fit a 2D translation (dx, dy) mapping pts2 onto pts1.

    Each match's own implied (dx, dy) = pts1 - pts2 is tried as a hypothesis;
    the one with the most other matches agreeing within
    TRANSLATION_TOLERANCE_PX wins, and the final estimate is the mean over
    just that inlier set. Match counts here are small (typically tens, not
    thousands), so trying every match as a hypothesis is cheap and exact -
    no need for randomized sampling.
    """
    diffs = pts1 - pts2
    best_count, best_inliers = 0, None
    for hyp in diffs:
        dist = np.linalg.norm(diffs - hyp, axis=1)
        inliers = dist <= TRANSLATION_TOLERANCE_PX
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers

    if best_inliers is None:
        return (0.0, 0.0), 0
    dx, dy = diffs[best_inliers].mean(axis=0)
    return (float(dx), float(dy)), best_count


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


def _hull_mask(shape: tuple[int, int], points_px: np.ndarray) -> np.ndarray:
    """A filled mask of the convex hull of `points_px` (canvas pixel coords)."""
    import cv2

    mask = np.zeros(shape, dtype=np.uint8)
    hull = cv2.convexHull(points_px.reshape(-1, 2).astype(np.float32))
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask


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
    measurable_mask: np.ndarray | None,
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

    # The mm measurement only covers the checkerboard's own detected extent
    # (see merge_and_measure_overlap) - outline that sub-region distinctly
    # from the full FOV-overlap outline above, and anchor the label there,
    # so it's clear the number describes this smaller box, not the whole
    # green outline.
    label_anchor = largest
    if measurable_mask is not None and np.count_nonzero(measurable_mask) > 0:
        m_contours, _ = cv2.findContours(measurable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if m_contours:
            m_largest = max(m_contours, key=cv2.contourArea)
            cv2.drawContours(canvas, [m_largest], -1, (0, 165, 255), 2)
            label_anchor = m_largest

    _draw_label(canvas, _format_measurement_text(mm_result), label_anchor)


def _draw_corners(canvas: np.ndarray, pts: np.ndarray, color: tuple[int, int, int]) -> None:
    import cv2

    for p in pts.reshape(-1, 2):
        cv2.circle(canvas, (int(round(p[0])), int(round(p[1]))), 4, color, -1, lineType=cv2.LINE_AA)


def _format_measurement_text(mm_result: OverlapMeasurement | None) -> str:
    if mm_result is None:
        return "Overlap size unavailable (no checkerboard detected)"
    return (
        f"Checkerboard region: {mm_result.width_mm:.1f} x {mm_result.height_mm:.1f} mm "
        f"(area {mm_result.area_mm2 / 100.0:.1f} cm2)"
    )


def _draw_label(canvas: np.ndarray, text: str, contour) -> None:
    import cv2

    if contour is not None:
        moments = cv2.moments(contour)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 20, 20
    else:
        cx, cy = 20, 20

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
