"""Checkerboard-based overlap measurement tab.

Fully self-contained: owns its own overlap computation, only shares the
generic ZoomableImageView/numpy_to_qimage helpers and raw_loader's
sidecar-based decoding/rotation (via overlap_processing.load_overlap_image
and raw_loader.rotate_image), same as HdrBurstTab. The one deliberate
coupling to the rest of the app is file selection: MainWindow pushes the
currently selected Viewer-tab folder's .raw files in via
`set_available_files()`, so this tab only ever offers files from that folder
rather than letting the user browse anywhere.
"""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .overlap_processing import (
    MANUAL_ALIGN_LOW_CONFIDENCE_THRESHOLD,
    OverlapCancelled,
    OverlapImage,
    OverlapResult,
    load_overlap_image,
    merge_and_measure_overlap,
)
from .raw_loader import RawFormatError, rotate_image
from .widgets import ZoomableImageView, numpy_to_qimage

THUMBNAIL_SIZE = 140

# The rig this tab is used with mounts the camera 90 degrees from the
# checkerboard's natural orientation, so every loaded image is rotated
# clockwise by this fixed amount, once, at load time - see
# _ImageSlot._on_selection_changed.
FIXED_ROTATION_DEGREES = 90

_PLACEHOLDER_TEXT = "Load two .raw images below to begin manual alignment"


class _ImageSlot(QWidget):
    """One image slot: a dropdown over the Viewer tab's current folder, plus a thumbnail.

    Every loaded image is rotated FIXED_ROTATION_DEGREES clockwise once here
    before being previewed or cached for compute - see _on_selection_changed.
    """

    item_changed = Signal()
    selection_failed = Signal(str)

    _PLACEHOLDER_EMPTY = "(choose a folder in the Viewer tab)"
    _PLACEHOLDER_PICK = "(select a file)"

    def __init__(self, title: str):
        super().__init__()
        self.item: OverlapImage | None = None
        self._files: list[Path] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel(title))

        self.combo = QComboBox()
        self.combo.addItem(self._PLACEHOLDER_EMPTY)
        self.combo.setEnabled(False)
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        layout.addWidget(self.combo)

        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        self.thumb_label.setStyleSheet("background-color: #202020;")
        layout.addWidget(self.thumb_label)

        self.caption = QLabel("No image loaded")
        self.caption.setFixedWidth(THUMBNAIL_SIZE)
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color: #cccccc; font-size: 10px;")
        layout.addWidget(self.caption)

    def set_available_files(self, files: list[Path]) -> None:
        self._files = files
        self.combo.blockSignals(True)
        self.combo.clear()
        if files:
            self.combo.addItem(self._PLACEHOLDER_PICK)
            self.combo.addItems([f.name for f in files])
            self.combo.setEnabled(True)
        else:
            self.combo.addItem(self._PLACEHOLDER_EMPTY)
            self.combo.setEnabled(False)
        self.combo.setCurrentIndex(0)
        self.combo.blockSignals(False)
        self.clear()

    def _on_selection_changed(self, index: int) -> None:
        if index <= 0 or index - 1 >= len(self._files):
            self.clear()
            return
        path = self._files[index - 1]
        try:
            item = load_overlap_image(path)
        except (ValueError, RawFormatError) as exc:
            self.clear()
            self.selection_failed.emit(str(exc))
            return
        item.image = rotate_image(item.image, FIXED_ROTATION_DEGREES)
        self.set_item(item)

    def set_item(self, item: OverlapImage) -> None:
        self.item = item
        pixmap = QPixmap.fromImage(numpy_to_qimage(item.image)).scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumb_label.setPixmap(pixmap)
        self.caption.setText(item.path.name)
        self.item_changed.emit()

    def clear(self) -> None:
        self.item = None
        self.thumb_label.setPixmap(QPixmap())
        self.caption.setText("No image loaded")
        self.item_changed.emit()


class _ManualAlignView(QGraphicsView):
    """Fixed base image (image 1) plus a draggable, ~50%-opacity overlay (image 2).

    The primary alignment input surface: the starting point for
    align_with_manual_guess's local pixel-correlation refinement - not a
    general-purpose viewer, so it's kept separate from the shared
    ZoomableImageView (which has different, single-pixmap pan/zoom semantics
    used by other tabs).
    """

    OVERLAY_OPACITY = 0.5
    NUDGE_STEP_PX = 1
    NUDGE_STEP_PX_SHIFT = 10
    MIN_ZOOM = 0.05
    MAX_ZOOM = 8.0
    ZOOM_STEP = 1.15

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._base_item = QGraphicsPixmapItem()
        self._base_item.setZValue(0)
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setZValue(1)
        self._overlay_item.setOpacity(self.OVERLAY_OPACITY)
        self._overlay_item.setFlag(QGraphicsItem.ItemIsMovable, True)
        self._scene.addItem(self._base_item)
        self._scene.addItem(self._overlay_item)

        self.setDragMode(QGraphicsView.NoDrag)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setBackgroundBrush(Qt.black)
        self.setMinimumSize(200, 200)

    def set_images(self, pixmap1: QPixmap, pixmap2: QPixmap) -> None:
        self._base_item.setPixmap(pixmap1)
        self._base_item.setPos(0, 0)
        self._overlay_item.setPixmap(pixmap2)
        self._overlay_item.setPos((pixmap1.width() - pixmap2.width()) / 2, (pixmap1.height() - pixmap2.height()) / 2)
        self._scene.setSceneRect(self._scene.itemsBoundingRect())
        self.resetTransform()
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.setFocus()

    def manual_offset(self) -> tuple[float, float]:
        pos = self._overlay_item.pos()
        return pos.x(), pos.y()

    def wheelEvent(self, event) -> None:
        factor = self.ZOOM_STEP if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_STEP
        current = self.transform().m11()
        if current <= 0:
            return
        target = max(self.MIN_ZOOM, min(self.MAX_ZOOM, current * factor))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(target / current, target / current)

    def keyPressEvent(self, event) -> None:
        step_map = {
            Qt.Key_Left: (-1, 0),
            Qt.Key_Right: (1, 0),
            Qt.Key_Up: (0, -1),
            Qt.Key_Down: (0, 1),
        }
        direction = step_map.get(event.key())
        if direction is None:
            super().keyPressEvent(event)
            return
        step = self.NUDGE_STEP_PX_SHIFT if event.modifiers() & Qt.ShiftModifier else self.NUDGE_STEP_PX
        self._overlay_item.moveBy(direction[0] * step, direction[1] * step)


class _OverlapWorker(QObject):
    """Runs merge_and_measure_overlap on a background thread.

    Keeps the UI thread free to repaint and respond to the Stop button while
    checkerboard detection (potentially the slowest part, trying many
    candidate pattern sizes) is running.
    """

    finished = Signal(object)  # OverlapResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        image1,
        image2,
        cancel_event: threading.Event,
        manual_guess: tuple[float, float],
    ):
        super().__init__()
        self._image1 = image1
        self._image2 = image2
        self._cancel_event = cancel_event
        self._manual_guess = manual_guess

    def run(self) -> None:
        try:
            result = merge_and_measure_overlap(
                self._image1,
                self._image2,
                self._manual_guess,
                cancel_event=self._cancel_event,
            )
        except OverlapCancelled:
            self.cancelled.emit()
        except Exception as exc:  # includes cv2.error alongside ValueError
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class OverlapTab(QWidget):
    """Checkerboard-based overlap measurement between two .raw images.

    Manual placement (drag image 2 over image 1) is always the first step;
    "Confirm & Refine" then runs align_with_manual_guess's local
    pixel-correlation refinement from that starting point. There is no
    automatic-alignment path - see align_with_manual_guess's docstring for
    why a human-supplied starting position is more robust here than
    feature-based methods.
    """

    def __init__(self):
        super().__init__()
        self._thread: QThread | None = None
        self._worker: _OverlapWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._has_result = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.result_label = ZoomableImageView()
        self.result_label.clear_image(_PLACEHOLDER_TEXT)
        self.result_label.zoom_changed.connect(self._on_zoom_changed)
        root.addWidget(self.result_label, stretch=1)

        self.manual_align_view = _ManualAlignView()
        self.manual_align_view.setVisible(False)
        root.addWidget(self.manual_align_view, stretch=1)

        self.zoom_row_widget = QWidget()
        zoom_row = QHBoxLayout(self.zoom_row_widget)
        zoom_row.setContentsMargins(0, 0, 0, 0)
        zoom_out_button = QPushButton("−")
        zoom_out_button.setFixedWidth(28)
        zoom_out_button.clicked.connect(self.result_label.zoom_out)
        zoom_row.addWidget(zoom_out_button)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setFixedWidth(50)
        zoom_row.addWidget(self.zoom_label)

        zoom_in_button = QPushButton("+")
        zoom_in_button.setFixedWidth(28)
        zoom_in_button.clicked.connect(self.result_label.zoom_in)
        zoom_row.addWidget(zoom_in_button)

        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.result_label.fit_to_window)
        zoom_row.addWidget(fit_button)

        actual_size_button = QPushButton("100%")
        actual_size_button.clicked.connect(self.result_label.zoom_actual_size)
        zoom_row.addWidget(actual_size_button)

        zoom_row.addStretch(1)
        root.addWidget(self.zoom_row_widget)

        result_controls_row = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        result_controls_row.addWidget(self.stop_button)

        self.realign_button = QPushButton("Re-align")
        self.realign_button.setVisible(False)
        self.realign_button.clicked.connect(self._on_realign_clicked)
        result_controls_row.addWidget(self.realign_button)

        result_controls_row.addStretch(1)
        root.addLayout(result_controls_row)

        manual_controls_row = QHBoxLayout()
        self.manual_confirm_button = QPushButton("Confirm && Refine")
        self.manual_confirm_button.setVisible(False)
        self.manual_confirm_button.clicked.connect(self._on_manual_confirm_clicked)
        manual_controls_row.addWidget(self.manual_confirm_button)

        self.manual_cancel_button = QPushButton("Cancel")
        self.manual_cancel_button.setVisible(False)
        self.manual_cancel_button.clicked.connect(self._on_manual_cancel_clicked)
        manual_controls_row.addWidget(self.manual_cancel_button)

        manual_controls_row.addStretch(1)
        root.addLayout(manual_controls_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.results_label)

        slots_row = QHBoxLayout()
        self.slot1 = _ImageSlot("Image 1")
        self.slot1.item_changed.connect(self._on_slot_item_changed)
        self.slot1.selection_failed.connect(lambda msg: self._set_status(msg, error=True))
        slots_row.addWidget(self.slot1)

        self.slot2 = _ImageSlot("Image 2")
        self.slot2.item_changed.connect(self._on_slot_item_changed)
        self.slot2.selection_failed.connect(lambda msg: self._set_status(msg, error=True))
        slots_row.addWidget(self.slot2)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear_clicked)
        slots_row.addWidget(self.clear_button, alignment=Qt.AlignBottom)

        slots_row.addStretch(1)
        root.addLayout(slots_row)

    # ---------- file list (pushed in from MainWindow's Viewer folder) ----------
    def set_available_files(self, files: list[Path]) -> None:
        self.slot1.set_available_files(files)
        self.slot2.set_available_files(files)
        self._has_result = False
        self._set_manual_mode(False)
        self.result_label.clear_image(_PLACEHOLDER_TEXT)
        self.results_label.setText("")
        self._set_status("")

    def _on_slot_item_changed(self) -> None:
        # A new image selection invalidates any prior result and drag position.
        self._has_result = False
        both_ready = self.slot1.item is not None and self.slot2.item is not None
        if both_ready:
            pixmap1 = QPixmap.fromImage(numpy_to_qimage(self.slot1.item.image))
            pixmap2 = QPixmap.fromImage(numpy_to_qimage(self.slot2.item.image))
            self.manual_align_view.set_images(pixmap1, pixmap2)
            self._set_manual_mode(True)
        else:
            self._set_manual_mode(False)
            self.result_label.clear_image(_PLACEHOLDER_TEXT)
            self.results_label.setText("")

    def _on_clear_clicked(self) -> None:
        self.slot1.combo.setCurrentIndex(0)
        self.slot2.combo.setCurrentIndex(0)

    # ---------- manual align / compute ----------
    def _on_manual_confirm_clicked(self) -> None:
        offset = self.manual_align_view.manual_offset()
        self._set_manual_mode(False)
        self._run_worker(offset)

    def _on_manual_cancel_clicked(self) -> None:
        self._set_manual_mode(False)

    def _on_realign_clicked(self) -> None:
        self._set_manual_mode(True)

    def _set_manual_mode(self, enabled: bool) -> None:
        self.manual_align_view.setVisible(enabled)
        self.manual_confirm_button.setVisible(enabled)
        self.manual_cancel_button.setVisible(enabled)
        self.result_label.setVisible(not enabled)
        self.zoom_row_widget.setVisible(not enabled)
        self.stop_button.setVisible(not enabled)
        self.realign_button.setVisible(not enabled and self._has_result)
        # Changing either image mid-drag would leave the drag view showing
        # stale pixmaps relative to the (now different) cached images.
        self.slot1.setEnabled(not enabled)
        self.slot2.setEnabled(not enabled)

    def _run_worker(self, manual_guess: tuple[float, float]) -> None:
        self._cancel_event = threading.Event()
        self._worker = _OverlapWorker(
            self.slot1.item.image, self.slot2.item.image, self._cancel_event, manual_guess
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)

        self.stop_button.setEnabled(True)
        self.clear_button.setEnabled(False)
        self._set_status("Computing overlap…", error=False)
        self._thread.start()

    def _on_stop_clicked(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.stop_button.setEnabled(False)
        self._set_status("Stopping…", error=False)

    def _on_worker_finished(self, result: OverlapResult) -> None:
        self.result_label.set_image(QPixmap.fromImage(numpy_to_qimage(result.merged_image)))
        self._has_result = True
        if (result.manual_confidence or 0.0) < MANUAL_ALIGN_LOW_CONFIDENCE_THRESHOLD:
            self._set_status(
                "Overlap computed - manual alignment confidence is low; inspect the result "
                "closely, or Re-align.",
                error=True,
            )
        else:
            self._set_status("Overlap computed.", error=False)
        self.results_label.setText(_format_results(result))
        self._cleanup_worker()
        self._set_manual_mode(False)

    def _on_worker_failed(self, message: str) -> None:
        self._set_status(f"Overlap computation failed: {message}", error=True)
        self._cleanup_worker()
        self._set_manual_mode(True)

    def _on_worker_cancelled(self) -> None:
        self._set_status("Overlap computation stopped.", error=True)
        self._cleanup_worker()
        self._set_manual_mode(True)

    def _cleanup_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self.stop_button.setEnabled(False)
        self.clear_button.setEnabled(True)

    # ---------- helpers ----------
    def _on_zoom_changed(self, percent: int) -> None:
        self.zoom_label.setText(f"{percent}%")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setStyleSheet(f"color: {'#c0392b' if error else '#2e7d32'};")
        self.status_label.setText(text)


def _format_results(result: OverlapResult) -> str:
    lines = [
        f"Overlap: {result.overlap_area_px:,} px "
        f"({result.percent_of_image1:.1f}% of image 1, {result.percent_of_image2:.1f}% of image 2)"
    ]
    if result.mm is not None:
        lines.append(
            f"Overlap size: {result.mm.width_mm:.1f} x {result.mm.height_mm:.1f} mm "
            f"(area {result.mm.area_mm2 / 100.0:.1f} cm2) - using the checkerboard's 25x25mm "
            f"squares as scale reference, most accurate where the scene is roughly flat"
        )
    else:
        lines.append("Physical size: unavailable (no checkerboard detected in either image)")
    lines.append(
        f"Checkerboard corners detected: image 1 = {result.corners_detected_1}, "
        f"image 2 = {result.corners_detected_2}"
    )
    lines.append(_format_alignment_line(result.manual_confidence))
    return "\n".join(lines)


def _format_alignment_line(confidence: float | None) -> str:
    conf_text = f"{confidence:.2f}" if confidence is not None else "unavailable"
    if confidence is not None and confidence < MANUAL_ALIGN_LOW_CONFIDENCE_THRESHOLD:
        return (
            f"Alignment: manual placement refined by local pixel correlation "
            f"(confidence {conf_text}, low) - inspect the result closely, or re-align"
        )
    return f"Alignment: manual placement refined by local pixel correlation (confidence {conf_text})"
