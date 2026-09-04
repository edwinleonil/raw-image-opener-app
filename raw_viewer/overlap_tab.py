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
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .overlap_processing import (
    OverlapCancelled,
    OverlapImage,
    OverlapResult,
    load_overlap_image,
    merge_and_measure_overlap,
)
from .raw_loader import RawFormatError, rotate_image
from .widgets import ZoomableImageView, numpy_to_qimage

THUMBNAIL_SIZE = 140

# Same options/order as MainWindow's rotation control (main_window.py) - both
# images are rotated the same way before alignment, for rigs whose sensor
# orientation doesn't match the physical scene (e.g. a portrait-mounted
# camera). Alignment itself makes no horizontal/vertical assumption, so this
# is purely a preprocessing convenience, not a requirement.
ROTATION_OPTIONS = [("None", 0), ("90° CW", 90), ("180°", 180), ("90° CCW", 270)]


class _ImageSlot(QWidget):
    """One image slot: a dropdown over the Viewer tab's current folder, plus a thumbnail."""

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


class _OverlapWorker(QObject):
    """Runs merge_and_measure_overlap on a background thread.

    Keeps the UI thread free to repaint and respond to the Stop button while
    checkerboard detection/alignment (potentially the slowest part, trying
    many candidate pattern sizes) is running.
    """

    finished = Signal(object)  # OverlapResult
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, image1, image2, cancel_event: threading.Event):
        super().__init__()
        self._image1 = image1
        self._image2 = image2
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:
            result = merge_and_measure_overlap(
                self._image1, self._image2, cancel_event=self._cancel_event
            )
        except OverlapCancelled:
            self.cancelled.emit()
        except Exception as exc:  # includes cv2.error alongside ValueError
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class OverlapTab(QWidget):
    """Checkerboard-based overlap measurement between two .raw images."""

    def __init__(self):
        super().__init__()
        self._thread: QThread | None = None
        self._worker: _OverlapWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._build_ui()
        self._update_compute_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.result_label = ZoomableImageView()
        self.result_label.clear_image("Load two .raw images below, then Compute Overlap")
        self.result_label.zoom_changed.connect(self._on_zoom_changed)
        root.addWidget(self.result_label, stretch=1)

        zoom_row = QHBoxLayout()
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
        root.addLayout(zoom_row)

        compute_row = QHBoxLayout()
        self.compute_button = QPushButton("Compute Overlap")
        self.compute_button.clicked.connect(self._on_compute_clicked)
        compute_row.addWidget(self.compute_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        compute_row.addWidget(self.stop_button)

        compute_row.addWidget(QLabel("Rotate before aligning:"))
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems([label for label, _ in ROTATION_OPTIONS])
        self.rotation_combo.setCurrentIndex(0)  # "None" - verified default; change per-rig if needed
        compute_row.addWidget(self.rotation_combo)
        compute_row.addStretch(1)
        root.addLayout(compute_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-weight: bold;")
        root.addWidget(self.results_label)

        slots_row = QHBoxLayout()
        self.slot1 = _ImageSlot("Image 1")
        self.slot1.item_changed.connect(self._update_compute_state)
        self.slot1.selection_failed.connect(lambda msg: self._set_status(msg, error=True))
        slots_row.addWidget(self.slot1)

        self.slot2 = _ImageSlot("Image 2")
        self.slot2.item_changed.connect(self._update_compute_state)
        self.slot2.selection_failed.connect(lambda msg: self._set_status(msg, error=True))
        slots_row.addWidget(self.slot2)

        slots_row.addStretch(1)
        root.addLayout(slots_row)

    # ---------- file list (pushed in from MainWindow's Viewer folder) ----------
    def set_available_files(self, files: list[Path]) -> None:
        self.slot1.set_available_files(files)
        self.slot2.set_available_files(files)
        self.result_label.clear_image("Load two .raw images below, then Compute Overlap")
        self.results_label.setText("")
        self._set_status("")

    def _update_compute_state(self) -> None:
        ready = self.slot1.item is not None and self.slot2.item is not None
        is_running = self._thread is not None
        self.compute_button.setEnabled(ready and not is_running)

    # ---------- compute ----------
    def _on_compute_clicked(self) -> None:
        degrees = ROTATION_OPTIONS[self.rotation_combo.currentIndex()][1]
        image1 = rotate_image(self.slot1.item.image, degrees) if degrees else self.slot1.item.image
        image2 = rotate_image(self.slot2.item.image, degrees) if degrees else self.slot2.item.image

        self._cancel_event = threading.Event()
        self._worker = _OverlapWorker(image1, image2, self._cancel_event)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.cancelled.connect(self._on_worker_cancelled)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.cancelled.connect(self._thread.quit)

        self.compute_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._set_status("Computing overlap…", error=False)
        self._thread.start()

    def _on_stop_clicked(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self.stop_button.setEnabled(False)
        self._set_status("Stopping…", error=False)

    def _on_worker_finished(self, result: OverlapResult) -> None:
        self.result_label.set_image(QPixmap.fromImage(numpy_to_qimage(result.merged_image)))
        self._set_status("Overlap computed.", error=False)
        self.results_label.setText(_format_results(result))
        self._cleanup_worker()

    def _on_worker_failed(self, message: str) -> None:
        self._set_status(f"Overlap computation failed: {message}", error=True)
        self._cleanup_worker()

    def _on_worker_cancelled(self) -> None:
        self._set_status("Overlap computation stopped.", error=True)
        self._cleanup_worker()

    def _cleanup_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._cancel_event = None
        self.stop_button.setEnabled(False)
        self._update_compute_state()

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
            f"Checkerboard region (image 1): {result.mm.width_mm:.1f} x {result.mm.height_mm:.1f} mm "
            f"(area {result.mm.area_mm2 / 100.0:.1f} cm2) - physical size only within the "
            f"detected board's own extent, not the whole overlap"
        )
    else:
        lines.append("Physical size: unavailable (no checkerboard detected in image 1)")
    lines.append(
        f"Checkerboard corners detected: image 1 = {result.corners_detected_1}, "
        f"image 2 = {result.corners_detected_2} | alignment inliers = {result.inlier_matches}"
    )
    return "\n".join(lines)
