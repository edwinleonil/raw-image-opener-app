"""Main window for the Raw Image Viewer."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .capture_folder import (
    find_capture_folder_info,
    format_f_number,
    format_working_distance,
)
from .hdr_tab import HdrBurstTab
from .ocr_processing import run_ocr
from .overlap_tab import OverlapTab
from .trials_tab import TrialsTab
from .raw_loader import (
    RawFormatError,
    adjust_brightness_contrast,
    load_sidecar_metadata,
    process_raw_file,
    rotate_image,
    sharpen_image,
)
from .widgets import ZoomableImageView, numpy_to_qimage

ORG_NAME = "RawImageOpener"
APP_NAME = "RawImageViewer"
RAW_EXTENSIONS = {".raw"}
ROTATION_OPTIONS = [("0°", 0), ("90° CW", 90), ("180°", 180), ("90° CCW (-90°)", 270)]


class _OcrWorker(QObject):
    """Runs OCR on a cropped image region on a background thread.

    Keeps the UI thread responsive while PaddleOCR loads its models (a
    one-time cost, paid on the first OCR run) and recognizes text.
    """

    finished = Signal(object)  # OcrResult
    failed = Signal(str)

    def __init__(self, crop):
        super().__init__()
        self._crop = crop

    @Slot()
    def run(self) -> None:
        try:
            result = run_ocr(self._crop)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raw Image Viewer")
        self.resize(1100, 750)

        self.settings = QSettings(ORG_NAME, APP_NAME)

        self.folder: Path | None = None
        self.files: list[Path] = []
        self.index: int = -1
        self._last_rendered_path: Path | None = None
        self._measurement_count: int = 0
        self._current_image8 = None
        self._ocr_thread: QThread | None = None
        self._ocr_worker: _OcrWorker | None = None
        self._ocr_result_valid = False
        self._close_pending = False

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._render_current)

        self._build_ui()
        self._load_settings()
        self._connect_shortcuts()
        self._update_nav_state()

    # ---------- UI construction ----------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)

        left = QVBoxLayout()
        self.image_label = ZoomableImageView()
        self.image_label.clear_image("Open a folder, or paste an image path, to begin")
        self.image_label.zoom_changed.connect(self._on_zoom_changed)
        self.image_label.measurement_added.connect(self._on_measurement_added)
        self.image_label.measurements_cleared.connect(self._on_measurements_cleared)
        self.image_label.ocr_region_selected.connect(self._on_ocr_region_selected)
        left.addWidget(self.image_label, stretch=1)

        zoom_bar = QHBoxLayout()
        zoom_out_button = QPushButton("−")
        zoom_out_button.setFixedWidth(28)
        zoom_out_button.clicked.connect(self.image_label.zoom_out)
        zoom_bar.addWidget(zoom_out_button)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setFixedWidth(50)
        zoom_bar.addWidget(self.zoom_label)

        zoom_in_button = QPushButton("+")
        zoom_in_button.setFixedWidth(28)
        zoom_in_button.clicked.connect(self.image_label.zoom_in)
        zoom_bar.addWidget(zoom_in_button)

        fit_button = QPushButton("Fit")
        fit_button.clicked.connect(self.image_label.fit_to_window)
        zoom_bar.addWidget(fit_button)

        actual_size_button = QPushButton("100%")
        actual_size_button.clicked.connect(self.image_label.zoom_actual_size)
        zoom_bar.addWidget(actual_size_button)

        self.measure_button = QPushButton("Measure")
        self.measure_button.setCheckable(True)
        self.measure_button.toggled.connect(self._on_measure_mode_toggled)
        zoom_bar.addWidget(self.measure_button)

        self.ocr_button = QPushButton("OCR")
        self.ocr_button.setCheckable(True)
        self.ocr_button.toggled.connect(self._on_ocr_mode_toggled)
        zoom_bar.addWidget(self.ocr_button)

        zoom_bar.addStretch(1)
        left.addLayout(zoom_bar)

        nav_bar = QHBoxLayout()
        self.prev_button = QPushButton("◀ Previous")
        self.next_button = QPushButton("Next ▶")
        self.status_label = QLabel("No folder selected")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.prev_button.clicked.connect(self.show_previous)
        self.next_button.clicked.connect(self.show_next)
        nav_bar.addWidget(self.prev_button)
        nav_bar.addWidget(self.status_label, stretch=1)
        nav_bar.addWidget(self.next_button)
        left.addLayout(nav_bar)

        root.addLayout(left, stretch=3)

        panel = QVBoxLayout()

        open_button = QPushButton("Open Folder…")
        open_button.clicked.connect(self.open_folder)
        panel.addWidget(open_button)

        self.folder_label = QLabel("No folder selected")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet("color: #888888;")
        panel.addWidget(self.folder_label)

        file_path_row = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText(
            "Paste a path to a .raw image or a folder, then Enter"
        )
        self.file_path_edit.setToolTip(
            "A .raw image loads its whole folder and jumps to that image; a "
            "folder loads from its first image. The box then tracks the image "
            "on screen."
        )
        self.file_path_edit.returnPressed.connect(self._load_typed_file_path)
        file_path_row.addWidget(self.file_path_edit, stretch=1)

        browse_file_button = QPushButton("Browse…")
        browse_file_button.clicked.connect(self._browse_for_file)
        file_path_row.addWidget(browse_file_button)
        panel.addLayout(file_path_row)

        metadata_box = QGroupBox("Capture metadata")
        metadata_form = QFormLayout(metadata_box)

        self.sidecar_label = QLabel("")
        self.sidecar_label.setWordWrap(True)
        self.sidecar_label.setStyleSheet("color: #666666;")
        metadata_form.addRow(self.sidecar_label)

        # Working distance and aperture come from the capture folder's name
        # (wd600f4_Capture_...), not from the sidecar - see capture_folder.py.
        self.working_distance_label = QLabel("—")
        metadata_form.addRow("Working distance", self.working_distance_label)

        self.aperture_label = QLabel("—")
        metadata_form.addRow("Aperture", self.aperture_label)

        self.exposure_value_label = QLabel("—")
        metadata_form.addRow("exposure_us", self.exposure_value_label)

        self.gain_value_label = QLabel("—")
        metadata_form.addRow("gain_db", self.gain_value_label)

        self.light_current_value_label = QLabel("—")
        metadata_form.addRow("light_current_ma", self.light_current_value_label)

        panel.addWidget(metadata_box)

        rotation_box = QGroupBox("Rotation")
        rotation_form = QFormLayout(rotation_box)
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems([label for label, _ in ROTATION_OPTIONS])
        rotation_form.addRow("Angle", self.rotation_combo)
        panel.addWidget(rotation_box)

        adjustments_box = QGroupBox("Adjustments")
        adjustments_form = QFormLayout(adjustments_box)

        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_value_label = QLabel("0")
        self.brightness_value_label.setFixedWidth(40)
        self.brightness_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        brightness_row = QHBoxLayout()
        brightness_row.addWidget(self.brightness_slider)
        brightness_row.addWidget(self.brightness_value_label)
        adjustments_form.addRow("Brightness", brightness_row)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_value_label = QLabel("100%")
        self.contrast_value_label.setFixedWidth(40)
        self.contrast_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        contrast_row = QHBoxLayout()
        contrast_row.addWidget(self.contrast_slider)
        contrast_row.addWidget(self.contrast_value_label)
        adjustments_form.addRow("Contrast", contrast_row)

        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(0)
        self.sharpness_value_label = QLabel("0")
        self.sharpness_value_label.setFixedWidth(40)
        self.sharpness_value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        sharpness_row = QHBoxLayout()
        sharpness_row.addWidget(self.sharpness_slider)
        sharpness_row.addWidget(self.sharpness_value_label)
        adjustments_form.addRow("Sharpness", sharpness_row)

        self.brightness_slider.valueChanged.connect(
            lambda v: self.brightness_value_label.setText(str(v))
        )
        self.contrast_slider.valueChanged.connect(
            lambda v: self.contrast_value_label.setText(f"{v + 100}%")
        )
        self.sharpness_slider.valueChanged.connect(
            lambda v: self.sharpness_value_label.setText(str(v))
        )

        self.adjustments_reset_button = QPushButton("Reset")
        self.adjustments_reset_button.clicked.connect(self._on_reset_adjustments)
        adjustments_form.addRow("", self.adjustments_reset_button)

        panel.addWidget(adjustments_box)

        measure_box = QGroupBox("Measure")
        measure_layout = QVBoxLayout(measure_box)

        self.measurements_list = QListWidget()
        self.measurements_list.setMaximumHeight(120)
        measure_layout.addWidget(self.measurements_list)

        self.clear_measurements_button = QPushButton("Clear All")
        self.clear_measurements_button.clicked.connect(self.image_label.clear_measurements)
        measure_layout.addWidget(self.clear_measurements_button)

        panel.addWidget(measure_box)

        ocr_box = QGroupBox("OCR")
        ocr_layout = QVBoxLayout(ocr_box)

        self.ocr_result_edit = QPlainTextEdit()
        self.ocr_result_edit.setReadOnly(True)
        self.ocr_result_edit.setMaximumHeight(60)
        self.ocr_result_edit.setPlaceholderText(
            "Enable OCR mode and drag a box around a stamped/engraved code"
        )
        ocr_layout.addWidget(self.ocr_result_edit)

        self.ocr_confidence_label = QLabel("")
        self.ocr_confidence_label.setStyleSheet("color: #666666;")
        ocr_layout.addWidget(self.ocr_confidence_label)

        self.ocr_clear_button = QPushButton("Clear")
        self.ocr_clear_button.clicked.connect(self._on_ocr_clear_clicked)
        ocr_layout.addWidget(self.ocr_clear_button)

        panel.addWidget(ocr_box)

        hint = QLabel(
            "Each .raw file needs a matching <name>.json sidecar (width, "
            "height, dtype, format, plus capture settings) next to it - "
            "files without one can't be opened. The panel above shows the "
            "capture settings from that sidecar."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666666;")
        panel.addWidget(hint)

        panel.addStretch(1)

        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #c0392b;")
        panel.addWidget(self.error_label)

        root.addLayout(panel, stretch=1)

        self.viewer_widget = central
        self.trials_tab = TrialsTab()
        self.trials_tab.folder_selected.connect(self._on_trial_folder_selected)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.trials_tab, "Load Data")
        self.tabs.addTab(central, "Viewer")
        self.tabs.addTab(HdrBurstTab(), "HDR / Burst Stacking")
        self.overlap_tab = OverlapTab()
        self.tabs.addTab(self.overlap_tab, "Overlap Measurement")
        self.setCentralWidget(self.tabs)

        for signal in (
            self.rotation_combo.currentIndexChanged,
            self.brightness_slider.valueChanged,
            self.contrast_slider.valueChanged,
            self.sharpness_slider.valueChanged,
        ):
            signal.connect(self._on_format_changed)

        self.rotation_combo.currentIndexChanged.connect(
            lambda *_: self.image_label.clear_measurements()
        )
        self.rotation_combo.currentIndexChanged.connect(
            lambda *_: self.image_label.clear_ocr_box()
        )

    def _connect_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self._on_next_shortcut)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self._on_previous_shortcut)

    # A plain single-key shortcut outranks the focused widget, so the path
    # box never sees its own arrow keys. Hand them back while it has focus -
    # otherwise pressing Left to inspect a pasted path would step back an
    # image and overwrite the text under the cursor.
    def _on_next_shortcut(self) -> None:
        if self.file_path_edit.hasFocus():
            self.file_path_edit.cursorForward(False, 1)
            return
        self.show_next()

    def _on_previous_shortcut(self) -> None:
        if self.file_path_edit.hasFocus():
            self.file_path_edit.cursorBackward(False, 1)
            return
        self.show_previous()

    # ---------- settings persistence ----------
    def _load_settings(self) -> None:
        s = self.settings
        self.rotation_combo.setCurrentIndex(int(s.value("rotation_index", 0)))
        self.brightness_slider.setValue(int(s.value("brightness", 0)))
        self.contrast_slider.setValue(int(s.value("contrast", 0)))
        self.sharpness_slider.setValue(int(s.value("sharpness", 0)))
        last_folder = s.value("last_folder", "")
        if last_folder and Path(last_folder).is_dir():
            self._set_folder(Path(last_folder))

        trials_parent_folder = s.value("trials_parent_folder", "")
        if trials_parent_folder and Path(trials_parent_folder).is_dir():
            self.trials_tab.set_parent_folder(Path(trials_parent_folder))

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("rotation_index", self.rotation_combo.currentIndex())
        s.setValue("brightness", self.brightness_slider.value())
        s.setValue("contrast", self.contrast_slider.value())
        s.setValue("sharpness", self.sharpness_slider.value())
        if self.folder:
            s.setValue("last_folder", str(self.folder))
        trials_parent_folder = self.trials_tab.current_parent_folder()
        if trials_parent_folder:
            s.setValue("trials_parent_folder", str(trials_parent_folder))

    def closeEvent(self, event) -> None:
        if self._ocr_thread is not None:
            self._close_pending = True
            self._on_ocr_clear_clicked()
            self.ocr_confidence_label.setText("Waiting for OCR to finish before closing...")
            event.ignore()
            return
        self._save_settings()
        super().closeEvent(event)

    # ---------- folder / navigation ----------
    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing .raw images")
        if folder:
            self._set_folder(Path(folder))

    def _browse_for_file(self) -> None:
        # Start where the user already is, so Browse... is a sibling picker
        # rather than a trip back to the home directory.
        start_dir = str(self.folder) if self.folder else ""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select image file", start_dir, "Raw images (*.raw);;All files (*)"
        )
        if file:
            self.file_path_edit.setText(file)
            self._load_file_path(Path(file))

    def _load_typed_file_path(self) -> None:
        text = self.file_path_edit.text().strip()
        if text:
            self._load_file_path(Path(text))

    def _load_file_path(self, path: Path) -> None:
        # A single image is loaded as its whole folder, so Next/Previous
        # still walks the other captures next to it. A folder path behaves
        # the same as Open Folder... on it.
        path = Path(str(path).strip().strip('"')).expanduser()
        if path.is_dir():
            self._set_folder(path)
            return
        if not path.is_file():
            self.error_label.setText(f"File not found: {path}")
            return
        if path.suffix.lower() not in RAW_EXTENSIONS:
            expected = ", ".join(sorted(RAW_EXTENSIONS))
            self.error_label.setText(
                f"Not an image this viewer opens (expected {expected}): {path.name}"
            )
            return
        self._set_folder(path.parent, select=path)
        # Set after the render, which clears error_label on success. Only
        # reachable if the file went away between is_file() and iterdir().
        if self.index < 0 or self.files[self.index] != path:
            self.error_label.setText(
                f"{path.name} is no longer in {path.parent} — showing that folder instead"
            )

    def _set_folder(self, folder: Path, select: Path | None = None) -> None:
        """Load every .raw image in `folder`, starting at `select` if given.

        `select` is how a pasted single-file path gets the whole folder
        behind Next/Previous while still opening the image that was asked
        for. Matched with `==`, which on Windows ignores case and separator
        style, so a pasted `top_1.raw` finds the on-disk `Top_1.raw`. An
        unmatched `select` falls back to the first image - the caller
        reports that, after the render, because `_render_current` clears
        `error_label` on success.
        """
        files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS
        )
        self.folder = folder
        self.files = files
        if not files:
            self.index = -1
        elif select is None:
            self.index = 0
        else:
            try:
                self.index = files.index(select)
            except ValueError:
                self.index = 0
        self.folder_label.setText(f"Folder: {folder.name}")
        self.folder_label.setToolTip(str(folder))
        self.overlap_tab.set_available_files(files)
        self._save_settings()
        if not files:
            self._current_image8 = None
            self._last_rendered_path = None
            self._on_ocr_clear_clicked()
            # Nothing is on screen, so the box must not keep pointing at an
            # image from the folder that was open before.
            self.file_path_edit.clear()
            self.image_label.clear_image(f"No .raw files found in:\n{folder}")
            self.status_label.setText("No .raw files found")
            self._update_nav_state()
            return
        self._render_current()

    def _on_trial_folder_selected(self, folder: Path) -> None:
        self._set_folder(folder)
        self.tabs.setCurrentWidget(self.viewer_widget)

    def show_next(self) -> None:
        if self.index + 1 < len(self.files):
            self.index += 1
            self._render_current()

    def show_previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._render_current()

    def _update_nav_state(self) -> None:
        has_files = bool(self.files)
        self.prev_button.setEnabled(has_files and self.index > 0)
        self.next_button.setEnabled(has_files and self.index + 1 < len(self.files))

    # ---------- format controls ----------
    def _on_format_changed(self) -> None:
        self._on_ocr_clear_clicked()
        self._refresh_timer.start()

    def _on_zoom_changed(self, percent: int) -> None:
        self.zoom_label.setText(f"{percent}%")

    def _on_reset_adjustments(self) -> None:
        for slider in (self.brightness_slider, self.contrast_slider, self.sharpness_slider):
            slider.setValue(0)

    def _on_measure_mode_toggled(self, enabled: bool) -> None:
        if enabled and self.ocr_button.isChecked():
            self.ocr_button.setChecked(False)
        self.image_label.set_measure_mode(enabled)

    def _on_ocr_mode_toggled(self, enabled: bool) -> None:
        if enabled and self.measure_button.isChecked():
            self.measure_button.setChecked(False)
        self.image_label.set_ocr_mode(enabled and self._ocr_thread is None)

    def _on_ocr_clear_clicked(self) -> None:
        self._ocr_result_valid = False
        self.image_label.clear_ocr_box()
        self.ocr_result_edit.setPlainText("")
        self.ocr_confidence_label.setText("")
        if self.error_label.text().startswith("OCR failed:"):
            self.error_label.setText("")

    def _on_ocr_region_selected(self, rect) -> None:
        if (
            self._current_image8 is None
            or self._ocr_thread is not None
            or self._close_pending
            or self._refresh_timer.isActive()
        ):
            return

        height, width = self._current_image8.shape[:2]
        x0, y0 = max(0, rect.x()), max(0, rect.y())
        x1, y1 = min(width, rect.x() + rect.width()), min(height, rect.y() + rect.height())
        if x1 <= x0 or y1 <= y0:
            return
        crop = self._current_image8[y0:y1, x0:x1].copy()

        self._ocr_result_valid = True
        self._ocr_worker = _OcrWorker(crop)
        self._ocr_thread = QThread(self)
        self._ocr_worker.moveToThread(self._ocr_thread)

        self._ocr_thread.started.connect(self._ocr_worker.run)
        self._ocr_worker.finished.connect(self._on_ocr_finished)
        self._ocr_worker.failed.connect(self._on_ocr_failed)
        self._ocr_worker.finished.connect(self._ocr_thread.quit)
        self._ocr_worker.failed.connect(self._ocr_thread.quit)
        self._ocr_thread.finished.connect(self._ocr_worker.deleteLater)
        self._ocr_thread.finished.connect(self._cleanup_ocr_worker)
        self._ocr_thread.finished.connect(self._ocr_thread.deleteLater)

        self.ocr_button.setEnabled(False)
        self.image_label.set_ocr_mode(False)
        self.error_label.setText("")
        self.ocr_result_edit.setPlainText("")
        self.ocr_confidence_label.setText("Reading text… (first run loads the OCR model)")
        self._ocr_thread.start()

    @Slot(object)
    def _on_ocr_finished(self, result) -> None:
        if not self._ocr_result_valid:
            return
        self.ocr_result_edit.setPlainText(result.text)
        self.ocr_confidence_label.setText(
            "No text found" if not result.text else f"{result.confidence * 100:.0f}% confidence"
        )

    @Slot(str)
    def _on_ocr_failed(self, message: str) -> None:
        if not self._ocr_result_valid:
            return
        self.error_label.setText(f"OCR failed: {message}")
        self.ocr_confidence_label.setText("")

    @Slot()
    def _cleanup_ocr_worker(self) -> None:
        self._ocr_thread.wait()
        self.ocr_button.setEnabled(True)
        self._ocr_thread = None
        self._ocr_worker = None
        if self._close_pending:
            self.close()
        else:
            self.image_label.set_ocr_mode(self.ocr_button.isChecked())

    def _on_measurement_added(self, measurement) -> None:
        self._measurement_count += 1
        p1, p2, dist = measurement.p1, measurement.p2, measurement.distance_px
        text = (
            f"{self._measurement_count}: ({p1.x()},{p1.y()}) → "
            f"({p2.x()},{p2.y()}) = {dist:.1f} px"
        )
        self.measurements_list.addItem(text)

    def _on_measurements_cleared(self) -> None:
        self.measurements_list.clear()
        self._measurement_count = 0

    def _current_adjustments(self) -> tuple[int, int, int]:
        return (
            self.brightness_slider.value(),
            self.contrast_slider.value(),
            self.sharpness_slider.value(),
        )

    # ---------- rendering ----------
    def _render_current(self) -> None:
        self._on_ocr_clear_clicked()
        self._current_image8 = None
        if self.index < 0 or self.index >= len(self.files):
            return
        path = self.files[self.index]
        # The box tracks the image on screen so its path can be copied back
        # out. setText emits neither returnPressed nor textEdited, so this
        # cannot re-enter _load_typed_file_path. Set before the failure
        # returns below: a file that won't decode is exactly the one whose
        # path you want to go and look at.
        self.file_path_edit.setText(str(path))
        is_new_image = path != self._last_rendered_path
        self._last_rendered_path = path

        sidecar = load_sidecar_metadata(path)
        self._update_metadata_panel(path, sidecar)
        if sidecar is None:
            self._current_image8 = None
            message = f"{path.name}: no matching {path.with_suffix('.json').name} sidecar found"
            self.image_label.clear_image(message)
            self.error_label.setText(message)
            self.status_label.setText(f"{self.index + 1} / {len(self.files)} — {path.name}")
            self._update_nav_state()
            return

        width, height = sidecar["width"], sidecar["height"]
        bytes_per_pixel = sidecar["bytes_per_pixel"]
        big_endian = sidecar["big_endian"]
        pattern = sidecar["pattern"]
        norm_mode = "Auto (min/max)"

        try:
            image8, _ = process_raw_file(
                path, width, height, bytes_per_pixel, big_endian, pattern, norm_mode
            )
        except RawFormatError as exc:
            self._current_image8 = None
            self.image_label.clear_image(str(exc))
            self.error_label.setText(str(exc))
            self.status_label.setText(f"{self.index + 1} / {len(self.files)} — {path.name}")
            self._update_nav_state()
            return

        self.error_label.setText("")

        brightness, contrast, sharpness = self._current_adjustments()
        if brightness or contrast:
            image8 = adjust_brightness_contrast(image8, brightness, contrast)
        if sharpness:
            image8 = sharpen_image(image8, sharpness)

        rotation_degrees = ROTATION_OPTIONS[self.rotation_combo.currentIndex()][1]
        if rotation_degrees:
            image8 = rotate_image(image8, rotation_degrees)

        self._current_image8 = image8
        qimage = numpy_to_qimage(image8)
        self.image_label.set_image(QPixmap.fromImage(qimage), reset_view=is_new_image)
        self.status_label.setText(f"{self.index + 1} / {len(self.files)} — {path.name}")
        self._update_nav_state()

    def _update_metadata_panel(self, path: Path, sidecar: dict | None) -> None:
        # Set before the no-sidecar return below: the folder name still says
        # how the shot was set up even when its sidecar is missing or broken.
        info = find_capture_folder_info(path)
        self.working_distance_label.setText(format_working_distance(info))
        self.aperture_label.setText(format_f_number(info))
        tooltip = f"From folder {info.folder_name}" if info else ""
        self.working_distance_label.setToolTip(tooltip)
        self.aperture_label.setToolTip(tooltip)

        if sidecar is None:
            self.sidecar_label.setText("")
            for label in (
                self.exposure_value_label,
                self.gain_value_label,
                self.light_current_value_label,
            ):
                label.setText("—")
            return

        self.sidecar_label.setText(f"From {sidecar['source']}")
        self.exposure_value_label.setText(
            "—" if sidecar["exposure_us"] is None else str(sidecar["exposure_us"])
        )
        self.gain_value_label.setText(
            "—" if sidecar["gain_db"] is None else str(sidecar["gain_db"])
        )
        self.light_current_value_label.setText(
            "—" if sidecar["light_current_ma"] is None else str(sidecar["light_current_ma"])
        )
