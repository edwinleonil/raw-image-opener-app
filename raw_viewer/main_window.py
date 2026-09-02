"""Main window for the Raw Image Viewer."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QSettings, QTimer
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .hdr_tab import HdrBurstTab
from .raw_loader import (
    BAYER_PATTERNS,
    NORMALIZATION_MODES,
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
        self.image_label.clear_image("Open a folder to begin")
        self.image_label.zoom_changed.connect(self._on_zoom_changed)
        self.image_label.measurement_added.connect(self._on_measurement_added)
        self.image_label.measurements_cleared.connect(self._on_measurements_cleared)
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
        self.measure_button.toggled.connect(self.image_label.set_measure_mode)
        zoom_bar.addWidget(self.measure_button)

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

        format_box = QGroupBox("Raw format")
        form = QFormLayout(format_box)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 20000)
        self.width_spin.setValue(5328)
        form.addRow("Width", self.width_spin)

        self.auto_height_check = QCheckBox("Auto (from file size)")
        self.auto_height_check.setChecked(True)
        form.addRow("", self.auto_height_check)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(1, 20000)
        self.height_spin.setValue(4608)
        self.height_spin.setEnabled(False)
        form.addRow("Height", self.height_spin)

        self.bpp_combo = QComboBox()
        self.bpp_combo.addItems(["1 byte (8-bit)", "2 bytes (16-bit)"])
        form.addRow("Bytes / pixel", self.bpp_combo)

        self.endian_combo = QComboBox()
        self.endian_combo.addItems(["Little-endian", "Big-endian"])
        form.addRow("Byte order", self.endian_combo)

        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(BAYER_PATTERNS)
        self.pattern_combo.setCurrentIndex(BAYER_PATTERNS.index("RGGB"))
        form.addRow("Bayer pattern", self.pattern_combo)

        self.norm_combo = QComboBox()
        self.norm_combo.addItems(NORMALIZATION_MODES)
        form.addRow("Display levels", self.norm_combo)

        self._format_controls = [
            self.width_spin,
            self.auto_height_check,
            self.height_spin,
            self.bpp_combo,
            self.endian_combo,
            self.pattern_combo,
        ]

        panel.addWidget(format_box)

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

        self.sidecar_label = QLabel("")
        self.sidecar_label.setWordWrap(True)
        self.sidecar_label.setStyleSheet("color: #2e7d32;")
        panel.addWidget(self.sidecar_label)

        hint = QLabel(
            "These .raw files have no header, so width/height/bit depth "
            "can't be detected automatically. Adjust the fields above until "
            "the preview looks correct - a wrong width shows up as diagonal "
            "tearing. If a matching <name>.json sidecar sits next to a .raw "
            "file, its format is used automatically instead."
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

        tabs = QTabWidget()
        tabs.addTab(central, "Viewer")
        tabs.addTab(HdrBurstTab(), "HDR / Burst Stacking")
        self.setCentralWidget(tabs)

        for signal in (
            self.width_spin.valueChanged,
            self.height_spin.valueChanged,
            self.bpp_combo.currentIndexChanged,
            self.endian_combo.currentIndexChanged,
            self.pattern_combo.currentIndexChanged,
            self.norm_combo.currentIndexChanged,
            self.rotation_combo.currentIndexChanged,
            self.brightness_slider.valueChanged,
            self.contrast_slider.valueChanged,
            self.sharpness_slider.valueChanged,
        ):
            signal.connect(self._on_format_changed)

        for signal in (
            self.width_spin.valueChanged,
            self.height_spin.valueChanged,
            self.auto_height_check.toggled,
            self.rotation_combo.currentIndexChanged,
        ):
            signal.connect(lambda *_: self.image_label.clear_measurements())

        self.auto_height_check.toggled.connect(self._on_auto_height_toggled)

    def _connect_shortcuts(self) -> None:
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.show_next)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.show_previous)

    # ---------- settings persistence ----------
    def _load_settings(self) -> None:
        s = self.settings
        self.width_spin.setValue(int(s.value("width", 5328)))
        self.height_spin.setValue(int(s.value("height", 4608)))
        self.auto_height_check.setChecked(str(s.value("auto_height", "true")) == "true")
        self.bpp_combo.setCurrentIndex(int(s.value("bpp_index", 0)))
        self.endian_combo.setCurrentIndex(int(s.value("endian_index", 0)))
        self.pattern_combo.setCurrentIndex(
            int(s.value("pattern_index", BAYER_PATTERNS.index("RGGB")))
        )
        self.norm_combo.setCurrentIndex(int(s.value("norm_index", 0)))
        self.rotation_combo.setCurrentIndex(int(s.value("rotation_index", 0)))
        self.brightness_slider.setValue(int(s.value("brightness", 0)))
        self.contrast_slider.setValue(int(s.value("contrast", 0)))
        self.sharpness_slider.setValue(int(s.value("sharpness", 0)))
        last_folder = s.value("last_folder", "")
        if last_folder and Path(last_folder).is_dir():
            self._set_folder(Path(last_folder))

    def _save_settings(self) -> None:
        s = self.settings
        s.setValue("width", self.width_spin.value())
        s.setValue("height", self.height_spin.value())
        s.setValue("auto_height", "true" if self.auto_height_check.isChecked() else "false")
        s.setValue("bpp_index", self.bpp_combo.currentIndex())
        s.setValue("endian_index", self.endian_combo.currentIndex())
        s.setValue("pattern_index", self.pattern_combo.currentIndex())
        s.setValue("norm_index", self.norm_combo.currentIndex())
        s.setValue("rotation_index", self.rotation_combo.currentIndex())
        s.setValue("brightness", self.brightness_slider.value())
        s.setValue("contrast", self.contrast_slider.value())
        s.setValue("sharpness", self.sharpness_slider.value())
        if self.folder:
            s.setValue("last_folder", str(self.folder))

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)

    # ---------- folder / navigation ----------
    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder containing .raw images")
        if folder:
            self._set_folder(Path(folder))

    def _set_folder(self, folder: Path) -> None:
        files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in RAW_EXTENSIONS
        )
        self.folder = folder
        self.files = files
        self.index = 0 if files else -1
        self.folder_label.setText(f"Folder: {folder.name}")
        self.folder_label.setToolTip(str(folder))
        self._save_settings()
        if not files:
            self.image_label.clear_image(f"No .raw files found in:\n{folder}")
            self.status_label.setText("No .raw files found")
            self._update_nav_state()
            return
        self._render_current()

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
    def _on_auto_height_toggled(self, checked: bool) -> None:
        self.height_spin.setEnabled(not checked)
        self._on_format_changed()

    def _on_format_changed(self) -> None:
        self._refresh_timer.start()

    def _on_zoom_changed(self, percent: int) -> None:
        self.zoom_label.setText(f"{percent}%")

    def _on_reset_adjustments(self) -> None:
        for slider in (self.brightness_slider, self.contrast_slider, self.sharpness_slider):
            slider.setValue(0)

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

    def _current_format(self):
        bytes_per_pixel = 1 if self.bpp_combo.currentIndex() == 0 else 2
        big_endian = self.endian_combo.currentIndex() == 1
        pattern = self.pattern_combo.currentText()
        norm_mode = self.norm_combo.currentText()
        width = self.width_spin.value()
        height = None if self.auto_height_check.isChecked() else self.height_spin.value()
        return width, height, bytes_per_pixel, big_endian, pattern, norm_mode

    def _current_adjustments(self) -> tuple[int, int, int]:
        return (
            self.brightness_slider.value(),
            self.contrast_slider.value(),
            self.sharpness_slider.value(),
        )

    # ---------- rendering ----------
    def _render_current(self) -> None:
        if self.index < 0 or self.index >= len(self.files):
            return
        path = self.files[self.index]
        is_new_image = path != self._last_rendered_path
        self._last_rendered_path = path
        norm_mode = self.norm_combo.currentText()

        sidecar = load_sidecar_metadata(path)
        self._apply_sidecar_to_controls(sidecar)
        if sidecar is not None:
            width, height = sidecar["width"], sidecar["height"]
            bytes_per_pixel = sidecar["bytes_per_pixel"]
            big_endian = sidecar["big_endian"]
            pattern = sidecar["pattern"]
        else:
            width, height, bytes_per_pixel, big_endian, pattern, _ = self._current_format()

        try:
            image8, resolved_height = process_raw_file(
                path, width, height, bytes_per_pixel, big_endian, pattern, norm_mode
            )
        except RawFormatError as exc:
            self.image_label.clear_image(str(exc))
            self.error_label.setText(str(exc))
            self.status_label.setText(f"{self.index + 1} / {len(self.files)} — {path.name}")
            self._update_nav_state()
            return

        self.error_label.setText("")
        if sidecar is None and self.auto_height_check.isChecked():
            self.height_spin.blockSignals(True)
            self.height_spin.setValue(resolved_height)
            self.height_spin.blockSignals(False)

        brightness, contrast, sharpness = self._current_adjustments()
        if brightness or contrast:
            image8 = adjust_brightness_contrast(image8, brightness, contrast)
        if sharpness:
            image8 = sharpen_image(image8, sharpness)

        rotation_degrees = ROTATION_OPTIONS[self.rotation_combo.currentIndex()][1]
        if rotation_degrees:
            image8 = rotate_image(image8, rotation_degrees)

        qimage = numpy_to_qimage(image8)
        self.image_label.set_image(QPixmap.fromImage(qimage), reset_view=is_new_image)
        self.status_label.setText(f"{self.index + 1} / {len(self.files)} — {path.name}")
        self._update_nav_state()

    def _apply_sidecar_to_controls(self, sidecar: dict | None) -> None:
        for widget in self._format_controls:
            widget.setEnabled(sidecar is None)
        if sidecar is None:
            self.sidecar_label.setText("")
            self.auto_height_check.setEnabled(True)
            self.height_spin.setEnabled(not self.auto_height_check.isChecked())
            return

        self.sidecar_label.setText(
            f"Using detected format from {sidecar['source']}: "
            f"{sidecar['width']}x{sidecar['height']}, {sidecar['pattern']}, "
            f"{sidecar['bytes_per_pixel'] * 8}-bit"
        )
        for widget, value in (
            (self.width_spin, sidecar["width"]),
            (self.height_spin, sidecar["height"]),
        ):
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)

        self.bpp_combo.blockSignals(True)
        self.bpp_combo.setCurrentIndex(0 if sidecar["bytes_per_pixel"] == 1 else 1)
        self.bpp_combo.blockSignals(False)

        self.endian_combo.blockSignals(True)
        self.endian_combo.setCurrentIndex(1 if sidecar["big_endian"] else 0)
        self.endian_combo.blockSignals(False)

        self.pattern_combo.blockSignals(True)
        self.pattern_combo.setCurrentIndex(BAYER_PATTERNS.index(sidecar["pattern"]))
        self.pattern_combo.blockSignals(False)

        self.auto_height_check.blockSignals(True)
        self.auto_height_check.setChecked(False)
        self.auto_height_check.blockSignals(False)
