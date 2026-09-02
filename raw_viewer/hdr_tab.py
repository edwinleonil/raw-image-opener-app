"""HDR exposure fusion / multi-frame burst stacking tab.

Fully self-contained: owns its own image selection, thumbnail previews and
merge logic. Independent of MainWindow and the Viewer tab - it only shares
the generic ImageLabel/numpy_to_qimage helpers and the raw decoding
functions in raw_loader.py.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .hdr_processing import (
    MAX_STACK_SIZE,
    MIN_STACK_SIZE,
    StackImage,
    load_stack_image,
    merge_exposure_fusion,
)
from .raw_loader import RawFormatError
from .widgets import ZoomableImageView, numpy_to_qimage

THUMBNAIL_SIZE = 140

# Where this rig's captures live - each trial subfolder has a
# FullSize_RAW_Images/ directory with the .raw + .json pairs. Used only as
# the starting point for the "Add Image(s)…" dialog; if it's not present
# (e.g. on another machine) we just fall back to the user's home folder.
DEFAULT_BROWSE_ROOT = Path(
    r"C:\Users\me1elar\Documents\GitHub\AI-23-27-NextGen-ImageCapturingApp\NextGenTrials"
)


class _ThumbnailWidget(QWidget):
    """One thumbnail in the selected-images strip: preview + caption + remove."""

    def __init__(self, item: StackImage, on_remove):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        thumb_label = QLabel()
        thumb_label.setAlignment(Qt.AlignCenter)
        thumb_label.setFixedSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE)
        thumb_label.setStyleSheet("background-color: #202020;")
        pixmap = QPixmap.fromImage(numpy_to_qimage(item.image)).scaled(
            THUMBNAIL_SIZE, THUMBNAIL_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        thumb_label.setPixmap(pixmap)
        layout.addWidget(thumb_label)

        caption = QLabel(
            f"{item.path.name}\n{item.exposure_us / 1000.0:.1f} ms · {item.gain_db:.1f} dB"
        )
        caption.setFixedWidth(THUMBNAIL_SIZE)
        caption.setAlignment(Qt.AlignCenter)
        caption.setWordWrap(True)
        caption.setStyleSheet("color: #cccccc; font-size: 10px;")
        layout.addWidget(caption)

        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(lambda: on_remove(item.path))
        layout.addWidget(remove_button)


class HdrBurstTab(QWidget):
    """HDR exposure fusion / burst stacking tab."""

    def __init__(self):
        super().__init__()
        self.items: list[StackImage] = []
        self._browse_dir = DEFAULT_BROWSE_ROOT if DEFAULT_BROWSE_ROOT.is_dir() else Path.home()
        self._build_ui()
        self._refresh_thumbnails()
        self._update_merge_state()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.result_label = ZoomableImageView()
        self.result_label.clear_image("Add 2-5 .raw images below, then Merge / Stack")
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

        merge_row = QHBoxLayout()
        self.align_checkbox = QCheckBox("Align images")
        self.align_checkbox.setChecked(True)
        merge_row.addWidget(self.align_checkbox)

        self.merge_button = QPushButton("Merge / Stack")
        self.merge_button.clicked.connect(self._on_merge_clicked)
        merge_row.addWidget(self.merge_button)
        merge_row.addStretch(1)
        root.addLayout(merge_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        controls_row = QHBoxLayout()
        add_button = QPushButton("Add Image(s)…")
        add_button.clicked.connect(self._on_add_images)
        controls_row.addWidget(add_button)

        clear_button = QPushButton("Clear All")
        clear_button.clicked.connect(self._on_clear_all)
        controls_row.addWidget(clear_button)

        self.count_label = QLabel()
        controls_row.addWidget(self.count_label)
        controls_row.addStretch(1)
        root.addLayout(controls_row)

        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_scroll.setFixedHeight(THUMBNAIL_SIZE + 100)
        self.thumbnail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumbnail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(self.thumbnail_scroll)

    # ---------- image selection ----------
    def _on_add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .raw images to stack", str(self._browse_dir), "Raw images (*.raw)"
        )
        if not paths:
            return

        # Remember where the user navigated to, so the next "Add Image(s)…"
        # picks up from there instead of resetting to the root every time.
        self._browse_dir = Path(paths[0]).parent

        existing_paths = {item.path for item in self.items}
        warnings: list[str] = []
        for path_str in paths:
            path = Path(path_str)
            if path in existing_paths:
                continue
            if len(self.items) >= MAX_STACK_SIZE:
                warnings.append(f"Only the first {MAX_STACK_SIZE} images are kept.")
                break
            try:
                item = load_stack_image(path)
            except (ValueError, RawFormatError) as exc:
                warnings.append(str(exc))
                continue

            if self.items and item.image.shape[:2] != self.items[0].image.shape[:2]:
                warnings.append(f"{path.name}: size doesn't match the other selected images - skipped.")
                continue

            self.items.append(item)
            existing_paths.add(path)

        self.items.sort(key=lambda item: item.exposure_us)
        self._set_status(" ".join(warnings), error=bool(warnings))
        self._refresh_thumbnails()
        self._update_merge_state()

    def _on_remove_item(self, path: Path) -> None:
        self.items = [item for item in self.items if item.path != path]
        self._refresh_thumbnails()
        self._update_merge_state()

    def _on_clear_all(self) -> None:
        self.items = []
        self.result_label.clear_image("Add 2-5 .raw images below, then Merge / Stack")
        self._set_status("")
        self._refresh_thumbnails()
        self._update_merge_state()

    def _refresh_thumbnails(self) -> None:
        strip = QWidget()
        layout = QHBoxLayout(strip)
        layout.setAlignment(Qt.AlignLeft)
        for item in self.items:
            layout.addWidget(_ThumbnailWidget(item, self._on_remove_item))
        layout.addStretch(1)
        self.thumbnail_scroll.setWidget(strip)
        self.count_label.setText(f"{len(self.items)} / {MAX_STACK_SIZE} selected")

    def _update_merge_state(self) -> None:
        self.merge_button.setEnabled(MIN_STACK_SIZE <= len(self.items) <= MAX_STACK_SIZE)

    # ---------- merging ----------
    def _on_merge_clicked(self) -> None:
        self.merge_button.setEnabled(False)
        self._set_status("Merging…", error=False)
        QApplication.processEvents()
        try:
            result = merge_exposure_fusion(
                [item.image for item in self.items], align=self.align_checkbox.isChecked()
            )
        except Exception as exc:  # includes cv2.error alongside ValueError
            self._set_status(f"Merge failed: {exc}", error=True)
            self._update_merge_state()
            return

        self.result_label.set_image(QPixmap.fromImage(numpy_to_qimage(result)))
        self._set_status(f"Merged {len(self.items)} images.", error=False)
        self._update_merge_state()

    # ---------- helpers ----------
    def _on_zoom_changed(self, percent: int) -> None:
        self.zoom_label.setText(f"{percent}%")

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setStyleSheet(f"color: {'#c0392b' if error else '#2e7d32'};")
        self.status_label.setText(text)
