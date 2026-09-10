"""Load Data tab: table of trial capture subfolders with capture settings.

Fully self-contained: given a parent trials folder, scans its immediate
subfolders for a FullSize_RAW_Images/ directory of .raw + .json sidecar
pairs (same layout the Viewer tab and raw_loader.load_sidecar_metadata
already understand) and lists one row per subfolder with the capture
settings read from its first sidecar. Double-clicking a row hands that
subfolder's raw_dir back to MainWindow via folder_selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .capture_folder import format_number, parse_capture_folder_name
from .raw_loader import load_sidecar_metadata

RAW_IMAGES_SUBDIR = "FullSize_RAW_Images"
COLUMN_HEADERS = [
    "Subfolder",
    "WD (mm)",
    "f-number",
    "Exposure (µs)",
    "Gain (dB)",
    "Light Current (mA)",
    "Light Brightness (%)",
    "Images",
    "Captured At",
]


@dataclass
class TrialRow:
    name: str
    raw_dir: Path
    image_count: int
    working_distance_mm: float | None
    f_number: float | None
    exposure_us: float | None
    gain_db: float | None
    light_current_ma: float | None
    light_brightness_pct: float | None
    captured_at: str | None


def scan_trials(parent: Path) -> list[TrialRow]:
    """Scan `parent`'s immediate subfolders for FullSize_RAW_Images/ dirs.

    Subfolders without one (or with no .raw files in it) are skipped. Only
    the first .raw file's sidecar is read - the rig's captures use one
    consistent exposure/gain/light setting per subfolder.
    """
    rows: list[TrialRow] = []
    for sub in sorted(p for p in parent.iterdir() if p.is_dir()):
        raw_dir = sub / RAW_IMAGES_SUBDIR
        if not raw_dir.is_dir():
            continue
        raw_files = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() == ".raw")
        if not raw_files:
            continue
        sidecar = load_sidecar_metadata(raw_files[0])
        # The working distance and aperture live in the subfolder's own name
        # (wd600f4_Capture_...), not in the sidecar - see capture_folder.py.
        capture = parse_capture_folder_name(sub.name)
        rows.append(
            TrialRow(
                name=sub.name,
                raw_dir=raw_dir,
                image_count=len(raw_files),
                working_distance_mm=capture.working_distance_mm if capture else None,
                f_number=capture.f_number if capture else None,
                exposure_us=sidecar["exposure_us"] if sidecar else None,
                gain_db=sidecar["gain_db"] if sidecar else None,
                light_current_ma=sidecar["light_current_ma"] if sidecar else None,
                light_brightness_pct=sidecar["light_brightness_pct"] if sidecar else None,
                captured_at=sidecar["captured_at"] if sidecar else None,
            )
        )
    return rows


def _cell_text(value) -> str:
    return "—" if value is None else str(value)


def _percent_text(value) -> str:
    return "—" if value is None else f"{value:.1f}%"


class _NumericTableWidgetItem(QTableWidgetItem):
    """A table item that sorts by a numeric value instead of its display text.

    QTableWidgetItem's default operator< compares data(Qt.DisplayRole), which
    for plain string items falls back to lexicographic string comparison -
    "100000.0" sorts before "800.0" because "1" < "8". Storing the real
    number here and comparing on that fixes ascending/descending sort for
    numeric columns (exposure, gain, light current, image count).
    """

    def __init__(self, text: str, value: float | None):
        super().__init__(text)
        # Missing values (no sidecar) sort below every real number,
        # consistently regardless of sort direction.
        self._sort_value = float("-inf") if value is None else value

    def __lt__(self, other: QTableWidgetItem) -> bool:
        if isinstance(other, _NumericTableWidgetItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class TrialsTab(QWidget):
    """Table of trial capture subfolders, scanned from a chosen parent folder."""

    folder_selected = Signal(Path)

    def __init__(self):
        super().__init__()
        self._parent_folder: Path | None = None
        self._rows: list[TrialRow] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Parent folder:"))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select a parent folder containing capture subfolders…")
        self.path_edit.returnPressed.connect(self._load_typed_path)
        path_row.addWidget(self.path_edit, stretch=1)

        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._on_browse)
        path_row.addWidget(browse_button)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self._on_refresh)
        path_row.addWidget(refresh_button)

        root.addLayout(path_row)

        self.table = QTableWidget(0, len(COLUMN_HEADERS))
        self.table.setHorizontalHeaderLabels(COLUMN_HEADERS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemDoubleClicked.connect(self._on_row_double_clicked)
        root.addWidget(self.table, stretch=1)

        self.status_label = QLabel(
            "Double-click a row to load that subfolder's images into the Viewer tab."
        )
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    # ---------- folder selection ----------
    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select parent trials folder")
        if folder:
            self.set_parent_folder(Path(folder))

    def _load_typed_path(self) -> None:
        text = self.path_edit.text().strip()
        if text:
            self.set_parent_folder(Path(text.strip('"')).expanduser())

    def _on_refresh(self) -> None:
        if self._parent_folder is not None:
            self.set_parent_folder(self._parent_folder)

    def set_parent_folder(self, folder: Path) -> None:
        self.path_edit.setText(str(folder))
        if not folder.is_dir():
            self._parent_folder = None
            self._populate_table([])
            self._set_status(f"Folder not found: {folder}", error=True)
            return

        self._parent_folder = folder
        rows = scan_trials(folder)
        self._populate_table(rows)
        if not rows:
            self._set_status(
                f"No subfolders with a {RAW_IMAGES_SUBDIR}\\ directory found in:\n{folder}",
                error=True,
            )
        else:
            self._set_status(
                f"{len(rows)} capture subfolder(s) found. "
                "Double-click a row to load it into the Viewer tab."
            )

    def current_parent_folder(self) -> Path | None:
        return self._parent_folder

    # ---------- table population ----------
    def _populate_table(self, rows: list[TrialRow]) -> None:
        self._rows = rows
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            items = [
                QTableWidgetItem(row.name),
                _NumericTableWidgetItem(
                    format_number(row.working_distance_mm), row.working_distance_mm
                ),
                _NumericTableWidgetItem(format_number(row.f_number), row.f_number),
                _NumericTableWidgetItem(_cell_text(row.exposure_us), row.exposure_us),
                _NumericTableWidgetItem(_cell_text(row.gain_db), row.gain_db),
                _NumericTableWidgetItem(_cell_text(row.light_current_ma), row.light_current_ma),
                _NumericTableWidgetItem(
                    _percent_text(row.light_brightness_pct), row.light_brightness_pct
                ),
                _NumericTableWidgetItem(str(row.image_count), row.image_count),
                QTableWidgetItem(_cell_text(row.captured_at)),
            ]
            for col_index, item in enumerate(items):
                item.setData(Qt.UserRole, row_index)
                self.table.setItem(row_index, col_index, item)
        self.table.setSortingEnabled(True)

    def _on_row_double_clicked(self, item: QTableWidgetItem) -> None:
        row_index = item.data(Qt.UserRole)
        if row_index is None or row_index >= len(self._rows):
            return
        self.folder_selected.emit(self._rows[row_index].raw_dir)

    def _set_status(self, text: str, error: bool = False) -> None:
        self.status_label.setStyleSheet(f"color: {'#c0392b' if error else '#666666'};")
        self.status_label.setText(text)
