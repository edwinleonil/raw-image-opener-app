import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from raw_viewer.trials_tab import COLUMN_HEADERS, TrialsTab, scan_trials

WIDTH, HEIGHT = 16, 12
WD_COLUMN = COLUMN_HEADERS.index("WD (mm)")
F_COLUMN = COLUMN_HEADERS.index("f-number")
SUBFOLDER_COLUMN = COLUMN_HEADERS.index("Subfolder")

# Deliberately not in sorted-by-working-distance order, and 1200 is chosen
# because "1200" < "600" lexicographically - the bug _NumericTableWidgetItem
# exists to prevent.
CAPTURE_FOLDERS = [
    "wd800f8_Capture_2026-09-02_12-10-30",
    "wd600f2.5_Capture_2026-09-01_09-47-42",
    "wd1200f4_Capture_2026-09-10_14-08-56",
]
PLAIN_FOLDER = "some_other_folder"


def _write_capture(parent: Path, name: str) -> None:
    raw_dir = parent / name / "FullSize_RAW_Images"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "Left_1.raw"
    raw_path.write_bytes(np.zeros((HEIGHT, WIDTH), dtype=np.uint8).tobytes())
    raw_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "width": WIDTH,
                "height": HEIGHT,
                "dtype": "uint8",
                "format": "Mono8",
                "exposure_us": 10000.0,
                "gain_db": 0.0,
                "light_current_ma": 200.0,
                "light_brightness_pct": 100.0,
                "captured_at": "2026-09-10T14:08:56",
            }
        )
    )


class ScanTrialsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parent = Path(self.tmp.name)
        for name in CAPTURE_FOLDERS + [PLAIN_FOLDER]:
            _write_capture(self.parent, name)
        self.rows = {row.name: row for row in scan_trials(self.parent)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_folder_name_supplies_working_distance_and_f_number(self):
        expected = {
            "wd800f8_Capture_2026-09-02_12-10-30": (800.0, 8.0),
            "wd600f2.5_Capture_2026-09-01_09-47-42": (600.0, 2.5),
            "wd1200f4_Capture_2026-09-10_14-08-56": (1200.0, 4.0),
        }
        for name, (wd, f_number) in expected.items():
            with self.subTest(name=name):
                self.assertEqual(self.rows[name].working_distance_mm, wd)
                self.assertEqual(self.rows[name].f_number, f_number)

    def test_unparseable_folder_leaves_both_none(self):
        row = self.rows[PLAIN_FOLDER]
        self.assertIsNone(row.working_distance_mm)
        self.assertIsNone(row.f_number)

    def test_sidecar_fields_are_still_populated(self):
        row = self.rows[PLAIN_FOLDER]
        self.assertEqual(row.exposure_us, 10000.0)
        self.assertEqual(row.gain_db, 0.0)
        self.assertEqual(row.light_current_ma, 200.0)
        self.assertEqual(row.light_brightness_pct, 100.0)
        self.assertEqual(row.captured_at, "2026-09-10T14:08:56")
        self.assertEqual(row.image_count, 1)


class TrialsTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.parent = Path(self.tmp.name)
        for name in CAPTURE_FOLDERS + [PLAIN_FOLDER]:
            _write_capture(self.parent, name)
        self.tab = TrialsTab()
        self.tab.set_parent_folder(self.parent)

    def tearDown(self):
        self.tab.deleteLater()
        self.tmp.cleanup()

    def _cell(self, row: int, column: int) -> str:
        return self.tab.table.item(row, column).text()

    def _row_of(self, folder_name: str) -> int:
        for row in range(self.tab.table.rowCount()):
            if self._cell(row, SUBFOLDER_COLUMN) == folder_name:
                return row
        raise AssertionError(f"{folder_name} not in the table")

    def test_cells_render_trimmed_numbers(self):
        row = self._row_of("wd600f2.5_Capture_2026-09-01_09-47-42")
        self.assertEqual(self._cell(row, WD_COLUMN), "600")
        self.assertEqual(self._cell(row, F_COLUMN), "2.5")

        row = self._row_of("wd800f8_Capture_2026-09-02_12-10-30")
        self.assertEqual(self._cell(row, WD_COLUMN), "800")
        self.assertEqual(self._cell(row, F_COLUMN), "8")

    def test_unparseable_folder_shows_placeholders(self):
        row = self._row_of(PLAIN_FOLDER)
        self.assertEqual(self._cell(row, WD_COLUMN), "—")
        self.assertEqual(self._cell(row, F_COLUMN), "—")

    def test_working_distance_sorts_numerically(self):
        self.tab.table.sortItems(WD_COLUMN, Qt.AscendingOrder)
        column = [self._cell(r, WD_COLUMN) for r in range(self.tab.table.rowCount())]
        # "1200" would sort first under a plain string comparison.
        self.assertEqual(column, ["—", "600", "800", "1200"])

    def test_f_number_sorts_numerically(self):
        self.tab.table.sortItems(F_COLUMN, Qt.AscendingOrder)
        column = [self._cell(r, F_COLUMN) for r in range(self.tab.table.rowCount())]
        self.assertEqual(column, ["—", "2.5", "4", "8"])


if __name__ == "__main__":
    unittest.main()
