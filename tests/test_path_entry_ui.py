import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from raw_viewer.main_window import MainWindow

CAPTURE_FOLDER = "wd800f8_Capture_2026-09-02_11-51-51"
WIDTH, HEIGHT = 16, 12
IMAGE_NAMES = ["Top_1.raw", "Top_2.raw", "Top_3.raw"]

SIDECAR = json.dumps(
    {
        "width": WIDTH,
        "height": HEIGHT,
        "dtype": "uint8",
        "format": "Mono8",
        "exposure_us": 10000.0,
        "gain_db": 0.0,
        "light_current_ma": 200.0,
    }
)


class PathEntryTests(unittest.TestCase):
    """The Viewer's path box loads the whole folder around a pasted image."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        # Keep the developer's real last-used folder out of the test.
        self.settings_load = patch.object(MainWindow, "_load_settings")
        self.settings_save = patch.object(MainWindow, "_save_settings")
        self.settings_load.start()
        self.settings_save.start()
        self.window = MainWindow()

        self.tmp = tempfile.TemporaryDirectory()
        self.raw_dir = Path(self.tmp.name) / CAPTURE_FOLDER / "FullSize_RAW_Images"
        self.raw_dir.mkdir(parents=True)
        pixels = np.zeros((HEIGHT, WIDTH), dtype=np.uint8).tobytes()
        for name in IMAGE_NAMES:
            raw_path = self.raw_dir / name
            raw_path.write_bytes(pixels)
            raw_path.with_suffix(".json").write_text(SIDECAR)
        self.top1, self.top2, self.top3 = (self.raw_dir / n for n in IMAGE_NAMES)
        (self.raw_dir / "notes.txt").write_text("not an image")

        self.empty_dir = Path(self.tmp.name) / "empty_folder"
        self.empty_dir.mkdir()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()
        self.settings_load.stop()
        self.settings_save.stop()

    def _paste(self, text) -> None:
        """Type `text` into the path box and press Enter."""
        self.window.file_path_edit.setText(str(text))
        self.window._load_typed_file_path()

    def _loaded_names(self) -> list[str]:
        return [p.name for p in self.window.files]

    def test_typed_file_path_loads_the_parent_folder_and_selects_the_image(self):
        self._paste(self.top2)
        self.assertEqual(self.window.folder, self.raw_dir)
        self.assertEqual(self._loaded_names(), IMAGE_NAMES)
        self.assertEqual(self.window.index, 1)
        self.assertEqual(self.window.status_label.text(), "2 / 3 — Top_2.raw")
        self.assertTrue(self.window.prev_button.isEnabled())
        self.assertTrue(self.window.next_button.isEnabled())
        self.assertEqual(self.window.error_label.text(), "")

    def test_navigation_walks_the_whole_folder_after_a_pasted_path(self):
        self._paste(self.top2)
        self.window.show_next()
        self.assertEqual(self.window.files[self.window.index], self.top3)
        self.assertFalse(self.window.next_button.isEnabled())
        self.window.show_previous()
        self.window.show_previous()
        self.assertEqual(self.window.files[self.window.index], self.top1)
        self.assertFalse(self.window.prev_button.isEnabled())

    def test_path_box_tracks_the_current_image(self):
        self._paste(self.top2)
        self.assertEqual(self.window.file_path_edit.text(), str(self.top2))
        self.window.show_next()
        self.assertEqual(self.window.file_path_edit.text(), str(self.top3))
        self.window.show_previous()
        self.assertEqual(self.window.file_path_edit.text(), str(self.top2))
        # The plain folder route tracks it too.
        self.window._set_folder(self.raw_dir)
        self.assertEqual(self.window.file_path_edit.text(), str(self.top1))

    def test_quoted_path_is_accepted(self):
        # What Explorer's "Copy as path" puts on the clipboard.
        self._paste(f'"{self.top2}"')
        self.assertEqual(self.window.index, 1)
        self.assertEqual(self.window.file_path_edit.text(), str(self.top2))

    def test_case_and_separator_differences_still_select_the_image(self):
        self._paste(str(self.top2).replace("\\", "/").lower())
        self.assertEqual(self.window.folder, self.raw_dir)
        self.assertEqual(self._loaded_names(), IMAGE_NAMES)
        self.assertEqual(self.window.index, 1)

    def test_folder_path_loads_the_folder_from_its_first_image(self):
        self._paste(self.raw_dir)
        self.assertEqual(self.window.folder, self.raw_dir)
        self.assertEqual(self._loaded_names(), IMAGE_NAMES)
        self.assertEqual(self.window.index, 0)
        self.assertEqual(self.window.file_path_edit.text(), str(self.top1))
        self.assertEqual(self.window.error_label.text(), "")

    def test_non_raw_path_reports_an_error_and_leaves_the_folder_alone(self):
        self.window._set_folder(self.raw_dir)
        self._paste(self.raw_dir / "notes.txt")
        self.assertIn(".raw", self.window.error_label.text())
        # The loaded folder is untouched, so the current image stays up.
        self.assertEqual(self._loaded_names(), IMAGE_NAMES)
        self.assertEqual(self.window.index, 0)
        self.assertEqual(self.window.folder, self.raw_dir)
        # The typo survives, so it can be corrected instead of retyped.
        self.assertTrue(self.window.file_path_edit.text().endswith("notes.txt"))

    def test_sidecar_json_path_is_rejected(self):
        self.window._set_folder(self.raw_dir)
        self._paste(self.top2.with_suffix(".json"))
        self.assertIn(".raw", self.window.error_label.text())
        self.assertEqual(self.window.index, 0)

    def test_missing_path_reports_file_not_found(self):
        self.window._set_folder(self.raw_dir)
        self._paste(self.raw_dir / "Top_9.raw")
        self.assertTrue(self.window.error_label.text().startswith("File not found"))
        self.assertEqual(self._loaded_names(), IMAGE_NAMES)
        self.assertEqual(self.window.index, 0)

    def test_empty_folder_clears_the_path_box(self):
        self._paste(self.top2)
        self._paste(self.empty_dir)
        self.assertEqual(self.window.files, [])
        self.assertEqual(self.window.index, -1)
        self.assertEqual(self.window.status_label.text(), "No .raw files found")
        self.assertEqual(self.window.file_path_edit.text(), "")
        self.assertFalse(self.window.prev_button.isEnabled())
        self.assertFalse(self.window.next_button.isEnabled())

    def test_overlap_tab_receives_the_parent_folder_files(self):
        self._paste(self.top2)
        for slot in (self.window.overlap_tab.slot1, self.window.overlap_tab.slot2):
            names = [slot.combo.itemText(i) for i in range(1, slot.combo.count())]
            self.assertEqual(names, IMAGE_NAMES)
            self.assertTrue(slot.combo.isEnabled())

    def test_pasted_path_records_its_folder_for_persistence(self):
        # This is the value _save_settings writes to last_folder; asserting
        # it here avoids un-patching _save_settings and touching the real
        # registry, which the suite deliberately never does.
        self._paste(self.top2)
        self.assertEqual(self.window.folder, self.raw_dir)
        self.assertTrue(self.window.folder.is_dir())

    def test_capture_folder_metadata_still_resolves_for_a_pasted_path(self):
        self._paste(self.top2)
        self.assertEqual(self.window.working_distance_label.text(), "800 mm")
        self.assertEqual(self.window.aperture_label.text(), "f/8")

    def test_arrow_keys_edit_the_path_box_while_it_has_focus(self):
        self._paste(self.top2)
        # hasFocus() needs a shown, active window - even offscreen - and the
        # box lives on the Viewer tab, which is not the one that opens first.
        self.window.show()
        self.window.activateWindow()
        self.window.tabs.setCurrentWidget(self.window.viewer_widget)
        self.window.file_path_edit.setFocus()
        self.app.processEvents()
        self.assertTrue(self.window.file_path_edit.hasFocus())
        self.window._on_next_shortcut()
        # Still on Top_2, and the text was not rewritten under the cursor.
        self.assertEqual(self.window.index, 1)
        self.assertEqual(self.window.file_path_edit.text(), str(self.top2))

    def test_arrow_keys_navigate_when_the_path_box_is_not_focused(self):
        self._paste(self.top2)
        self.window.image_label.setFocus()
        self.window._on_next_shortcut()
        self.assertEqual(self.window.index, 2)
        self.window._on_previous_shortcut()
        self.assertEqual(self.window.index, 1)


if __name__ == "__main__":
    unittest.main()
