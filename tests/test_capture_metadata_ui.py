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

CAPTURE_FOLDER = "wd600f4_Capture_2026-09-10_14-08-56"
WIDTH, HEIGHT = 16, 12


class CaptureMetadataPanelTests(unittest.TestCase):
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
        self.raw_path = self.raw_dir / "Left_1.raw"
        self.raw_path.write_bytes(np.zeros((HEIGHT, WIDTH), dtype=np.uint8).tobytes())
        self.sidecar_path = self.raw_path.with_suffix(".json")
        self.sidecar_path.write_text(
            json.dumps(
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
        )

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.tmp.cleanup()
        self.settings_load.stop()
        self.settings_save.stop()

    def test_folder_name_populates_working_distance_and_aperture(self):
        self.window._set_folder(self.raw_dir)
        self.assertEqual(self.window.working_distance_label.text(), "600 mm")
        self.assertEqual(self.window.aperture_label.text(), "f/4")
        self.assertEqual(self.window.exposure_value_label.text(), "10000.0")
        self.assertIn(CAPTURE_FOLDER, self.window.working_distance_label.toolTip())

    def test_values_survive_a_missing_sidecar(self):
        self.sidecar_path.unlink()
        self.window._set_folder(self.raw_dir)
        # The image can't be decoded without the sidecar...
        self.assertEqual(self.window.exposure_value_label.text(), "—")
        # ...but the folder name still says how the shot was set up.
        self.assertEqual(self.window.working_distance_label.text(), "600 mm")
        self.assertEqual(self.window.aperture_label.text(), "f/4")

    def test_unparseable_folder_name_shows_placeholders(self):
        plain_dir = Path(self.tmp.name) / "some_folder"
        plain_dir.mkdir()
        raw_path = plain_dir / "Left_1.raw"
        raw_path.write_bytes(self.raw_path.read_bytes())
        raw_path.with_suffix(".json").write_text(self.sidecar_path.read_text())

        self.window._set_folder(plain_dir)
        self.assertEqual(self.window.working_distance_label.text(), "—")
        self.assertEqual(self.window.aperture_label.text(), "—")
        self.assertEqual(self.window.working_distance_label.toolTip(), "")
        # The sidecar-driven rows are unaffected.
        self.assertEqual(self.window.exposure_value_label.text(), "10000.0")


if __name__ == "__main__":
    unittest.main()
