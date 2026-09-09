import os
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QCloseEvent, QPixmap
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

from raw_viewer.main_window import MainWindow
from raw_viewer.ocr_processing import OcrResult
from raw_viewer.widgets import ZoomableImageView, numpy_to_qimage


class OcrWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.settings_load = patch.object(MainWindow, "_load_settings")
        self.settings_save = patch.object(MainWindow, "_save_settings")
        self.settings_load.start()
        self.settings_save.start()
        self.window = MainWindow()
        self.window._current_image8 = np.zeros((100, 160), dtype=np.uint8)
        self.window.image_label.set_image(
            QPixmap.fromImage(numpy_to_qimage(self.window._current_image8))
        )
        self.window.ocr_button.setChecked(True)
        self.release = threading.Event()

        def recognize(crop):
            if not self.release.wait(5):
                raise TimeoutError("Test worker was not released")
            return OcrResult("K0680\nFW6099\n110249", 0.99)

        self.recognizer = patch("raw_viewer.main_window.run_ocr", side_effect=recognize)
        self.recognizer.start()

    def tearDown(self):
        self.release.set()
        self.wait_for_worker()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.recognizer.stop()
        self.settings_save.stop()
        self.settings_load.stop()

    def wait_for_worker(self):
        deadline = time.monotonic() + 8
        while self.window._ocr_thread is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertIsNone(self.window._ocr_thread, "OCR worker did not finish")

    def start_worker(self):
        self.window._on_ocr_region_selected(QRect(10, 10, 80, 40))
        self.assertIsNotNone(self.window._ocr_thread)

    def test_success_displays_all_lines(self):
        self.start_worker()
        self.release.set()
        self.wait_for_worker()
        self.assertEqual(self.window.ocr_result_edit.toPlainText(), "K0680\nFW6099\n110249")
        self.assertEqual(self.window.ocr_confidence_label.text(), "99% confidence")
        self.assertTrue(self.window.ocr_button.isEnabled())

    def test_clear_discards_in_flight_result(self):
        self.start_worker()
        self.window._on_ocr_clear_clicked()
        self.release.set()
        self.wait_for_worker()
        self.assertEqual(self.window.ocr_result_edit.toPlainText(), "")
        self.assertEqual(self.window.ocr_confidence_label.text(), "")

    def test_format_change_discards_in_flight_result(self):
        self.start_worker()
        self.window._on_format_changed()
        self.release.set()
        self.wait_for_worker()
        self.assertEqual(self.window.ocr_result_edit.toPlainText(), "")

    def test_image_change_discards_in_flight_result(self):
        self.start_worker()
        self.window._render_current()
        self.release.set()
        self.wait_for_worker()
        self.assertEqual(self.window.ocr_result_edit.toPlainText(), "")

    def test_failure_then_retry(self):
        self.recognizer.stop()
        with patch("raw_viewer.main_window.run_ocr", side_effect=RuntimeError("test error")):
            self.start_worker()
            self.wait_for_worker()
        self.assertEqual(self.window.error_label.text(), "OCR failed: test error")
        self.assertTrue(self.window.image_label._ocr_mode)
        self.recognizer.start()
        self.start_worker()
        self.release.set()
        self.wait_for_worker()
        self.assertEqual(self.window.error_label.text(), "")
        self.assertIn("K0680", self.window.ocr_result_edit.toPlainText())

    def test_clear_discards_in_flight_error(self):
        self.recognizer.stop()

        def fail(crop):
            self.release.wait(5)
            raise RuntimeError("stale error")

        with patch("raw_viewer.main_window.run_ocr", side_effect=fail):
            self.start_worker()
            self.window._on_ocr_clear_clicked()
            self.release.set()
            self.wait_for_worker()
        self.assertEqual(self.window.error_label.text(), "")

    def test_busy_worker_disables_canvas_selection(self):
        self.start_worker()
        self.assertFalse(self.window.image_label._ocr_mode)

    def test_close_is_deferred_until_worker_finishes(self):
        self.window.show()
        self.start_worker()
        event = QCloseEvent()
        self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        self.assertTrue(self.window.isVisible())
        self.release.set()
        self.wait_for_worker()
        self.assertFalse(self.window.isVisible())

    def test_finished_worker_and_thread_are_deleted(self):
        self.start_worker()
        worker = self.window._ocr_worker
        thread = self.window._ocr_thread
        self.release.set()
        self.wait_for_worker()
        QTest.qWait(20)
        self.assertFalse(isValid(worker))
        self.assertFalse(isValid(thread))


class OcrSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.view = ZoomableImageView()
        self.view.resize(500, 400)
        pixmap = QPixmap(160, 100)
        pixmap.fill(Qt.white)
        self.view.set_image(pixmap)
        self.view.show()
        self.app.processEvents()
        self.view.zoom_actual_size()
        self.view.set_ocr_mode(True)
        self.regions = []
        self.view.ocr_region_selected.connect(self.regions.append)

    def tearDown(self):
        self.view.close()
        self.view.deleteLater()
        self.app.processEvents()

    def drag(self, start, end):
        start_point = self.view.mapFromScene(QPointF(*start))
        end_point = self.view.mapFromScene(QPointF(*end))
        QTest.mousePress(self.view.viewport(), Qt.LeftButton, pos=start_point)
        QTest.mouseMove(self.view.viewport(), end_point)
        QTest.mouseRelease(self.view.viewport(), Qt.LeftButton, pos=end_point)

    def test_forward_and_reverse_selection(self):
        for start, end in [((10, 10), (90, 50)), ((90, 50), (10, 10))]:
            self.drag(start, end)
            self.assertEqual(self.regions[-1], QRect(10, 10, 80, 40))

    def test_clamps_selection_to_image(self):
        self.drag((-20, -10), (180, 120))
        self.assertEqual(self.regions, [QRect(0, 0, 160, 100)])

    def test_rejects_tiny_and_outside_boxes(self):
        self.drag((10, 10), (12, 12))
        self.drag((-40, -40), (-10, -10))
        self.assertEqual(self.regions, [])

    def test_zoomed_selection_uses_image_pixels(self):
        self.view.scale(2, 2)
        self.drag((10, 10), (90, 50))
        self.assertEqual(self.regions, [QRect(10, 10, 80, 40)])

    def test_disabling_mode_cancels_drag(self):
        start = self.view.mapFromScene(QPointF(10, 10))
        QTest.mousePress(self.view.viewport(), Qt.LeftButton, pos=start)
        self.view.set_ocr_mode(False)
        QTest.mouseRelease(self.view.viewport(), Qt.LeftButton, pos=start)
        self.assertIsNone(self.view._ocr_drag_start)
        self.assertEqual(self.regions, [])


if __name__ == "__main__":
    unittest.main()