import os
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from raw_viewer.ocr_processing import run_ocr


@unittest.skipUnless(os.environ.get("OCR_REAL_TESTS") == "1", "Set OCR_REAL_TESTS=1 to use cached OCR models")
class RealOcrTests(unittest.TestCase):
    def make_image(self):
        image = np.full((600, 1000, 3), 255, dtype=np.uint8)
        codes = ["K0680", "FW6099", "110249"]
        for index, code in enumerate(codes):
            cv2.putText(image, code, (90, 150 + index * 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5, cv2.LINE_AA)
        return image, codes

    def test_multiline_and_blank_offline(self):
        image, codes = self.make_image()
        with patch("socket.socket.connect", side_effect=AssertionError("OCR attempted network access")) as connect:
            result = run_ocr(image)
            self.assertEqual(result.text.splitlines(), codes)
            self.assertGreater(result.confidence, 0.9)
            blank = run_ocr(np.full_like(image, 255))
            self.assertEqual(blank.text, "")
            self.assertEqual(blank.confidence, 0.0)
            connect.assert_not_called()

    def test_worker_reuses_engine_across_requests(self):
        from PySide6.QtCore import QRect
        from PySide6.QtGui import QPixmap
        from PySide6.QtTest import QTest
        from PySide6.QtWidgets import QApplication

        from raw_viewer.main_window import MainWindow
        from raw_viewer.widgets import numpy_to_qimage

        app = QApplication.instance() or QApplication([])
        image, codes = self.make_image()
        with (
            patch.object(MainWindow, "_load_settings"),
            patch.object(MainWindow, "_save_settings"),
            patch("socket.socket.connect", side_effect=AssertionError("OCR attempted network access")) as connect,
        ):
            window = MainWindow()
            window._current_image8 = image
            window.image_label.set_image(QPixmap.fromImage(numpy_to_qimage(image)))
            window.ocr_button.setChecked(True)
            try:
                for request in range(2):
                    with self.subTest(request=request):
                        window._on_ocr_region_selected(QRect(0, 0, image.shape[1], image.shape[0]))
                        deadline = time.monotonic() + 90
                        while window._ocr_thread is not None and time.monotonic() < deadline:
                            QTest.qWait(20)
                        self.assertIsNone(window._ocr_thread, "Real OCR worker timed out")
                        self.assertEqual(window.error_label.text(), "")
                        self.assertEqual(window.ocr_result_edit.toPlainText().splitlines(), codes)
                        self.assertTrue(window.image_label._ocr_mode)
                connect.assert_not_called()
            finally:
                window.close()
                if window._ocr_thread is None:
                    window.deleteLater()
                app.processEvents()


if __name__ == "__main__":
    unittest.main()