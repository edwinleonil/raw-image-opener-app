import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from raw_viewer.ocr_processing import run_ocr


@unittest.skipUnless(os.environ.get("OCR_REAL_TESTS") == "1", "Set OCR_REAL_TESTS=1 to use cached OCR models")
class RealOcrTests(unittest.TestCase):
    def test_multiline_and_blank_offline(self):
        image = np.full((600, 1000, 3), 255, dtype=np.uint8)
        codes = ["K0680", "FW6099", "110249"]
        for index, code in enumerate(codes):
            cv2.putText(image, code, (90, 150 + index * 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5, cv2.LINE_AA)
        with patch("socket.socket.connect", side_effect=AssertionError("OCR attempted network access")) as connect:
            result = run_ocr(image)
            self.assertEqual(result.text.splitlines(), codes)
            self.assertGreater(result.confidence, 0.9)
            blank = run_ocr(np.full_like(image, 255))
            self.assertEqual(blank.text, "")
            self.assertEqual(blank.confidence, 0.0)
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()