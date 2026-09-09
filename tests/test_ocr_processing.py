import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from raw_viewer import ocr_processing as ocr


class OcrProcessingTests(unittest.TestCase):
    def test_rgb_is_converted_to_bgr(self):
        crop = np.zeros((600, 600, 3), dtype=np.uint8)
        crop[:] = (255, 0, 0)
        processed = ocr.preprocess_for_ocr(crop)
        self.assertEqual(processed[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(crop[0, 0].tolist(), [255, 0, 0])

    def test_upscale_is_bounded_for_small_and_narrow_crops(self):
        for shape, expected in [((10, 20), (60, 120, 3)), ((4, 10000), (2, 4000, 3))]:
            with self.subTest(shape=shape):
                self.assertEqual(ocr.preprocess_for_ocr(np.zeros(shape, dtype=np.uint8)).shape, expected)

    def test_rotated_noncontiguous_crop(self):
        crop = np.rot90(np.zeros((600, 800, 3), dtype=np.uint8))
        self.assertFalse(crop.flags.c_contiguous)
        self.assertTrue(ocr.preprocess_for_ocr(crop).flags.c_contiguous)

    def test_invalid_inputs_have_clear_errors(self):
        for crop in (np.zeros((0, 10), dtype=np.uint8), np.zeros((5, 5)), np.zeros((5, 5, 4), dtype=np.uint8)):
            with self.subTest(shape=crop.shape), self.assertRaises(ValueError):
                ocr.preprocess_for_ocr(crop)

    def test_empty_crop_does_not_load_engine(self):
        with patch.object(ocr, "get_ocr_engine") as engine:
            self.assertEqual(ocr.run_ocr(np.zeros((0, 10), dtype=np.uint8)), ocr.OcrResult("", 0.0))
            engine.assert_not_called()

    def test_multiline_result_and_numpy_scores(self):
        engine = Mock()
        engine.predict.return_value = [{
            "rec_texts": ["K0680", "FW6099", "110249"],
            "rec_scores": np.array([0.98, 0.99, 1.0]),
        }]
        with patch.object(ocr, "get_ocr_engine", return_value=engine):
            result = ocr.run_ocr(np.zeros((100, 200), dtype=np.uint8))
        self.assertEqual(result.text, "K0680\nFW6099\n110249")
        self.assertAlmostEqual(result.confidence, 0.99)

    def test_no_detections(self):
        for predictions in ([], [{"rec_texts": [], "rec_scores": []}]):
            with self.subTest(predictions=predictions):
                engine = Mock()
                engine.predict.return_value = predictions
                with patch.object(ocr, "get_ocr_engine", return_value=engine):
                    self.assertEqual(ocr.run_ocr(np.zeros((10, 20), dtype=np.uint8)), ocr.OcrResult("", 0.0))

    def test_missing_models_fail_before_paddle_import(self):
        with tempfile.TemporaryDirectory() as cache:
            constructor = Mock()
            with (
                patch.dict(os.environ, {"PADDLE_PDX_CACHE_HOME": cache}),
                patch.object(ocr, "_engine", None),
                patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=constructor)}),
            ):
                with self.assertRaisesRegex(RuntimeError, "OCR is offline; missing or incomplete models"):
                    ocr.get_ocr_engine()
                constructor.assert_not_called()

    def test_explicit_model_paths_and_cached_engine(self):
        with tempfile.TemporaryDirectory() as cache:
            for name in ocr._MODEL_NAMES.values():
                directory = Path(cache) / "official_models" / name
                directory.mkdir(parents=True)
                for filename in ("inference.json", "inference.pdiparams", "inference.yml"):
                    (directory / filename).touch()
            constructor = Mock()
            with (
                patch.dict(os.environ, {"PADDLE_PDX_CACHE_HOME": cache}),
                patch.object(ocr, "_engine", None),
                patch.object(ocr, "_register_cuda_dll_directories"),
                patch.dict("sys.modules", {"paddleocr": SimpleNamespace(PaddleOCR=constructor)}),
            ):
                self.assertIs(ocr.get_ocr_engine(), ocr.get_ocr_engine())
                constructor.assert_called_once()
                for component, name in ocr._MODEL_NAMES.items():
                    self.assertEqual(
                        Path(constructor.call_args.kwargs[f"{component}_model_dir"]),
                        Path(cache) / "official_models" / name,
                    )


if __name__ == "__main__":
    unittest.main()