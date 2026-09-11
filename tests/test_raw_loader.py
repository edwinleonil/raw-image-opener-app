import unittest

import numpy as np

from raw_viewer.raw_loader import denoise_image


class DenoiseImageTests(unittest.TestCase):
    def _noisy_rgb(self):
        rng = np.random.default_rng(0)
        base = np.full((64, 64, 3), 128, dtype=np.float32)
        noisy = base + rng.normal(0, 25, base.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def _noisy_mono(self):
        rng = np.random.default_rng(0)
        base = np.full((64, 64), 128, dtype=np.float32)
        noisy = base + rng.normal(0, 25, base.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def test_zero_amount_is_noop(self):
        image = self._noisy_rgb()
        result = denoise_image(image, 0)
        self.assertIs(result, image)

    def test_negative_amount_is_noop(self):
        image = self._noisy_rgb()
        result = denoise_image(image, -5)
        self.assertIs(result, image)

    def test_reduces_noise_on_rgb_image(self):
        image = self._noisy_rgb()
        result = denoise_image(image, 60)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertLess(result.std(), image.std())

    def test_reduces_noise_on_mono_image(self):
        image = self._noisy_mono()
        result = denoise_image(image, 60)
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertLess(result.std(), image.std())


if __name__ == "__main__":
    unittest.main()
