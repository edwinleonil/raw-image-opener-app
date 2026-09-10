import unittest
from pathlib import Path

from raw_viewer import capture_folder as cf


class ParseCaptureFolderNameTests(unittest.TestCase):
    def test_accepted_names(self):
        cases = [
            ("wd600f4_Capture_2026-09-10_14-08-56", 600.0, 4.0),
            ("wd600f2.5_Capture_2026-09-01_09-47-42", 600.0, 2.5),
            ("wd1200f11_Capture_2026-09-01_11-26-12", 1200.0, 11.0),
            ("WD600F4_Capture_2026-09-10_14-08-56", 600.0, 4.0),
        ]
        for name, wd, f_number in cases:
            with self.subTest(name=name):
                info = cf.parse_capture_folder_name(name)
                self.assertIsNotNone(info)
                self.assertEqual(info.folder_name, name)
                self.assertEqual(info.working_distance_mm, wd)
                self.assertEqual(info.f_number, f_number)

    def test_rejected_names(self):
        for name in [
            "WD600_f2.8_Capture_2026-09-10_14-08-56",  # separator between the numbers
            "Capture_2026-09-10_14-08-56",  # no prefix
            "FullSize_RAW_Images",
            "trial_wd600f4_Capture_2026-09-10_14-08-56",  # prefix not at the start
            "wd600f4_2026-09-10_14-08-56",  # missing _Capture_
            "wdf4_Capture_2026-09-10_14-08-56",  # no working distance
            "",
        ]:
            with self.subTest(name=name):
                self.assertIsNone(cf.parse_capture_folder_name(name))


class FindCaptureFolderInfoTests(unittest.TestCase):
    def test_real_rig_layout(self):
        path = Path(
            r"C:\trials\wd600f4_Capture_2026-09-10_14-08-56\FullSize_RAW_Images\Left_1.raw"
        )
        info = cf.find_capture_folder_info(path)
        self.assertIsNotNone(info)
        self.assertEqual(info.working_distance_mm, 600.0)
        self.assertEqual(info.f_number, 4.0)

    def test_raw_directly_in_capture_folder(self):
        path = Path(r"C:\trials\wd800f8_Capture_2026-09-10_14-08-56\Left_1.raw")
        info = cf.find_capture_folder_info(path)
        self.assertIsNotNone(info)
        self.assertEqual(info.working_distance_mm, 800.0)
        self.assertEqual(info.f_number, 8.0)

    def test_no_capture_ancestor(self):
        path = Path(r"C:\images\some_folder\Left_1.raw")
        self.assertIsNone(cf.find_capture_folder_info(path))

    def test_does_not_walk_past_max_depth(self):
        deep = Path(r"C:\wd600f4_Capture_2026-09-10_14-08-56\a\b\c\d\Left_1.raw")
        self.assertIsNone(cf.find_capture_folder_info(deep))


class FormattingTests(unittest.TestCase):
    def test_working_distance(self):
        info = cf.parse_capture_folder_name("wd600f4_Capture_2026-09-10_14-08-56")
        self.assertEqual(cf.format_working_distance(info), "600 mm")
        self.assertEqual(cf.format_working_distance(None), "—")

    def test_f_number_drops_trailing_zero_but_keeps_fraction(self):
        whole = cf.parse_capture_folder_name("wd600f4_Capture_2026-09-10_14-08-56")
        fractional = cf.parse_capture_folder_name("wd600f2.5_Capture_2026-09-01_09-47-42")
        self.assertEqual(cf.format_f_number(whole), "f/4")
        self.assertEqual(cf.format_f_number(fractional), "f/2.5")
        self.assertEqual(cf.format_f_number(None), "—")


if __name__ == "__main__":
    unittest.main()
