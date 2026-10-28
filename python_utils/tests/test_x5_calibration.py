import tempfile
import unittest
from pathlib import Path

import numpy as np
from x5_calibration import object_points, read_cache, select_views, validate_sync, write_cache


class PreparationTests(unittest.TestCase):
    def test_timestamp_mismatch_is_rejected(self):
        self.assertAlmostEqual(validate_sync([0, 0.2], [0, 0.200001], 1e-5), 1e-6)
        with self.assertRaises(ValueError):
            validate_sync([0, 0.2], [0], 1e-5)
        with self.assertRaises(ValueError):
            validate_sync([0, 0.2], [0, 0.21], 1e-5)

    def test_whole_board_selection_and_roundtrip(self):
        center = np.array([600, 400])
        inside = {
            "frame": 7,
            "board": 2,
            "path": "Cam_001/frame_000007.png",
            "ids": np.array([0, 2, 5, 7]),
            "points": center + np.array([[0, 0], [10, 0], [0, 10], [10, 10]]),
        }
        edge = dict(inside, frame=9, points=inside["points"].copy())
        edge["points"][-1] = center + [350, 0]
        selected = select_views([inside, edge], (1200, 800), 0.8, 4)
        self.assertEqual([o["frame"] for o in selected], [7])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.yml"
            write_cache(path, [((1200, 800), selected)] * 2)
            loaded = read_cache(path)
            self.assertEqual(loaded[1][0], (1200, 800))
            self.assertEqual(loaded[1][1][0]["path"], inside["path"])
            np.testing.assert_array_equal(loaded[0][1][0]["ids"], inside["ids"])
            np.testing.assert_array_equal(loaded[0][1][0]["points"], inside["points"])

    def test_non_square_board_corner_mapping(self):
        actual = object_points(np.array([0, 4, 5, 9]), 6, 0.02)
        np.testing.assert_allclose(actual, [[0, 0, 0], [0.08, 0, 0], [0, 0.02, 0], [0.08, 0.02, 0]])


if __name__ == "__main__":
    unittest.main()
