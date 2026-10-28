import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from visualization_data import camera_in_object, load_scene


def fixture(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("schema_version", 1)
    fs.startWriteStruct("cameras", cv2.FileNode_SEQ)
    for cid in (0, 1):
        fs.startWriteStruct("", cv2.FileNode_MAP)
        fs.write("camera_id", cid)
        fs.write("camera_group_id", 3)
        T = np.eye(4)
        if cid:
            T[:3, :3] = np.diag([-1.0, 1.0, -1.0])
            T[2, 3] = -0.03
        fs.write("camera_to_group", T)
        fs.write("width", 640)
        fs.write("height", 480)
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct("objects", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("object_id", 2)
    fs.write("points", np.array([[0.0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0.03, 0.03, 0]]))
    fs.startWriteStruct("boards", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("board_id", 4)
    fs.write("squares_x", 3)
    fs.write("squares_y", 3)
    fs.startWriteStruct("point_indices", cv2.FileNode_SEQ)
    for i in range(4):
        fs.write("", i)
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct("frames", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("frame_id", 7)
    fs.startWriteStruct("images", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("camera_id", 0)
    fs.write("path", "/container/Cam_001/image_7.png")
    fs.endWriteStruct()
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("camera_id", 1)
    fs.write("path", "")
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct("groups", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("camera_group_id", 3)
    fs.startWriteStruct("poses", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("object_id", 2)
    T = np.eye(4)
    T[:3, 3] = [1, 2, 3]
    fs.write("object_to_group", T)
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct("observations", cv2.FileNode_SEQ)
    fs.startWriteStruct("", cv2.FileNode_MAP)
    fs.write("camera_id", 0)
    fs.write("object_id", 2)
    fs.startWriteStruct("point_indices", cv2.FileNode_SEQ)
    fs.write("", 0)
    fs.endWriteStruct()
    fs.write("detected", np.array([[20.0, 30.0]]))
    fs.write("projected", np.array([[20.3, 30.4]]))
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.release()


class ViewerDataTests(unittest.TestCase):
    def test_native_poses_errors_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            export = root / "visualization_data.yml"
            fixture(export)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "frames": [
                            {"frame_id": 7, "file": "image_7.png", "timestamp_seconds": 0.7},
                            {"frame_id": 9, "file": "image_9.png", "timestamp_seconds": 0.9},
                        ]
                    }
                )
            )
            scene = load_scene(
                export, 2, 3, root / "images", manifest, camera_names=["Front", "Rear"]
            )
            self.assertEqual([f["frame"] for f in scene["frames"]], [7, 9])
            current, missing = scene["frames"]
            np.testing.assert_allclose(current["cameras"][0]["center"], [-1, -2, -3])
            np.testing.assert_allclose(current["cameras"][1]["center"], [-1, -2, -3.03])
            np.testing.assert_allclose(current["cameras"][1]["R"], np.diag([-1, 1, -1]))
            self.assertAlmostEqual(current["overlays"][0]["rms"], 0.5)
            self.assertEqual(current["overlays"][0]["corners"][0]["board"], 4)
            self.assertIsNone(missing["cameras"])
            self.assertEqual(missing["overlays"][0]["corners"], [])
            self.assertEqual(current["images"][0], str(root / "images/Cam_001/image_7.png"))
            self.assertEqual(current["images"][1], str(root / "images/Cam_002/image_7.png"))
            self.assertEqual(scene["camera_sizes"], [(640, 480), (640, 480)])
            no_manifest = load_scene(export)
            self.assertIsNone(no_manifest["frames"][0]["images"][1])
            self.assertIsNone(no_manifest["frames"][0]["time"])
            millimetres = load_scene(export, units_per_meter=1000)
            np.testing.assert_allclose(
                millimetres["frames"][0]["cameras"][0]["center"], [-0.001, -0.002, -0.003]
            )

    def test_transform_composition_not_inverse_camera_pose(self):
        obj = np.eye(4)
        obj[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        obj[:3, 3] = [1, 2, 3]
        cam = np.eye(4)
        cam[:3, 3] = [0.02, 0.03, 0.04]
        result = camera_in_object(obj, cam)
        np.testing.assert_allclose(obj @ result, cam)

    def test_reject_legacy_pose_export(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.yml"
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
            fs.write("old_poses", np.eye(4))
            fs.release()
            with self.assertRaisesRegex(ValueError, "legacy poses"):
                load_scene(path)


if __name__ == "__main__":
    unittest.main()
