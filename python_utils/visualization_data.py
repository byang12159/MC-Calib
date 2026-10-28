"""Load final native calibration exports without estimating or refitting poses."""

import json
from pathlib import Path

import cv2
import numpy as np


def entries(node):
    return [node.at(i) for i in range(node.size())]


def integers(node):
    return [int(n.real()) for n in entries(node)]


def transform(node):
    matrix = node.mat()
    if matrix is None or matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 pose matrix")
    if (
        not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
        or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Invalid rigid pose matrix")
    return matrix


def camera_in_object(object_to_group, camera_to_group):
    """Compose camera->object, preserving the camera's optical coordinate axes."""
    return np.linalg.inv(object_to_group) @ camera_to_group


def load_scene(
    path,
    object_id=None,
    group_id=None,
    image_root=None,
    manifest=None,
    units_per_meter=1.0,
    camera_names=None,
):
    if not np.isfinite(units_per_meter) or units_per_meter <= 0:
        raise ValueError("units_per_meter must be positive")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"Cannot open {path}; rerun calibration with visualization export enabled")
    try:
        if int(fs.getNode("schema_version").real()) != 1:
            raise ValueError(
                "Expected visualization_data.yml schema version 1; legacy poses lack frame IDs"
            )
        cameras = {}
        for n in entries(fs.getNode("cameras")):
            cid = int(n.getNode("camera_id").real())
            cameras[cid] = {
                "id": cid,
                "group": int(n.getNode("camera_group_id").real()),
                "pose": transform(n.getNode("camera_to_group")),
                "size": (int(n.getNode("width").real()), int(n.getNode("height").real())),
            }
        if not cameras:
            raise ValueError("No calibrated cameras in export")
        objects = {int(n.getNode("object_id").real()): n for n in entries(fs.getNode("objects"))}
        if not objects:
            raise ValueError("No reconstructed objects in export")
        object_id = min(objects) if object_id is None else object_id
        group_id = min(c["group"] for c in cameras.values()) if group_id is None else group_id
        camera_ids = sorted(cid for cid, c in cameras.items() if c["group"] == group_id)
        if object_id not in objects or not camera_ids:
            raise ValueError("Requested object or camera group is not in the export")
        if camera_names and len(camera_names) != len(camera_ids):
            raise ValueError("Provide one camera name per selected camera (ascending camera ID)")
        names = camera_names or [f"Camera {cid}" for cid in camera_ids]
        obj = objects[object_id]
        points = obj.getNode("points").mat()
        if (
            points is None
            or points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
        ):
            raise ValueError("Invalid reconstructed object points")
        points = points / units_per_meter
        boards = []
        corner_lookup = {}
        for b in entries(obj.getNode("boards")):
            bid = int(b.getNode("board_id").real())
            sx, sy = (int(b.getNode(key).real()) for key in ("squares_x", "squares_y"))
            indices = integers(b.getNode("point_indices"))
            if (
                sx < 3
                or sy < 3
                or len(indices) != (sx - 1) * (sy - 1)
                or any(i < 0 or i >= len(points) for i in indices)
            ):
                raise ValueError("Board topology does not match reconstructed points")
            p = points[indices]
            boards.append(
                {
                    "id": bid,
                    "origin": p[0].tolist(),
                    "u": ((p[sx - 2] - p[0]) / (sx - 2)).tolist(),
                    "v": ((p[(sy - 2) * (sx - 1)] - p[0]) / (sy - 2)).tolist(),
                    "points": p.tolist(),
                    "squares_x": sx,
                    "squares_y": sy,
                }
            )
            corner_lookup.update({idx: (bid, i) for i, idx in enumerate(indices)})
        records = {}
        for n in entries(fs.getNode("frames")):
            fid = int(n.getNode("frame_id").real())
            images = {
                int(i.getNode("camera_id").real()): i.getNode("path").string()
                for i in entries(n.getNode("images"))
            }
            record = {
                "frame": fid,
                "cameras": None,
                "images": images,
                "overlays": [{"corners": [], "rms": None, "max": None} for _ in camera_ids],
                "reason": "Selected object has no optimized pose in this camera group",
            }
            for group in entries(n.getNode("groups")):
                if int(group.getNode("camera_group_id").real()) != group_id:
                    continue
                for pose in entries(group.getNode("poses")):
                    if int(pose.getNode("object_id").real()) != object_id:
                        continue
                    object_to_group = transform(pose.getNode("object_to_group"))
                    record["cameras"] = []
                    for cid, name in zip(camera_ids, names):
                        world = camera_in_object(object_to_group, cameras[cid]["pose"])
                        record["cameras"].append(
                            {
                                "name": name,
                                "center": (world[:3, 3] / units_per_meter).tolist(),
                                "R": world[:3, :3].tolist(),
                            }
                        )
                for observation in entries(group.getNode("observations")):
                    cid = int(observation.getNode("camera_id").real())
                    if (
                        cid not in camera_ids
                        or int(observation.getNode("object_id").real()) != object_id
                    ):
                        continue
                    measured = observation.getNode("detected").mat()
                    projected = observation.getNode("projected").mat()
                    ids = integers(observation.getNode("point_indices"))
                    if (
                        measured is None
                        or projected is None
                        or measured.shape != projected.shape
                        or measured.shape != (len(ids), 2)
                    ):
                        raise ValueError("Malformed native reprojection observation")
                    if not np.isfinite(measured).all() or not np.isfinite(projected).all():
                        raise ValueError("Non-finite native corner coordinates")
                    for idx, xy, pred in zip(ids, measured, projected):
                        bid, corner = corner_lookup[idx]
                        record["overlays"][camera_ids.index(cid)]["corners"].append(
                            {
                                "board": bid,
                                "id": corner,
                                "detected": xy.tolist(),
                                "projected": pred.tolist(),
                                "error": float(np.linalg.norm(pred - xy)),
                            }
                        )
            records[fid] = record
        times = {}
        if manifest:
            metadata = json.loads(Path(manifest).read_text())
            for index, info in enumerate(metadata["frames"]):
                fid = info.get("frame_id", index)
                if fid not in records:
                    records[fid] = {
                        "frame": fid,
                        "cameras": None,
                        "images": {},
                        "overlays": [{"corners": [], "rms": None, "max": None} for _ in camera_ids],
                        "reason": "No optimized pose for this sample",
                    }
                record = records[fid]
                times[fid] = float(info["timestamp_seconds"])
                record["source_frame"] = info.get("source_frame_index")
                if image_root:
                    for cid in camera_ids:
                        if not record["images"].get(cid):
                            record["images"][cid] = str(
                                Path(image_root) / f"Cam_{cid+1:03d}" / Path(info["file"]).name
                            )
        for fid, record in records.items():
            record["time"] = times.get(fid)
            paths = []
            for cid in camera_ids:
                raw = record["images"].get(cid)
                target = Path(raw) if raw else None
                if target is not None and image_root:
                    target = Path(image_root) / target.parent.name / target.name
                paths.append(str(target) if target is not None else None)
            record["images"] = paths
            all_errors = []
            for lens in record["overlays"]:
                errors = [c["error"] for c in lens["corners"]]
                if errors:
                    lens["rms"] = float(np.sqrt(np.mean(np.square(errors))))
                    lens["max"] = max(errors)
                    all_errors.extend(errors)
            record["rms"] = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else None
            record["points"] = len(all_errors)
        frames = [records[fid] for fid in sorted(records)]
        if not any(f["cameras"] for f in frames):
            raise ValueError("Selected object has no poses in the selected group")
        return {
            "boards": boards,
            "frames": frames,
            "camera_ids": camera_ids,
            "camera_names": names,
            "camera_sizes": [cameras[cid]["size"] for cid in camera_ids],
            "object_id": object_id,
            "group_id": group_id,
        }
    finally:
        fs.release()
