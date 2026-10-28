"""Extract synchronized raw X5 streams and prepare a central-field KB4 calibration."""

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def probe_frames(source, stream):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        f"v:{stream}",
        "-show_entries",
        "stream=width,height:frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(source),
    ]
    info = json.loads(subprocess.check_output(command))
    if len(info.get("streams", [])) != 1:
        raise ValueError(f"Missing video stream {stream}")
    size = tuple(info["streams"][0][key] for key in ("width", "height"))
    times = np.array(
        [float(frame["best_effort_timestamp_time"]) for frame in info.get("frames", [])]
    )
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Stream {stream} has missing or non-monotonic timestamps")
    return size, times


def validate_sync(front, rear, tolerance):
    if len(front) != len(rear) or not len(front):
        raise ValueError("Streams have different or empty frame counts")
    difference = float(np.max(np.abs(np.asarray(front) - np.asarray(rear))))
    if difference > tolerance:
        raise ValueError(f"Stream timestamps differ by {difference:.6f}s; limit {tolerance}s")
    return difference


def extract(args):
    if args.every < 1 or args.sync_tolerance < 0:
        raise ValueError("Sampling interval must be positive; tolerance must be nonnegative")
    front_size, front = probe_frames(args.source, 0)
    rear_size, rear = probe_frames(args.source, 1)
    if front_size != rear_size:
        raise ValueError("Front/rear stream dimensions differ")
    difference = validate_sync(front, rear, args.sync_tolerance)
    # Never mix old images with a new extraction.
    args.output.mkdir(parents=True, exist_ok=False)
    for camera in range(2):
        folder = args.output / f"Cam_{camera + 1:03d}"
        folder.mkdir()
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-n",
                "-noautorotate",
                "-i",
                str(args.source),
                "-map",
                f"0:v:{camera}",
                "-vf",
                f"select=not(mod(n\\,{args.every}))",
                "-vsync",
                "vfr",
                "-start_number",
                "0",
                str(folder / "frame_%06d.png"),
            ],
            check=True,
        )
        expected = len(range(0, len(front), args.every))
        if len(list(folder.glob("frame_*.png"))) != expected:
            raise RuntimeError(f"Stream {camera}: extracted frame count does not match timestamps")
    manifest = {
        "source": str(args.source.resolve()),
        "resolution": list(front_size),
        "streams": {
            "Cam_001": {"stream_index": 0, "lens": "front"},
            "Cam_002": {"stream_index": 1, "lens": "rear"},
        },
        "sampling": f"every {args.every} decoded frames; no flip, crop, or rotation",
        "max_presentation_timestamp_difference_seconds": difference,
        "frames": [
            {
                "file": f"frame_{i:06d}.png",
                "source_frame_index": n,
                "timestamp_seconds": float(front[n] - front[0]),
            }
            for i, n in enumerate(range(0, len(front), args.every))
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def sequence(node):
    return [node.at(i).real() for i in range(node.size())]


def read_cache(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f"Cannot open detection cache: {path}")
    cameras = []
    try:
        for camera in range(int(fs.getNode("nb_camera").real())):
            node = fs.getNode(f"camera_{camera}")
            size = (int(node.getNode("img_width").real()), int(node.getNode("img_height").real()))
            observations = []
            frames = sequence(node.getNode("frame_idxs"))
            for i, frame in enumerate(frames):
                points = np.array(sequence(node.getNode("pts_2d").at(i))).reshape(-1, 2)
                ids = np.array(sequence(node.getNode("charuco_idxs").at(i)), dtype=int)
                if len(points) != len(ids) or len(set(ids)) != len(ids):
                    raise ValueError("Inconsistent corner IDs in cache")
                observations.append(
                    {
                        "frame": int(frame),
                        "board": int(node.getNode("board_idxs").at(i).real()),
                        "path": node.getNode("frame_paths").at(i).string(),
                        "points": points,
                        "ids": ids,
                    }
                )
            cameras.append((size, observations))
    finally:
        fs.release()
    if len(cameras) != 2 or any(min(size) <= 0 for size, _ in cameras):
        raise ValueError("X5 preparation requires exactly two cameras with valid image sizes")
    return cameras


def select_views(observations, size, radius, minimum):
    center = np.array(size) / 2
    limit = radius * min(size) / 2
    return [
        o
        for o in observations
        if len(o["ids"]) >= minimum
        and np.isfinite(o["points"]).all()
        and np.max(np.linalg.norm(o["points"] - center, axis=1)) <= limit
    ]


def object_points(ids, squares_x, square_size):
    return (
        np.column_stack((ids % (squares_x - 1), ids // (squares_x - 1), np.zeros(len(ids))))
        * square_size
    )


def bootstrap(observations, size, squares_x, square_size, max_views):
    if len(observations) < 15:
        raise ValueError(f"Only {len(observations)} central views; at least 15 are required")
    observations = sorted(observations, key=lambda o: (o["frame"], o["board"]))
    selected = [
        observations[i]
        for i in np.linspace(0, len(observations) - 1, min(max_views, len(observations)), dtype=int)
    ]
    objects = [
        object_points(o["ids"], squares_x, square_size).astype(np.float32).reshape(1, -1, 3)
        for o in selected
    ]
    pixels = [o["points"].astype(np.float32).reshape(1, -1, 2) for o in selected]
    width, height = size
    K = np.array([[width / np.pi, 0, width / 2], [0, width / np.pi, height / 2], [0, 0, 1.0]])
    # OpenCV 4 exposes fisheye flags in cv2.fisheye; OpenCV 5 moved them to cv2.
    constants = cv2.fisheye if hasattr(cv2.fisheye, "CALIB_FIX_SKEW") else cv2
    flags = (
        constants.CALIB_USE_INTRINSIC_GUESS
        | constants.CALIB_RECOMPUTE_EXTRINSIC
        | constants.CALIB_FIX_SKEW
    )
    rms, K, D, _, _ = cv2.fisheye.calibrate(
        objects,
        pixels,
        size,
        K,
        np.zeros((4, 1)),
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8),
    )
    if (
        not np.isfinite(K).all()
        or not np.isfinite(D).all()
        or not np.isfinite(rms)
        or rms > 5
        or not 0 < K[0, 0] < 2 * width
        or not 0 < K[1, 1] < 2 * height
    ):
        raise ValueError(f"Unusable central intrinsic fit (RMS {rms})")
    return K, D, {"rms_px": rms, "selected_views": len(selected)}


def write_cache(path, cameras):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise ValueError(f"Cannot write {path}")

    def write_sequence(name, values):
        fs.startWriteStruct(name, cv2.FileNode_SEQ)
        for value in values:
            fs.write("", value)
        fs.endWriteStruct()

    fs.write("nb_camera", len(cameras))
    for camera, (size, observations) in enumerate(cameras):
        fs.startWriteStruct(f"camera_{camera}", cv2.FileNode_MAP)
        fs.write("img_width", size[0])
        fs.write("img_height", size[1])
        for name, key in [
            ("frame_idxs", "frame"),
            ("board_idxs", "board"),
            ("frame_paths", "path"),
        ]:
            write_sequence(name, [o[key] for o in observations])
        for name, key in [("pts_2d", "points"), ("charuco_idxs", "ids")]:
            fs.startWriteStruct(name, cv2.FileNode_SEQ)
            for o in observations:
                write_sequence("", o[key].ravel().tolist())
            fs.endWriteStruct()
        fs.endWriteStruct()
    fs.release()


def prepare(args):
    if (
        not np.isfinite(args.square_size)
        or args.square_size <= 0
        or min(args.squares_x, args.squares_y) < 3
        or args.boards < 1
        or not 0 < args.bootstrap_radius <= args.radius <= 1
        or not 4 <= args.min_corners <= (args.squares_x - 1) * (args.squares_y - 1)
        or args.max_views < 15
    ):
        raise ValueError("Invalid board geometry, radius, or view selection options")
    cameras = read_cache(args.cache)
    cv2.setNumThreads(1)
    seeds, filtered, diagnostics = [], [], []
    for size, observations in cameras:
        for o in observations:
            if (
                not 0 <= o["board"] < args.boards
                or np.any(o["ids"] < 0)
                or np.any(o["ids"] >= (args.squares_x - 1) * (args.squares_y - 1))
            ):
                raise ValueError("Cache IDs do not match the specified boards")
        initial = select_views(observations, size, args.bootstrap_radius, args.min_corners)
        K, D, report = bootstrap(initial, size, args.squares_x, args.square_size, args.max_views)
        retained = select_views(observations, size, args.radius, args.min_corners)
        seeds.append((K, D))
        filtered.append((size, retained))
        report.update(original_views=len(observations), retained_views=len(retained))
        diagnostics.append(report)
    # Do not leave a partially prepared run when fitting fails or overwrite previous results.
    args.output.mkdir(parents=True, exist_ok=False)
    write_cache(args.output / "central_detections.yml", filtered)
    fs = cv2.FileStorage(str(args.output / "initial_intrinsics.yml"), cv2.FILE_STORAGE_WRITE)
    fs.write("nb_camera", 2)
    for camera, (K, D) in enumerate(seeds):
        fs.startWriteStruct(f"camera_{camera}", cv2.FileNode_MAP)
        fs.write("camera_matrix", K)
        fs.write("distortion_vector", D.T)
        fs.endWriteStruct()
    fs.release()
    config = {
        "number_x_square": args.squares_x,
        "number_y_square": args.squares_y,
        "resolution_x": 4820,
        "resolution_y": 4820,
        "length_square": 0.04,
        "length_marker": 0.03,
        "number_board": args.boards,
        "square_size": args.square_size,
        "distortion_model": 1,
        "number_camera": 2,
        "refine_corner": 1,
        "min_perc_pts": args.min_corners / ((args.squares_x - 1) * (args.squares_y - 1)),
        "fix_intrinsic": 0,
        "root_path": str(args.images.resolve()),
        "cam_prefix": "Cam_",
        "cam_params_path": str((args.output / "initial_intrinsics.yml").resolve()),
        "keypoints_path": str((args.output / "central_detections.yml").resolve()),
        "quaternion_averaging": 1,
        "ransac_threshold": 3,
        "number_iterations": 1000,
        "he_approach": 0,
        "save_path": str((args.output / "results").resolve()),
        "save_detection": 0,
        "save_reprojection": 0,
        "camera_params_file_name": "",
    }
    fs = cv2.FileStorage(str(args.output / "calibration.yml"), cv2.FILE_STORAGE_WRITE)
    for key, value in config.items():
        fs.write(key, value)
    for key in [
        "boards_index",
        "number_x_square_per_board",
        "number_y_square_per_board",
        "square_size_per_board",
        "resolution_x_per_board",
        "resolution_y_per_board",
        "distortion_per_camera",
    ]:
        fs.startWriteStruct(key, cv2.FileNode_SEQ)
        fs.endWriteStruct()
    fs.release()
    report = {
        "scope": "central-field initialization and selection, not full-FOV validation",
        "bootstrap_radius_fraction": args.bootstrap_radius,
        "radius_fraction": args.radius,
        "square_size": args.square_size,
        "cameras": diagnostics,
    }
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("extract", help="Extract two timestamp-checked raw video streams")
    command.add_argument("source", type=Path)
    command.add_argument("output", type=Path)
    command.add_argument("--every", type=int, default=24)
    command.add_argument("--sync-tolerance", type=float, default=1e-5)
    command.set_defaults(run=extract)
    command = commands.add_parser(
        "prepare", help="Seed KB4 intrinsics and filter whole board observations"
    )
    command.add_argument("cache", type=Path)
    command.add_argument("--images", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument(
        "--square-size", type=float, required=True, help="Measured square size in metres"
    )
    command.add_argument("--squares-x", type=int, default=5)
    command.add_argument("--squares-y", type=int, default=5)
    command.add_argument("--boards", type=int, default=4)
    command.add_argument(
        "--radius",
        type=float,
        default=800 / 960,
        help="Fraction of half the smaller image dimension",
    )
    command.add_argument("--bootstrap-radius", type=float, default=0.65)
    command.add_argument("--min-corners", type=int, default=8)
    command.add_argument("--max-views", type=int, default=100)
    command.set_defaults(run=prepare)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
