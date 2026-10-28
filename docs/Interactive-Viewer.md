# Interactive calibration inspection

Inspect reconstructed boards and final optimized camera poses with Viser, alongside synchronized images and native reprojection errors.
The viewer works with any calibrated camera group and reconstructed object; it does not depend on the X5 preparation workflow.

## Install and launch

Use a separate Python 3.10+ environment to avoid replacing OpenCV in an existing calibration environment.

```bash
python3 -m venv .venv-viewer
.venv-viewer/bin/python -m pip install -r python_utils/requirements_viewer.txt
.venv-viewer/bin/python python_utils/view_calibration.py \
  --data output/my_calibration/visualization_data.yml \
  --image-root data/my_images
```

Rebuild and rerun calibration to produce `visualization_data.yml`.
Existing camera/object/pose exports remain unchanged, but the legacy pose file cannot be used as a substitute: it lacks frame IDs and contains per-camera poses rather than final optimized group poses.
The server listens on `127.0.0.1:8086` by default.
Open that address in a browser; stop the server with Ctrl+C.

| Option | Purpose |
|---|---|
| `--object ID` | Fixed reconstructed object, default lowest ID |
| `--group ID` | Camera group, default lowest ID |
| `--image-root PATH` | Replace exported path prefixes, retaining camera-folder and filename |
| `--manifest PATH` | Optional frame timestamps and samples without detections |
| `--units-per-meter N` | 1 for calibration in metres, 1000 for calibration in millimetres |
| `--camera-names NAME ...` | Labels in ascending camera-ID order, for example `Front Rear` |
| `--port PORT` | Local HTTP/WebSocket port |
| `--frame-period SECONDS` | Playback interval when timestamps are unavailable |

Camera and object coordinates are converted to metres using the supplied unit scale.
No assumption is made that all reconstructed objects share a common coordinate system: choose one object at a time.
For multiple disconnected camera groups, choose a group that observed the selected object.
Image paths must identify the same synchronized samples used in calibration; the viewer never estimates a time offset or pairs images by nearest timestamp.
Missing images are labeled, while missing object poses hide the current camera glyphs without interpolating motion.

The optional JSON manifest contains a `frames` array with `file` and `timestamp_seconds` fields, and optionally `frame_id` and `source_frame_index`.
When `frame_id` is absent, array indices are the calibration frame IDs.
For manifest-only samples, images are located under `--image-root/Cam_001`, `Cam_002`, and so on, according to zero-based camera IDs; this convention matches the X5 extraction helper.
Without a manifest, the slider traverses exported frame IDs in sorted order and makes no claim about elapsed recording time.

## Inspect the result

Use the frame slider or playback to move the rig while the reconstructed object stays fixed.
The first two cameras are blue and red; additional cameras receive distinct colors.
Historical positions are individual dots at 50% opacity, and larger solid dots identify the current frame.
Dots do not select frames when clicked.
The camera glyphs indicate orientation, not lens FOV or the physical housing dimensions.

Each camera image has independent wheel zoom and drag-to-pan, with double-click and toolbar controls to reset the axes.
Hover over a corner to see its board ID, corner ID, and pixel error.
Cyan rings show detected corners; crosses show projected corners.
Projection crosses are green below 0.5 px, amber from 0.5 to 1.5 px, and red at or above 1.5 px.
These colors are display thresholds, not calibration acceptance criteria.
The images and numeric errors use final native MC-Calib projections of observations belonging to the selected object.
They are in-sample residuals, not held-out validation.

The image inspector and controls are independently resizable docked panels.
Zoom and pan are retained across frame changes; switching between full-image and corner-detail mode resets the image bounds.
The corner-detail bounds may crop printed board edges because they frame the detected corners rather than the physical paper.

## Export conventions

The additive `visualization_data.yml` schema version 1 contains cameras, reconstructed objects with board topology, and frames with image paths, group poses, and observations.
Transforms are named by direction: `camera_to_group` maps camera coordinates into the reference camera's coordinates, and `object_to_group` maps reconstructed object coordinates into that same group.
The fixed-object camera pose is `inverse(object_to_group) @ camera_to_group`.
Translations use the configured square-size units; the export does not guess whether those units are metres or millimetres.

The exporter reads `CameraGroupObs::object_pose_`, the state optimized by Ceres, and uses the native projection function for both perspective and fisheye cameras.
It exports each group/object pose once per frame, rather than duplicating poses for individual lens observations.
No Python pose fit or camera-model approximation is performed.

## Checks

```bash
PYTHONPATH=python_utils .venv-viewer/bin/python -m unittest discover \
  -s python_utils/tests -p 'test_visualization_data.py'
cd build
./tests/boost_tests_run --run_test=CheckMcCalibLight
```
