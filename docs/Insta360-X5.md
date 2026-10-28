# Insta360 X5

Calibrate both raw fisheye streams jointly using fixed ChArUco boards.
This workflow supplies central-view KB4 intrinsic seeds and excludes whole board observations outside a chosen image radius.
It does not change MC-Calib's camera model or establish calibration accuracy over the full lens FOV.

## Requirements

Build MC-Calib using the repository's Docker instructions or a supported native toolchain.
Install FFmpeg/ffprobe and Python with NumPy and OpenCV (the production Docker image already contains the Python dependencies).
Use the repository's board generator and measure the printed square size.
The supplied configuration assumes four 5×5 boards with 35 mm squares and the generator's 0.75 marker/square ratio.
Keep the boards fixed relative to one another and capture simultaneous observations that connect the boards, potentially through different lenses.

## Extract synchronized images

```bash
python3 python_utils/x5_calibration.py extract recording.insv data/x5 --every 24
```

The command checks every presentation timestamp and rejects mismatched stream counts, dimensions, or timing before extraction.
It decodes video stream 0 into `Cam_001` (front) and stream 1 into `Cam_002` (rear), without rotation, flipping, cropping, or stitching.
Verify the lens assignment visually for your recording mode.
The manifest records source-frame indices and elapsed timestamps.
The output directory must not already exist, including after a partially failed extraction.

## Detect, prepare, and calibrate

Edit `configs/insta360_x5.yml` to match your board layout, physical dimensions, image directory, and output directory.
Run from the repository root:

```bash
./build/apps/calibrate/calibrate configs/insta360_x5.yml --detect-only
python3 python_utils/x5_calibration.py prepare \
  output/x5_detections/detected_keypoints_data.yml \
  --images data/x5 --output output/x5_prepared \
  --square-size 0.035 --squares-x 5 --squares-y 5 --boards 4
./build/apps/calibrate/calibrate output/x5_prepared/calibration.yml
```

Use paths visible in the execution environment: when calibrating in Docker, run preparation in that same container mount so the generated absolute paths resolve there.
The `--detect-only` mode saves native detections and exits before intrinsic initialization.
Regular calibration also saves detections immediately after extraction, preserving them if a later stage fails.
The preparation command requires a fresh output directory and fits both cameras before writing outputs.
It produces intrinsic seeds, a filtered native cache, the joint calibration configuration, and selection diagnostics.
MC-Calib then jointly refines intrinsics, extrinsics, and board geometry; the seeds are not held fixed.

## Selection and limits

| Option | Default | Meaning |
|---|---:|---|
| `--bootstrap-radius` | 0.65 | Maximum corner radius for intrinsic initialization, as a fraction of half the smaller image dimension |
| `--radius` | 0.833333 | Maximum corner radius for the joint calibration cache; 800 px for 1920×1920 images |
| `--min-corners` | 8 | Minimum corners per board observation |
| `--max-views` | 100 | Maximum evenly sampled board observations per intrinsic seed |

At least 15 central board observations per lens are required for initialization.
The radii are geometric selection criteria, not residual-based rejection thresholds or guaranteed angle limits.
The configured radius applies to every corner of a board observation: a partially out-of-bounds observation is excluded entirely.
Initialization can still fail on poorly distributed views; collect more varied central views rather than treating arbitrary seeds as a successful calibration.

On the local X5 experiment that motivated this workflow, 509 synchronized image pairs produced a joint calibration with approximately 0.49 px in-sample RMS, a 27.34 mm baseline, and observed ray angles up to approximately 82° per lens.
These are results from one private recording, not expected tolerances for other devices or held-out validation.
MC-Calib's current fisheye optimization is not suitable for the complete greater-than-180° raw view; this workflow validates neither the outer rim nor rays behind the camera plane.
Disconnected board observation groups remain separate reconstructed objects, even if the boards were physically stationary.

## Checks

```bash
PYTHONPATH=python_utils python3 -m unittest discover -s python_utils/tests -p 'test_x5_calibration.py'
```
