"""Inspect final MC-Calib geometry, poses, images and residuals in Viser."""

import argparse
import time
import threading
import base64
import io
import plotly.graph_objects as go
from functools import lru_cache

from PIL import Image
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf
from visualization_data import load_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8086)
    parser.add_argument(
        "--data", type=Path, required=True, help="visualization_data.yml from final calibration"
    )
    parser.add_argument(
        "--image-root", type=Path, help="Rebase exported image paths under this directory"
    )
    parser.add_argument("--manifest", type=Path, help="Optional frame/timestamp manifest")
    parser.add_argument("--object", type=int, default=None, help="Fixed reconstructed object ID")
    parser.add_argument("--group", type=int, default=None, help="Camera group ID")
    parser.add_argument(
        "--units-per-meter",
        type=float,
        default=1.0,
        help="1 for square_size in metres, 1000 for millimetres",
    )
    parser.add_argument("--camera-names", nargs="+", help="Names in ascending camera-ID order")
    parser.add_argument(
        "--frame-period",
        type=float,
        default=0.2,
        help="Playback seconds per sample when timestamps are absent",
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.frame_period <= 0:
        parser.error("Port and frame period must be positive and valid")
    data = load_scene(
        args.data,
        args.object,
        args.group,
        args.image_root,
        args.manifest,
        args.units_per_meter,
        args.camera_names,
    )
    frames = data["frames"]
    overlays = [f["overlays"] for f in frames]
    names = data["camera_names"]
    colors = [(35, 115, 245), (235, 55, 55), (140, 85, 220), (40, 160, 120), (220, 140, 40)]
    points = np.array([p for b in data["boards"] for p in b["points"]])
    center = (points.min(0) + points.max(0)) / 2
    extent = max(float(np.linalg.norm(points.max(0) - points.min(0))), 0.1)
    home_position = center + np.array([-0.9, 0.7, -1.2]) * extent
    scale = max(float(np.median([np.linalg.norm(b["u"]) for b in data["boards"]])), 0.001) / 0.035
    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="MC-Calib inspection")
    server.gui.configure_theme(control_width="large", show_share_button=False)
    server.gui.main_panel.dock_left()
    server.gui.main_panel.set_width(260)
    server.scene.set_up_direction("+y")
    server.initial_camera.position = home_position
    server.initial_camera.look_at = center
    server.scene.add_frame(
        "/board_coordinates", axes_length=0.10 * scale, axes_radius=0.0015 * scale
    )
    palette = [(80, 155, 235), (235, 160, 70), (85, 190, 135)]
    for board_index, board in enumerate(data["boards"]):
        color = palette[board_index % len(palette)]
        o, u, v = [np.array(board[k]) for k in ("origin", "u", "v")]
        for parity in (0, 1):
            vertices, faces = [], []
            for y in range(-1, board["squares_y"] - 1):
                for x in range(-1, board["squares_x"] - 1):
                    if (x + y) % 2 != parity:
                        continue
                    n = len(vertices)
                    vertices.extend(
                        [
                            o + u * a + v * b
                            for a, b in [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
                        ]
                    )
                    faces.extend([(n, n + 1, n + 2), (n, n + 2, n + 3)])
            server.scene.add_mesh_simple(
                f'/boards/{board["id"]}/squares_{parity}',
                np.array(vertices),
                np.array(faces),
                color=color if parity == 0 else (225, 230, 235),
                side="double",
            )
        server.scene.add_label(
            f'/boards/{board["id"]}/label',
            f'Board {board["id"]}',
            position=o + (board["squares_x"] - 2) / 2 * u + (board["squares_y"] - 0.5) * v,
            anchor="bottom-center",
        )
        server.scene.add_point_cloud(
            f'/boards/{board["id"]}/corners',
            points=np.array(board["points"]),
            colors=(25, 25, 25),
            point_size=0.003 * scale,
        )
    rig = server.scene.add_frame("/rig", show_axes=False)
    camera_nodes = []
    for camera_index, name in enumerate(names):
        color = colors[camera_index % len(colors)]
        base = f'/rig/camera_{data["camera_ids"][camera_index]}'
        node = server.scene.add_frame(base, axes_length=0.025 * scale, axes_radius=0.0007 * scale)
        server.scene.add_camera_frustum(
            base + "/camera",
            fov=np.deg2rad(60),
            aspect=1.0,
            scale=0.025 * scale,
            color=color,
            thickness=0.001 * scale,
        )
        server.scene.add_line_segments(
            base + "/optical_axis",
            points=np.array([[[0, 0, 0], [0, 0, 0.075]]]) * scale,
            colors=color,
            thickness=0.0015 * scale,
        )
        server.scene.add_label(
            base + "/label", name + " +Z", position=(0, 0, 0.085 * scale), anchor="bottom-center"
        )
        server.scene.add_icosphere(
            base + "/selected_position",
            radius=0.007 * scale,
            color=color,
            subdivisions=2,
            cast_shadow=False,
            receive_shadow=False,
        )
        camera_nodes.append(node)
    trails = []
    dot_mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.002 * scale)
    for index in range(len(names)):
        color = colors[index % len(colors)]
        positions = np.array([f["cameras"][index]["center"] for f in frames if f["cameras"]])
        trails.append(
            server.scene.add_batched_meshes_simple(
                f"/positions/{index}",
                vertices=np.asarray(dot_mesh.vertices),
                faces=np.asarray(dot_mesh.faces),
                batched_wxyzs=np.tile([1.0, 0.0, 0.0, 0.0], (len(positions), 1)),
                batched_positions=positions,
                batched_colors=color,
                opacity=0.50,
                cast_shadow=False,
                receive_shadow=False,
                lod="off",
            )
        )
    server.gui.add_markdown(
        f'**Object {data["object_id"]} fixed** · Group {data["group_id"]} · Coordinates in metres'
    )
    slider = server.gui.add_slider(
        "Sampled frame", min=0, max=len(frames) - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Play", initial_value=False)
    show_trail = server.gui.add_checkbox("Show frame positions", initial_value=True)
    status = server.gui.add_markdown("")
    show_overlay = server.gui.add_checkbox("Corner overlays", initial_value=True)
    image_view = server.gui.add_dropdown(
        "Image view", options=("Full image", "Corner detail"), initial_value="Corner detail"
    )
    error_status = server.gui.add_markdown("")

    @lru_cache(maxsize=24)
    def image_source(frame_index, cam):
        raw = frames[frame_index]["images"][cam]
        if raw is None or not Path(raw).is_file():
            return None
        path = Path(raw)
        with Image.open(path) as source:
            stream = io.BytesIO()
            source.convert("RGB").save(stream, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode()

    def image_figure(frame_index, cam, overlay_enabled, view):
        corners = overlays[frame_index][cam]["corners"]
        source = image_source(frame_index, cam)
        figure = go.Figure()
        if source:
            figure.add_trace(go.Image(source=source, hoverinfo="skip"))
        else:
            figure.add_annotation(
                text="Image unavailable", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
            )
        if overlay_enabled and corners:
            xy = np.array([c["detected"] for c in corners])
            figure.add_trace(
                go.Scatter(
                    x=xy[:, 0].tolist(),
                    y=xy[:, 1].tolist(),
                    mode="markers",
                    name="Detected",
                    marker=dict(symbol="circle-open", size=10, color="#00dfff", line=dict(width=2)),
                    text=[f'Board {c["board"]}, corner {c["id"]}' for c in corners],
                    hovertemplate="%{text}<br>Detected (%{x:.2f}, %{y:.2f})<extra></extra>",
                )
            )
            projected = [c for c in corners if c["projected"] is not None]
            if projected:
                xy = np.array([c["projected"] for c in projected])
                colors = [
                    (
                        "#3cce45"
                        if c["error"] < 0.5
                        else ("#e7a500" if c["error"] < 1.5 else "#ff3434")
                    )
                    for c in projected
                ]
                figure.add_trace(
                    go.Scatter(
                        x=xy[:, 0].tolist(),
                        y=xy[:, 1].tolist(),
                        mode="markers",
                        name="Reprojected",
                        marker=dict(symbol="cross", size=6, color=colors),
                        text=[
                            f'Board {c["board"]}, corner {c["id"]}<br>Error {c["error"]:.3f} px'
                            for c in projected
                        ],
                        hovertemplate="%{text}<br>Projected (%{x:.2f}, %{y:.2f})<extra></extra>",
                    )
                )
        width, height = data["camera_sizes"][cam]
        xr, yr = [0, width], [height, 0]
        if view == "Corner detail" and corners:
            pts = np.array([c["detected"] for c in corners])
            lo, hi = pts.min(0), pts.max(0)
            xr = [max(0, float(lo[0] - 80)), min(width, float(hi[0] + 80))]
            yr = [min(height, float(hi[1] + 80)), max(0, float(lo[1] - 80))]
        name = names[cam]
        stamp = frames[frame_index]["time"]
        suffix = f" · {stamp:.2f} s" if stamp is not None else ""
        figure.update_layout(
            title=dict(
                text=f'{name} · frame {frames[frame_index]["frame"]}{suffix}', font=dict(size=14)
            ),
            margin=dict(l=35, r=10, t=38, b=25),
            dragmode="pan",
            uirevision=f"{cam}-{view}",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="black",
            font=dict(color="#333"),
        )
        figure.update_xaxes(range=xr, showgrid=False, zeroline=False, constrain="domain")
        figure.update_yaxes(
            range=yr,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
            constrain="domain",
        )
        return figure

    inspector = server.gui.add_panel()
    inspector.dock_right()
    inspector.set_width(560)
    plot_config = {
        "scrollZoom": True,
        "displaylogo": False,
        "displayModeBar": True,
        "modeBarButtonsToRemove": ["select2d", "lasso2d"],
        "doubleClick": "reset",
    }
    with inspector.add_tab("Lens inspection"):
        server.gui.add_markdown(
            "**Wheel:** zoom · **Drag:** pan · **Double-click:** reset. Hover corners for pixel error. Each lens zooms independently."
        )
        plots = [
            server.gui.add_plotly(
                image_figure(0, cam, True, "Corner detail"), config=plot_config, aspect=1.55
            )
            for cam in range(len(names))
        ]
        server.gui.add_markdown(
            "Cyan rings: detected. Crosses: reprojected. Green <0.5 px · amber 0.5–1.5 px · red ≥1.5 px. Native final reprojections for the selected reconstructed object. No pose refitting."
        )
    update_lock = threading.Lock()
    home = server.gui.add_button("Whole scene")
    close = server.gui.add_button("Inspect lenses")
    server.gui.add_markdown(
        "Large solid dots mark the selected frame; historical dots are 50% opaque. Camera colors follow image order (blue, red, then additional colors). Glyphs indicate orientation, not lens FOV. Axes: X red, Y green, Z blue."
    )

    def update():
        # Serialize image decoding and discard stale slider callbacks before publishing.
        with update_lock:
            frame_index = slider.value
            settings = (show_overlay.value, image_view.value)
            images = [image_figure(frame_index, cam, *settings) for cam in range(len(names))]
            if slider.value != frame_index or settings != (show_overlay.value, image_view.value):
                return
            publish(frame_index, images)

    def publish(frame_index, images):
        record = frames[frame_index]
        with server.atomic():
            summaries = []
            for name, lens in zip(names, overlays[frame_index]):
                count = len(lens["corners"])
                matched = sum(c["projected"] is not None for c in lens["corners"])
                metrics = (
                    f'RMS {lens["rms"]:.3f} px · max {lens["max"]:.2f} px'
                    if lens["rms"] is not None
                    else "No reprojection available"
                )
                summaries.append(f"**{name}**: {matched}/{count} corners projected · {metrics}")
            error_status.content = "\n\n".join(summaries)
            for plot, figure in zip(plots, images):
                plot.figure = figure
            rig.visible = record["cameras"] is not None
            close.disabled = not rig.visible
            status.content = f'**Frame {record["frame"]}**'
            if record["time"] is not None:
                status.content += f' · {record["time"]:.2f} s'
            status.content += "\n\n"
            if record["cameras"]:
                for node, camera in zip(camera_nodes, record["cameras"]):
                    node.position = np.array(camera["center"])
                    node.wxyz = tf.SO3.from_matrix(np.array(camera["R"])).wxyz
                if record["rms"] is not None:
                    status.content += (
                        f'Native RMS **{record["rms"]:.3f} px** · {record["points"]} corners'
                    )
            else:
                status.content += "**Pose unavailable**\n\n" + record["reason"]

    @slider.on_update
    def _(_event):
        update()

    @show_overlay.on_update
    def _(_event):
        update()

    @image_view.on_update
    def _(_event):
        update()

    @show_trail.on_update
    def _(_event):
        for trail in trails:
            trail.visible = show_trail.value

    @home.on_click
    def _(event):
        if event.client:
            event.client.camera.position = home_position
            event.client.camera.look_at = center

    @close.on_click
    def _(event):
        record = frames[slider.value]
        if event.client and record["cameras"]:
            center = np.mean([c["center"] for c in record["cameras"]], axis=0)
            event.client.camera.position = center + np.array([-0.13, 0.10, -0.15])
            event.client.camera.look_at = center

    update()
    while True:
        current = slider.value
        delay = args.frame_period
        if (
            current + 1 < len(frames)
            and frames[current]["time"] is not None
            and frames[current + 1]["time"] is not None
        ):
            delay = max(0.01, frames[current + 1]["time"] - frames[current]["time"])
        time.sleep(delay)
        if playing.value:
            if slider.value == len(frames) - 1:
                playing.value = False
            else:
                slider.value += 1


if __name__ == "__main__":
    main()
