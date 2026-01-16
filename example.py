import numpy as np

from duotact_simulator import DuotactSimulator
from geometry import DuotactGeometry
from rendering import DuotactVisualizer
from sensor import CameraPose, SimpleCamera


if __name__ == "__main__":
    geo = DuotactGeometry(
        length=0.04,
        width=0.04,
        height=0.08,
        ridge_width=0.005,
        ridge_length=0.028,
        nx=25,
    )
    sim = DuotactSimulator(geo, stiffness=180.0, spring_damping=3.0, mass=0.02, global_damping=0.04)

    # Press from left side toward +Y (inward), mid-height.
    press_center = np.array([geo.length * 0.5, -geo.width * 0.5, geo.height * 0.5])
    press_dir = np.array([0.0, 1.0, 0.0])
    presses = [
        {
            "center": press_center,
            "direction": press_dir,
            "magnitude": 0.08,
            "radius": 0.015,
        }
    ]

    mesh = sim.static_deform(presses)

    # Markers on left side as a grid (columns along length, rows from base to ridge).
    stride = 3  # column spacing along length
    rows = 5    # number of rows from base (0) to ridge (1)
    n = geo.nx
    col_idx = np.arange(0, n, stride)
    ts = np.linspace(0.0, 1.0, rows)
    markers_list = []
    for i in col_idx:
        base = mesh.vertices[i]
        ridge = mesh.vertices[2 * n + i]
        for t in ts:
            markers_list.append(base * (1.0 - t) + ridge * t)
    marker_positions = np.vstack(markers_list) if markers_list else np.zeros((0, 3))

    # save the marker positions
    np.savetxt("marker_world_coor.txt", marker_positions, fmt="%.3f")

    viz = DuotactVisualizer()

    # Virtual camera placed below the frame, looking upward to capture both side surfaces.
    cam = SimpleCamera(width=1280, height=960, fov_y_deg=100)
    pose = CameraPose(
        eye=np.array([geo.length * 0.5, 0.0, -0.04]),
        target=np.array([geo.length * 0.5, 0.0, 0.02]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    cam.render_mesh(mesh, pose, out_path="camera_view.png", face_alpha=0.7, outline_only=True, markers=marker_positions)

    # Save camera intrinsics and marker coordinates.
    K = np.array(
        [
            [cam.fx, 0.0, cam.width * 0.5],
            [0.0, cam.fy, cam.height * 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    np.savetxt("cam_intrinsic.txt", K, fmt="%.6f")

    marker_cam = cam.to_camera_coords(marker_positions, pose)
    np.savetxt("marker_3D_coor.txt", marker_cam, fmt="%.6f")

    uv, z_mark = cam.project(marker_positions, pose)
    marker_img = np.hstack([uv, z_mark.reshape(-1, 1)])
    np.savetxt("marker_img_coor.txt", marker_img, fmt="%.3f")

    # Visualize camera and frustum together with the object.
    frustum = cam.frustum_segments(pose, near=0.005, far=0.08)
    # Camera axes: Z axis points upward as requested.
    forward = pose.target - pose.eye
    forward = forward / (np.linalg.norm(forward) + 1e-9)
    z_cam = forward
    y_cam = -pose.up / (np.linalg.norm(pose.up) + 1e-9)
    x_cam = np.cross(y_cam, z_cam)
    x_cam = x_cam / (np.linalg.norm(x_cam) + 1e-9)
    cam_axes = np.stack([x_cam, y_cam, z_cam], axis=1)

    viz.show_static(
        mesh,
        title="Duotact + camera frustum",
        force_vectors=[
            {
                "center": press_center,
                "direction": press_dir,
                "scale": 0.02,
                "color": "orange",
                "name": "press",
            }
        ],
        frustum_segments=frustum,
        camera_point=pose.eye,
        markers=marker_positions,
        camera_axes=cam_axes,
        camera_axes_scale=0.02,
    )
