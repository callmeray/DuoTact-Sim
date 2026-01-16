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

    # 3D visualization for reference.
    viz = DuotactVisualizer()
    viz.show_static(
        mesh,
        title="Duotact side press (static deformation)",
        force_vectors=[
            {
                "center": press_center,
                "direction": press_dir,
                "scale": 0.02,
                "color": "orange",
                "name": "press",
            }
        ],
        markers=marker_positions,
    )

    # Virtual camera placed below the frame, looking upward to capture both side surfaces.
    cam = SimpleCamera(width=1280, height=960, fov_y_deg=100)
    pose = CameraPose(
        eye=np.array([geo.length * 0.5, 0.0, -0.04]),
        target=np.array([geo.length * 0.5, 0.0, 0.02]),
        up=np.array([0.0, 1.0, 0.0]),
    )
    cam.render_mesh(mesh, pose, out_path="camera_view.png", face_alpha=0.7, outline_only=True, markers=marker_positions)

    # Visualize camera and frustum together with the object.
    frustum = cam.frustum_segments(pose, near=0.005, far=0.08)
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
    )
