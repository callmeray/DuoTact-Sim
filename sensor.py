import math
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from geometry import DuotactMesh


@dataclass
class CameraPose:
    eye: np.ndarray  # (3,)
    target: np.ndarray  # (3,)
    up: np.ndarray  # (3,)


class SimpleCamera:
    def __init__(self, width: int = 1280, height: int = 960, fov_y_deg: float = 60.0) -> None:
        self.width = width
        self.height = height
        self.fov_y = math.radians(fov_y_deg)
        self.fx = 0.5 * width / math.tan(self.fov_y * 0.5)
        self.fy = self.fx

    @property
    def fov_x(self) -> float:
        return 2.0 * math.atan(0.5 * self.width / self.fx)

    @staticmethod
    def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        f = target - eye
        f = f / (np.linalg.norm(f) + 1e-9)
        u = up / (np.linalg.norm(up) + 1e-9)
        s = np.cross(f, u)
        s = s / (np.linalg.norm(s) + 1e-9)
        u2 = np.cross(s, f)
        # Camera space: x=s, y=u2, z=-f
        return np.stack([s, u2, -f], axis=0)

    def project(self, points: np.ndarray, pose: CameraPose) -> Tuple[np.ndarray, np.ndarray]:
        R = self._look_at(pose.eye, pose.target, pose.up)
        pc = (R @ (points - pose.eye).T).T  # (N,3)
        z = pc[:, 2:3]
        # Perspective projection; ignore points behind camera.
        u = self.fx * (pc[:, 0:1] / (z + 1e-9)) + self.width * 0.5
        v = self.fy * (pc[:, 1:2] / (z + 1e-9)) + self.height * 0.5
        return np.hstack([u, v]), z.ravel()


    def frustum_segments(
        self,
        pose: CameraPose,
        near: float = 0.01,
        far: float = 0.1,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return line segments (p0,p1) describing camera frustum in world space."""
        R = self._look_at(pose.eye, pose.target, pose.up)  # rows: cam axes in world
        fov_x = self.fov_x
        fov_y = self.fov_y
        tx = math.tan(fov_x * 0.5)
        ty = math.tan(fov_y * 0.5)

        def cam_dir(nx: float, ny: float) -> np.ndarray:
            d = np.array([nx * tx, ny * ty, -1.0])
            return d / (np.linalg.norm(d) + 1e-9)

        corners_cam = [
            cam_dir(-1, -1),
            cam_dir(1, -1),
            cam_dir(1, 1),
            cam_dir(-1, 1),
        ]

        segs: List[Tuple[np.ndarray, np.ndarray]] = []
        for d in corners_cam:
            pn = pose.eye + (R.T @ (d * near))
            pf = pose.eye + (R.T @ (d * far))
            segs.append((pose.eye, pn))
            segs.append((pn, pf))

        # Connect near and far rectangles.
        near_pts = [pose.eye + (R.T @ (d * near)) for d in corners_cam]
        far_pts = [pose.eye + (R.T @ (d * far)) for d in corners_cam]
        for i in range(4):
            segs.append((near_pts[i], near_pts[(i + 1) % 4]))
            segs.append((far_pts[i], far_pts[(i + 1) % 4]))
        return segs

    def render_mesh(
        self,
        mesh: DuotactMesh,
        pose: CameraPose,
        out_path: str,
        face_alpha: float = 0.6,
        outline_only: bool = False,
        markers: np.ndarray | None = None,
    ) -> None:
        uv, z = self.project(mesh.vertices, pose)
        tris = mesh.triangles

        if outline_only:
            fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
            n = len(mesh.ridge_loops[0]) if mesh.ridge_loops else 0
            if n > 0 and mesh.vertices.shape[0] >= 4 * n:
                # Build closed side polygons: base rail forward, ridge rail backward to close.
                left_base = list(range(0, n))
                left_ridge = list(range(2 * n, 3 * n))
                right_base = list(range(n, 2 * n))
                right_ridge = list(range(3 * n, 4 * n))

                def draw_side(base_idx, ridge_idx):
                    idxs = base_idx + ridge_idx[::-1]
                    pts = uv[idxs]
                    zs_side = z[idxs]
                    # drop points behind camera
                    mask = zs_side < -1e-6
                    if not np.any(mask):
                        return
                    pts = pts[mask]
                    if pts.shape[0] < 3:
                        return
                    # close poly
                    xs = list(pts[:, 0]) + [pts[0, 0]]
                    ys = list(pts[:, 1]) + [pts[0, 1]]
                    ax.plot(xs, ys, color="dimgray", linewidth=1.4)

                draw_side(left_base, left_ridge)
                draw_side(right_base, right_ridge)
            # markers projection in outline mode
            if markers is not None and markers.size > 0:
                uv_m, z_m = self.project(markers, pose)
                mask_m = z_m < -1e-6
                if np.any(mask_m):
                    ax.scatter(uv_m[mask_m, 0], uv_m[mask_m, 1], color="orange", s=12)
            ax.set_xlim(0, self.width)
            ax.set_ylim(self.height, 0)
            ax.set_aspect("equal")
            ax.axis("off")
            plt.tight_layout(pad=0)
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            return

        polys: List[np.ndarray] = []
        depths: List[float] = []
        for tri in tris:
            tri_z = z[tri]
            if np.any(tri_z >= -1e-6):
                # keep only triangles in front of camera (camera looks toward -Z in its space)
                continue
            polys.append(uv[tri])
            depths.append(np.mean(tri_z))

        if not polys:
            return

        # Painter's algorithm: far to near (z is negative; sort ascending -> far first)
        order = np.argsort(depths)
        polys_sorted = [polys[i] for i in order]

        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
        from matplotlib.collections import PolyCollection

        coll = PolyCollection(polys_sorted, facecolors="lightsteelblue", edgecolors="dimgray", linewidths=0.5, alpha=face_alpha)
        ax.add_collection(coll)
        if markers is not None and markers.size > 0:
            uv_m, z_m = self.project(markers, pose)
            mask_m = z_m < -1e-6
            if np.any(mask_m):
                ax.scatter(uv_m[mask_m, 0], uv_m[mask_m, 1], color="orange", s=12)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    def render_vectors(self, vectors: List[dict], pose: CameraPose, out_path: str, color: str = "orange", lw: float = 2.0) -> None:
        starts = []
        ends = []
        for v in vectors:
            c = np.asarray(v.get("center", (0, 0, 0)), dtype=float)
            d = np.asarray(v.get("direction", (0, 0, 1)), dtype=float)
            scale = float(v.get("scale", 0.02))
            starts.append(c)
            ends.append(c + d * scale)
        points = np.vstack([starts, ends]) if starts else np.zeros((0, 3))
        if points.shape[0] == 0:
            return
        uv, z = self.project(points, pose)
        half = uv.shape[0] // 2
        fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100), dpi=100)
        for i in range(half):
            s = uv[i]
            e = uv[half + i]
            if z[i] >= -1e-6 and z[half + i] >= -1e-6:
                continue
            ax.plot([s[0], e[0]], [s[1], e[1]], color=color, linewidth=lw)
            ax.scatter([s[0]], [s[1]], color=color, s=8)
        ax.set_xlim(0, self.width)
        ax.set_ylim(self.height, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
