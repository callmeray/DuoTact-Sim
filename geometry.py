import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DuotactMesh:
    vertices: np.ndarray  # (N, 3)
    triangles: np.ndarray  # (M, 3) int32 indices (front/back caps)
    ridge_loops: List[List[int]]  # one loop per ridge rail
    frame_edges: List[Tuple[int, int]]


class DuotactGeometry:
    def __init__(
        self,
        length: float = 0.12,
        width: float = 0.04,
        height: float = 0.02,
        ridge_width: float = 0.01,
        ridge_length: float = None,
        nx: int = 25,
    ) -> None:
        """
        Hollow duotact frame: two base rails (left/right) and a narrow top beam (ridge),
        plus front/back end caps as thin surfaces. Only perimeter/frame is modeled, not a
        solid volume.
        nx controls resolution along the length direction.
        """
        self.length = length
        self.width = width
        self.height = height
        self.ridge_length = ridge_length if ridge_length is not None else length
        self.ridge_width = max(1e-4, min(ridge_width, width))
        self.nx = nx

        # Clamp ridge length to [tiny, length].
        self.ridge_length = max(1e-4, min(self.ridge_length, self.length))

        (
            self.base_indices,
            self.ridge_loops,
            self.frame_edges,
            self.rest_positions,
        ) = self._build_nodes()
        self.springs = self._build_springs(self.rest_positions)
        self.triangles = self._build_triangles()

    def _build_nodes(self) -> Tuple[np.ndarray, List[List[int]], List[Tuple[int, int]], np.ndarray]:
        x_base = np.linspace(0.0, self.length, self.nx)
        ridge_offset = 0.5 * (self.length - self.ridge_length)
        x_ridge = np.linspace(ridge_offset, ridge_offset + self.ridge_length, self.nx)
        half_w = self.width * 0.5
        half_rw = self.ridge_width * 0.5

        left = np.stack([x_base, -np.full_like(x_base, half_w), np.zeros_like(x_base)], axis=-1)
        right = np.stack([x_base, np.full_like(x_base, half_w), np.zeros_like(x_base)], axis=-1)
        ridge_l = np.stack([x_ridge, -np.full_like(x_ridge, half_rw), np.full_like(x_ridge, self.height)], axis=-1)
        ridge_r = np.stack([x_ridge, np.full_like(x_ridge, half_rw), np.full_like(x_ridge, self.height)], axis=-1)

        base_indices = np.arange(0, left.shape[0] + right.shape[0])
        ridge_start = base_indices.size
        ridge_loops = [
            list(range(ridge_start, ridge_start + ridge_l.shape[0])),
            list(range(ridge_start + ridge_l.shape[0], ridge_start + ridge_l.shape[0] + ridge_r.shape[0])),
        ]

        positions = np.vstack([left, right, ridge_l, ridge_r])

        edges: List[Tuple[int, int]] = []
        n = self.nx
        # Rails along length.
        for i in range(n - 1):
            edges.append((i, i + 1))  # left rail
            edges.append((n + i, n + i + 1))  # right rail
            edges.append((2 * n + i, 2 * n + i + 1))  # ridge left
            edges.append((3 * n + i, 3 * n + i + 1))  # ridge right
        # Cross-section beams.
        for i in range(n):
            left_i = i
            right_i = n + i
            ridge_l_i = 2 * n + i
            ridge_r_i = 3 * n + i
            edges.append((left_i, right_i))
            edges.append((left_i, ridge_l_i))
            edges.append((right_i, ridge_r_i))
            edges.append((ridge_l_i, ridge_r_i))

        return base_indices, ridge_loops, edges, positions

    def _build_springs(self, rest: np.ndarray) -> List[Tuple[int, int, float]]:
        springs: List[Tuple[int, int, float]] = []
        n = self.nx
        # Rails along length.
        for i in range(n - 1):
            springs.append((i, i + 1, np.linalg.norm(rest[i] - rest[i + 1])))  # left base
            springs.append((n + i, n + i + 1, np.linalg.norm(rest[n + i] - rest[n + i + 1])))  # right base
            springs.append((2 * n + i, 2 * n + i + 1, np.linalg.norm(rest[2 * n + i] - rest[2 * n + i + 1])))  # ridge left
            springs.append((3 * n + i, 3 * n + i + 1, np.linalg.norm(rest[3 * n + i] - rest[3 * n + i + 1])))  # ridge right

        # Cross-section braces.
        for i in range(n):
            left_i = i
            right_i = n + i
            ridge_l_i = 2 * n + i
            ridge_r_i = 3 * n + i
            springs.append((left_i, right_i, np.linalg.norm(rest[left_i] - rest[right_i])))
            springs.append((left_i, ridge_l_i, np.linalg.norm(rest[left_i] - rest[ridge_l_i])))
            springs.append((right_i, ridge_r_i, np.linalg.norm(rest[right_i] - rest[ridge_r_i])))
            springs.append((ridge_l_i, ridge_r_i, np.linalg.norm(rest[ridge_l_i] - rest[ridge_r_i])))

        # Optional diagonals for stability (across neighboring cross-sections).
        for i in range(n - 1):
            springs.append((i, n + i + 1, np.linalg.norm(rest[i] - rest[n + i + 1])))
            springs.append((n + i, i + 1, np.linalg.norm(rest[n + i] - rest[i + 1])))
            springs.append((2 * n + i, n + i + 1, np.linalg.norm(rest[2 * n + i] - rest[n + i + 1])))
            springs.append((3 * n + i, i + 1, np.linalg.norm(rest[3 * n + i] - rest[i + 1])))
        return springs

    def _build_triangles(self) -> np.ndarray:
        # Keep left/right side surfaces; leave front/back open.
        tris: List[Tuple[int, int, int]] = []
        n = self.nx

        # Left side panels (between base left rail and ridge left rail).
        for i in range(n - 1):
            left_a = i
            left_b = i + 1
            ridge_a = 2 * n + i
            ridge_b = 2 * n + i + 1
            tris.append((left_a, left_b, ridge_b))
            tris.append((left_a, ridge_b, ridge_a))

        # Right side panels (between base right rail and ridge right rail).
        for i in range(n - 1):
            right_a = n + i
            right_b = n + i + 1
            ridge_a = 3 * n + i
            ridge_b = 3 * n + i + 1
            tris.append((right_a, ridge_b, right_b))
            tris.append((right_a, ridge_a, ridge_b))

        return np.asarray(tris, dtype=np.int32)

    def mesh_from_positions(self, positions: np.ndarray) -> DuotactMesh:
        return DuotactMesh(
            vertices=positions,
            triangles=self.triangles,
            ridge_loops=self.ridge_loops,
            frame_edges=self.frame_edges,
        )
