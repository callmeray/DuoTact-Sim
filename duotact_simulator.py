import numpy as np
from typing import Callable, List, Optional

from geometry import DuotactGeometry, DuotactMesh

ExternalForce = Optional[Callable[[np.ndarray], np.ndarray]]


def _gaussian_weights(points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    center = center.reshape(1, 3)
    radius_sq = max(radius, 1e-6) ** 2
    dist_sq = np.sum((points - center) ** 2, axis=1, keepdims=True)
    return np.exp(-dist_sq / (2.0 * radius_sq))


class DuotactSimulator:
    def __init__(
        self,
        geometry: DuotactGeometry,
        stiffness: float = 120.0,
        spring_damping: float = 2.0,
        mass: float = 0.02,
        global_damping: float = 0.02,
        fix_base: bool = True,
    ) -> None:
        self.geo = geometry
        self.pos = geometry.rest_positions.copy()
        self.vel = np.zeros_like(self.pos)

        self.k = stiffness
        self.c = spring_damping
        self.mass = mass
        self.global_damping = global_damping
        self.fix_base = fix_base

        self.springs = geometry.springs
        self.base_indices = geometry.base_indices

    def reset(self) -> None:
        self.pos = self.geo.rest_positions.copy()
        self.vel[...] = 0.0

    def _spring_forces(self) -> np.ndarray:
        forces = np.zeros_like(self.pos)
        for i, j, rest_len in self.springs:
            delta = self.pos[j] - self.pos[i]
            dist = np.linalg.norm(delta) + 1e-8
            dir_vec = delta / dist
            extension = dist - rest_len
            # Hooke + damping along spring direction.
            rel_vel = np.dot(self.vel[j] - self.vel[i], dir_vec)
            f = self.k * extension + self.c * rel_vel
            forces[i] += f * dir_vec
            forces[j] -= f * dir_vec
        return forces

    def _apply_fixed(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        if not self.fix_base:
            return
        positions[self.base_indices] = self.geo.rest_positions[self.base_indices]
        velocities[self.base_indices] = 0.0

    def step(self, dt: float, external: ExternalForce = None) -> None:
        forces = self._spring_forces()
        if external is not None:
            forces += external(self.pos)
        forces -= self.global_damping * self.vel

        acc = forces / self.mass
        self.vel += acc * dt
        self.pos += self.vel * dt
        self._apply_fixed(self.pos, self.vel)

    def simulate(
        self,
        steps: int,
        dt: float,
        external: ExternalForce = None,
    ) -> List[DuotactMesh]:
        frames: List[DuotactMesh] = []
        for _ in range(steps):
            self.step(dt, external)
            frames.append(self.geo.mesh_from_positions(self.pos.copy()))
        return frames

    # Lightweight, non-physical deformation: sum Gaussian displacements.
    # presses: iterable of dicts with keys center, direction, magnitude, radius.
    def static_deform(self, presses: List[dict]) -> DuotactMesh:
        deformed = self.geo.rest_positions.copy()
        for p in presses:
            center = np.asarray(p.get("center", (0, 0, 0)), dtype=float)
            direction = np.asarray(p.get("direction", (0, 0, -1)), dtype=float)
            magnitude = float(p.get("magnitude", 0.0))
            radius = float(p.get("radius", 0.01))

            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                dir_unit = np.array([0.0, 0.0, -1.0])
            else:
                dir_unit = direction / norm

            w = _gaussian_weights(deformed, center, radius)
            deformed += magnitude * w * dir_unit

        return self.geo.mesh_from_positions(deformed)

    # Convenience: Gaussian press with configurable direction (default: -Z).
    def make_press_force(
        self,
        center: np.ndarray,
        radius: float,
        magnitude: float,
        direction: np.ndarray | List[float] = (0.0, 0.0, -1.0),
    ) -> Callable[[np.ndarray], np.ndarray]:
        center = np.asarray(center, dtype=float).reshape(1, 3)
        radius_sq = max(radius, 1e-4) ** 2
        dir_vec = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-8:
            dir_unit = np.array([0.0, 0.0, -1.0])
        else:
            dir_unit = dir_vec / norm

        def force_field(positions: np.ndarray) -> np.ndarray:
            diff = positions - center
            dist_sq = np.sum(diff**2, axis=1)
            weights = np.exp(-dist_sq / (2.0 * radius_sq))[:, None]
            f = magnitude * dir_unit * weights
            return f

        return force_field
