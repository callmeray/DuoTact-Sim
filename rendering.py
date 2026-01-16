from typing import List, Optional

import numpy as np
import plotly.graph_objects as go

from geometry import DuotactMesh


class DuotactVisualizer:
    def __init__(self) -> None:
        pass

    def _mesh_trace(self, mesh: DuotactMesh, color: str = "royalblue", opacity: float = 0.5) -> go.Mesh3d:
        v = mesh.vertices
        tri = mesh.triangles
        return go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=tri[:, 0],
            j=tri[:, 1],
            k=tri[:, 2],
            color=color,
            opacity=opacity,
            flatshading=True,
        )

    def _ridge_trace(self, mesh: DuotactMesh, color: str = "crimson") -> go.Scatter3d:
        v = mesh.vertices
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for loop in mesh.ridge_loops:
            xs.extend(v[loop, 0].tolist())
            ys.extend(v[loop, 1].tolist())
            zs.extend(v[loop, 2].tolist())
            xs.append(None)
            ys.append(None)
            zs.append(None)
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=6),
            name="ridge",
        )

    def _frame_trace(self, mesh: DuotactMesh, color: str = "#444", width: int = 3) -> go.Scatter3d:
        v = mesh.vertices
        segments = []
        for a, b in mesh.frame_edges:
            segments.append((v[a], v[b]))
        if not segments:
            return go.Scatter3d()
        xs = []
        ys = []
        zs = []
        for p0, p1 in segments:
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            name="frame",
            showlegend=False,
        )

    def _segments_trace(
        self,
        segments: List[tuple],
        color: str = "orange",
        width: int = 3,
        name: str = "segments",
    ) -> go.Scatter3d:
        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        for p0, p1 in segments:
            xs.extend([p0[0], p1[0], None])
            ys.extend([p0[1], p1[1], None])
            zs.extend([p0[2], p1[2], None])
        return go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(color=color, width=width),
            name=name,
            showlegend=False,
        )

    def _point_trace(self, point: np.ndarray, color: str = "orange", size: int = 5, name: str = "cam") -> go.Scatter3d:
        return go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="markers",
            marker=dict(color=color, size=size, symbol="diamond"),
            name=name,
            showlegend=False,
        )

    def _axes_trace(
        self,
        origin: np.ndarray,
        axes: np.ndarray,
        scale: float = 0.02,
        colors: tuple = ("red", "green", "blue"),
        name: str = "axes",
    ) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []
        for i in range(3):
            axis_vec = axes[:, i]
            p1 = origin + axis_vec * scale
            traces.append(
                go.Scatter3d(
                    x=[origin[0], p1[0]],
                    y=[origin[1], p1[1]],
                    z=[origin[2], p1[2]],
                    mode="lines",
                    line=dict(color=colors[i], width=4),
                    name=f"{name}_{i}",
                    showlegend=False,
                )
            )
        return traces

    def _markers_trace(self, points: np.ndarray, color: str = "orange", size: int = 5, name: str = "markers") -> go.Scatter3d:
        return go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(color=color, size=size, symbol="circle"),
            name=name,
            showlegend=False,
        )

    def _force_traces(self, vectors: List[dict]) -> List[go.Scatter3d]:
        traces: List[go.Scatter3d] = []
        for vec in vectors:
            center = np.asarray(vec.get("center", (0, 0, 0)), dtype=float)
            direction = np.asarray(vec.get("direction", (0, 0, -1)), dtype=float)
            scale = float(vec.get("scale", 0.02))
            color = vec.get("color", "orange")
            name = vec.get("name", "force")
            p1 = center + direction * scale
            traces.append(
                go.Scatter3d(
                    x=[center[0], p1[0]],
                    y=[center[1], p1[1]],
                    z=[center[2], p1[2]],
                    mode="lines+markers",
                    line=dict(color=color, width=6),
                    marker=dict(size=4, color=color),
                    name=name,
                )
            )
        return traces

    def show_static(
        self,
        mesh: DuotactMesh,
        title: str = "Duotact",
        force_vectors: Optional[List[dict]] = None,
        frustum_segments: Optional[List[tuple]] = None,
        camera_point: Optional[np.ndarray] = None,
        markers: Optional[np.ndarray] = None,
        camera_axes: Optional[np.ndarray] = None,
        camera_axes_scale: float = 0.02,
    ) -> None:
        data = [self._mesh_trace(mesh), self._frame_trace(mesh), self._ridge_trace(mesh)]
        if force_vectors:
            data.extend(self._force_traces(force_vectors))
        if frustum_segments:
            data.append(self._segments_trace(frustum_segments, color="#ff7f0e", width=2, name="frustum"))
        if camera_point is not None:
            data.append(self._point_trace(camera_point, color="#d62728", size=6, name="camera"))
        if markers is not None and markers.size > 0:
            data.append(self._markers_trace(markers, color="orange", size=5, name="markers"))
        if camera_point is not None and camera_axes is not None:
            data.extend(self._axes_trace(camera_point, camera_axes, scale=camera_axes_scale, name="cam_axes"))
        fig = go.Figure(data=data)
        fig.update_layout(
            title=title,
            scene_aspectmode="data",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        )
        fig.show()

    def animate(self, frames: List[DuotactMesh], title: str = "Duotact animation", step_ms: int = 40, force_vectors: Optional[List[dict]] = None) -> None:
        if not frames:
            return
        first = frames[0]
        mesh_trace = self._mesh_trace(first)
        frame_trace = self._frame_trace(first)
        ridge_trace = self._ridge_trace(first)
        force_traces = self._force_traces(force_vectors or [])

        plotly_frames = []
        for idx, f in enumerate(frames):
            xs_r: List[float] = []
            ys_r: List[float] = []
            zs_r: List[float] = []
            for loop in f.ridge_loops:
                xs_r.extend(f.vertices[loop, 0].tolist())
                ys_r.extend(f.vertices[loop, 1].tolist())
                zs_r.extend(f.vertices[loop, 2].tolist())
                xs_r.append(None)
                ys_r.append(None)
                zs_r.append(None)
            plotly_frames.append(
                go.Frame(
                    data=[
                        go.Mesh3d(
                            x=f.vertices[:, 0],
                            y=f.vertices[:, 1],
                            z=f.vertices[:, 2],
                            i=f.triangles[:, 0],
                            j=f.triangles[:, 1],
                            k=f.triangles[:, 2],
                            color=mesh_trace.color,
                            opacity=mesh_trace.opacity,
                            flatshading=True,
                            name="surface",
                        ),
                        go.Scatter3d(
                            x=frame_trace.x,
                            y=frame_trace.y,
                            z=frame_trace.z,
                            mode="lines",
                            line=frame_trace.line,
                            name="frame",
                            showlegend=False,
                        ),
                        go.Scatter3d(
                            x=xs_r,
                            y=ys_r,
                            z=zs_r,
                            mode="lines",
                            line=dict(color="crimson", width=6),
                            name="ridge",
                        ),
                        *force_traces,
                    ],
                    name=str(idx),
                )
            )

        fig = go.Figure(
            data=[mesh_trace, frame_trace, ridge_trace, *force_traces],
            frames=plotly_frames,
            layout=go.Layout(
                title=title,
                scene_aspectmode="data",
                updatemenus=[
                    dict(
                        type="buttons",
                        buttons=[
                            dict(label="Play", method="animate", args=[[f.name for f in plotly_frames], {"frame": {"duration": step_ms, "redraw": True}, "fromcurrent": True}]),
                            dict(label="Pause", method="animate", args=[[None], {"mode": "immediate"}]),
                        ],
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        currentvalue={"prefix": "Frame: "},
                        steps=[dict(method="animate", args=[[f.name], {"mode": "immediate", "frame": {"duration": step_ms, "redraw": True}}], label=f.name) for f in plotly_frames],
                    )
                ],
            ),
        )
        fig.show()
