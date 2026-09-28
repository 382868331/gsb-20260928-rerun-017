from __future__ import annotations

from dataclasses import dataclass
from math import tan
from typing import Sequence

from .math3d import Mat4, cross, dot, normalize, sub, vec3


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    fovy_degrees: float = 60.0
    near: float = 0.1
    far: float = 100.0

    def __post_init__(self) -> None:
        if self.near <= 0.0:
            raise ValueError("camera near plane must be positive")
        if self.far <= self.near:
            raise ValueError("camera far plane must be greater than near")
        if not 0.0 < self.fovy_degrees < 180.0:
            raise ValueError("fovy must be between 0 and 180 degrees")

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            position=vec3(data["position"]),
            target=vec3(data["target"]),
            up=vec3(data.get("up", (0.0, 1.0, 0.0))),
            fovy_degrees=float(data.get("fovy_degrees", 60.0)),
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 100.0)),
        )

    def to_dict(self) -> dict:
        return {
            "position": list(self.position),
            "target": list(self.target),
            "up": list(self.up),
            "fovy_degrees": self.fovy_degrees,
            "near": self.near,
            "far": self.far,
        }


def look_at(position: Sequence[float], target: Sequence[float], up: Sequence[float]) -> Mat4:
    eye = vec3(position)
    forward = normalize(sub(vec3(target), eye))
    right = normalize(cross(forward, normalize(vec3(up))))
    camera_up = cross(right, forward)

    return (
        (right[0], right[1], right[2], -dot(right, eye)),
        (camera_up[0], camera_up[1], camera_up[2], -dot(camera_up, eye)),
        (-forward[0], -forward[1], -forward[2], dot(forward, eye)),
        (0.0, 0.0, 0.0, 1.0),
    )


def perspective_matrix(fovy_degrees: float, aspect: float, near: float, far: float) -> Mat4:
    y_scale = 1.0 / tan(fovy_degrees * 3.141592653589793 / 360.0)
    x_scale = y_scale / aspect
    depth = far + near
    depth_range = far - near
    return (
        (x_scale, 0.0, 0.0, 0.0),
        (0.0, y_scale, 0.0, 0.0),
        (0.0, 0.0, -depth / depth_range, -2.0 * far * near / depth_range),
        (0.0, 0.0, -1.0, 0.0),
    )


def view_projection_matrix(camera: Camera, aspect: float) -> Mat4:
    view = look_at(camera.position, camera.target, camera.up)
    projection = perspective_matrix(
        camera.fovy_degrees, aspect, camera.near, camera.far
    )
    from .math3d import mat4_mul

    return mat4_mul(projection, view)
