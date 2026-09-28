from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Vec3 = Tuple[float, float, float]


def subtract(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def normalize(v: Vec3) -> Vec3:
    length = math.sqrt(dot(v, v))
    if length <= 1e-12:
        raise ValueError("camera direction has zero length")
    return (v[0] / length, v[1] / length, v[2] / length)


@dataclass(frozen=True)
class Camera:
    name: str
    eye: Vec3 = (0.0, 0.0, 0.0)
    target: Vec3 = (0.0, 0.0, 1.0)
    up: Vec3 = (0.0, 1.0, 0.0)
    fov_y_degrees: float = 45.0
    near: float = 0.1
    far: float = 100.0

    def __post_init__(self) -> None:
        if not 0.0 < self.fov_y_degrees < 180.0:
            raise ValueError("fov_y_degrees must be in (0, 180)")
        if not 0.0 < self.near < self.far:
            raise ValueError("camera requires 0 < near < far")

    def frame(self) -> Tuple[Vec3, Vec3, Vec3]:
        forward = normalize(subtract(self.target, self.eye))
        right = normalize(cross(self.up, forward))
        camera_up = cross(forward, right)
        return right, camera_up, forward

    def view_position(self, world_position: Vec3) -> Tuple[float, float, float]:
        right, camera_up, forward = self.frame()
        relative = subtract(world_position, self.eye)
        return dot(relative, right), dot(relative, camera_up), dot(relative, forward)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "eye": list(self.eye),
            "target": list(self.target),
            "up": list(self.up),
            "fov_y_degrees": self.fov_y_degrees,
            "near": self.near,
            "far": self.far,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Camera":
        return cls(
            name=data["name"],
            eye=tuple(data.get("eye", (0.0, 0.0, -5.0))),
            target=tuple(data.get("target", (0.0, 0.0, 0.0))),
            up=tuple(data.get("up", (0.0, 1.0, 0.0))),
            fov_y_degrees=float(data.get("fov_y_degrees", 45.0)),
            near=float(data.get("near", 0.1)),
            far=float(data.get("far", 100.0)),
        )
