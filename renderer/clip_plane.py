from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

Vec3 = Tuple[float, float, float]
RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class ClipPlane:
    """A world-space clipping plane with its own section (cap) appearance.

    Geometry on the keep side (dot(p - point, normal) >= 0) is retained;
    the cut boundary is closed with an opaque section face drawn in
    section_color so it reads differently from any transparent surface.
    """

    name: str
    point: Vec3
    normal: Vec3
    section_color: RGB = (240, 210, 60)
    enabled: bool = True

    def __post_init__(self) -> None:
        if len(self.point) != 3 or len(self.normal) != 3:
            raise ValueError("clip plane point and normal must be 3-component")
        length = math.sqrt(sum(component * component for component in self.normal))
        if length <= 1e-12:
            raise ValueError("clip plane normal has zero length")
        object.__setattr__(self, "normal", tuple(component / length for component in self.normal))
        if len(self.section_color) != 3 or any(not 0 <= c <= 255 for c in self.section_color):
            raise ValueError("section_color must be three channels in [0, 255]")

    def signed_distance(self, position: Vec3) -> float:
        return sum(self.normal[k] * (position[k] - self.point[k]) for k in range(3))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "point": list(self.point),
            "normal": list(self.normal),
            "section_color": list(self.section_color),
            **({"enabled": False} if not self.enabled else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClipPlane":
        return cls(
            name=str(data["name"]),
            point=tuple(data["point"]),
            normal=tuple(data["normal"]),
            section_color=tuple(data.get("section_color", (240, 210, 60))),
            enabled=bool(data.get("enabled", True)),
        )
