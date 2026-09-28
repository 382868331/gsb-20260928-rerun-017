from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .camera import Camera
from .math3d import vec3

Color = tuple[int, int, int]


def color(value: Sequence[float] | None = None) -> Color:
    if value is None:
        return 255, 255, 255
    if len(value) != 3:
        raise ValueError("an RGB color requires exactly three components")
    channels = tuple(int(round(float(component))) for component in value)
    if any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError("color channels must be in the 0..255 range")
    return channels  # type: ignore[return-value]


@dataclass(frozen=True)
class Mesh:
    name: str
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    color: Color = (255, 255, 255)

    def __post_init__(self) -> None:
        for face in self.faces:
            if len(face) != 3:
                raise ValueError(f"mesh {self.name!r} contains a non-triangle face")
            for index in face:
                if not 0 <= index < len(self.vertices):
                    raise ValueError(f"mesh {self.name!r} contains an invalid vertex index")

    @classmethod
    def from_dict(cls, data: dict) -> "Mesh":
        return cls(
            name=str(data["name"]),
            vertices=tuple(vec3(vertex) for vertex in data["vertices"]),
            faces=tuple(tuple(int(index) for index in face) for face in data["faces"]),  # type: ignore[arg-type]
            color=color(data.get("color", (255, 255, 255))),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vertices": [list(vertex) for vertex in self.vertices],
            "faces": [list(face) for face in self.faces],
            "color": list(self.color),
        }


@dataclass
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    cameras: dict[str, Camera] = field(default_factory=dict)
    default_camera: str = "main"

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        cameras = {
            name: Camera.from_dict(camera_data)
            for name, camera_data in data.get("cameras", {}).items()
        }
        default_camera = data.get("default_camera", "main")
        if cameras and default_camera not in cameras:
            raise ValueError("default_camera does not name a camera in the scene")
        return cls(
            meshes=[Mesh.from_dict(mesh) for mesh in data.get("meshes", [])],
            cameras=cameras,
            default_camera=default_camera,
        )

    def to_dict(self) -> dict:
        return {
            "default_camera": self.default_camera,
            "cameras": {name: item.to_dict() for name, item in self.cameras.items()},
            "meshes": [mesh.to_dict() for mesh in self.meshes],
        }

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def save(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)
            stream.write("\n")
