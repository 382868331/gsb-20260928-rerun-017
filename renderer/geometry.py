from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class Triangle:
    indices: Tuple[int, int, int]
    color: Tuple[int, int, int] = (220, 220, 220)
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if len(self.indices) != 3:
            raise ValueError("a triangle must contain exactly three vertex indices")
        if len(self.color) != 3 or any(not 0 <= channel <= 255 for channel in self.color):
            raise ValueError("color must be three channels in [0, 255]")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0, 1]")


@dataclass
class Mesh:
    name: str
    vertices: List[Vec3]
    triangles: List[Triangle]
    uvs: Optional[List[Vec2]] = None
    texture_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.uvs is not None and len(self.uvs) != len(self.vertices):
            raise ValueError(f"mesh {self.name!r} needs one UV per vertex")
        for triangle in self.triangles:
            for index in triangle.indices:
                if not 0 <= index < len(self.vertices):
                    raise ValueError(f"triangle in mesh {self.name!r} references missing vertex {index}")

    @classmethod
    def quad(
        cls,
        name: str,
        corners: Sequence[Vec3],
        color: Tuple[int, int, int],
        opacity: float = 1.0,
        texture_name: Optional[str] = None,
    ) -> "Mesh":
        if len(corners) != 4:
            raise ValueError("quad requires four corners")
        vertices = list(corners)
        uvs = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        return cls(
            name=name,
            vertices=vertices,
            triangles=[
                Triangle((0, 1, 2), color, opacity),
                Triangle((0, 2, 3), color, opacity),
            ],
            uvs=uvs,
            texture_name=texture_name,
        )

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "vertices": [list(vertex) for vertex in self.vertices],
            "triangles": [
                {
                    "indices": list(triangle.indices),
                    "color": list(triangle.color),
                    "opacity": triangle.opacity,
                }
                for triangle in self.triangles
            ],
        }
        if self.uvs is not None:
            data["uvs"] = [list(uv) for uv in self.uvs]
        if self.texture_name is not None:
            data["texture_name"] = self.texture_name
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Mesh":
        uvs = data.get("uvs")
        return cls(
            name=str(data["name"]),
            vertices=[tuple(vertex) for vertex in data["vertices"]],
            triangles=[
                Triangle(
                    tuple(item["indices"]),
                    tuple(item.get("color", (220, 220, 220))),
                    float(item.get("opacity", 1.0)),
                )
                for item in data["triangles"]
            ],
            uvs=[tuple(uv) for uv in uvs] if uvs is not None else None,
            texture_name=data.get("texture_name"),
        )


def merge_meshes(meshes: Iterable[Mesh]) -> Mesh:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    names = []
    for mesh in meshes:
        names.append(mesh.name)
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        triangles.extend(
            Triangle(
                (triangle.indices[0] + offset, triangle.indices[1] + offset, triangle.indices[2] + offset),
                triangle.color,
                triangle.opacity,
            )
            for triangle in mesh.triangles
        )
        if mesh.uvs is not None:
            raise ValueError("merge_meshes does not currently merge UV-mapped meshes")
    return Mesh("+".join(names), vertices, triangles)
