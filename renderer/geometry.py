from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class Triangle:
    indices: Tuple[int, int, int]
    color: Tuple[int, int, int] = (220, 220, 220)

    def __post_init__(self) -> None:
        if len(self.indices) != 3:
            raise ValueError("a triangle must contain exactly three vertex indices")
        if len(self.color) != 3 or any(not 0 <= channel <= 255 for channel in self.color):
            raise ValueError("color must be three channels in [0, 255]")


@dataclass
class Mesh:
    name: str
    vertices: List[Vec3]
    triangles: List[Triangle]

    def __post_init__(self) -> None:
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
    ) -> "Mesh":
        if len(corners) != 4:
            raise ValueError("quad requires four corners")
        vertices = list(corners)
        return cls(
            name=name,
            vertices=vertices,
            triangles=[
                Triangle((0, 1, 2), color),
                Triangle((0, 2, 3), color),
            ],
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "vertices": [list(vertex) for vertex in self.vertices],
            "triangles": [
                {
                    "indices": list(triangle.indices),
                    "color": list(triangle.color),
                }
                for triangle in self.triangles
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Mesh":
        return cls(
            name=str(data["name"]),
            vertices=[tuple(vertex) for vertex in data["vertices"]],
            triangles=[
                Triangle(
                    tuple(item["indices"]),
                    tuple(item.get("color", (220, 220, 220))),
                )
                for item in data["triangles"]
            ],
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
            )
            for triangle in mesh.triangles
        )
    return Mesh("+".join(names), vertices, triangles)
