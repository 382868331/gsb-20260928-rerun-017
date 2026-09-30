from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
UV = Tuple[float, float]


@dataclass(frozen=True)
class Triangle:
    indices: Tuple[int, int, int]
    color: Tuple[int, int, int] = (220, 220, 220)
    uvs: Optional[Tuple[UV, UV, UV]] = None
    texture_name: Optional[str] = None
    opacity: float = 1.0

    def __post_init__(self) -> None:
        if len(self.indices) != 3:
            raise ValueError("a triangle must contain exactly three vertex indices")
        if len(self.color) != 3 or any(not 0 <= channel <= 255 for channel in self.color):
            raise ValueError("color must be three channels in [0, 255]")
        if self.uvs is not None and len(self.uvs) != 3:
            raise ValueError("triangle uvs must contain exactly three UV pairs")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("opacity must be in [0, 1]")


@dataclass
class Mesh:
    name: str
    vertices: List[Vec3]
    triangles: List[Triangle]
    textures: Optional[List] = None

    def __post_init__(self) -> None:
        texture_names = set()
        if self.textures:
            from .texture import Texture

            for texture in self.textures:
                if not isinstance(texture, Texture):
                    raise ValueError("mesh textures must be Texture objects")
                if texture.name in texture_names:
                    raise ValueError(f"duplicate texture name {texture.name!r} in mesh {self.name!r}")
                texture_names.add(texture.name)
        for triangle in self.triangles:
            for index in triangle.indices:
                if not 0 <= index < len(self.vertices):
                    raise ValueError(f"triangle in mesh {self.name!r} references missing vertex {index}")
            if triangle.texture_name is not None and triangle.texture_name not in texture_names:
                raise ValueError(f"triangle in mesh {self.name!r} references missing texture {triangle.texture_name!r}")
            if (triangle.texture_name is None) != (triangle.uvs is None):
                raise ValueError("triangle uvs and texture_name must be provided together")

    @classmethod
    def quad(
        cls,
        name: str,
        corners: Sequence[Vec3],
        color: Tuple[int, int, int],
        uvs: Optional[Sequence[UV]] = None,
        texture_name: Optional[str] = None,
        opacity: float = 1.0,
        textures: Optional[List] = None,
    ) -> "Mesh":
        if len(corners) != 4:
            raise ValueError("quad requires four corners")
        vertices = list(corners)
        if uvs is None:
            first_uvs = second_uvs = None
        else:
            if len(uvs) != 4:
                raise ValueError("quad uvs require four UV pairs")
            first_uvs = (uvs[0], uvs[1], uvs[2])
            second_uvs = (uvs[0], uvs[2], uvs[3])
        return cls(
            name=name,
            vertices=vertices,
            triangles=[
                Triangle((0, 1, 2), color, first_uvs, texture_name, opacity),
                Triangle((0, 2, 3), color, second_uvs, texture_name, opacity),
            ],
            textures=textures,
        )

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "vertices": [list(vertex) for vertex in self.vertices],
            "triangles": [
                {
                    "indices": list(triangle.indices),
                    "color": list(triangle.color),
                    **(
                        {
                            "uvs": [list(uv) for uv in triangle.uvs],
                            "texture_name": triangle.texture_name,
                        }
                        if triangle.uvs is not None
                        else {}
                    ),
                    **({"opacity": triangle.opacity} if triangle.opacity != 1.0 else {}),
                }
                for triangle in self.triangles
            ],
        }
        if self.textures:
            data["textures"] = [texture.to_dict() for texture in self.textures]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Mesh":
        textures = None
        if data.get("textures"):
            from .texture import Texture

            textures = [Texture.from_dict(item) for item in data["textures"]]
        return cls(
            name=str(data["name"]),
            vertices=[tuple(vertex) for vertex in data["vertices"]],
            triangles=[
                Triangle(
                    tuple(item["indices"]),
                    tuple(item.get("color", (220, 220, 220))),
                    tuple(tuple(uv) for uv in item["uvs"]) if "uvs" in item else None,
                    item.get("texture_name"),
                    float(item.get("opacity", 1.0)),
                )
                for item in data["triangles"]
            ],
            textures=textures,
        )


def merge_meshes(meshes: Iterable[Mesh]) -> Mesh:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    textures = []
    names = []
    for mesh in meshes:
        names.append(mesh.name)
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        if mesh.textures:
            textures.extend(mesh.textures)
        triangles.extend(
            Triangle(
                (triangle.indices[0] + offset, triangle.indices[1] + offset, triangle.indices[2] + offset),
                triangle.color,
                triangle.uvs,
                triangle.texture_name,
                triangle.opacity,
            )
            for triangle in mesh.triangles
        )
    return Mesh("+".join(names), vertices, triangles, textures or None)
