from __future__ import annotations

from typing import List, Tuple

from renderer import Camera, Mesh, Scene, Texture, Triangle
from renderer.texture import checker_texture, stripe_texture


def _grid_mesh(
    name: str,
    corners,
    columns: int,
    rows: int,
    color,
    texture_name: str,
    diagonal_flip: bool = False,
) -> Mesh:
    """Regular tessellation of the same physical quad with per-vertex UVs."""
    x0, y0, z0 = corners[0]
    x1, _, _ = corners[1]
    _, y1, z1 = corners[2]
    vertices: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    for row in range(rows + 1):
        ty = row / rows
        for column in range(columns + 1):
            tx = column / columns
            vertices.append(
                (
                    x0 + (x1 - x0) * tx,
                    y0 + (y1 - y0) * ty,
                    z0 + (z1 - z0) * ty,
                )
            )
            uvs.append((tx, ty))
    triangles = []
    for row in range(rows):
        for column in range(columns):
            a = row * (columns + 1) + column
            b = a + 1
            c = a + (columns + 1)
            d = c + 1
            if (row + column) % 2 == (1 if diagonal_flip else 0):
                triangles.append(Triangle((a, b, d), color))
                triangles.append(Triangle((a, d, c), color))
            else:
                triangles.append(Triangle((a, b, c), color))
                triangles.append(Triangle((b, d, c), color))
    return Mesh(name, vertices, triangles, uvs=uvs, texture_name=texture_name)


def build_texture_demo_scene() -> Scene:
    checker = checker_texture("checker", cells=6, cell_pixels=8)
    stripes = stripe_texture("stripes", stripes=5, stripe_pixels=8)

    # The two panels occupy identical world geometry but use opposite diagonal
    # choices and a different cell count; correct perspective UV interpolation
    # must produce the same texel pattern along every shared internal edge.
    triangulation_a = _grid_mesh(
        "triangulation-2x2",
        [(-2.35, -1.0, 2.6), (-0.65, -1.0, 2.6), (-0.65, 1.0, 2.6), (-2.35, 1.0, 2.6)],
        columns=2,
        rows=2,
        color=(255, 255, 255),
        texture_name="checker",
        diagonal_flip=False,
    )
    triangulation_b = _grid_mesh(
        "triangulation-4x4",
        [(0.65, -1.0, 2.6), (2.35, -1.0, 2.6), (2.35, 1.0, 2.6), (0.65, 1.0, 2.6)],
        columns=4,
        rows=4,
        color=(255, 255, 255),
        texture_name="checker",
        diagonal_flip=True,
    )

    # A single textured quad tilted so both triangles straddle the near plane.
    near_textured = Mesh.quad(
        "near-textured-quad",
        [
            (-1.15, -0.95, 0.15),
            (0.55, -0.95, 1.35),
            (0.55, 0.35, 1.35),
            (-1.15, 0.35, 0.15),
        ],
        (255, 255, 255),
        texture_name="checker",
    )

    # Opaque base behind two overlapping translucent sheets.
    opaque_base = Mesh.quad(
        "opaque-base",
        [(-0.7, -0.95, 2.2), (1.7, -0.95, 2.2), (1.7, 0.9, 2.2), (-0.7, 0.9, 2.2)],
        (238, 224, 188),
    )
    back_sheet = Mesh.quad(
        "translucent-back",
        [(-0.45, -0.72, 1.75), (1.45, -0.72, 1.75), (1.45, 0.65, 1.75), (-0.45, 0.65, 1.75)],
        (64, 132, 224),
        opacity=0.45,
    )
    front_sheet = Mesh.quad(
        "translucent-front",
        [(0.05, -0.5, 1.35), (1.95, -0.5, 1.35), (1.95, 0.82, 1.35), (0.05, 0.82, 1.35)],
        (224, 86, 104),
        opacity=0.4,
        texture_name=None,
    )
    textured_sheet = Mesh.quad(
        "translucent-striped",
        [(-0.9, -0.42, 1.55), (0.95, -0.42, 1.55), (0.95, 0.48, 1.55), (-0.9, 0.48, 1.55)],
        (255, 255, 255),
        opacity=0.55,
        texture_name="stripes",
    )

    front = Camera(name="front", eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), near=0.5, far=12.0)
    moved = Camera(
        name="moved",
        eye=(0.8, 0.25, -0.6),
        target=(0.0, -0.05, 1.4),
        near=0.5,
        far=12.0,
    )
    return Scene(
        name="textured-translucent-demo",
        meshes=[
            triangulation_a,
            triangulation_b,
            opaque_base,
            back_sheet,
            textured_sheet,
            near_textured,
            front_sheet,
        ],
        cameras=[front, moved],
        textures=[checker, stripes],
        background=(14, 16, 24),
    )
