from __future__ import annotations

from typing import Sequence, Tuple

from renderer import Camera, ClipPlane, Mesh, Scene, Triangle

Vec3 = Tuple[float, float, float]
RGB = Tuple[int, int, int]


def _quad_triangles(a: int, b: int, c: int, d: int, color: RGB, opacity: float, diagonal: int):
    if diagonal == 0:
        return [
            Triangle((a, b, c), color, opacity=opacity),
            Triangle((a, c, d), color, opacity=opacity),
        ]
    return [
        Triangle((a, b, d), color, opacity=opacity),
        Triangle((b, c, d), color, opacity=opacity),
    ]


def box_mesh(
    name: str,
    center: Vec3,
    size: Vec3,
    color: RGB,
    opacity: float = 1.0,
    diagonal: int = 0,
) -> Mesh:
    cx, cy, cz = center
    hx, hy, hz = (extent * 0.5 for extent in size)
    x0, x1 = cx - hx, cx + hx
    y0, y1 = cy - hy, cy + hy
    z0, z1 = cz - hz, cz + hz
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    quads = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (2, 6, 7, 3),
        (1, 5, 6, 2),
        (0, 3, 7, 4),
    ]
    triangles = []
    for quad in quads:
        triangles.extend(_quad_triangles(*quad, color, opacity, diagonal))
    return Mesh(name, vertices, triangles)


def square_tube_mesh(
    name: str,
    center: Vec3,
    half_length: float,
    outer_half: float,
    inner_half: float,
    color: RGB,
    opacity: float = 1.0,
    diagonal: int = 0,
) -> Mesh:
    """Tube along the z axis with open ends; a cross-section is an annulus."""
    cx, cy, cz = center
    z0, z1 = cz - half_length, cz + half_length
    o, i = outer_half, inner_half
    vertices = [
        (cx - o, cy - o, z0),
        (cx + o, cy - o, z0),
        (cx + o, cy + o, z0),
        (cx - o, cy + o, z0),
        (cx - o, cy - o, z1),
        (cx + o, cy - o, z1),
        (cx + o, cy + o, z1),
        (cx - o, cy + o, z1),
        (cx - i, cy - i, z0),
        (cx + i, cy - i, z0),
        (cx + i, cy + i, z0),
        (cx - i, cy + i, z0),
        (cx - i, cy - i, z1),
        (cx + i, cy - i, z1),
        (cx + i, cy + i, z1),
        (cx - i, cy + i, z1),
    ]
    quads = [
        (0, 1, 5, 4),  # outer walls
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
        (8, 12, 13, 9),  # inner walls
        (9, 13, 14, 10),
        (10, 14, 15, 11),
        (11, 15, 12, 8),
    ]
    triangles = []
    for quad in quads:
        triangles.extend(_quad_triangles(*quad, color, opacity, diagonal))
    return Mesh(name, vertices, triangles)


def build_clip_scene(diagonal: int = 0) -> Scene:
    box_a = box_mesh("box-a", (-0.35, 0.05, 2.6), (1.2, 1.2, 1.2), (90, 160, 230), 0.45, diagonal)
    box_b = box_mesh("box-b", (0.4, 0.2, 2.6), (1.0, 1.0, 1.0), (120, 220, 140), 0.45, diagonal)
    tube = square_tube_mesh("tube", (0.0, -0.75, 2.4), 0.7, 0.32, 0.16, (230, 170, 90), 0.6, diagonal)
    panel = Mesh.quad(
        "open-panel",
        [(-1.0, -0.9, 3.4), (1.0, -0.9, 3.4), (1.0, 0.9, 3.4), (-1.0, 0.9, 3.4)],
        (200, 120, 220),
        opacity=0.5,
    )
    planes = [
        ClipPlane("cut-x", (0.05, 0.0, 0.0), (1.0, 0.0, 0.0), (250, 205, 60)),
        ClipPlane("cut-y", (0.0, -0.2, 0.0), (0.0, 1.0, 0.0), (90, 220, 235)),
        ClipPlane("cut-z", (0.0, 0.0, 2.4), (0.0, 0.0, 1.0), (235, 130, 230)),
    ]
    cameras = [
        Camera("front", eye=(0.0, 0.15, -1.4), target=(0.0, 0.0, 2.6), near=0.1, far=20.0),
        Camera("west", eye=(-3.0, 0.7, 2.1), target=(0.0, -0.1, 2.6), near=0.1, far=20.0),
        Camera("east", eye=(3.0, 0.7, 2.1), target=(0.0, -0.1, 2.6), near=0.1, far=20.0),
    ]
    return Scene(
        "clip-section-demo",
        [box_a, box_b, tube, panel],
        cameras,
        background=(12, 14, 20),
        clip_planes=planes,
    )
