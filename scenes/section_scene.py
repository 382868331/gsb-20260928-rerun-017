"""Joint clipping-plane demo: caps, holes, open boundaries and control groups."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from renderer import Camera, ClipGroup, ClipPlane, Mesh, Scene, Triangle

Vec3 = Tuple[float, float, float]


def _box_mesh(
    name: str,
    low: Vec3,
    high: Vec3,
    color: Tuple[int, int, int],
    opacity: float = 1.0,
) -> Mesh:
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return Mesh(
        name,
        vertices,
        [Triangle(face, color, opacity=opacity) for face in faces],
    )


def _quad(points: Sequence[Vec3], color: Tuple[int, int, int], opacity: float = 1.0) -> Mesh:
    return Mesh(
        "open-sheet",
        list(points),
        [
            Triangle((0, 1, 2), color, opacity=opacity),
            Triangle((0, 2, 3), color, opacity=opacity),
        ],
    )


def _annulus_walls(
    name: str,
    outer: float,
    inner: float,
    z_low: float,
    z_high: float,
    color: Tuple[int, int, int],
) -> Mesh:
    """A square tube (outer box minus a square hole), built only from quads."""
    o, i = outer, inner
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []

    def add_quad(points: Sequence[Vec3], reversed_: bool = False) -> None:
        offset = len(vertices)
        vertices.extend(points)
        if reversed_:
            indices = ((0, 2, 1), (0, 3, 2))
        else:
            indices = ((0, 1, 2), (0, 2, 3))
        for face in indices:
            triangles.append(Triangle(tuple(offset + value for value in face), color))

    for z, reversed_ in ((z_low, False), (z_high, True)):
        add_quad([(-o, -o, z), (o, -o, z), (o, o, z), (-o, o, z)], reversed_)
        add_quad([(-i, -i, z), (-i, i, z), (i, i, z), (i, -i, z)], reversed_)

    # Outer walls.
    add_quad([(o, -o, z_low), (o, -o, z_high), (o, o, z_high), (o, o, z_low)])
    add_quad([(-o, o, z_low), (-o, o, z_high), (-o, -o, z_high), (-o, -o, z_low)])
    add_quad([(-o, -o, z_low), (-o, -o, z_high), (o, -o, z_high), (o, -o, z_low)])
    add_quad([(o, o, z_low), (o, o, z_high), (-o, o, z_high), (-o, o, z_low)])
    # Inner walls face back into the tunnel.
    add_quad([(i, -i, z_low), (i, -i, z_high), (i, i, z_high), (i, i, z_low)], True)
    add_quad([(-i, i, z_low), (-i, i, z_high), (-i, -i, z_high), (-i, -i, z_low)], True)
    add_quad([(-i, -i, z_low), (i, -i, z_low), (i, -i, z_high), (-i, -i, z_high)], True)
    add_quad([(i, i, z_low), (-i, i, z_low), (-i, i, z_high), (i, i, z_high)], True)
    return Mesh(name, vertices, triangles)


def section_scene() -> Scene:
    """Two transparent solids, a square-tube body and an open reference sheet."""
    blue_body = _box_mesh("blue-solid", (-1.15, -1.0, -1.2), (0.15, 0.9, 1.1), (90, 140, 230), 0.45)
    amber_body = _box_mesh("amber-solid", (-0.15, -0.95, -1.0), (1.15, 0.95, 1.2), (235, 170, 70), 0.4)
    tube = _annulus_walls("square-tube", 0.95, 0.38, -1.3, 1.3, (120, 210, 150))
    sheet = _quad(
        [(-1.4, 1.05, -0.9), (1.4, 1.05, -0.9), (1.4, 1.05, 0.9), (-1.4, 1.05, 0.9)],
        (210, 210, 215),
        0.35,
    )

    paired = ClipGroup(
        "paired",
        (
            ClipPlane("cut-x", (-1.0, 0.0, 0.0), 0.0, (255, 90, 90)),
            ClipPlane("cut-z", (0.0, 0.0, -1.0), 0.15, (120, 200, 255)),
        ),
    )
    control = paired.copied("control-moved-x").with_plane("cut-x", offset=0.45)

    cameras = [
        Camera("front", eye=(2.6, 1.8, -3.2), target=(0.0, -0.1, 0.0), near=0.2, far=20.0),
        Camera("behind", eye=(-2.6, -1.6, 3.2), target=(0.0, -0.1, 0.0), near=0.2, far=20.0),
    ]
    return Scene(
        "joint-section-demo",
        [blue_body, amber_body, tube, sheet],
        cameras,
        background=(10, 12, 18),
        clip_groups=[paired, control],
    )
