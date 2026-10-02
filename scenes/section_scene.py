from __future__ import annotations

import math
from typing import List, Tuple

from renderer import Camera, ClipGroup, ClipPlane, Mesh, Scene, Triangle


def _ring(cx, cy, cz, radius, sides, z):
    return [
        (cx + radius * math.cos(2 * math.pi * i / sides), cy + radius * math.sin(2 * math.pi * i / sides), z)
        for i in range(sides)
    ]


def hollow_tube(name, outer_radius=1.0, inner_radius=0.45, height=2.0, sides=16, diagonal_b=False):
    """A closed tube with a real through-hole (annular caps)."""
    z0, z1 = -height / 2, height / 2
    outer_bottom = _ring(0, 0, 0, outer_radius, sides, z0)
    inner_bottom = _ring(0, 0, 0, inner_radius, sides, z0)
    outer_top = _ring(0, 0, 0, outer_radius, sides, z1)
    inner_top = _ring(0, 0, 0, inner_radius, sides, z1)
    vertices: List[Tuple[float, float, float]] = outer_bottom + inner_bottom + outer_top + inner_top
    ob, ib, ot = 0, sides, 2 * sides
    it = 3 * sides
    triangles: List[Triangle] = []
    color = (120, 150, 230)

    def add(a, b, c):
        triangles.append(Triangle((a, b, c), color, opacity=1.0))

    for i in range(sides):
        j = (i + 1) % sides
        # Outer wall, inner wall.
        add(ob + i, ot + j, ob + j)
        add(ob + i, ot + i, ot + j)
        add(ib + i, ib + j, it + j)
        add(ib + i, it + j, it + i)
        # Annular top and bottom (quads connect outer ring to inner ring).
        quad = (ot + i, ot + j, it + j, it + i)
        top_quad = (ob + i, ob + j, ib + j, ib + i)
        if diagonal_b:
            for q in (quad, top_quad):
                add(q[0], q[2], q[1])
                add(q[0], q[3], q[2])
        else:
            for q in (quad, top_quad):
                add(q[0], q[1], q[2])
                add(q[0], q[2], q[3])
    return Mesh(name, vertices, triangles)


def solid_box(name, center, half=(0.6, 0.6, 0.9), color=(220, 160, 90)):
    cx, cy, cz = center
    hx, hy, hz = half
    vertices = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return Mesh(name, vertices, [Triangle(face, color) for face in faces])


def open_sheet(name, color=(90, 200, 150)):
    """A single open quad surface; a cut through it cannot be capped."""
    return Mesh.quad(
        name,
        [(-2.2, 0.9, -0.4), (2.4, 0.9, 0.2), (2.4, 1.8, 0.9), (-2.2, 1.8, 0.3)],
        color,
    )


def _groups():
    # Two jointly acting planes. Group B is a duplicated control group with
    # only one plane moved; palettes/opacity stay independent per group.
    planes_a = [
        ClipPlane.through("front-cut", (0.0, 0.0, 1.0), (0.0, 0.0, 0.1)),
        ClipPlane.through("side-cut", (-1.0, 0.0, 0.0), (0.4, 0.0, 0.0)),
    ]
    group_a = ClipGroup("pair-a", planes_a, palette=[(255, 120, 60), (90, 200, 255)])
    planes_b = [
        ClipPlane.through("front-cut", (0.0, 0.0, 1.0), (0.0, 0.0, 0.6)),
        ClipPlane.through("side-cut", (-1.0, 0.0, 0.0), (0.4, 0.0, 0.0)),
    ]
    group_b = ClipGroup("pair-b", planes_b, palette=[(255, 120, 60), (90, 200, 255)])
    return [group_a, group_b]


def build_section_scene(diagonal_b=False):
    tube = hollow_tube("tube", diagonal_b=diagonal_b)
    box = solid_box("box", (-1.8, -1.4, 0.2), half=(0.9, 0.7, 1.0))
    sheet = open_sheet("sheet")
    cameras = [
        Camera("front", eye=(0.0, -1.2, -5.5), target=(0.0, 0.0, 0.2), near=0.1, far=30.0),
        Camera("back", eye=(0.0, 1.2, 5.5), target=(0.0, 0.0, 0.2), near=0.1, far=30.0),
    ]
    return Scene(
        "multi-plane-sections",
        [tube, box, sheet],
        cameras,
        background=(14, 16, 22),
        clip_groups=_groups(),
    )
