from __future__ import annotations

from renderer import Camera, Mesh, Scene, Triangle, checkerboard, stripe_card


def _cameras():
    return [
        Camera(name="front", eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0), near=0.5, far=10.0),
        Camera(name="moved", eye=(0.9, 0.25, -0.8), target=(0.0, 0.0, 1.0), near=0.5, far=10.0),
    ]


# Same tilted, non-screen-parallel quad corners for every triangulation.
QUAD_CORNERS = [
    # Coplanar tilted plane (z = 2.5 - 0.2x + 0.15y): only then alternative
    # diagonal triangulations describe the exact same surface.
    (-1.35, -0.95, 2.6275),
    (1.25, -0.8, 2.13),
    (1.35, 0.9, 2.365),
    (-1.2, 1.05, 2.8975),
]
QUAD_UVS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def triangulation_scene(diagonal: str) -> Scene:
    """Same surface with the diagonal cut one way or the other."""
    texture = checkerboard("checker", cells=8, cell_pixels=8)
    if diagonal == "a":
        triangles = [
            Triangle((0, 1, 2), (255, 255, 255), tuple(QUAD_UVS[i] for i in (0, 1, 2)), "checker"),
            Triangle((0, 2, 3), (255, 255, 255), tuple(QUAD_UVS[i] for i in (0, 2, 3)), "checker"),
        ]
    else:
        triangles = [
            Triangle((0, 3, 2), (255, 255, 255), tuple(QUAD_UVS[i] for i in (0, 3, 2)), "checker"),
            Triangle((0, 2, 1), (255, 255, 255), tuple(QUAD_UVS[i] for i in (0, 2, 1)), "checker"),
        ]
    mesh = Mesh(
        name=f"tiled-quad-diagonal-{diagonal}",
        vertices=list(QUAD_CORNERS),
        triangles=triangles,
        textures=[texture],
    )
    return Scene(f"triangulation-{diagonal}", [mesh], _cameras(), background=(12, 14, 20))


def near_textured_scene() -> Scene:
    """Textured quad straddling the near plane; UVs survive clipping."""
    texture = stripe_card("stripes")
    corners = [
        (-1.7, -1.05, 0.08),
        (0.35, -1.05, 0.08),
        (0.45, 1.1, 1.7),
        (-1.6, 1.1, 1.7),
    ]
    mesh = Mesh.quad(
        "near-textured-quad",
        corners,
        (255, 255, 255),
        uvs=QUAD_UVS,
        texture_name="stripes",
        textures=[texture],
    )
    return Scene("near-textured", [mesh], _cameras(), background=(12, 14, 20))


def transparent_overlap_scene() -> Scene:
    """Opaque back panel plus two crossing translucent layers."""
    checker = checkerboard("back-checker", cells=6, cell_pixels=8,
                           color_a=(205, 205, 210), color_b=(70, 72, 90))
    back = Mesh.quad(
        "opaque-back",
        [(-1.5, -1.0, 3.2), (1.5, -1.0, 3.2), (1.5, 1.0, 3.2), (-1.5, 1.0, 3.2)],
        (255, 255, 255),
        uvs=QUAD_UVS,
        texture_name="back-checker",
        textures=[checker],
    )

    layer_a = Mesh.quad(
        "translucent-red",
        [(-1.25, -0.9, 2.3), (0.9, -0.55, 1.55), (1.05, 0.7, 1.55), (-1.1, 0.95, 2.3)],
        (255, 90, 80),
        opacity=0.55,
    )
    layer_b = Mesh.quad(
        "translucent-blue",
        [(-0.9, -0.95, 1.7), (1.25, -0.7, 2.45), (1.1, 0.85, 2.45), (-1.05, 0.6, 1.7)],
        (90, 150, 255),
        opacity=0.45,
    )
    return Scene(
        "transparent-overlap",
        [back, layer_a, layer_b],
        _cameras(),
        background=(12, 14, 20),
    )
