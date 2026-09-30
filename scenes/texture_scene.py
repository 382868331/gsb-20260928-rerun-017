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


# Flat colours make a single source-over blend analytically exact: any pixel
# covered twice along a shared edge produces a visibly different third colour,
# while a gap exposes the backdrop, so seams cannot hide in the imagery.
SEAM_BACKDROP = (10, 20, 30)
SEAM_SURFACE = (60, 200, 120)
SEAM_OPACITY = 0.5
SEAM_BLENDED = tuple(
    int(round(back * (1.0 - SEAM_OPACITY) + surface * SEAM_OPACITY))
    for back, surface in zip(SEAM_BACKDROP, SEAM_SURFACE)
)

# A tilted plane that crosses the near plane and every screen edge depending
# on the camera. The two far corners sit at different depths so the internal
# diagonal is slanted in screen space instead of axis aligned.
SEAM_CORNERS = [
    (-3.0, -0.12, 0.28),
    (3.2, -0.05, 0.34),
    (3.4, 1.9, 3.0),
    (-3.2, 1.7, 2.8),
]


def transparent_seam_scene(diagonal: str = "a", reverse: bool = False) -> Scene:
    """One translucent plane over a full-frame flat opaque backdrop.

    `diagonal` selects the triangulation of the shared internal edge and
    `reverse` submits the triangles in the opposite order; the rendered
    coverage and colour must be identical in every case.
    """
    backdrop = Mesh.quad(
        "seam-backdrop",
        [(-8.0, -8.0, 4.0), (8.0, -8.0, 4.0), (8.0, 8.0, 4.0), (-8.0, 8.0, 4.0)],
        SEAM_BACKDROP,
    )
    if diagonal == "a":
        triangles = [
            Triangle((0, 1, 2), SEAM_SURFACE, opacity=SEAM_OPACITY),
            Triangle((0, 2, 3), SEAM_SURFACE, opacity=SEAM_OPACITY),
        ]
    else:
        triangles = [
            Triangle((0, 3, 2), SEAM_SURFACE, opacity=SEAM_OPACITY),
            Triangle((0, 2, 1), SEAM_SURFACE, opacity=SEAM_OPACITY),
        ]
    if reverse:
        triangles = list(reversed(triangles))
    surface = Mesh("seam-surface", list(SEAM_CORNERS), triangles)
    return Scene(f"transparent-seam-{diagonal}", [backdrop, surface], _cameras(), background=(0, 0, 0))


def transparent_layers_scene(diagonal: str = "a", reverse_layers: bool = False) -> Scene:
    """Two stacked translucent planes (one always nearer) over a backdrop.

    Both planes share their own four vertices between their triangles and
    spill past the screen edges. Layer order is reversed by flag; fragment
    depth sorting must make the result independent of submission order.
    """
    backdrop = Mesh.quad(
        "layers-backdrop",
        [(-8.0, -8.0, 4.5), (8.0, -8.0, 4.5), (8.0, 8.0, 4.5), (-8.0, 8.0, 4.5)],
        (24, 24, 40),
    )

    def plane(name: str, corners, color, opacity: float) -> Mesh:
        if diagonal == "a":
            triangles = [
                Triangle((0, 1, 2), color, opacity=opacity),
                Triangle((0, 2, 3), color, opacity=opacity),
            ]
        else:
            triangles = [
                Triangle((0, 3, 2), color, opacity=opacity),
                Triangle((0, 2, 1), color, opacity=opacity),
            ]
        return Mesh(name, list(corners), triangles)

    far_layer = plane(
        "layers-far",
        [(-2.6, -1.7, 3.0), (2.4, -1.5, 2.7), (2.6, 1.6, 3.1), (-2.4, 1.8, 3.3)],
        (220, 70, 70),
        0.45,
    )
    near_layer = plane(
        "layers-near",
        [(-2.2, -1.8, 1.9), (2.8, -1.6, 1.7), (2.9, 1.9, 2.0), (-2.3, 1.7, 2.1)],
        (70, 200, 120),
        0.4,
    )
    layers = [backdrop, far_layer, near_layer]
    if reverse_layers:
        layers = [backdrop, near_layer, far_layer]
    return Scene(f"transparent-layers-{diagonal}", layers, _cameras(), background=(0, 0, 0))
