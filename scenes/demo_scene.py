from __future__ import annotations

from renderer import Camera, Mesh, Scene, Triangle


def build_demo_scene() -> Scene:
    near_crossing = Mesh(
        name="near-crossing-quad",
        vertices=[
            (-1.8, -1.15, 0.06),
            (0.15, -1.15, 0.06),
            (0.15, 1.15, 1.55),
            (-1.8, 1.15, 1.55),
        ],
        triangles=[
            Triangle((0, 1, 2), (62, 170, 224)),
            Triangle((0, 2, 3), (62, 170, 224)),
        ],
    )

    back_panel = Mesh.quad(
        "back-panel",
        [
            (-0.65, -0.75, 2.0),
            (1.55, -0.75, 2.0),
            (1.55, 0.95, 2.0),
            (-0.65, 0.95, 2.0),
        ],
        (226, 190, 84),
    )
    front_panel = Mesh.quad(
        "front-panel",
        [
            (-0.15, -0.42, 1.25),
            (0.95, -0.42, 1.25),
            (0.95, 0.55, 1.25),
            (-0.15, 0.55, 1.25),
        ],
        (226, 92, 84),
    )

    partly_offscreen = Mesh(
        name="partly-offscreen-triangle",
        vertices=[
            (1.25, -1.4, 3.0),
            (4.0, -1.4, 3.0),
            (4.0, 1.4, 3.0),
        ],
        triangles=[Triangle((0, 1, 2), (112, 210, 132))],
    )

    front_camera = Camera(
        name="front",
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
        near=0.5,
        far=10.0,
    )
    moved_camera = Camera(
        name="moved",
        eye=(0.75, 0.12, -0.7),
        target=(0.0, 0.0, 1.0),
        near=0.5,
        far=10.0,
    )
    return Scene(
        name="triangle-pipeline-demo",
        meshes=[back_panel, partly_offscreen, near_crossing, front_panel],
        cameras=[front_camera, moved_camera],
        background=(12, 14, 20),
    )
