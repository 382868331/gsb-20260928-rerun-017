from __future__ import annotations

import json
from pathlib import Path

from mini_render.camera import view_projection_matrix
from mini_render.geometry import Scene
from mini_render.image import write_png
from mini_render.math3d import transform_point
from mini_render.rasterizer import (
    ClipVertex,
    RenderSettings,
    clip_triangle,
    render,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE_PATH = ROOT / "scenes" / "validation.json"


def pixel(image: bytes, width: int, x: int, y: int) -> tuple[int, int, int]:
    offset = (y * width + x) * 3
    return image[offset], image[offset + 1], image[offset + 2]


def count_color(image: bytes, color: tuple[int, int, int]) -> int:
    return sum(
        image[offset:offset + 3] == bytes(color)
        for offset in range(0, len(image), 3)
    )


def test_scene_roundtrip_and_saved_camera(tmp_path: Path) -> None:
    scene = Scene.load(SCENE_PATH)
    saved = tmp_path / "scene.json"
    scene.save(saved)

    assert json.loads(saved.read_text(encoding="utf-8")) == json.loads(
        SCENE_PATH.read_text(encoding="utf-8")
    )
    assert set(scene.cameras) == {"main", "side"}


def test_depth_occlusion_and_partial_outside_coverage() -> None:
    scene = Scene.load(SCENE_PATH)
    settings = RenderSettings(width=160, height=120)
    image = render(scene, settings)

    assert count_color(image, settings.background) < 160 * 120
    assert count_color(image, (45, 205, 210)) > 100
    assert count_color(image, (225, 70, 65)) > 50
    assert count_color(image, (235, 205, 70)) > 5
    assert pixel(image, 160, 80, 60) == (225, 70, 65)


def test_camera_change_moves_coverage() -> None:
    scene = Scene.load(SCENE_PATH)
    main_image = render(scene, RenderSettings(width=160, height=120, camera="main"))
    side_image = render(scene, RenderSettings(width=160, height=120, camera="side"))

    assert main_image != side_image
    assert count_color(side_image, (45, 205, 210)) > 0


def test_clipped_shared_edge_leaves_no_background_gap() -> None:
    scene = Scene.load(SCENE_PATH)
    shared_edge_scene = Scene(
        meshes=[scene.meshes[0]],
        cameras=scene.cameras,
        default_camera="main",
    )
    image = render(shared_edge_scene, RenderSettings(width=160, height=120))

    assert count_color(image, (18, 20, 26)) == 0


def test_triangle_crossing_near_is_clipped_not_discarded() -> None:
    scene = Scene.load(SCENE_PATH)
    camera = scene.cameras["main"]
    matrix = view_projection_matrix(camera, 4.0 / 3.0)
    mesh = scene.meshes[0]

    clipped = clip_triangle(
        tuple(
            ClipVertex(transform_point(matrix, mesh.vertices[index]))
            for index in mesh.faces[0]
        )
    )

    assert len(clipped) >= 1
    for triangle in clipped:
        for vertex in triangle:
            x, y, z, w = vertex.position
            assert w > 0.0
            assert -w - 1e-9 <= x <= w + 1e-9
            assert -w - 1e-9 <= y <= w + 1e-9
            assert -w - 1e-9 <= z <= w + 1e-9


def test_shared_edge_near_clip_intersections_match() -> None:
    behind = ClipVertex((0.0, 0.0, -2.0, 2.0))
    beyond = ClipVertex((0.0, 0.0, 2.0, 0.2))
    left = ClipVertex((-1.0, 0.0, -2.0, 2.0))
    right = ClipVertex((1.0, 0.0, -2.0, 2.0))

    first = clip_triangle((behind, beyond, left))
    second = clip_triangle((behind, right, beyond))

    assert first and second
    first_cut = {
        vertex.position
        for triangle in first
        for vertex in triangle
        if abs(vertex.position[2] + vertex.position[3]) < 1e-9
    }
    second_cut = {
        vertex.position
        for triangle in second
        for vertex in triangle
        if abs(vertex.position[2] + vertex.position[3]) < 1e-9
    }
    assert first_cut & second_cut


def test_png_writer_emits_real_png(tmp_path: Path) -> None:
    output = tmp_path / "image.png"
    write_png(output, bytes([255, 0, 0, 0, 255, 0]), 2, 1)

    data = output.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in data and b"IDAT" in data and b"IEND" in data
