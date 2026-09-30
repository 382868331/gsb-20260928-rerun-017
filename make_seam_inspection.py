from __future__ import annotations

import tempfile
from pathlib import Path

from renderer import Camera, Mesh, RenderOptions, Scene, Triangle, render
from renderer.image import Image


CORNERS = [
    (-1.4, -1.1, 0.3),
    (1.9, -1.1, 2.8),
    (1.1, 1.2, 2.1),
    (-1.9, 1.2, 1.1),
]
GREEN = (0, 210, 0)
OPACITY = 0.5


def surface_variants():
    return {
        "diagonal_a": [Mesh("surface", CORNERS, [
            Triangle((0, 1, 2), GREEN, opacity=OPACITY),
            Triangle((0, 2, 3), GREEN, opacity=OPACITY),
        ])],
        "diagonal_b": [Mesh("surface", CORNERS, [
            Triangle((0, 3, 2), GREEN, opacity=OPACITY),
            Triangle((0, 2, 1), GREEN, opacity=OPACITY),
        ])],
        "split_meshes": [
            Mesh("surface_a", CORNERS, [Triangle((0, 1, 2), GREEN, opacity=OPACITY)]),
            Mesh("surface_b", CORNERS, [Triangle((0, 2, 3), GREEN, opacity=OPACITY)]),
        ],
    }


def _place(target: Image, source: Image, x: int, y: int) -> None:
    for row in range(source.height):
        for col in range(source.width):
            target.set_pixel(x + col, y + row, source.get_pixel(col, row))


def _zoom(source: Image, left: int, top: int, span: int, scale: int) -> Image:
    zoomed = Image(span * scale, span * scale, (30, 30, 30))
    for row in range(span):
        for col in range(span):
            pixel = source.get_pixel(left + col, top + row)
            for sy in range(scale):
                for sx in range(scale):
                    zoomed.set_pixel(col * scale + sx, row * scale + sy, pixel)
    return zoomed


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    camera = Camera("front", near=1.5, far=20, fov_y_degrees=55)
    width, height = 320, 240
    options = RenderOptions(width, height, "front")

    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)
        renders = {}
        for name, meshes in surface_variants().items():
            scene = Scene(f"seam-{name}", meshes, [camera], background=(0, 0, 0))
            scene.save(temp_dir / f"{name}.json")
            reloaded = Scene.load(temp_dir / f"{name}.json")
            renders[name] = render(reloaded, options)
            renders[name].save_png(output_dir / f"seam_{name}.png")

    reference = renders["diagonal_a"].pixels
    for name, image in renders.items():
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(reference, image.pixels)) if a != b]
        if differing:
            raise SystemExit(f"seam inspection mismatch for {name}: {differing[:4]}")

    # The near cut forms the left silhouette of the visible surface: pick an
    # interior row and its leftmost covered column, so the zoomed crop
    # straddles the boundary with uncovered pixels immediately on its left.
    image = renders["diagonal_a"]
    span, scale = 24, 8
    boundary_row = height // 3
    covered_columns = [col for col in range(width) if image.get_pixel(col, boundary_row) != (0, 0, 0)]
    if not covered_columns:
        raise SystemExit("could not locate an interior clip boundary to zoom")
    boundary_column = covered_columns[0]
    top = max(0, boundary_row - span // 2)
    left = max(0, boundary_column - span // 2)

    labels = ("diagonal_a", "diagonal_b", "split_meshes")
    sheet = Image(span * scale * len(labels), span * scale, background=(30, 30, 30))
    for index, name in enumerate(labels):
        crop = _zoom(renders[name], left, top, span, scale)
        _place(sheet, crop, index * span * scale, 0)
    sheet.save_png(output_dir / "seam_boundary_zoom.png")
    print(f"wrote boundary zoom crops near pixel ({boundary_column},{boundary_row}) to output/seam_boundary_zoom.png")


if __name__ == "__main__":
    main()
