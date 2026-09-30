from __future__ import annotations

import tempfile
from pathlib import Path

from renderer import RenderOptions, Scene, render
from renderer.image import Image
from scenes.texture_scene import (
    near_textured_scene,
    transparent_overlap_scene,
    triangulation_scene,
)


def _save_and_reload_render(scene: Scene, options: RenderOptions, path: Path) -> Image:
    scene.save(path)
    return render(Scene.load(path), options)


def _place(target: Image, source: Image, x: int, y: int) -> None:
    for row in range(source.height):
        for col in range(source.width):
            target.set_pixel(x + col, y + row, source.get_pixel(col, row))


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    width, height = 200, 150
    options = RenderOptions(width, height, "front")
    moved_options = RenderOptions(width, height, "moved")

    diagonal_a = triangulation_scene("a")
    diagonal_b = triangulation_scene("b")
    near_scene = near_textured_scene()
    overlap = transparent_overlap_scene()

    with tempfile.TemporaryDirectory() as directory:
        temp_dir = Path(directory)
        # Persistent demo artifacts.
        diagonal_a.save(output_dir / "triangulation_a.json")
        diagonal_b.save(output_dir / "triangulation_b.json")
        near_scene.save(output_dir / "near_textured.json")
        overlap.save(output_dir / "transparent_overlap.json")

        image_a = _save_and_reload_render(diagonal_a, options, temp_dir / "a.json")
        image_b = _save_and_reload_render(diagonal_b, options, temp_dir / "b.json")
        image_near = _save_and_reload_render(near_scene, options, temp_dir / "near.json")
        image_layers = _save_and_reload_render(overlap, options, temp_dir / "layers.json")
        image_moved = _save_and_reload_render(overlap, moved_options, temp_dir / "layers_moved.json")

    contact = Image(width * 3, height * 2, background=(12, 14, 20))
    _place(contact, image_a, 0, 0)
    _place(contact, image_b, width, 0)
    _place(contact, image_near, width * 2, 0)
    _place(contact, image_layers, 0, height)
    _place(contact, image_moved, width, height)
    contact.save_png(output_dir / "feature_comparisons.png")
    image_a.save_png(output_dir / "triangulation_a.png")
    image_b.save_png(output_dir / "triangulation_b.png")
    image_near.save_png(output_dir / "near_textured.png")
    image_layers.save_png(output_dir / "transparent_overlap_front.png")
    image_moved.save_png(output_dir / "transparent_overlap_moved.png")
    print("wrote texture/transparency comparison images to output/")


if __name__ == "__main__":
    main()
