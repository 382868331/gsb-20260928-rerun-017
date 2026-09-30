from __future__ import annotations

from pathlib import Path

from renderer import RenderOptions, Scene, render
from scenes.demo_scene import build_demo_scene
from scenes.texture_demo_scene import build_texture_demo_scene


def main() -> None:
    output_dir = Path("output")
    scene = build_demo_scene()
    scene_path = output_dir / "demo_scene.json"
    scene.save(scene_path)

    for camera_name in ("front", "moved"):
        image = render(scene, RenderOptions(320, 240, camera_name))
        image.save_png(output_dir / f"demo_{camera_name}.png")
    print(f"saved {scene_path} and camera views in {output_dir}/")

    textured_scene = build_texture_demo_scene()
    textured_path = output_dir / "texture_scene.json"
    textured_scene.save(textured_path)
    for camera_name in ("front", "moved"):
        image = render(textured_scene, RenderOptions(320, 240, camera_name))
        image.save_png(output_dir / f"texture_{camera_name}.png")

    reloaded = Scene.load(textured_path)
    render(reloaded, RenderOptions(320, 240, "front")).save_png(output_dir / "texture_reloaded.png")
    print(f"saved {textured_path} and textured comparison views in {output_dir}/")


if __name__ == "__main__":
    main()
