from __future__ import annotations

from pathlib import Path

from renderer import RenderOptions, render
from scenes.demo_scene import build_demo_scene


def main() -> None:
    output_dir = Path("output")
    scene = build_demo_scene()
    scene_path = output_dir / "demo_scene.json"
    scene.save(scene_path)

    for camera_name in ("front", "moved"):
        image = render(scene, RenderOptions(320, 240, camera_name))
        image.save_png(output_dir / f"demo_{camera_name}.png")
    print(f"saved {scene_path} and camera views in {output_dir}/")


if __name__ == "__main__":
    main()
