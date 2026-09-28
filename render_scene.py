from __future__ import annotations

import argparse
from pathlib import Path

from renderer import RenderOptions, Scene, render


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a saved JSON scene with the built-in rasterizer.")
    parser.add_argument("scene", type=Path, help="path to a scene JSON file")
    parser.add_argument("output", type=Path, help="output PNG path")
    parser.add_argument("--width", type=int, default=240)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--camera", default=None, help="camera name; defaults to the first saved camera")
    args = parser.parse_args()

    scene = Scene.load(args.scene)
    image = render(scene, RenderOptions(args.width, args.height, args.camera))
    image.save_png(args.output)
    print(f"rendered {args.output} with camera {scene.camera(args.camera).name}")


if __name__ == "__main__":
    main()
