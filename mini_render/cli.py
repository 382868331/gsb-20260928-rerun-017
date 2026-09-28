from __future__ import annotations

import argparse

from .geometry import Scene
from .geometry import color as parse_color
from .image import write_png
from .rasterizer import RenderSettings, render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a saved scene to PNG")
    parser.add_argument("scene", help="path to a scene JSON file")
    parser.add_argument("output", help="path to the output PNG file")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--camera", help="named camera in the scene")
    parser.add_argument(
        "--background",
        type=lambda value: parse_color(
            tuple(int(part) for part in value.split(","))
        ),
        default=(18, 20, 26),
        help="comma-separated RGB background, for example 18,20,26",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene = Scene.load(args.scene)
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        background=args.background,
        camera=args.camera,
    )
    image = render(scene, settings)
    write_png(args.output, image, args.width, args.height)


if __name__ == "__main__":
    main()
