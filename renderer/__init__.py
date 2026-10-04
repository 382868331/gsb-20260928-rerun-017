"""A small self-contained offline triangle rasterizer."""

from .camera import Camera
from .geometry import Mesh, Triangle
from .rasterizer import RenderOptions, render
from .scene import Scene
from .section import ClipGroup, ClipPlane, build_section
from .texture import Texture, checkerboard, stripe_card

__all__ = [
    "Camera",
    "Mesh",
    "Triangle",
    "RenderOptions",
    "render",
    "Scene",
    "ClipGroup",
    "ClipPlane",
    "build_section",
    "Texture",
    "checkerboard",
    "stripe_card",
]
