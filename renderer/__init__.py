"""A small self-contained offline triangle rasterizer."""

from .camera import Camera
from .clip_plane import ClipPlane
from .geometry import Mesh, Triangle
from .rasterizer import RenderOptions, render, render_with_report
from .scene import Scene
from .section import GapInfo, RenderReport, SectionInfo
from .texture import Texture, checkerboard, stripe_card

__all__ = [
    "Camera",
    "ClipPlane",
    "GapInfo",
    "Mesh",
    "RenderOptions",
    "RenderReport",
    "SectionInfo",
    "Triangle",
    "render",
    "render_with_report",
    "Scene",
    "Texture",
    "checkerboard",
    "stripe_card",
]
