"""A small self-contained offline triangle rasterizer."""

from .camera import Camera
from .geometry import Mesh, Triangle
from .rasterizer import RenderOptions, render
from .scene import Scene
from .section import ClipGroup, ClipPlane, SectionReport, apply_clip_group, section_signature
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
    "SectionReport",
    "apply_clip_group",
    "section_signature",
    "Texture",
    "checkerboard",
    "stripe_card",
]
