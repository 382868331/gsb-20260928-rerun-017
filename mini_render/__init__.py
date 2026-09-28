"""A small dependency-free offline software renderer."""

from .camera import Camera, view_projection_matrix
from .geometry import Mesh, Scene
from .rasterizer import RenderSettings, render

__all__ = [
    "Camera",
    "Mesh",
    "RenderSettings",
    "Scene",
    "render",
    "view_projection_matrix",
]
