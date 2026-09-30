"""A small self-contained offline triangle rasterizer."""

from .camera import Camera
from .geometry import Mesh, Triangle
from .rasterizer import RenderOptions, render
from .scene import Scene
from .texture import Texture

__all__ = ["Camera", "Mesh", "Triangle", "RenderOptions", "render", "Scene", "Texture"]
