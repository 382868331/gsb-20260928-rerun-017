from __future__ import annotations

import math
from typing import List, Sequence, Tuple

RGB = Tuple[int, int, int]
UV = Tuple[float, float]


class Texture:
    """Procedurally defined 2D texture sampled with perspective-corrected UVs."""

    def __init__(self, name: str, width: int, height: int, pixels: Sequence[RGB]):
        if width <= 0 or height <= 0:
            raise ValueError("texture dimensions must be positive")
        if len(pixels) != width * height:
            raise ValueError("texture pixel count does not match dimensions")
        self.name = name
        self.width = width
        self.height = height
        self.pixels: List[RGB] = [tuple(value) for value in pixels]

    def sample(self, u: float, v: float) -> RGB:
        wrapped_u = u - math.floor(u)
        wrapped_v = v - math.floor(v)
        x = min(self.width - 1, max(0, int(wrapped_u * self.width)))
        y = min(self.height - 1, max(0, int(wrapped_v * self.height)))
        return self.pixels[y * self.width + x]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "pixels": [list(pixel) for pixel in self.pixels],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Texture":
        width = int(data["width"])
        height = int(data["height"])
        return cls(str(data["name"]), width, height, [tuple(pixel) for pixel in data["pixels"]])


def checkerboard(
    name: str,
    cells: int = 8,
    cell_pixels: int = 8,
    color_a: RGB = (235, 235, 235),
    color_b: RGB = (45, 45, 45),
) -> Texture:
    size = cells * cell_pixels
    pixels: List[RGB] = []
    for y in range(size):
        for x in range(size):
            cell_x = x // cell_pixels
            cell_y = y // cell_pixels
            on_grid_line = x % cell_pixels == 0 or y % cell_pixels == 0
            if on_grid_line:
                pixels.append((180, 60, 60))
            elif (cell_x + cell_y) % 2 == 0:
                pixels.append(color_a)
            else:
                pixels.append(color_b)
    return Texture(name, size, size, pixels)


def stripe_card(name: str, width: int = 64, height: int = 64) -> Texture:
    """Horizontal stripes plus a border: distortion or a seam shows immediately."""
    pixels: List[RGB] = []
    for y in range(height):
        for x in range(width):
            border = x == 0 or y == 0 or x == width - 1 or y == height - 1
            if border:
                color = (200, 80, 80)
            elif (y // 4) % 2 == 0:
                color = (70, 130, 220)
            else:
                color = (230, 220, 120)
            pixels.append(color)
    return Texture(name, width, height, pixels)
