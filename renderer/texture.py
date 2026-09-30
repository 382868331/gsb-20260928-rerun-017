from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

RGBA = Tuple[int, int, int, int]


class Texture:
    """A small RGBA texture whose texels are computed, never loaded from images."""

    def __init__(self, name: str, width: int, height: int, texels: Sequence[RGBA]):
        if not name:
            raise ValueError("texture requires a name")
        if width <= 0 or height <= 0:
            raise ValueError("texture dimensions must be positive")
        if len(texels) != width * height:
            raise ValueError("texel count must equal width * height")
        for texel in texels:
            if len(texel) != 4 or any(not 0 <= channel <= 255 for channel in texel):
                raise ValueError("each texel must be four channels in [0, 255]")
        self.name = name
        self.width = width
        self.height = height
        self.texels: List[RGBA] = [tuple(texel) for texel in texels]

    @classmethod
    def procedural(
        cls,
        name: str,
        width: int,
        height: int,
        generator: Callable[[int, int], RGBA],
    ) -> "Texture":
        return cls(
            name,
            width,
            height,
            [generator(x, y) for y in range(height) for x in range(width)],
        )

    def sample(self, u: float, v: float) -> RGBA:
        # Repeat wrapping keeps the shared physical surface identical regardless of
        # how its triangles were authored.
        u = u % 1.0
        v = v % 1.0
        epsilon = 1e-7
        x = min(self.width - 1, max(0, int(u * self.width + epsilon)))
        y = min(self.height - 1, max(0, int(v * self.height + epsilon)))
        return self.texels[y * self.width + x]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            # Flat RGBA rows keep saved scenes a complete, renderable record.
            "texels": [channel for texel in self.texels for channel in texel],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Texture":
        width = int(data["width"])
        height = int(data["height"])
        raw = data["texels"]
        if len(raw) != width * height * 4:
            raise ValueError("saved texture does not contain width * height * 4 channels")
        texels = [
            tuple(raw[index:index + 4])
            for index in range(0, len(raw), 4)
        ]
        return cls(str(data["name"]), width, height, texels)


def checker_texture(
    name: str,
    cells: int = 8,
    first: RGBA = (232, 226, 214, 255),
    second: RGBA = (58, 74, 104, 255),
    cell_pixels: int = 8,
) -> Texture:
    """High-frequency checkerboard: triangulation distortion is easy to spot."""

    def texel(x: int, y: int) -> RGBA:
        return first if ((x // cell_pixels) + (y // cell_pixels)) % 2 == 0 else second

    return Texture.procedural(name, cells * cell_pixels, cells * cell_pixels, texel)


def stripe_texture(
    name: str,
    stripes: int = 6,
    foreground: RGBA = (226, 112, 96, 255),
    background: RGBA = (240, 236, 224, 255),
    stripe_pixels: int = 10,
) -> Texture:
    def texel(x: int, y: int) -> RGBA:
        return foreground if (x // stripe_pixels) % 2 == 0 else background

    return Texture.procedural(name, stripes * stripe_pixels * 2, 64, texel)
