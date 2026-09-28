from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import List, Sequence, Tuple

RGB = Tuple[int, int, int]


class Image:
    def __init__(self, width: int, height: int, background: RGB = (0, 0, 0)):
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        self.width = width
        self.height = height
        self.pixels: List[RGB] = [background for _ in range(width * height)]

    def set_pixel(self, x: int, y: int, color: RGB) -> None:
        self.pixels[y * self.width + x] = tuple(max(0, min(255, int(round(value)))) for value in color)

    def get_pixel(self, x: int, y: int) -> RGB:
        return self.pixels[y * self.width + x]

    def save_png(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            row_start = y * self.width
            for x in range(self.width):
                raw.extend(self.pixels[row_start + x])

        def chunk(kind: bytes, data: bytes) -> bytes:
            checksum = zlib.crc32(kind + data)
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        png += chunk(b"IEND", b"")
        target.write_bytes(png)


def read_png_dimensions(path: str | Path) -> Tuple[int, int]:
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    return struct.unpack(">II", data[16:24])
