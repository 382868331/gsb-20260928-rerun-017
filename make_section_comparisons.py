"""Contact sheet for joint clipping: paired cut, moved control, front/behind.

The paired group and its moved control carry independent section colors; the
front and behind views of each group state the same mesh/section ownership.
"""
from __future__ import annotations

from pathlib import Path

from renderer import RenderOptions, render
from renderer.image import Image
from scenes.section_scene import section_scene


def _place(target: Image, source: Image, x: int, y: int) -> None:
    for row in range(source.height):
        for col in range(source.width):
            target.set_pixel(x + col, y + row, source.get_pixel(col, row))


def _label(target: Image, x: int, y: int, text: str, color=(235, 235, 240)) -> None:
    # A dependency-free 3x5 pixel font, just enough for short tags.
    glyphs = {
        "A": ("010", "101", "111", "101", "101"),
        "B": ("110", "101", "110", "101", "110"),
        "C": ("111", "100", "100", "100", "111"),
        "D": ("110", "101", "101", "101", "110"),
        "E": ("111", "100", "110", "100", "111"),
        "F": ("111", "100", "110", "100", "100"),
        "P": ("110", "101", "110", "100", "100"),
        "R": ("110", "101", "110", "101", "101"),
        "O": ("111", "101", "101", "101", "111"),
        "N": ("101", "111", "111", "111", "101"),
        "T": ("111", "010", "010", "010", "010"),
        " ": ("000", "000", "000", "000", "000"),
    }
    cursor_x = x
    for character in text.upper():
        pattern = glyphs.get(character, glyphs[" "])
        for row, bits in enumerate(pattern):
            for col, bit in enumerate(bits):
                if bit == "1":
                    target.set_pixel(cursor_x + col, y + row, color)
        cursor_x += 4


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    scene = section_scene()
    scene.save(output_dir / "section_scene.json")

    width, height = 260, 195
    views = [
        ("paired", "front", "A FRONT"),
        ("paired", "behind", "B BEHIND"),
        ("control-moved-x", "front", "C MOVED FRONT"),
        ("control-moved-x", "behind", "D MOVED BEHIND"),
    ]
    sheet = Image(width * 2, height * 2 + 14, scene.background)
    for index, (group, camera, label) in enumerate(views):
        image = render(scene, RenderOptions(width, height, camera, group))
        column = (index % 2) * width
        row = (index // 2) * (height + 7)
        _place(sheet, image, column, row + 7)
        _label(sheet, column + 4, row + 1, label)
    sheet.save_png(output_dir / "section_comparisons.png")
    print(f"saved {output_dir / 'section_comparisons.png'}")


if __name__ == "__main__":
    main()
