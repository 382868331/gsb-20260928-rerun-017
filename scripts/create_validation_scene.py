from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mini_render import Camera, Mesh, Scene



def main() -> None:
    scene = Scene(default_camera="main")
    scene.cameras = {
        "main": Camera(
            position=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 1.2),
            fovy_degrees=60.0,
            near=0.5,
            far=30.0,
        ),
        "side": Camera(
            position=(-3.0, 0.4, 5.0),
            target=(0.0, 0.0, 1.2),
            fovy_degrees=60.0,
            near=0.5,
            far=30.0,
        ),
    }
    scene.meshes = [
        Mesh(
            name="large_cyan_near_crossing_quad",
            vertices=(
                (-5.0, -4.0, -1.0),
                (5.0, -4.0, -1.0),
                (5.0, 4.0, 4.8),
                (-5.0, 4.0, 4.8),
            ),
            faces=((0, 1, 2), (0, 2, 3)),
            color=(45, 205, 210),
        ),
        Mesh(
            name="front_red_overlap",
            vertices=(
                (-0.25, -0.22, 4.0),
                (0.25, -0.22, 4.0),
                (0.25, 0.22, 4.0),
                (-0.25, 0.22, 4.0),
            ),
            faces=((0, 1, 2), (0, 2, 3)),
            color=(225, 70, 65),
        ),
        Mesh(
            name="right_yellow_partial_outside",
            vertices=(
                (0.90, 0.30, 3.8),
                (1.30, 0.30, 3.8),
                (1.30, 0.43, 3.8),
                (0.90, 0.43, 3.8),
            ),
            faces=((0, 1, 2), (0, 2, 3)),
            color=(235, 205, 70),
        ),
    ]

    target = ROOT / "scenes" / "validation.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    scene.save(target)


if __name__ == "__main__":
    main()
