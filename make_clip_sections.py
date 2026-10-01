from __future__ import annotations

from pathlib import Path

from renderer import RenderOptions, Scene, render_with_report
from scenes.clip_scene import build_clip_scene


def describe(report) -> None:
    for section in report.sections:
        print(
            f"  section: mesh={section.mesh_name} plane={section.plane_name} "
            f"loops={section.loop_count} vertices={section.vertex_count} color={section.color}"
        )
    for gap in report.gaps:
        print(
            f"  gap: mesh={gap.mesh_name} plane={gap.plane_name} "
            f"open chain {gap.start} -> {gap.end} length={gap.length:.4f} (not filled)"
        )


def main() -> None:
    out = Path("output")
    out.mkdir(exist_ok=True)
    scene = build_clip_scene()
    scene.save(out / "clip_scene.json")
    for camera_name in ("front", "west", "east"):
        image, report = render_with_report(scene, RenderOptions(320, 240, camera_name))
        image.save_png(out / f"clip_{camera_name}.png")
        print(f"camera {camera_name}:")
        describe(report)
    reloaded = Scene.load(out / "clip_scene.json")
    image, report = render_with_report(reloaded, RenderOptions(320, 240, "front"))
    image.save_png(out / "clip_reloaded_front.png")
    print("reloaded scene front render:")
    describe(report)


if __name__ == "__main__":
    main()
