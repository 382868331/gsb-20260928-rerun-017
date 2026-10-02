from __future__ import annotations

import json
from pathlib import Path

from renderer import RenderOptions, apply_clip_group, render, section_signature
from scenes.section_scene import build_section_scene


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    base_scene = build_section_scene()
    base_scene.save(output_dir / "section_scene.json")
    summary = {}
    for group in base_scene.clip_groups:
        derived, report = apply_clip_group(base_scene, group)
        summary[group.name] = {
            "signature": section_signature(report),
            "report": report.to_dict(),
        }
        for camera_name in ("front", "back"):
            image = render(derived, RenderOptions(320, 240, camera_name))
            image.save_png(output_dir / f"sections_{group.name}_{camera_name}.png")
    # Same conditions, alternative triangulation: signatures must match so a
    # pure re-triangulation is distinguishable from a condition change.
    alt_scene = build_section_scene(diagonal_b=True)
    alt_summary = {}
    for group in alt_scene.clip_groups:
        _, report = apply_clip_group(alt_scene, group)
        alt_summary[group.name] = section_signature(report)
    summary["triangulation_b_signatures"] = alt_summary
    (output_dir / "section_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
