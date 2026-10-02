from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .camera import Camera
from .geometry import Mesh


class Scene:
    def __init__(
        self,
        name: str,
        meshes: Iterable[Mesh],
        cameras: Iterable[Camera],
        background=(12, 14, 20),
        clip_groups=None,
    ):
        self.name = name
        self.meshes = list(meshes)
        self.cameras = list(cameras)
        self.background = tuple(background)
        from .section import ClipGroup

        self.clip_groups = list(clip_groups) if clip_groups else []
        for group in self.clip_groups:
            if not isinstance(group, ClipGroup):
                raise ValueError("clip_groups must contain ClipGroup objects")
        if not self.cameras:
            raise ValueError("scene needs at least one camera")
        if len(self.background) != 3 or any(not 0 <= value <= 255 for value in self.background):
            raise ValueError("background must be three channels in [0, 255]")
        self._ensure_unique(mesh.name for mesh in self.meshes)
        self._ensure_unique(camera.name for camera in self.cameras)
        self._ensure_unique(group.name for group in self.clip_groups)

    @staticmethod
    def _ensure_unique(names: Iterable[str]) -> None:
        seen = set()
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate name: {name}")
            seen.add(name)

    def camera(self, name: Optional[str] = None) -> Camera:
        if name is None:
            return self.cameras[0]
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(f"camera {name!r} is not in scene {self.name!r}")

    def to_dict(self) -> dict:
        data = {
            "format": "mini-offline-renderer-scene",
            "version": 2 if self.clip_groups else 1,
            "name": self.name,
            "background": list(self.background),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
        }
        if self.clip_groups:
            data["clip_groups"] = [group.to_dict() for group in self.clip_groups]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        if data.get("format") != "mini-offline-renderer-scene":
            raise ValueError("unsupported scene format")
        if data.get("version", 1) not in (1, 2):
            raise ValueError("unsupported scene version")
        from .section import ClipGroup

        clip_groups = [ClipGroup.from_dict(item) for item in data.get("clip_groups", [])]
        return cls(
            name=data["name"],
            meshes=[Mesh.from_dict(item) for item in data["meshes"]],
            cameras=[Camera.from_dict(item) for item in data["cameras"]],
            background=tuple(data.get("background", (12, 14, 20))),
            clip_groups=clip_groups,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
