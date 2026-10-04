from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .camera import Camera
from .geometry import Mesh
from .section import ClipGroup


class Scene:
    def __init__(
        self,
        name: str,
        meshes: Iterable[Mesh],
        cameras: Iterable[Camera],
        background=(12, 14, 20),
        clip_groups: Optional[Iterable[ClipGroup]] = None,
    ):
        self.name = name
        self.meshes = list(meshes)
        self.cameras = list(cameras)
        self.background = tuple(background)
        self.clip_groups = list(clip_groups) if clip_groups is not None else []
        if not self.cameras:
            raise ValueError("scene needs at least one camera")
        if len(self.background) != 3 or any(not 0 <= value <= 255 for value in self.background):
            raise ValueError("background must be three channels in [0, 255]")
        self._ensure_unique(mesh.name for mesh in self.meshes)
        self._ensure_unique(camera.name for camera in self.cameras)

    @staticmethod
    def _ensure_unique(names: Iterable[str]) -> None:
        seen = set()
        for name in names:
            if name in seen:
                raise ValueError(f"duplicate name: {name}")
            seen.add(name)

    def clip_group(self, name: Optional[str] = None) -> ClipGroup:
        if name is None:
            if not self.clip_groups:
                raise KeyError(f"scene {self.name!r} has no clip groups")
            return self.clip_groups[0]
        for group in self.clip_groups:
            if group.name == name:
                return group
        raise KeyError(f"clip group {name!r} is not in scene {self.name!r}")

    def camera(self, name: Optional[str] = None) -> Camera:
        if name is None:
            return self.cameras[0]
        for camera in self.cameras:
            if camera.name == name:
                return camera
        raise KeyError(f"camera {name!r} is not in scene {self.name!r}")

    def to_dict(self) -> dict:
        return {
            "format": "mini-offline-renderer-scene",
            "version": 2,
            "name": self.name,
            "background": list(self.background),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "meshes": [mesh.to_dict() for mesh in self.meshes],
            **(
                {"clip_groups": [group.to_dict() for group in self.clip_groups]}
                if self.clip_groups
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Scene":
        if data.get("format") != "mini-offline-renderer-scene":
            raise ValueError("unsupported scene format")
        version = data.get("version", 1)
        if version not in (1, 2):
            raise ValueError("unsupported scene version")
        return cls(
            name=data["name"],
            meshes=[Mesh.from_dict(item) for item in data["meshes"]],
            cameras=[Camera.from_dict(item) for item in data["cameras"]],
            background=tuple(data.get("background", (12, 14, 20))),
            clip_groups=[ClipGroup.from_dict(item) for item in data.get("clip_groups", [])],
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
