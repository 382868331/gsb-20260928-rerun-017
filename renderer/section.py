from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .clip_plane import ClipPlane
from .geometry import Mesh, Vec3

RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class SectionInfo:
    """A closed section face produced by one mesh on one clip plane."""

    mesh_name: str
    plane_name: str
    loop_count: int
    vertex_count: int
    color: RGB


@dataclass(frozen=True)
class GapInfo:
    """An open section chain: the model cannot be closed on this plane."""

    mesh_name: str
    plane_name: str
    start: Vec3
    end: Vec3
    length: float


@dataclass
class RenderReport:
    sections: List[SectionInfo] = field(default_factory=list)
    gaps: List[GapInfo] = field(default_factory=list)

    def section_for(self, mesh_name: str, plane_name: str) -> SectionInfo | None:
        for section in self.sections:
            if section.mesh_name == mesh_name and section.plane_name == plane_name:
                return section
        return None


def _quantize(point: Vec3, tolerance: float) -> Tuple[int, int, int]:
    return tuple(int(round(component / tolerance)) for component in point)


def _section_segments(mesh: Mesh, plane: ClipPlane, scale: float) -> Dict[tuple, Tuple[Vec3, Vec3]]:
    """One segment per triangle/plane intersection, deduplicated by endpoints.

    Distances are snapped to zero within a scale-relative epsilon so a plane
    resting exactly on a vertex or sliding along an existing edge produces
    the same segments as an infinitesimally offset plane.
    """
    epsilon = 1e-9 * scale
    tolerance = 1e-6 * scale
    segments: Dict[tuple, Tuple[Vec3, Vec3]] = {}
    for triangle in mesh.triangles:
        points = [mesh.vertices[index] for index in triangle.indices]
        distances = []
        for point in points:
            distance = plane.signed_distance(point)
            distances.append(0.0 if abs(distance) <= epsilon else distance)
        polygon: List[Vec3] = []
        for i in range(3):
            j = (i + 1) % 3
            da, db = distances[i], distances[j]
            a, b = points[i], points[j]
            if da == 0.0 and db == 0.0:
                continue  # edge lies in the plane; neighbours emit its endpoints
            if da == 0.0:
                polygon.append(a)
            elif db == 0.0:
                polygon.append(b)
            elif (da > 0.0) != (db > 0.0):
                t = da / (da - db)
                polygon.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
        unique: List[Vec3] = []
        for point in polygon:
            if not unique or _quantize(unique[-1], tolerance) != _quantize(point, tolerance):
                unique.append(point)
        if len(unique) > 1 and _quantize(unique[0], tolerance) == _quantize(unique[-1], tolerance):
            unique.pop()
        if len(unique) != 2:
            continue
        key_a = _quantize(unique[0], tolerance)
        key_b = _quantize(unique[1], tolerance)
        if key_a == key_b:
            continue
        if key_a <= key_b:
            segments.setdefault((key_a, key_b), (unique[0], unique[1]))
        else:
            segments.setdefault((key_b, key_a), (unique[1], unique[0]))
    return segments


def _chain_segments(
    segments: Dict[tuple, Tuple[Vec3, Vec3]],
) -> Tuple[List[List[Vec3]], List[List[Vec3]]]:
    """Chain segments into closed loops; leftover open chains are returned too."""
    adjacency: Dict[tuple, List[tuple]] = {}
    for key in segments:
        adjacency.setdefault(key[0], []).append(key)
        adjacency.setdefault(key[1], []).append(key)
    unused = set(segments)
    loops: List[List[Vec3]] = []
    chains: List[List[Vec3]] = []
    while unused:
        first = unused.pop()
        keys = [first[0], first[1]]
        points = [segments[first][0], segments[first][1]]
        closed = False
        for extend_tail in (True, False):  # extend the tail, then the head
            while True:
                end_key = keys[-1] if extend_tail else keys[0]
                candidate = next((key for key in adjacency.get(end_key, []) if key in unused), None)
                if candidate is None:
                    break
                unused.discard(candidate)
                pa, pb = segments[candidate]
                join = candidate[0] if candidate[0] == end_key else candidate[1]
                other = candidate[1] if join == candidate[0] else candidate[0]
                point = pb if join == candidate[0] else pa
                if extend_tail:
                    keys.append(other)
                    points.append(point)
                else:
                    keys.insert(0, other)
                    points.insert(0, point)
                if keys[0] == keys[-1]:
                    closed = True
                    break
            if closed:
                break
        if closed:
            points.pop()  # last point repeats the first
            loops.append(points)
        else:
            chains.append(points)
    return loops, chains


def extract_section(
    mesh: Mesh,
    plane: ClipPlane,
    scale: float,
) -> Tuple[List[List[Vec3]], List[GapInfo]]:
    """Closed section loops plus gap reports for chains that cannot close."""
    segments = _section_segments(mesh, plane, scale)
    loops, chains = _chain_segments(segments)
    gaps = []
    for chain in chains:
        if len(chain) < 2:
            continue
        start, end = chain[0], chain[-1]
        length = math.sqrt(sum((end[k] - start[k]) ** 2 for k in range(3)))
        gaps.append(GapInfo(mesh.name, plane.name, start, end, length))
    return loops, gaps
