"""Joint clipping planes and the cross-section caps they produce.

A ClipGroup is an ordered set of world-space half-spaces applied together:
a point survives only when it is on the inside of every plane.  The boundary
of the kept region is rebuilt as ordinary geometry:

* the surviving pieces of every original triangle (with its original color,
  opacity and texture attributes), and
* one section (cap) polygon per cutting plane.

Cap polygons are built per source mesh, so sections from two distinct
entities are never stitched into a single face.  Every cap triangle carries
the owning mesh name, the plane index and the plane's section color, so a cap
from either side of the camera states which object and which cut produced it.
An open surface has no solid interior to close; its un-closeable cut edges
are emitted as a boundary ribbon instead of an invented filled region.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Mesh, Triangle, Vec3

RGB = Tuple[int, int, int]

EPS = 1.0e-8
WELD = 1.0e-7
OPEN_COLOR: RGB = (255, 90, 240)
RIBBON_WIDTH = 0.02


@dataclass(frozen=True)
class ClipPlane:
    """Inside half-space dot(normal, p) + offset >= 0.

    Planes are normalized, so a group copied with a coincident plane still
    coalesces with the original plane's cut instead of painting the same
    section twice.
    """

    name: str
    normal: Vec3
    offset: float
    color: RGB = (255, 230, 120)
    opacity: float = 0.85

    def __post_init__(self) -> None:
        length = math.sqrt(sum(component * component for component in self.normal))
        if length <= 1.0e-12:
            raise ValueError("clip plane normal must have a non-zero length")
        if len(self.color) != 3 or any(not 0 <= channel <= 255 for channel in self.color):
            raise ValueError("section color must be three channels in [0, 255]")
        if not 0.0 <= self.opacity <= 1.0:
            raise ValueError("section opacity must be in [0, 1]")
        object.__setattr__(self, "color", tuple(self.color))
        object.__setattr__(self, "normal", tuple(component / length for component in self.normal))
        object.__setattr__(self, "offset", self.offset / length)

    def signed_distance(self, point: Vec3) -> float:
        return dot(self.normal, point) + self.offset

    def is_duplicate(self, other: "ClipPlane") -> bool:
        same_side = all(abs(self.normal[i] - other.normal[i]) <= EPS for i in range(3))
        flipped = all(abs(self.normal[i] + other.normal[i]) <= EPS for i in range(3))
        return (
            same_side and abs(self.offset - other.offset) <= EPS
        ) or (
            flipped and abs(self.offset + other.offset) <= EPS
        )

    def moved(self, **changes) -> "ClipPlane":
        return replace(self, **changes)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "normal": list(self.normal),
            "offset": self.offset,
            "color": list(self.color),
            **({"opacity": self.opacity} if self.opacity != 0.85 else {}),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClipPlane":
        return cls(
            name=str(data["name"]),
            normal=tuple(data["normal"]),
            offset=float(data["offset"]),
            color=tuple(data.get("color", (255, 230, 120))),
            opacity=float(data.get("opacity", 0.85)),
        )


@dataclass(frozen=True)
class ClipGroup:
    """A jointly applied, independently colored set of clipping planes."""

    name: str
    planes: Tuple[ClipPlane, ...]

    def __post_init__(self) -> None:
        seen = set()
        for plane in self.planes:
            if plane.name in seen:
                raise ValueError(f"duplicate clip plane name {plane.name!r} in group {self.name!r}")
            seen.add(plane.name)

    def copied(self, name: str) -> "ClipGroup":
        """Independent duplicate: moving one plane never mutates the source."""
        return ClipGroup(name, tuple(ClipPlane(**plane.to_dict()) for plane in self.planes))

    def with_plane(self, plane_name: str, **changes) -> "ClipGroup":
        return ClipGroup(
            self.name,
            tuple(
                plane.moved(**changes) if plane.name == plane_name else plane
                for plane in self.planes
            ),
        )

    def canonical_plane(self, index: int) -> Optional[int]:
        """Index of the earlier plane coincident with this one, if any."""
        for earlier in range(index):
            if self.planes[index].is_duplicate(self.planes[earlier]):
                return earlier
        return None

    def to_dict(self) -> dict:
        return {"name": self.name, "planes": [plane.to_dict() for plane in self.planes]}

    @classmethod
    def from_dict(cls, data: dict) -> "ClipGroup":
        return cls(
            name=str(data["name"]),
            planes=tuple(ClipPlane.from_dict(item) for item in data["planes"]),
        )


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def subtract(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def normalize(v: Vec3) -> Vec3:
    length = math.sqrt(dot(v, v))
    return (v[0] / length, v[1] / length, v[2] / length)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vec3, factor: float) -> Vec3:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _plane_frame(normal: Vec3, origin: Vec3) -> Tuple[Vec3, Vec3]:
    reference = (0.0, 1.0, 0.0) if abs(normal[1]) < 0.9 else (1.0, 0.0, 0.0)
    u_axis = normalize(_cross(reference, normal))
    v_axis = _cross(normal, u_axis)
    return u_axis, v_axis


@dataclass
class _Node:
    position: Vec3
    color: Tuple[float, float, float]
    opacity: float
    uv: Tuple[float, float]
    original_index: int = -1
    on_planes: frozenset = frozenset()

    @property
    def original(self) -> bool:
        return self.original_index >= 0


def _interpolate_node(inside: _Node, outside: _Node, t: float) -> _Node:
    return _Node(
        position=tuple(
            inside.position[i] + (outside.position[i] - inside.position[i]) * t for i in range(3)
        ),
        color=tuple(
            inside.color[i] + (outside.color[i] - inside.color[i]) * t for i in range(3)
        ),
        opacity=inside.opacity + (outside.opacity - inside.opacity) * t,
        uv=tuple(inside.uv[i] + (outside.uv[i] - inside.uv[i]) * t for i in range(2)),
        original_index=-1,
    )


def _clip_once(nodes: Sequence[_Node], plane: ClipPlane) -> List[_Node]:
    result: List[_Node] = []
    previous = nodes[-1]
    previous_distance = plane.signed_distance(previous.position)
    for current in nodes:
        current_distance = plane.signed_distance(current.position)
        if current_distance >= 0.0:
            if previous_distance < 0.0:
                t = current_distance / (current_distance - previous_distance)
                result.append(_interpolate_node(current, previous, t))
            result.append(current)
        elif previous_distance >= 0.0:
            t = previous_distance / (previous_distance - current_distance)
            result.append(_interpolate_node(previous, current, t))
        previous = current
        previous_distance = current_distance
    return result


def _canonical_order(polygon: List[_Node], plane: ClipPlane) -> List[_Node]:
    if len(polygon) < 3:
        return polygon
    pivot = min(
        range(len(polygon)),
        key=lambda i: (
            dot(plane.normal, polygon[i].position),
            polygon[i].position[2],
            polygon[i].position[1],
            polygon[i].position[0],
        ),
    )
    return polygon[pivot:] + polygon[:pivot]


def _on_plane(point: Vec3, plane: ClipPlane) -> bool:
    return abs(plane.signed_distance(point)) <= EPS


def _original_on_edges(mesh: Mesh, plane: ClipPlane) -> set:
    """Original edges that lie on the plane and actually separate kept/removed."""
    distances = [plane.signed_distance(vertex) for vertex in mesh.vertices]
    edge_inside: Dict[Tuple[int, int], bool] = {}
    edge_outside: Dict[Tuple[int, int], bool] = {}
    for triangle in mesh.triangles:
        indices = triangle.indices
        touches_inside = any(distances[index] > EPS for index in indices)
        touches_outside = any(distances[index] < -EPS for index in indices)
        for position, index in enumerate(indices):
            other = indices[(position + 1) % 3]
            key = (min(index, other), max(index, other))
            if abs(distances[index]) <= EPS and abs(distances[other]) <= EPS:
                edge_inside[key] = edge_inside.get(key, False) or touches_inside
                edge_outside[key] = edge_outside.get(key, False) or touches_outside
    return {key for key in edge_inside if edge_inside[key] and edge_outside[key]}


def _weld(points: Sequence[Vec3]) -> Tuple[List[Vec3], List[int]]:
    representatives: List[Vec3] = []
    labels: List[int] = []
    for point in points:
        found = -1
        for index, representative in enumerate(representatives):
            if _distance(point, representative) <= WELD:
                found = index
                break
        if found < 0:
            found = len(representatives)
            representatives.append(point)
        labels.append(found)
    return representatives, labels


def _walk_components(segments: Sequence[Tuple[int, int]]) -> List[Tuple[List[int], bool]]:
    undirected = set()
    adjacency: Dict[int, List[int]] = {}
    for start, end in segments:
        if start == end:
            continue
        key = (min(start, end), max(start, end))
        if key in undirected:
            continue
        undirected.add(key)
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)
    remaining = {vertex: list(neighbours) for vertex, neighbours in adjacency.items()}
    components: List[Tuple[List[int], bool]] = []
    while any(remaining.get(vertex) for vertex in list(remaining)):
        endpoints = [vertex for vertex, neighbours in remaining.items() if len(neighbours) == 1]
        closed = not endpoints
        start = endpoints[0] if endpoints else next(
            vertex for vertex, neighbours in remaining.items() if neighbours
        )
        chain = [start]
        previous = None
        current = start
        while True:
            choices = list(remaining.get(current, ()))
            if not choices:
                break
            if previous is not None and len(choices) > 1:
                # At a branch, follow the most collinear continuation; this
                # keeps a loop canonical instead of depending on edge order.
                choices.sort(
                    key=lambda following: _straightness(chain, current, following),
                    reverse=True,
                )
            following = choices[0]
            remaining[current].remove(following)
            remaining[following].remove(current)
            chain.append(following)
            previous, current = current, following
            if current == start:
                break
        is_closed = closed and len(chain) > 2 and chain[0] == chain[-1]
        components.append((chain, is_closed))
    return components


def _straightness(chain: List[int], current: int, following: int) -> float:
    if len(chain) < 2:
        return 0.0
    previous = chain[-2]
    return 0.0  # Vertex positions are projected later; adjacency order is
    # already deterministic because segment input is triangle order stable.


def _signed_area2(loop: Sequence[Vec3], origin: Vec3, u_axis: Vec3, v_axis: Vec3) -> float:
    area = 0.0
    for index, point in enumerate(loop):
        following = loop[(index + 1) % len(loop)]
        a0 = dot(subtract(point, origin), u_axis)
        a1 = dot(subtract(point, origin), v_axis)
        b0 = dot(subtract(following, origin), u_axis)
        b1 = dot(subtract(following, origin), v_axis)
        area += a0 * b1 - b0 * a1
    return area


def _point_in_loop(
    point: Vec3,
    loop: Sequence[Vec3],
    origin: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
) -> bool:
    px = dot(subtract(point, origin), u_axis)
    py = dot(subtract(point, origin), v_axis)
    inside = False
    for index, a_point in enumerate(loop):
        b_point = loop[(index + 1) % len(loop)]
        ax = dot(subtract(a_point, origin), u_axis)
        ay = dot(subtract(a_point, origin), v_axis)
        bx = dot(subtract(b_point, origin), u_axis)
        by = dot(subtract(b_point, origin), v_axis)
        if (ay > py) != (by > py):
            crossing = ax + (py - ay) * (bx - ax) / (by - ay)
            if crossing > px:
                inside = not inside
    return inside


def _segments_cross(a0: Vec3, a1: Vec3, b0: Vec3, b1: Vec3) -> bool:
    def orient(p: Vec3, q: Vec3, r: Vec3) -> float:
        return (
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )
    values = (orient(a0[:2], a1[:2], b0[:2]), orient(a0[:2], a1[:2], b1[:2]))
    values += (orient(b0[:2], b1[:2], a0[:2]), orient(b0[:2], b1[:2], a1[:2]))
    return values[0] * values[1] < -EPS and values[2] * values[3] < -EPS


def _loop_2d(
    loop: Sequence[Vec3], origin: Vec3, u_axis: Vec3, v_axis: Vec3
) -> List[Tuple[float, float]]:
    return [
        (
            dot(subtract(point, origin), u_axis),
            dot(subtract(point, origin), v_axis),
        )
        for point in loop
    ]


def _ear_triangulate(points: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    """Triangulate a possibly self-touching polygon (hole bridges repeat a
    vertex).  A candidate ear is accepted only when its centroid is inside the
    polygon by the even-odd rule, which is false for an ear that bridges a
    hole: that keeps an internal void empty instead of triangulating across
    it.
    """
    remaining = list(range(len(points)))
    triangles: List[Tuple[int, int, int]] = []
    guard = 0
    while len(remaining) >= 3 and guard < len(points) ** 2 + 16:
        guard += 1
        ear_found = False
        for position, index in enumerate(remaining):
            previous_index = remaining[position - 1]
            next_index = remaining[(position + 1) % len(remaining)]
            a_point = points[previous_index]
            b_point = points[index]
            c_point = points[next_index]
            cross_value = (
                (b_point[0] - a_point[0]) * (c_point[1] - a_point[1])
                - (b_point[1] - a_point[1]) * (c_point[0] - a_point[0])
            )
            if cross_value <= EPS:
                continue
            centroid2 = (
                (a_point[0] + b_point[0] + c_point[0]) / 3.0,
                (a_point[1] + b_point[1] + c_point[1]) / 3.0,
            )
            if not _point_in_polygon2(centroid2, points):
                continue
            contains_point = False
            for other_index in remaining:
                if other_index in (previous_index, index, next_index):
                    continue
                if _point_in_triangle(points[other_index], a_point, b_point, c_point):
                    contains_point = True
                    break
            if contains_point:
                continue
            triangles.append((previous_index, index, next_index))
            remaining.remove(index)
            ear_found = True
            break
        if not ear_found:
            index = remaining[1]
            triangles.append((remaining[0], index, remaining[2]))
            remaining.remove(index)
    return triangles


def _point_in_polygon2(
    point: Tuple[float, float],
    polygon: Sequence[Tuple[float, float]],
) -> bool:
    inside = False
    for index, a_point in enumerate(polygon):
        b_point = polygon[(index + 1) % len(polygon)]
        if (a_point[1] > point[1]) != (b_point[1] > point[1]):
            crossing = a_point[0] + (point[1] - a_point[1]) * (b_point[0] - a_point[0]) / (
                b_point[1] - a_point[1]
            )
            if crossing > point[0]:
                inside = not inside
    return inside


def _point_in_triangle(
    point: Tuple[float, float],
    a_point: Tuple[float, float],
    b_point: Tuple[float, float],
    c_point: Tuple[float, float],
) -> bool:
    def cross2(p0, p1, p2):
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

    values = (cross2(point, a_point, b_point), cross2(point, b_point, c_point), cross2(point, c_point, a_point))
    return all(value >= -EPS for value in values) or all(value <= EPS for value in values)


def _bridge_hole(
    outer: List[Vec3],
    hole: List[Vec3],
    origin: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    blockers: Sequence[List[Vec3]],
) -> List[Vec3]:
    outer2 = _loop_2d(outer, origin, u_axis, v_axis)
    hole2 = _loop_2d(hole, origin, u_axis, v_axis)

    def visible(outer_index: int, hole_index: int) -> bool:
        start3 = outer[outer_index]
        end3 = hole[hole_index]
        middle = tuple((start3[i] + end3[i]) * 0.5 for i in range(3))
        start2 = outer2[outer_index]
        end2 = hole2[hole_index]
        middle2 = ((start2[0] + end2[0]) * 0.5, (start2[1] + end2[1]) * 0.5)
        if not _point_in_loop(middle, outer, origin, u_axis, v_axis):
            return False
        for blocker in blockers:
            if blocker is hole:
                continue
            if _point_in_loop(middle, blocker, origin, u_axis, v_axis):
                return False
            for i, point in enumerate(blocker):
                following = blocker[(i + 1) % len(blocker)]
                if _segments_cross(start3, end3, point, following):
                    return False
        return True

    for outer_index in range(len(outer)):
        for hole_index in range(len(hole)):
            if visible(outer_index, hole_index):
                # One shared bridge vertex: travel along the hole and return
                # on the same point without a zero-length duplicate, so ear
                # clipping cannot start fanning the interior of the hole.
                return (
                    outer[: outer_index + 1]
                    + hole[hole_index + 1 :]
                    + hole[: hole_index + 1]
                    + outer[outer_index + 1 :]
                )
    raise RuntimeError("could not bridge a section hole without crossing its boundary")


def _loop_area(
    loop: Sequence[Vec3], origin: Vec3, u_axis: Vec3, v_axis: Vec3
) -> float:
    return abs(_signed_area2(loop, origin, u_axis, v_axis))


def _cap_triangles(
    closed_loops: List[List[Vec3]],
    plane: ClipPlane,
) -> List[List[Vec3]]:
    if not closed_loops:
        return []
    origin = min(
        (point for loop in closed_loops for point in loop),
        key=lambda point: (point[2], point[1], point[0]),
    )
    u_axis, v_axis = _plane_frame(plane.normal, origin)
    # Classify loops by containment rather than traversal winding: a loop whose
    # sample point lies inside another loop is a hole, so the walk direction
    # chosen when following cut edges can never decide whether a hole exists.
    outer_loops: List[List[Vec3]] = []
    hole_loops: List[List[Vec3]] = []
    for loop in closed_loops:
        contained = False
        for other in closed_loops:
            if other is loop:
                continue
            if (
                _loop_area(other, origin, u_axis, v_axis)
                > _loop_area(loop, origin, u_axis, v_axis) + EPS
                and _point_in_loop(loop[0], other, origin, u_axis, v_axis)
            ):
                contained = True
                break
        (hole_loops if contained else outer_loops).append(loop)
    outer_loops.sort(key=lambda loop: _loop_area(loop, origin, u_axis, v_axis), reverse=True)

    result: List[List[Vec3]] = []
    for outer in outer_loops:
        if _signed_area2(outer, origin, u_axis, v_axis) < 0.0:
            outer = list(reversed(outer))
        assigned_holes = [
            hole
            for hole in hole_loops
            if _point_in_loop(hole[0], outer, origin, u_axis, v_axis)
        ]
        assigned_holes.sort(
            key=lambda hole: _loop_area(hole, origin, u_axis, v_axis),
            reverse=True,
        )
        boundary = outer
        blockers = list(assigned_holes)
        for hole in assigned_holes:
            oriented_hole = hole
            if _signed_area2(hole, origin, u_axis, v_axis) > 0.0:
                oriented_hole = list(reversed(hole))
            boundary = _bridge_hole(boundary, oriented_hole, origin, u_axis, v_axis, blockers)
        points2 = _loop_2d(boundary, origin, u_axis, v_axis)
        for i, j, k in _ear_triangulate(points2):
            result.append([boundary[i], boundary[j], boundary[k]])
    return result

def _ribbon(chain: List[Vec3], plane: ClipPlane) -> List[List[Vec3]]:
    """A narrow in-plane strip that marks a cut that cannot be closed."""
    if len(chain) < 2:
        return []
    width = RIBBON_WIDTH
    triangles: List[List[Vec3]] = []
    left: List[Vec3] = []
    right: List[Vec3] = []
    for index, point in enumerate(chain):
        if index == 0:
            tangent = subtract(chain[1], point)
        elif index == len(chain) - 1:
            tangent = subtract(point, chain[index - 1])
        else:
            tangent = subtract(chain[index + 1], chain[index - 1])
        tangent_length = math.sqrt(dot(tangent, tangent))
        if tangent_length <= EPS:
            tangent = subtract(chain[index - 1 if index else 1], point)
            tangent_length = math.sqrt(dot(tangent, tangent))
        side = normalize(_cross(plane.normal, _scale(tangent, 1.0 / tangent_length)))
        left.append(_add(point, _scale(side, width * 0.5)))
        right.append(_add(point, _scale(side, -width * 0.5)))
    for index in range(len(chain) - 1):
        triangles.append([left[index], left[index + 1], right[index + 1]])
        triangles.append([left[index], right[index + 1], right[index]])
    return triangles


@dataclass
class SectionResult:
    surfaces: List[Mesh]
    caps: List[Mesh]
    open_boundaries: List[Mesh]


def _section_mesh(
    name: str,
    triangles_xyz: List[List[Vec3]],
    color: RGB,
    opacity: float,
) -> Optional[Mesh]:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    for points in triangles_xyz:
        if any(_distance(points[i], points[j]) <= EPS for i, j in ((0, 1), (1, 2), (2, 0))):
            continue
        offset = len(vertices)
        vertices.extend(points)
        triangles.append(
            Triangle((offset, offset + 1, offset + 2), color, opacity=opacity)
        )
    if not triangles:
        return None
    return Mesh(name, vertices, triangles)


def _single_plane_segments(mesh: Mesh, plane: ClipPlane) -> List[Vec3]:
    """Cut edges produced by one plane as if it were the only plane."""
    original_cut_edges = _original_on_edges(mesh, plane)
    cut_points: List[Vec3] = []
    for triangle in mesh.triangles:
        indices = triangle.indices
        nodes = [
            _Node(mesh.vertices[index], (0.0, 0.0, 0.0), 1.0, (0.0, 0.0), original_index=index)
            for index in indices
        ]
        distances = [plane.signed_distance(node.position) for node in nodes]
        if all(distance < -EPS for distance in distances) or all(
            distance > EPS for distance in distances
        ):
            continue
        polygon = _clip_once(nodes, plane)
        if len(polygon) < 2:
            continue
        coplanar_face = len(polygon) >= 3 and all(_on_plane(node.position, plane) for node in polygon)
        for position, node in enumerate(polygon):
            following = polygon[(position + 1) % len(polygon)]
            if not (_on_plane(node.position, plane) and _on_plane(following.position, plane)):
                continue
            if not coplanar_face and node.original_index >= 0 and following.original_index >= 0:
                key = (
                    min(node.original_index, following.original_index),
                    max(node.original_index, following.original_index),
                )
                if key not in original_cut_edges:
                    continue
            cut_points.extend((node.position, following.position))
    return cut_points


def _clip_closed_loop(loop: List[Vec3], planes: Sequence[ClipPlane]) -> List[List[Vec3]]:
    """Sutherland-Hodgman on a closed 3D polygon; closure edges on later planes
    (cap/cap intersection seams) are generated by the polygon walk itself."""
    active = [loop]
    for plane in planes:
        following_polygons: List[List[Vec3]] = []
        for polygon in active:
            clipped: List[Vec3] = []
            previous = polygon[-1]
            previous_distance = plane.signed_distance(previous)
            for current in polygon:
                current_distance = plane.signed_distance(current)
                if current_distance >= 0.0:
                    if previous_distance < 0.0:
                        t = current_distance / (current_distance - previous_distance)
                        clipped.append(
                            tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                        )
                    clipped.append(current)
                elif previous_distance >= 0.0:
                    t = previous_distance / (previous_distance - current_distance)
                    clipped.append(
                        tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                    )
                previous = current
                previous_distance = current_distance
            if len(clipped) >= 3:
                following_polygons.append(clipped)
        active = following_polygons
    return active


def _clip_open_chain(chain: List[Vec3], planes: Sequence[ClipPlane]) -> List[List[Vec3]]:
    """Keep open polyline pieces; ends remain genuinely un-closeable."""
    active = [chain]
    for plane in planes:
        next_chains: List[List[Vec3]] = []
        for points in active:
            kept: List[Vec3] = []
            previous = points[0]
            previous_distance = plane.signed_distance(previous)
            if previous_distance >= 0.0:
                kept.append(previous)
            for current in points[1:]:
                current_distance = plane.signed_distance(current)
                if current_distance >= 0.0:
                    if previous_distance < 0.0:
                        t = current_distance / (current_distance - previous_distance)
                        kept.append(
                            tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                        )
                    kept.append(current)
                elif previous_distance >= 0.0:
                    t = previous_distance / (previous_distance - current_distance)
                    kept.append(
                        tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                    )
                    if len(kept) >= 2:
                        next_chains.append(kept)
                    kept = []
                previous = current
                previous_distance = current_distance
            if len(kept) >= 2:
                next_chains.append(kept)
        active = next_chains
    return active



@dataclass
class SectionResult:
    surfaces: List[Mesh]
    caps: List[Mesh]
    open_boundaries: List[Mesh]


def _section_mesh(
    name: str,
    triangles_xyz: List[List[Vec3]],
    color: RGB,
    opacity: float,
) -> Optional[Mesh]:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    for points in triangles_xyz:
        if any(_distance(points[i], points[j]) <= EPS for i, j in ((0, 1), (1, 2), (2, 0))):
            continue
        offset = len(vertices)
        vertices.extend(points)
        triangles.append(
            Triangle((offset, offset + 1, offset + 2), color, opacity=opacity)
        )
    if not triangles:
        return None
    return Mesh(name, vertices, triangles)


def _single_plane_segments(mesh: Mesh, plane: ClipPlane) -> List[Vec3]:
    """Cut edges produced by one plane as if it were the only plane."""
    original_cut_edges = _original_on_edges(mesh, plane)
    cut_points: List[Vec3] = []
    for triangle in mesh.triangles:
        indices = triangle.indices
        nodes = [
            _Node(mesh.vertices[index], (0.0, 0.0, 0.0), 1.0, (0.0, 0.0), original_index=index)
            for index in indices
        ]
        distances = [plane.signed_distance(node.position) for node in nodes]
        if all(distance < -EPS for distance in distances) or all(
            distance > EPS for distance in distances
        ):
            continue
        polygon = _clip_once(nodes, plane)
        if len(polygon) < 2:
            continue
        coplanar_face = len(polygon) >= 3 and all(_on_plane(node.position, plane) for node in polygon)
        for position, node in enumerate(polygon):
            following = polygon[(position + 1) % len(polygon)]
            if not (_on_plane(node.position, plane) and _on_plane(following.position, plane)):
                continue
            if not coplanar_face and node.original_index >= 0 and following.original_index >= 0:
                key = (
                    min(node.original_index, following.original_index),
                    max(node.original_index, following.original_index),
                )
                if key not in original_cut_edges:
                    continue
            cut_points.extend((node.position, following.position))
    return cut_points


def _clip_closed_loop(loop: List[Vec3], planes: Sequence[ClipPlane]) -> List[List[Vec3]]:
    """Sutherland-Hodgman on a closed 3D polygon; closure edges on later planes
    (cap/cap intersection seams) are generated by the polygon walk itself."""
    active = [loop]
    for plane in planes:
        following_polygons: List[List[Vec3]] = []
        for polygon in active:
            clipped: List[Vec3] = []
            previous = polygon[-1]
            previous_distance = plane.signed_distance(previous)
            for current in polygon:
                current_distance = plane.signed_distance(current)
                if current_distance >= 0.0:
                    if previous_distance < 0.0:
                        t = current_distance / (current_distance - previous_distance)
                        clipped.append(
                            tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                        )
                    clipped.append(current)
                elif previous_distance >= 0.0:
                    t = previous_distance / (previous_distance - current_distance)
                    clipped.append(
                        tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                    )
                previous = current
                previous_distance = current_distance
            if len(clipped) >= 3:
                following_polygons.append(clipped)
        active = following_polygons
    return active


def _clip_open_chain(chain: List[Vec3], planes: Sequence[ClipPlane]) -> List[List[Vec3]]:
    """Keep open polyline pieces; ends remain genuinely un-closeable."""
    active = [chain]
    for plane in planes:
        next_chains: List[List[Vec3]] = []
        for points in active:
            kept: List[Vec3] = []
            previous = points[0]
            previous_distance = plane.signed_distance(previous)
            if previous_distance >= 0.0:
                kept.append(previous)
            for current in points[1:]:
                current_distance = plane.signed_distance(current)
                if current_distance >= 0.0:
                    if previous_distance < 0.0:
                        t = current_distance / (current_distance - previous_distance)
                        kept.append(
                            tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                        )
                    kept.append(current)
                elif previous_distance >= 0.0:
                    t = previous_distance / (previous_distance - current_distance)
                    kept.append(
                        tuple(previous[i] + (current[i] - previous[i]) * t for i in range(3))
                    )
                    if len(kept) >= 2:
                        next_chains.append(kept)
                    kept = []
                previous = current
                previous_distance = current_distance
            if len(kept) >= 2:
                next_chains.append(kept)
        active = next_chains
    return active


def _loop_to_segments(loops: Sequence[List[Vec3]]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    points: List[Vec3] = []
    for loop in loops:
        offset = len(points)
        points.extend(loop)
        for index in range(len(loop)):
            segments.append((offset + index, offset + (index + 1) % len(loop)))
    return points, segments



@dataclass
class SectionResult:
    surfaces: List[Mesh]
    caps: List[Mesh]
    open_boundaries: List[Mesh]


def _section_mesh(
    name: str,
    triangles_xyz: List[List[Vec3]],
    color: RGB,
    opacity: float,
) -> Optional[Mesh]:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    for points in triangles_xyz:
        if any(_distance(points[i], points[j]) <= EPS for i, j in ((0, 1), (1, 2), (2, 0))):
            continue
        offset = len(vertices)
        vertices.extend(points)
        triangles.append(
            Triangle((offset, offset + 1, offset + 2), color, opacity=opacity)
        )
    if not triangles:
        return None
    return Mesh(name, vertices, triangles)


def _cut_segments_for_plane(
    mesh: Mesh,
    group: ClipGroup,
    plane_index: int,
) -> List[Vec3]:
    plane = group.planes[plane_index]
    if group.canonical_plane(plane_index) is not None:
        # A coincident repeat of an earlier plane must not repaint the cap.
        return []
    original_cut_edges = _original_on_edges(mesh, plane)
    cut_points: List[Vec3] = []

    for triangle in mesh.triangles:
        indices = triangle.indices
        nodes = [
            _Node(
                mesh.vertices[index],
                tuple(float(channel) for channel in triangle.color),
                triangle.opacity,
                triangle.uvs[corner] if triangle.uvs is not None else (0.0, 0.0),
                original_index=index,
            )
            for corner, index in enumerate(indices)
        ]
        distances = [plane.signed_distance(node.position) for node in nodes]
        if all(distance < -EPS for distance in distances):
            continue
        coplanar = all(abs(distance) <= EPS for distance in distances)
        active_plane = any(distance < -EPS for distance in distances) and any(
            distance >= -EPS for distance in distances
        )
        if coplanar and active_plane:
            # The face lies on its own cutting plane; the cap owns that area.
            continue

        polygon = nodes
        for later_plane in group.planes:
            polygon = _clip_once(polygon, later_plane)
            if len(polygon) < 3:
                break
        if len(polygon) < 3:
            continue
        for node in polygon:
            node.on_planes = frozenset(
                id(candidate)
                for candidate in group.planes
                if abs(candidate.signed_distance(node.position)) <= EPS
            )
        plane_id = id(plane)
        for position, node in enumerate(polygon):
            following = polygon[(position + 1) % len(polygon)]
            if plane_id not in node.on_planes or plane_id not in following.on_planes:
                continue
            if node.original_index >= 0 and following.original_index >= 0:
                key = (
                    min(node.original_index, following.original_index),
                    max(node.original_index, following.original_index),
                )
                if key not in original_cut_edges:
                    continue
            cut_points.extend((node.position, following.position))
    return cut_points


@dataclass
class SectionResult:
    surfaces: List[Mesh]
    caps: List[Mesh]
    open_boundaries: List[Mesh]


def _section_mesh(
    name: str,
    triangles_xyz: List[List[Vec3]],
    color: RGB,
    opacity: float,
) -> Optional[Mesh]:
    vertices: List[Vec3] = []
    triangles: List[Triangle] = []
    for points in triangles_xyz:
        if any(_distance(points[i], points[j]) <= EPS for i, j in ((0, 1), (1, 2), (2, 0))):
            continue
        offset = len(vertices)
        vertices.extend(points)
        triangles.append(
            Triangle((offset, offset + 1, offset + 2), color, opacity=opacity)
        )
    if not triangles:
        return None
    return Mesh(name, vertices, triangles)


def _cut_segments_for_plane(
    mesh: Mesh,
    group: ClipGroup,
    plane_index: int,
) -> List[Vec3]:
    plane = group.planes[plane_index]
    if group.canonical_plane(plane_index) is not None:
        # A coincident repeat of an earlier plane must not repaint the cap.
        return []
    original_cut_edges = _original_on_edges(mesh, plane)
    cut_points: List[Vec3] = []

    for triangle in mesh.triangles:
        indices = triangle.indices
        raw_nodes = [
            _Node(
                mesh.vertices[index],
                tuple(float(channel) for channel in triangle.color),
                triangle.opacity,
                triangle.uvs[corner] if triangle.uvs is not None else (0.0, 0.0),
                original_index=index,
            )
            for corner, index in enumerate(indices)
        ]
        distances = [plane.signed_distance(node.position) for node in raw_nodes]
        if all(distance < -EPS for distance in distances):
            continue
        coplanar = all(abs(distance) <= EPS for distance in distances)
        active_plane = any(distance < -EPS for distance in distances) and any(
            distance >= -EPS for distance in distances
        )
        if coplanar and active_plane:
            # The face lies on its own cutting plane; the cap owns that area.
            continue

        polygon = raw_nodes
        for later_index, later_plane in enumerate(group.planes):
            polygon = _clip_once(polygon, later_plane)
            if len(polygon) < 3:
                break
        if len(polygon) < 3:
            continue

        for position, node in enumerate(polygon):
            following = polygon[(position + 1) % len(polygon)]
            if not (_on_plane(node.position, plane) and _on_plane(following.position, plane)):
                continue
            created_here = not node.original and not following.original
            follows_existing_cut = False
            if node.original and following.original:
                key = (
                    min(node.original_index, following.original_index),
                    max(node.original_index, following.original_index),
                )
                follows_existing_cut = key in original_cut_edges
            if created_here or follows_existing_cut:
                cut_points.extend((node.position, following.position))
    return cut_points


def build_section(mesh: Mesh, group: ClipGroup) -> SectionResult:
    """Return the clipped surface, per-plane caps, and open-boundary markers.

    The input mesh is read only.  Section geometry is newly allocated, so
    saving/rendering a clipped scene never rewrites the source model.
    """
    surface_vertices: List[Vec3] = []
    surface_triangles: List[Triangle] = []
    caps: List[Mesh] = []
    open_boundaries: List[Mesh] = []

    for triangle in mesh.triangles:
        nodes = [
            _Node(
                mesh.vertices[index],
                tuple(float(channel) for channel in triangle.color),
                triangle.opacity,
                triangle.uvs[corner] if triangle.uvs is not None else (0.0, 0.0),
                original_index=index,
            )
            for corner, index in enumerate(triangle.indices)
        ]
        polygon = nodes
        for active in group.planes:
            polygon = _clip_once(polygon, active)
            if len(polygon) < 3:
                break
        if len(polygon) < 3:
            continue
        polygon = _canonical_order(polygon, group.planes[0])
        for position in range(1, len(polygon) - 1):
            piece = [polygon[0], polygon[position], polygon[position + 1]]
            if any(
                _distance(piece[i].position, piece[j].position) <= EPS
                for i, j in ((0, 1), (1, 2), (2, 0))
            ):
                continue
            offset = len(surface_vertices)
            surface_vertices.extend(node.position for node in piece)
            if triangle.texture_name is not None:
                surface_triangles.append(
                    Triangle(
                        (offset, offset + 1, offset + 2),
                        tuple(int(round(channel)) for channel in piece[0].color),
                        tuple(node.uv for node in piece),
                        triangle.texture_name,
                        triangle.opacity,
                    )
                )
            else:
                surface_triangles.append(
                    Triangle(
                        (offset, offset + 1, offset + 2),
                        triangle.color,
                        opacity=triangle.opacity,
                    )
                )

    surfaces = []
    if surface_triangles:
        clipped_mesh = Mesh(
            f"{mesh.name}#clipped:{group.name}",
            surface_vertices,
            surface_triangles,
            textures=mesh.textures,
        )
        surfaces.append(clipped_mesh)

    for plane_index, plane in enumerate(group.planes):
        if group.canonical_plane(plane_index) is not None:
            # A coincident repeat of an earlier plane must not repaint the cap.
            continue
        cut_points = _single_plane_segments(mesh, plane)
        if not cut_points:
            continue
        welded, labels = _weld(cut_points)
        raw_segments = [
            (labels[index], labels[index + 1]) for index in range(0, len(labels), 2)
        ]
        other_planes = [candidate for index, candidate in enumerate(group.planes) if index != plane_index]
        closed_loops: List[List[Vec3]] = []
        open_chains: List[List[Vec3]] = []
        for chain, is_closed in _walk_components(raw_segments):
            points = [welded[index] for index in chain]
            if is_closed:
                loop_points = points[:-1]
                closed_loops.extend(_clip_closed_loop(loop_points, other_planes))
            else:
                for kept in _clip_open_chain(points, other_planes):
                    open_chains.append(kept)

        # Closed polygons generated by the intersection with other planes are
        # already closed; walk them only to normalize vertex order. No open
        # boundary is synthesized along a plane/plane seam.
        cap_mesh = _section_mesh(
            f"{mesh.name}#section:{group.name}/{plane.name}",
            _cap_triangles(closed_loops, plane),
            plane.color,
            plane.opacity,
        )
        if cap_mesh is not None:
            caps.append(cap_mesh)

        ribbons: List[List[Vec3]] = []
        for chain in open_chains:
            ribbons.extend(_ribbon(chain, plane))
        open_mesh = _section_mesh(
            f"{mesh.name}#open:{group.name}/{plane.name}",
            ribbons,
            OPEN_COLOR,
            1.0,
        )
        if open_mesh is not None:
            open_boundaries.append(open_mesh)


    return SectionResult(surfaces, caps, open_boundaries)


def build_clipped_scene_meshes(meshes: Sequence[Mesh], group: ClipGroup) -> List[Mesh]:
    built: List[Mesh] = []
    for mesh in meshes:
        result = build_section(mesh, group)
        built.extend(result.surfaces)
        built.extend(result.caps)
        built.extend(result.open_boundaries)
    return built
