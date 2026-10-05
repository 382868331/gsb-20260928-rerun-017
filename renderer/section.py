from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .geometry import Mesh, Triangle, Vec3
from .scene import Scene

EPS = 1.0e-9
WELD_EPS = 1.0e-5


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v):
    return math.sqrt(_dot(v, v))


def _normalized(v):
    length = _norm(v)
    return (v[0] / length, v[1] / length, v[2] / length)


@dataclass(frozen=True)
class ClipPlane:
    """Half space n.p + d >= 0 is retained. Coincident planes are equal
    regardless of equation scaling/sign after canonical normalization."""

    name: str
    normal: Tuple[float, float, float]
    d: float

    def __post_init__(self):
        length = _norm(self.normal)
        if length <= 1.0e-12:
            raise ValueError("clip plane normal must be non-zero")
        n = tuple(component / length for component in self.normal)
        dd = self.d / length
        object.__setattr__(self, "normal", n)
        object.__setattr__(self, "d", dd)

    def distance(self, p) -> float:
        return _dot(self.normal, p) + self.d

    @staticmethod
    def through(name: str, normal, point) -> "ClipPlane":
        n = _normalized(normal)
        return ClipPlane(name, n, -_dot(n, point))

    def key(self):
        return tuple(round(component, 7) for component in self.normal) + (round(self.d, 7),)

    def equivalent(self, other: "ClipPlane") -> bool:
        """Two planes describe the same geometric plane when their equations
        are equal up to a common sign. The retained half space is *not*
        normalized away: this test is used only to remove a second plane
        whose equation is exactly redundant in the same orientation."""
        tolerance = 1.0e-6
        same = all(abs(a - b) <= tolerance for a, b in zip(self.normal, other.normal)) and abs(
            self.d - other.d
        ) <= tolerance
        opposite = all(abs(a + b) <= tolerance for a, b in zip(self.normal, other.normal)) and abs(
            self.d + other.d
        ) <= tolerance
        return same or opposite

    def frame(self):
        """Orthonormal (u, v) in the plane; normal = u x v."""
        n = self.normal
        helper = (0.0, 0.0, 1.0) if abs(n[2]) < 0.9 else (0.0, 1.0, 0.0)
        u = _normalized(_cross(helper, n))
        v = _cross(n, u)
        return u, v

    def project(self, p):
        u, v = self.frame()
        rel = _add(p, _scale(self.normal, self.d))
        return (_dot(rel, u), _dot(rel, v))

    def to_dict(self):
        return {"name": self.name, "normal": list(self.normal), "d": self.d}

    @classmethod
    def from_dict(cls, data):
        return cls(data["name"], tuple(data["normal"]), float(data["d"]))


@dataclass
class ClipGroup:
    """A named set of planes applied jointly. Each group keeps its own copy
    of the planes and its own palette/opacity, so duplicating a group and
    moving one plane never mutates the other group."""

    name: str
    planes: List[ClipPlane] = field(default_factory=list)
    palette: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 120, 60), (90, 200, 255), (255, 220, 90), (170, 120, 255),
    ])
    shell_opacity: float = 0.25
    cap_opacity: float = 1.0

    def __post_init__(self):
        if not 0.0 < self.shell_opacity <= 1.0:
            raise ValueError("shell_opacity must be in (0, 1]")
        if not 0.0 < self.cap_opacity <= 1.0:
            raise ValueError("cap_opacity must be in (0, 1]")
        unique = []
        for plane in self.planes:
            if any(plane.equivalent(existing) for existing in unique):
                continue
            unique.append(plane)
        self.planes = unique
        if not self.planes:
            raise ValueError("clip group needs at least one unique plane")

    def copy_with(self, name=None, planes=None) -> "ClipGroup":
        return ClipGroup(
            name=name if name is not None else self.name,
            planes=list(planes if planes is not None else self.planes),
            palette=list(self.palette),
            shell_opacity=self.shell_opacity,
            cap_opacity=self.cap_opacity,
        )

    def to_dict(self):
        return {
            "name": self.name,
            "planes": [plane.to_dict() for plane in self.planes],
            "palette": [list(color) for color in self.palette],
            "shell_opacity": self.shell_opacity,
            "cap_opacity": self.cap_opacity,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            planes=[ClipPlane.from_dict(item) for item in data["planes"]],
            palette=[tuple(color) for color in data.get("palette", [])],
            shell_opacity=float(data.get("shell_opacity", 0.25)),
            cap_opacity=float(data.get("cap_opacity", 1.0)),
        )


# ---------------------------------------------------------------------------
# World-space clipping
# ---------------------------------------------------------------------------

# A clip vertex is (position, tag) where tag is None, a non-negative plane
# index for a genuine in/out intersection, or ~index (negative minus one) for
# an original vertex that merely lies on the plane. Only intersections form
# section caps; on-plane vertices belong to the model's own surface and must
# never create or fill section geometry.
CVertex = Tuple[Vec3, Optional[int]]


def _clip_polygon(polygon: List[CVertex], plane: ClipPlane, tag: int) -> List[CVertex]:
    """Sutherland-Hodgman against one half space. Every point that lies on
    this plane is tagged with this plane's index, including original
    vertices that happen to sit exactly on the cut."""
    result: List[CVertex] = []
    if not polygon:
        return result
    prev = polygon[-1]
    prev_dist = plane.distance(prev[0])
    prev_inside = prev_dist >= -EPS
    for current in polygon:
        cur_dist = plane.distance(current[0])
        cur_inside = cur_dist >= -EPS
        if cur_inside:
            if not prev_inside:
                denom = prev_dist - cur_dist
                t = prev_dist / denom if denom != 0.0 else 0.0
                t = min(1.0, max(0.0, t))
                hit = _add(prev[0], _scale(_sub(current[0], prev[0]), t))
                result.append((hit, tag))
            point, _old_tag = current
            if abs(cur_dist) <= WELD_EPS:
                result.append((point, ~tag))
            else:
                result.append(current)
        elif prev_inside:
            denom = prev_dist - cur_dist
            t = prev_dist / denom if denom != 0.0 else 0.0
            t = min(1.0, max(0.0, t))
            hit = _add(prev[0], _scale(_sub(current[0], prev[0]), t))
            result.append((hit, tag))
        prev = current
        prev_dist = cur_dist
        prev_inside = cur_inside
    # Drop consecutive duplicated points (a vertex landing on the plane can
    # otherwise be emitted twice).
    deduped: List[CVertex] = []
    for vertex in result:
        if deduped and _norm(_sub(deduped[-1][0], vertex[0])) < WELD_EPS:
            continue
        deduped.append(vertex)
    if len(deduped) >= 2 and _norm(_sub(deduped[0][0], deduped[-1][0])) < WELD_EPS:
        deduped.pop()
    return deduped


@dataclass
class ClipResult:
    polygons: List[List[CVertex]]
    discarded: int


def clip_triangle_vertices(
    vertices: Sequence[Vec3], planes: Sequence[ClipPlane]
) -> ClipResult:
    polygons: List[List[CVertex]] = [[(tuple(v), None) for v in vertices]]
    for index, plane in enumerate(planes):
        next_polygons: List[List[CVertex]] = []
        for polygon in polygons:
            clipped = _clip_polygon(polygon, plane, index)
            if len(clipped) >= 3:
                next_polygons.append(clipped)
        polygons = next_polygons
        if not polygons:
            return ClipResult([], 1)
    return ClipResult(polygons, 0)


# ---------------------------------------------------------------------------
# Planar polygons with holes (ear clipping via bridges)
# ---------------------------------------------------------------------------


def _polygon_area2(ring: Sequence[Tuple[float, float]]) -> float:
    area = 0.0
    for i, point in enumerate(ring):
        nxt = ring[(i + 1) % len(ring)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return area


def _point_in_ring(point, ring) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > point[1]) != (yj > point[1]):
            x_cross = xi + (xj - xi) * (point[1] - yi) / (yj - yi)
            if point[0] < x_cross:
                inside = not inside
        j = i
    return inside


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def _ear_clip_indices(ring: List[Tuple[float, float]]) -> List[int]:
    n = len(ring)
    if n < 3:
        return []
    indices = _drop_collinear(ring, list(range(n)))
    if len(indices) < 3:
        return []
    triangles: List[int] = []
    guard = 0
    expected = len(indices) - 2
    while len(indices) > 2 and guard < expected * 4 + 10:
        guard += 1
        clipped = False
        for pos in range(len(indices)):
            i_prev = indices[(pos - 1) % len(indices)]
            i_cur = indices[pos]
            i_next = indices[(pos + 1) % len(indices)]
            a, b, c = ring[i_prev], ring[i_cur], ring[i_next]
            cross2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross2 <= 1e-12:
                continue
            mid = ((b[0] + a[0] + c[0]) / 3.0, (b[1] + a[1] + c[1]) / 3.0)
            if not _point_in_ring(mid, [ring[index] for index in indices]):
                continue
            blocked = False
            for q in range(len(indices)):
                p1 = ring[indices[q]]
                p2 = ring[indices[(q + 1) % len(indices)]]
                if p1 in (a, b) or p2 in (a, b) or p1 in (b, c) or p2 in (b, c):
                    continue
                if _segments_intersect(a, c, p1, p2):
                    blocked = True
                    break
            if blocked:
                continue
            triangles.extend((i_prev, i_cur, i_next))
            del indices[pos]
            clipped = True
            break
        if not clipped:
            return []
    if len(triangles) // 3 != expected:
        return []
    return triangles


def _drop_collinear(ring, indices):
    # Remove vertices lying on the segment between their neighbours; they
    # carry no outline information and would stall the ear clipper.
    idx = list(indices)
    changed = True
    while changed and len(idx) > 3:
        changed = False
        for pos in range(len(idx)):
            a = ring[idx[(pos - 1) % len(idx)]]
            b = ring[idx[pos]]
            c = ring[idx[(pos + 1) % len(idx)]]
            cross2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            lab = math.hypot(b[0] - a[0], b[1] - a[1])
            lbc = math.hypot(c[0] - b[0], c[1] - b[1])
            if lab <= 0.0 or lbc <= 0.0:
                continue
            if abs(cross2) > 1.0e-9 * lab * lbc:
                continue
            # Only drop genuine between-points, never spike tips.
            if (b[0] - a[0]) * (c[0] - b[0]) + (b[1] - a[1]) * (c[1] - b[1]) >= 0.0:
                del idx[pos]
                changed = True
                break
    return idx


def triangulate_2d_with_holes(
    outer: List[Tuple[float, float]], holes: List[List[Tuple[float, float]]]
) -> List[Tuple[int, int, int]]:
    """Triangulate an outer contour keeping interior holes empty.

    The outer ring is ear-clipped. An ear is admissible only when its
    centroid is outside every hole and its ear diagonal does not cross a
    hole boundary, so real holes are never filled even under multi-plane
    cuts. Returns global indices into [outer, *holes]."""
    outer = list(outer)
    if _polygon_area2(outer) < 0.0:
        outer.reverse()
    normalized_holes = []
    for hole in holes:
        hole = list(hole)
        if _polygon_area2(hole) > 0.0:
            hole.reverse()
        normalized_holes.append(hole)

    if not normalized_holes:
        count = len(outer)
        if count < 3:
            return []
        # Ear-clip the outer ring: a plain (i, i+1, i+2) walk is not a fan
        # and both overlaps and leaves gaps on anything but a triangle.
        flat = _ear_clip_indices(outer)
        return [(flat[i], flat[i + 1], flat[i + 2]) for i in range(0, len(flat), 3)]

    # Fold each hole into the outer ring with a zero-width bridge: from the
    # anchor vertex out to the hole's leftmost vertex, once around the hole,
    # and straight back to the anchor. Anchor and hole entry point appear
    # twice, so the result is one simple ring that the ear clipper can walk;
    # the channel itself has zero area and never fills the hole. Multiple
    # holes are folded in one at a time against the growing ring.
    ring2: List[Tuple[float, float]] = list(outer)
    # ring index -> global combined index
    ring_global: List[int] = list(range(len(outer)))
    hole_base = len(outer)
    for hole_number, hole in enumerate(normalized_holes):
        h_local = min(range(len(hole)), key=lambda i: (hole[i][0], hole[i][1]))
        h = hole[h_local]
        candidate = min(
            range(len(ring2)),
            key=lambda i: (h[0] - ring2[i][0]) ** 2 + (h[1] - ring2[i][1]) ** 2
            if ring2[i][0] <= h[0] + 1e-9
            else 1e30,
        )
        base = hole_base + sum(len(x) for x in normalized_holes[:hole_number])
        ordered_local = [(h_local + k) % len(hole) for k in range(len(hole))]
        insert_points = [hole[index] for index in ordered_local]
        insert_points += [hole[h_local], ring2[candidate]]
        insert_globals = [base + index for index in ordered_local]
        insert_globals += [base + h_local, ring_global[candidate]]
        ring2[candidate + 1 : candidate + 1] = insert_points
        ring_global[candidate + 1 : candidate + 1] = insert_globals

    tris = _ear_clip_indices(ring2)
    accepted: List[Tuple[int, int, int]] = []
    for a, b, c in (tris[i : i + 3] for i in range(0, len(tris), 3)):
        area2 = (
            (ring2[b][0] - ring2[a][0]) * (ring2[c][1] - ring2[a][1])
            - (ring2[b][1] - ring2[a][1]) * (ring2[c][0] - ring2[a][0])
        )
        if abs(area2) <= 1.0e-12:
            continue
        centroid = (
            (ring2[a][0] + ring2[b][0] + ring2[c][0]) / 3.0,
            (ring2[a][1] + ring2[b][1] + ring2[c][1]) / 3.0,
        )
        if not _point_in_ring(centroid, outer):
            continue
        if any(_point_in_ring(centroid, hole) for hole in normalized_holes):
            continue
        accepted.append((ring_global[a], ring_global[b], ring_global[c]))
    return accepted


# ---------------------------------------------------------------------------
# Welding cut chords into loops / open chains, and building cap meshes
# ---------------------------------------------------------------------------


@dataclass
class CapRecord:
    mesh: str
    plane: str
    closed: bool
    loops: int
    holes: int
    area: float
    open_segments: int = 0

    def to_dict(self):
        return {
            "mesh": self.mesh,
            "plane": self.plane,
            "closed": self.closed,
            "loops": self.loops,
            "holes": self.holes,
            "area": round(self.area, 7),
            "open_segments": self.open_segments,
        }


@dataclass
class SectionReport:
    group: str
    caps: List[CapRecord] = field(default_factory=list)
    open_boundaries: List[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "group": self.group,
            "caps": [cap.to_dict() for cap in self.caps],
            "open_boundaries": list(self.open_boundaries),
        }


def weld_chords(
    segments: List[Tuple[Vec3, Vec3]], plane: ClipPlane
) -> Tuple[List[List[Vec3]], List[Tuple[Vec3, Vec3]]]:
    """Merge undirected segments into closed loops and open chains.

    Vertices are welded inside WELD_EPS. Duplicate segments are removed, so
    two coincident faces never leave double fill, while touching chords from
    *different meshes* are fed in separately and therefore never stitch
    different entities into one face. Branch points (more than two incident
    segments) are handled per 2D turn angle so the true outline is walked.
    """
    points: List[Vec3] = []
    xy: List[Tuple[float, float]] = []

    def point_index(p):
        for i, q in enumerate(points):
            if _norm(_sub(p, q)) < WELD_EPS:
                return i
        points.append(p)
        xy.append(plane.project(p))
        return len(points) - 1

    edges = set()
    for a, b in segments:
        if _norm(_sub(a, b)) < WELD_EPS:
            continue
        ia, ib = point_index(a), point_index(b)
        if ia == ib:
            continue
        edges.add((min(ia, ib), max(ia, ib)))
    adjacency: Dict[int, List[int]] = {i: [] for i in range(len(points))}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    used: set = set()
    loops: List[List[Vec3]] = []
    open_chains: List[Tuple[Vec3, Vec3]] = []

    def angle_to(current, previous, nxt):
        v1 = (xy[current][0] - xy[previous][0], xy[current][1] - xy[previous][1])
        v2 = (xy[nxt][0] - xy[current][0], xy[nxt][1] - xy[current][1])
        cross2 = v1[0] * v2[1] - v1[1] * v2[0]
        dot2 = v1[0] * v2[0] + v1[1] * v2[1]
        return math.atan2(cross2, dot2)

    # Open chains first (odd/degree-1 endpoints).
    endpoints = [i for i, nbrs in adjacency.items() if len(nbrs) == 1]
    for start in endpoints:
        if not adjacency[start]:
            continue
        previous = start
        current = adjacency[start][0]
        edge_key = (min(previous, current), max(previous, current))
        if edge_key in used:
            continue
        chain = [points[previous], points[current]]
        used.add(edge_key)
        while len(adjacency[current]) >= 2:
            choices = [n for n in adjacency[current] if (min(current, n), max(current, n)) not in used]
            if not choices:
                break
            nxt = min(choices, key=lambda n: angle_to(current, previous, n))
            used.add((min(current, nxt), max(current, nxt)))
            previous, current = current, nxt
            chain.append(points[current])
        open_chains.append((chain[0], chain[-1]))

    for edge in sorted(edges):
        if edge in used:
            continue
        a, b = edge
        loop_indices = [a, b]
        used.add(edge)
        previous, current = a, b
        ok = True
        while current != a:
            choices = [n for n in adjacency[current] if (min(current, n), max(current, n)) not in used]
            if not choices:
                ok = False
                break
            nxt = min(choices, key=lambda n: angle_to(current, previous, n))
            used.add((min(current, nxt), max(current, nxt)))
            previous, current = current, nxt
            loop_indices.append(current)
        if ok and len(loop_indices) >= 3:
            loops.append([points[i] for i in loop_indices[:-1]])
    return loops, open_chains


def stitch_plane_junctions(
    chords_by_plane: Dict[int, List[Tuple[Vec3, Vec3]]],
    planes: Sequence[ClipPlane],
) -> Dict[int, List[Tuple[Vec3, Vec3]]]:
    """Close section caps where cut planes meet.

    When a plane's chords end on another cut plane, the cap boundary on the
    intersection line is a straight segment between pairs of such open
    endpoints. Each component of the section intersects the line in two
    points (an outer boundary) or zero (a hole that never reaches the cut);
    pairing endpoints along the line by nearest neighbour therefore closes
    each boundary exactly once and never bridges across an interior hole.
    The added seam lies on both planes, but is appended to only the plane
    whose cap it bounds, so the two caps never duplicate the same fill.
    """
    from collections import Counter

    stitched: Dict[int, List[Tuple[Vec3, Vec3]]] = {k: list(v) for k, v in chords_by_plane.items()}
    if len(planes) < 2:
        return stitched

    def open_endpoints(segments):
        points: List[Vec3] = []

        def pid(p):
            for i, q in enumerate(points):
                if _norm(_sub(p, q)) < WELD_EPS:
                    return i
            points.append(p)
            return len(points) - 1

        degree = Counter()
        for a, b in segments:
            degree[pid(a)] += 1
            degree[pid(b)] += 1
        return [points[i] for i, d in degree.items() if d == 1]

    # Weld each plane's chords first, then close every open chain whose two
    # endpoints both terminate on the same other cut plane. Chains that
    # stay open belong to an unclosed (open-surface) model.
    def close_with_segments(plane: ClipPlane, segments, other_planes) -> List[Tuple[Vec3, Vec3]]:
        _loops, chains = weld_chords(segments, plane)
        added = []
        for a, b in chains:
            terminator = None
            for other in other_planes:
                if abs(other.distance(a)) <= WELD_EPS and abs(other.distance(b)) <= WELD_EPS:
                    terminator = other
                    break
            if terminator is None:
                continue
            added.append((a, b))
        return added

    for ai, plane in enumerate(planes):
        others = [other for bi, other in enumerate(planes) if bi != ai]
        seams = close_with_segments(plane, stitched[ai], others)
        stitched[ai].extend(seams)
    return stitched


def _hsv(h, s, v):
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i %= 6
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return (int(r * 255), int(g * 255), int(b * 255))


OPEN_COLOR = (255, 40, 200)


def _cap_color(group: "ClipGroup", plane_index: int, mesh_index: int, mesh_count: int):
    # Hue spreads planes across the group palette's hue range; mesh index adds
    # a small deterministic lightness-free hue offset so two objects cut by
    # the same plane are attributable, while the plane family still reads as
    # one color. Saturation/value stay fixed so caps never mimic surfaces.
    base = group.palette[plane_index % max(1, len(group.palette))]
    import colorsys

    r, g, b = (component / 255.0 for component in base)
    hh, ss, vv = colorsys.rgb_to_hsv(r, g, b)
    offset = (mesh_index / mesh_count) * 0.08 if mesh_count > 1 else 0.0
    hh = (hh + offset) % 1.0
    return _hsv(hh, max(0.55, ss), max(0.6, vv))


def _ribbon_mesh(name, a, b, width, color):
    """A thin quad ribbon along an open 3D segment, marking a boundary that
    could not be capped."""
    direction = _sub(b, a)
    length = _norm(direction)
    if length < EPS:
        return None
    w = _normalized(direction)
    helper = (0.0, 0.0, 1.0) if abs(w[2]) < 0.9 else (0.0, 1.0, 0.0)
    side = _normalized(_cross(helper, w))
    side = _scale(side, width * 0.5)
    corners = [_add(a, side), _add(b, side), _add(b, _scale(side, -1.0)), _add(a, _scale(side, -1.0))]
    return Mesh.quad(name, corners, color, opacity=1.0)


def apply_clip_group(scene: Scene, group: ClipGroup) -> Tuple[Scene, SectionReport]:
    """Return a new Scene in which the group's planes act jointly on every
    mesh, plus a report describing every cap. The source scene and its meshes
    are never mutated."""
    report = SectionReport(group.name)
    result_meshes: List[Mesh] = []
    mesh_count = len(scene.meshes)

    for mesh_index, mesh in enumerate(scene.meshes):
        shell_vertices: List[Vec3] = []
        shell_triangles: List[Triangle] = []
        # Section chords per cut plane. After welding, open endpoints that
        # lie on the intersection line of two cut planes are stitched
        # together with a seam segment so jointly acting planes close their
        # caps; genuinely open surfaces keep unpaired open endpoints.
        chords: Dict[int, List[Tuple[Vec3, Vec3]]] = {i: [] for i in range(len(group.planes))}
        discarded = 0

        for triangle in mesh.triangles:
            source = [mesh.vertices[index] for index in triangle.indices]
            clip_result = clip_triangle_vertices(source, group.planes)
            discarded += clip_result.discarded
            for polygon in clip_result.polygons:
                base = len(shell_vertices)
                shell_vertices.extend(vertex for vertex, _tag in polygon)
                for i in range(1, len(polygon) - 1):
                    shell_triangles.append(
                        Triangle(
                            (base, base + i, base + i + 1),
                            triangle.color,
                            opacity=group.shell_opacity,
                        )
                    )
                for i, (point, tag) in enumerate(polygon):
                    nxt_point, nxt_tag = polygon[(i + 1) % len(polygon)]
                    for cut_index, cut_plane in enumerate(group.planes):
                        if abs(cut_plane.distance(point)) > WELD_EPS:
                            continue
                        if abs(cut_plane.distance(nxt_point)) > WELD_EPS:
                            continue
                        genuine = (tag is not None and tag >= 0) or (nxt_tag is not None and nxt_tag >= 0)
                        merely_coplanar = tag == ~cut_index and nxt_tag == ~cut_index
                        if genuine and not merely_coplanar:
                            chords[cut_index].append((point, nxt_point))

        if shell_vertices:
            result_meshes.append(
                Mesh(
                    f"{mesh.name}::shell::{group.name}",
                    shell_vertices,
                    shell_triangles,
                )
            )

        stitched = stitch_plane_junctions(chords, group.planes)
        for plane_index, plane in enumerate(group.planes):
            loops, open_ends = weld_chords(stitched[plane_index], plane)
            if not loops and not open_ends:
                continue
            color = _cap_color(group, plane_index, mesh_index, mesh_count)
            cap_vertices: List[Vec3] = []
            cap_triangles: List[Triangle] = []
            hole_count = 0
            total_area = 0.0
            closed_loops = 0

            if loops:
                projected = [[plane.project(p) for p in loop] for loop in loops]
                # Group loops into outer contours and their holes. A loop is
                # an outer contour if no other loop contains its centroid;
                # otherwise it belongs to the smallest containing outer.
                areas = [abs(_polygon_area2(ring)) for ring in projected]
                centroids = [
                    (sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring))
                    for ring in projected
                ]
                outers: List[int] = []
                for i in range(len(loops)):
                    containers = [
                        j
                        for j in range(len(loops))
                        if j != i and areas[j] > areas[i] and _point_in_ring(centroids[i], projected[j])
                    ]
                    if not containers:
                        outers.append(i)
                # Map each non-outer loop to its smallest containing outer.
                holes_of: Dict[int, List[int]] = {i: [] for i in outers}
                for i in range(len(loops)):
                    if i in outers:
                        continue
                    parent = min(
                        (j for j in outers if _point_in_ring(centroids[i], projected[j])),
                        key=lambda j: areas[j],
                    )
                    holes_of[parent].append(i)
                for outer_index in outers:
                    outer_2d = projected[outer_index]
                    if _polygon_area2(outer_2d) < 0.0:
                        outer_2d = list(reversed(outer_2d))
                        loops[outer_index] = list(reversed(loops[outer_index]))
                    holes_2d: List[List[Tuple[float, float]]] = []
                    hole_3d: List[List[Vec3]] = []
                    for i in holes_of[outer_index]:
                        ring_2d = projected[i]
                        if _polygon_area2(ring_2d) > 0.0:
                            ring_2d = list(reversed(ring_2d))
                            loops[i] = list(reversed(loops[i]))
                        holes_2d.append(ring_2d)
                        hole_3d.append(loops[i])
                    tris_2d = triangulate_2d_with_holes(outer_2d, holes_2d)
                    ordered_3d = [loops[outer_index]] + hole_3d
                    base_index = len(cap_vertices)
                    for ring in ordered_3d:
                        cap_vertices.extend(ring)
                    # triangulate indices are local to [outer, *holes].
                    local_offset = [0]
                    for ring in ordered_3d:
                        local_offset.append(local_offset[-1] + len(ring))
                    for tri_index, tris in enumerate(tris_2d):
                        a, b, c = tris
                        cap_triangles.append(
                            Triangle((base_index + a, base_index + b, base_index + c), color, opacity=group.cap_opacity)
                        )
                    hole_count += len(holes_2d)
                    total_area += abs(_polygon_area2(outer_2d)) * 0.5 - sum(
                        abs(_polygon_area2(hole)) * 0.5 for hole in holes_2d
                    )
                closed_loops = len(outers)

            # Caps from different meshes are always emitted as separate
            # meshes, even when they touch, so distinct entities never fuse.
            if cap_triangles:
                result_meshes.append(
                    Mesh(
                        f"{mesh.name}::cap::{group.name}::{plane.name}",
                        cap_vertices,
                        cap_triangles,
                    )
                )
            open_ribbons: List[Mesh] = []
            for endpoint_index, (a, b) in enumerate(open_ends):
                ribbon = _ribbon_mesh(
                    f"{mesh.name}::open::{group.name}::{plane.name}::{endpoint_index}",
                    a,
                    b,
                    0.03,
                    OPEN_COLOR,
                )
                if ribbon is not None:
                    open_ribbons.append(ribbon)
                    result_meshes.append(ribbon)
            if cap_triangles:
                report.caps.append(
                    CapRecord(
                        mesh=mesh.name,
                        plane=plane.name,
                        closed=True,
                        loops=closed_loops,
                        holes=hole_count,
                        area=total_area,
                        open_segments=len(open_ribbons),
                    )
                )
            if open_ribbons:
                report.open_boundaries.append(
                    {
                        "mesh": mesh.name,
                        "plane": plane.name,
                        "segments": len(open_ribbons),
                    }
                )
                report.caps.append(
                    CapRecord(
                        mesh=mesh.name,
                        plane=plane.name,
                        closed=False,
                        loops=0,
                        holes=0,
                        area=0.0,
                        open_segments=len(open_ribbons),
                    )
                )

    derived = Scene(
        name=f"{scene.name}@{group.name}",
        meshes=result_meshes,
        cameras=scene.cameras,
        background=scene.background,
    )
    return derived, report


def section_signature(report: SectionReport) -> str:
    """Triangulation-independent fingerprint of a cut: it depends only on
    object/plane attribution, closed-loop count, holes, and section area,
    never on how the original surface or the cap was triangulated."""
    rows = []
    for cap in report.caps:
        rows.append((cap.mesh, cap.plane, cap.closed, cap.loops, cap.holes, round(cap.area, 5)))
    rows.sort()
    digest = hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()
    return digest[:16]
