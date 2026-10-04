from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .camera import Camera
from .geometry import Mesh
from .image import Image
from .scene import Scene
from .texture import Texture
from .section import ClipGroup, build_clipped_scene_meshes


@dataclass(frozen=True)
class RenderOptions:
    width: int = 240
    height: int = 180
    camera_name: Optional[str] = None
    clip_group_name: Optional[str] = None


# Clip vertex layout:
# 0..3: clip x, y, z, w (w is camera-space depth)
# 5..7: color / w; 8..9: texture u,v / w; 10: opacity / w
ClipVertex = List[float]


def _clip_vertex(
    camera: Camera,
    position: Tuple[float, float, float],
    color: Tuple[float, float, float],
    aspect_ratio: float,
    u: float = 0.0,
    v: float = 0.0,
    opacity: float = 1.0,
) -> ClipVertex:
    x, y, z = camera.view_position(position)
    aspect_scale = math.tan(math.radians(camera.fov_y_degrees) * 0.5)
    near, far = camera.near, camera.far
    depth_scale = (far + near) / (far - near)
    depth_offset = -2.0 * far * near / (far - near)
    inv_depth = 1.0 / z
    return [
        x / (aspect_scale * aspect_ratio),
        y / aspect_scale,
        z * depth_scale + depth_offset,
        z,
        z,
        color[0] * inv_depth,
        color[1] * inv_depth,
        color[2] * inv_depth,
        u * inv_depth,
        v * inv_depth,
        opacity * inv_depth,
    ]


_CLIP_TESTS = (
    lambda v: v[3] + v[0],
    lambda v: v[3] - v[0],
    lambda v: v[3] + v[1],
    lambda v: v[3] - v[1],
    lambda v: v[2] + v[3],
    lambda v: v[3] - v[2],
)


def _interpolate(a: ClipVertex, b: ClipVertex, t: float) -> ClipVertex:
    result = [a_value + (b_value - a_value) * t for a_value, b_value in zip(a[:5], b[:5])]
    # Slice 5+ stores attribute / w. The clip factor t is linear in clip
    # space, so recover the plain attribute, interpolate it, then divide by
    # the new w; linearly blending attribute / w here overshoots on a near
    # cut between a near and a far vertex.
    new_w = result[3]
    for index in range(5, 11):
        attribute_a = a[index] * a[3]
        attribute_b = b[index] * b[3]
        attribute = attribute_a + (attribute_b - attribute_a) * t
        result.append(attribute / new_w)
    return result


def _clip_intersection(
    inside: ClipVertex,
    outside: ClipVertex,
    inside_distance: float,
    outside_distance: float,
) -> ClipVertex:
    # Always interpolate from the inside endpoint toward the outside one.
    # Two neighbouring triangles share an edge in opposite traversal
    # directions; a direction-dependent formula would otherwise generate two
    # slightly different vertices for the same geometric clip point and leave
    # a gap or a double-blended seam.
    t = inside_distance / (inside_distance - outside_distance)
    vertex = _interpolate(inside, outside, t)
    # A crossing can land on the near plane to floating-point precision, where
    # 1/w is undefined. Such a point has zero screen area, so drop it.
    if vertex[3] <= 0.0:
        return []
    return [vertex]


def clip_triangle(vertices: Sequence[ClipVertex]) -> List[List[ClipVertex]]:
    active = [list(vertices)]
    for inside_distance in _CLIP_TESTS:
        next_polygons: List[List[ClipVertex]] = []
        for polygon in active:
            clipped: List[ClipVertex] = []
            previous = polygon[-1]
            previous_distance = inside_distance(previous)
            for current in polygon:
                current_distance = inside_distance(current)
                if current_distance >= 0.0:
                    if previous_distance < 0.0:
                        clipped.extend(_clip_intersection(current, previous, current_distance, previous_distance))
                    clipped.append(current)
                elif previous_distance >= 0.0:
                    clipped.extend(_clip_intersection(previous, current, previous_distance, current_distance))
                previous = current
                previous_distance = current_distance
            if len(clipped) >= 3:
                next_polygons.append(clipped)
        active = next_polygons
    fan_triangles: List[List[ClipVertex]] = []
    for polygon in active:
        # Canonicalize winding in clip-space xy (orientation is preserved by
        # the perspective divide for w > 0), then fan from the vertex smallest
        # in (y, x). Reversed or reordered copies of the same triangle then
        # decompose a clipped polygon into the exact same fan pieces.
        area2 = 0.0
        for i, vertex in enumerate(polygon):
            nxt = polygon[(i + 1) % len(polygon)]
            area2 += vertex[0] * nxt[1] - nxt[0] * vertex[1]
        if area2 < 0.0:
            polygon = list(reversed(polygon))
        pivot = min(range(len(polygon)), key=lambda i: (polygon[i][1], polygon[i][0]))
        ordered_polygon = polygon[pivot:] + polygon[:pivot]
        fan_triangles.extend(
            [ordered_polygon[0], ordered_polygon[i], ordered_polygon[i + 1]]
            for i in range(1, len(ordered_polygon) - 1)
        )
    return fan_triangles


ProjectedVertex = Tuple[float, float, float, Tuple[float, float], float, Tuple[float, float, float]]


def _project_vertex(vertex: ClipVertex, width: int, height: int) -> ProjectedVertex:
    inv_w = 1.0 / vertex[3]
    ndc_x = vertex[0] * inv_w
    ndc_y = vertex[1] * inv_w
    pixel_x = (ndc_x + 1.0) * 0.5 * width
    pixel_y = (1.0 - ndc_y) * 0.5 * height
    # Keep attributes in attribute/w form: screen barycentric interpolation
    # of those values, divided by the interpolated 1/w, is perspective correct.
    color = (vertex[5], vertex[6], vertex[7])
    uv = (vertex[8], vertex[9])
    return pixel_x, pixel_y, inv_w, uv, vertex[10], color


def _edge(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


Fragment = Tuple[int, int, float, float, Tuple[float, float], Tuple[float, float, float]]


def _collect_fragments(
    projected: Sequence[ProjectedVertex],
    width: int,
    height: int,
) -> List[Fragment]:
    # Order corners by screen position before rasterizing. The same
    # geometric triangle submitted reversed, rotated, or as part of a
    # differently triangulated surface must then walk its edges and sum its
    # perspective attributes in one canonical order, so seam pixels carry
    # identical coverage and shading (including the green channel) rather
    # than a last-binary-digit flicker.
    ordered = sorted(projected, key=lambda vertex: (vertex[1], vertex[0], vertex[2]))
    points = [(vertex[0], vertex[1]) for vertex in ordered]
    area = _edge(points[0], points[1], points[2])
    if abs(area) < 1e-10:
        return []
    orientation = 1.0 if area > 0.0 else -1.0
    # A shared edge belongs to the triangle for which it is a top/left edge,
    # so neighbouring triangulations never double-blend or leave a gap.
    biases = []
    for i in range(3):
        j = (i + 1) % 3
        edge_y = points[j][1] - points[i][1]
        edge_x = points[j][0] - points[i][0]
        if orientation > 0.0:
            top_left = edge_y < 0.0 or (edge_y == 0.0 and edge_x > 0.0)
        else:
            top_left = edge_y > 0.0 or (edge_y == 0.0 and edge_x < 0.0)
        biases.append(0.0 if top_left else -1e-9)

    min_x = max(0, int(math.floor(min(point[0] for point in points) - 0.5)))
    max_x = min(width - 1, int(math.ceil(max(point[0] for point in points) - 0.5)))
    min_y = max(0, int(math.floor(min(point[1] for point in points) - 0.5)))
    max_y = min(height - 1, int(math.ceil(max(point[1] for point in points) - 0.5)))

    edges = (
        (points[1], points[2], biases[0]),
        (points[2], points[0], biases[1]),
        (points[0], points[1], biases[2]),
    )
    fragments: List[Fragment] = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            sample = (x + 0.5, y + 0.5)
            weights = [_edge(a, b, sample) * orientation + bias for a, b, bias in edges]
            if any(weight < 0.0 for weight in weights):
                continue
            denominator = area * orientation
            b0 = weights[0] / denominator
            b1 = weights[1] / denominator
            b2 = weights[2] / denominator
            inv_w = b0 * ordered[0][2] + b1 * ordered[1][2] + b2 * ordered[2][2]
            camera_depth = 1.0 / inv_w
            uv_over_w = (
                b0 * ordered[0][3][0] + b1 * ordered[1][3][0] + b2 * ordered[2][3][0],
                b0 * ordered[0][3][1] + b1 * ordered[1][3][1] + b2 * ordered[2][3][1],
            )
            opacity_over_w = b0 * ordered[0][4] + b1 * ordered[1][4] + b2 * ordered[2][4]
            color_over_w = tuple(
                b0 * ordered[0][5][channel] + b1 * ordered[1][5][channel] + b2 * ordered[2][5][channel]
                for channel in range(3)
            )
            uv = (uv_over_w[0] / inv_w, uv_over_w[1] / inv_w)
            opacity = max(0.0, min(1.0, opacity_over_w / inv_w))
            color = tuple(channel / inv_w for channel in color_over_w)
            fragments.append((x, y, camera_depth, opacity, uv, color))
    return fragments


def _shade(fragment: Fragment, texture: Optional[Texture]) -> Tuple[float, float, float]:
    _, _, _, _, uv, color = fragment
    if texture is None:
        return color
    texel = texture.sample(uv[0], uv[1])
    return (
        color[0] * texel[0] / 255.0,
        color[1] * texel[1] / 255.0,
        color[2] * texel[2] / 255.0,
    )


def render(scene: Scene, options: RenderOptions = RenderOptions()) -> Image:
    camera = scene.camera(options.camera_name)
    if options.clip_group_name is None and scene.clip_groups:
        clip_group = scene.clip_groups[0]
    elif options.clip_group_name is not None:
        clip_group = scene.clip_group(options.clip_group_name)
    else:
        clip_group = None
    render_meshes = (
        build_clipped_scene_meshes(scene.meshes, clip_group)
        if clip_group is not None
        else scene.meshes
    )
    image = Image(options.width, options.height, scene.background)
    # Camera-space depth (larger = farther) keeps opaque/transparent ordering
    # consistent after transparent pieces are split by the near plane.
    depth = [math.inf for _ in range(options.width * options.height)]
    aspect_ratio = options.width / options.height
    transparent_fragments: List[Fragment] = []

    for mesh in render_meshes:
        textures: Dict[str, Texture] = {}
        if mesh.textures:
            textures = {texture.name: texture for texture in mesh.textures}
        for triangle in mesh.triangles:
            texture = textures.get(triangle.texture_name) if triangle.texture_name is not None else None
            source_vertices = []
            for corner, index in enumerate(triangle.indices):
                uv = triangle.uvs[corner] if triangle.uvs is not None else (0.0, 0.0)
                vertex = _clip_vertex(
                    camera,
                    mesh.vertices[index],
                    tuple(float(channel) for channel in triangle.color),
                    aspect_ratio,
                    uv[0],
                    uv[1],
                    triangle.opacity,
                )
                source_vertices.append(vertex)
            for clipped_triangle in clip_triangle(source_vertices):
                projected = [_project_vertex(vertex, options.width, options.height) for vertex in clipped_triangle]
                fragments = _collect_fragments(projected, options.width, options.height)
                if triangle.opacity >= 1.0:
                    for fragment in fragments:
                        x, y, fragment_depth, _, _, _ = fragment
                        offset = y * options.width + x
                        if fragment_depth < depth[offset]:
                            image.set_pixel(x, y, _shade(fragment, texture))
                            depth[offset] = fragment_depth
                else:
                    for fragment in fragments:
                        x, y, fragment_depth, opacity, uv, color = fragment
                        if fragment_depth < depth[y * options.width + x]:
                            transparent_fragments.append(
                                (x, y, fragment_depth, opacity, uv, _shade(fragment, texture))
                            )

    # Supported transparency: one source-over pass, per-fragment back-to-front.
    # Order-independent whole-surface blending (weighted OIT, refraction) is
    # intentionally out of scope.
    transparent_fragments.sort(key=lambda item: (item[0], item[1], -item[2]))
    # A physical surface may arrive as two coincident pieces (alternative
    # diagonal, reversed submission, or triangles from separate meshes with
    # shared vertices). After clipping their shared edge can differ by a
    # rounding unit, so the top/left fill rule alone lets both pieces emit the
    # seam pixel. Blending them twice brightens the line and re-applies the
    # material; coalesce coincident fragments before compositing instead of
    # hiding the edge. Fragments count as one surface point only when they
    # share a pixel, the same camera depth, and the same material response,
    # so genuinely stacked layers are kept separate.
    coalesced: List[Fragment] = []
    for fragment in transparent_fragments:
        x, y, fragment_depth, opacity, uv, shaded = fragment
        if coalesced and coalesced[-1][0] == x and coalesced[-1][1] == y:
            previous = coalesced[-1]
            same_depth = abs(previous[2] - fragment_depth) <= max(1.0e-6, 1.0e-5 * fragment_depth)
            same_material = (
                abs(previous[3] - opacity) <= 1.0e-7
                and all(abs(previous[5][channel] - shaded[channel]) <= 1.0e-6 for channel in range(3))
            )
            if same_depth and same_material:
                continue
        coalesced.append(fragment)
    coalesced.sort(key=lambda item: item[2], reverse=True)
    for x, y, fragment_depth, opacity, _, shaded in coalesced:
        offset = y * options.width + x
        if fragment_depth >= depth[offset]:
            continue
        base = image.get_pixel(x, y)
        blended = tuple(base[channel] * (1.0 - opacity) + shaded[channel] * opacity for channel in range(3))
        image.set_pixel(x, y, blended)
    return image
