from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .camera import Camera
from .geometry import Mesh
from .image import Image
from .scene import Scene
from .texture import Texture


@dataclass(frozen=True)
class RenderOptions:
    width: int = 240
    height: int = 180
    camera_name: str | None = None


ClipVertex = List[float]


def _clip_vertex(
    camera: Camera,
    position: Tuple[float, float, float],
    color: Tuple[float, float, float],
    aspect_ratio: float,
    uv: Tuple[float, float] = (0.0, 0.0),
    opacity: float = 1.0,
) -> ClipVertex:
    x, y, z = camera.view_position(position)
    aspect_scale = math.tan(math.radians(camera.fov_y_degrees) * 0.5)
    near, far = camera.near, camera.far
    depth_scale = (far + near) / (far - near)
    depth_offset = -2.0 * far * near / (far - near)
    return [
        x / (aspect_scale * aspect_ratio),
        y / aspect_scale,
        z * depth_scale + depth_offset,
        z,
        z,
        color[0],
        color[1],
        color[2],
        uv[0],
        uv[1],
        opacity,
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
    return [a_value + (b_value - a_value) * t for a_value, b_value in zip(a, b)]


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
                        denominator = current_distance - previous_distance
                        clipped.append(_interpolate(previous, current, -previous_distance / denominator))
                    clipped.append(current)
                elif previous_distance >= 0.0:
                    denominator = current_distance - previous_distance
                    clipped.append(_interpolate(previous, current, -previous_distance / denominator))
                previous = current
                previous_distance = current_distance
            if len(clipped) >= 3:
                next_polygons.append(clipped)
        active = next_polygons
    return [[polygon[0], polygon[i], polygon[i + 1]] for polygon in active for i in range(1, len(polygon) - 1)]


def _project_vertex(vertex: ClipVertex, width: int, height: int):
    inv_w = 1.0 / vertex[3]
    ndc_x = vertex[0] * inv_w
    ndc_y = vertex[1] * inv_w
    ndc_z = vertex[2] * inv_w
    pixel_x = (ndc_x + 1.0) * 0.5 * width
    pixel_y = (1.0 - ndc_y) * 0.5 * height
    color = (vertex[5] * inv_w, vertex[6] * inv_w, vertex[7] * inv_w)
    uv_over_w = (vertex[8] * inv_w, vertex[9] * inv_w)
    opacity_over_w = vertex[10] * inv_w
    return pixel_x, pixel_y, ndc_z, inv_w, color, uv_over_w, opacity_over_w


def _is_top_left(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    # Direct3D-style tie-break: a pixel exactly on a horizontal or left-facing
    # shared edge belongs to one triangle only, so abutting triangulations do
    # not leave background slivers.
    rise = b[1] - a[1]
    run = b[0] - a[0]
    return rise > 0.0 or (rise == 0.0 and run > 0.0)


def _edge(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def _raster_triangle(
    projected,
    image: Image,
    depth: List[float],
    texture: Optional[Texture],
) -> None:
    points = [(vertex[0], vertex[1]) for vertex in projected]
    area = _edge(points[0], points[1], points[2])
    if abs(area) < 1e-10:
        return
    orientation = 1.0 if area > 0.0 else -1.0
    edge_specs = ((1, 2), (2, 0), (0, 1))
    top_left = [_is_top_left(points[start], points[end]) for start, end in edge_specs]

    min_x = max(0, int(math.floor(min(point[0] for point in points) - 0.5)))
    max_x = min(image.width - 1, int(math.ceil(max(point[0] for point in points) - 0.5)))
    min_y = max(0, int(math.floor(min(point[1] for point in points) - 0.5)))
    max_y = min(image.height - 1, int(math.ceil(max(point[1] for point in points) - 0.5)))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            sample = (x + 0.5, y + 0.5)
            raw = [
                _edge(points[start], points[end], sample)
                for (start, end) in edge_specs
            ]
            covered = True
            for value, is_top_left_edge in zip(raw, top_left):
                signed = value * orientation
                on_edge = abs(signed) < 1e-9
                if signed < -1e-9 or (on_edge and not is_top_left_edge):
                    covered = False
                    break
            if not covered:
                continue
            clamped = [0.0 if abs(value) < 1e-9 else value for value in raw]
            b0, b1, b2 = (value / area for value in clamped)
            sample_depth = b0 * projected[0][2] + b1 * projected[1][2] + b2 * projected[2][2]
            offset = y * image.width + x
            if sample_depth >= depth[offset]:
                continue
            inv_w = b0 * projected[0][3] + b1 * projected[1][3] + b2 * projected[2][3]
            color_over_w = tuple(
                b0 * projected[0][4][channel]
                + b1 * projected[1][4][channel]
                + b2 * projected[2][4][channel]
                for channel in range(3)
            )
            color = tuple(channel / inv_w for channel in color_over_w)
            uv_over_w = (
                b0 * projected[0][5][0] + b1 * projected[1][5][0] + b2 * projected[2][5][0],
                b0 * projected[0][5][1] + b1 * projected[1][5][1] + b2 * projected[2][5][1],
            )
            u = uv_over_w[0] / inv_w
            v = uv_over_w[1] / inv_w
            if texture is not None:
                texel = texture.sample(u, v)
                color = (
                    color[0] * texel[0] / 255.0,
                    color[1] * texel[1] / 255.0,
                    color[2] * texel[2] / 255.0,
                )
            image.set_pixel(x, y, color)
            depth[offset] = sample_depth


def _collect_triangle(
    projected,
    image: Image,
) -> List[tuple]:
    """Return (x, y, depth, shade, camera-z) fragments for deferred sorting."""
    points = [(vertex[0], vertex[1]) for vertex in projected]
    area = _edge(points[0], points[1], points[2])
    if abs(area) < 1e-10:
        return []
    orientation = 1.0 if area > 0.0 else -1.0
    edge_specs = ((1, 2), (2, 0), (0, 1))
    top_left = [_is_top_left(points[start], points[end]) for start, end in edge_specs]

    min_x = max(0, int(math.floor(min(point[0] for point in points) - 0.5)))
    max_x = min(image.width - 1, int(math.ceil(max(point[0] for point in points) - 0.5)))
    min_y = max(0, int(math.floor(min(point[1] for point in points) - 0.5)))
    max_y = min(image.height - 1, int(math.ceil(max(point[1] for point in points) - 0.5)))

    fragments = []
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            sample = (x + 0.5, y + 0.5)
            raw = [
                _edge(points[start], points[end], sample)
                for start, end in edge_specs
            ]
            covered = True
            for value, is_top_left_edge in zip(raw, top_left):
                signed = value * orientation
                on_edge = abs(signed) < 1e-9
                if signed < -1e-9 or (on_edge and not is_top_left_edge):
                    covered = False
                    break
            if not covered:
                continue
            clamped = [0.0 if abs(value) < 1e-9 else value for value in raw]
            b0, b1, b2 = (value / area for value in clamped)
            sample_depth = b0 * projected[0][2] + b1 * projected[1][2] + b2 * projected[2][2]
            inv_w = b0 * projected[0][3] + b1 * projected[1][3] + b2 * projected[2][3]
            camera_z = 1.0 / inv_w
            color_over_w = tuple(
                b0 * projected[0][4][channel]
                + b1 * projected[1][4][channel]
                + b2 * projected[2][4][channel]
                for channel in range(3)
            )
            color = tuple(channel / inv_w for channel in color_over_w)
            uv_over_w = (
                b0 * projected[0][5][0] + b1 * projected[1][5][0] + b2 * projected[2][5][0],
                b0 * projected[0][5][1] + b1 * projected[1][5][1] + b2 * projected[2][5][1],
            )
            opacity_over_w = (
                b0 * projected[0][6] + b1 * projected[1][6] + b2 * projected[2][6]
            )
            fragments.append(
                (x, y, sample_depth, color, uv_over_w[0] / inv_w, uv_over_w[1] / inv_w,
                 opacity_over_w / inv_w, camera_z)
            )
    return fragments


def render(scene: Scene, options: RenderOptions = RenderOptions()) -> Image:
    camera = scene.camera(options.camera_name)
    image = Image(options.width, options.height, scene.background)
    depth = [1.0 for _ in range(options.width * options.height)]
    aspect_ratio = options.width / options.height
    textures: Dict[str, Texture] = {texture.name: texture for texture in scene.textures}

    # Transparent layer: grouped per clipped triangle and sorted back-to-front.
    # Supported scope is the standard painter's "over" blend against the depth
    # buffer produced by the opaque pass.
    transparent_layers: List[Tuple[float, List[tuple], Optional[Texture]]] = []

    for mesh in scene.meshes:
        mesh_texture = textures.get(mesh.texture_name) if mesh.texture_name is not None else None
        mesh_uvs = mesh.uvs if mesh.uvs is not None else [(0.0, 0.0) for _ in mesh.vertices]
        vertex_cache = [
            _clip_vertex(camera, vertex, (0.0, 0.0, 0.0), aspect_ratio, uv)
            for vertex, uv in zip(mesh.vertices, mesh_uvs)
        ]
        for triangle in mesh.triangles:
            source_vertices = []
            for index, uv in zip(triangle.indices, (mesh_uvs[i] for i in triangle.indices)):
                vertex = list(vertex_cache[index])
                vertex[5] = float(triangle.color[0])
                vertex[6] = float(triangle.color[1])
                vertex[7] = float(triangle.color[2])
                vertex[8] = uv[0]
                vertex[9] = uv[1]
                vertex[10] = triangle.opacity
                source_vertices.append(vertex)
            clipped = clip_triangle(source_vertices)
            for clipped_triangle in clipped:
                projected = [_project_vertex(vertex, options.width, options.height) for vertex in clipped_triangle]
                if triangle.opacity >= 1.0:
                    _raster_triangle(projected, image, depth, mesh_texture)
                else:
                    fragments = _collect_triangle(projected, image)
                    if fragments:
                        sort_depth = max(fragment[7] for fragment in fragments)
                        transparent_layers.append((sort_depth, fragments, mesh_texture))

    # Back-to-front: the layer with the largest camera-space z is farthest and
    # blends first; nearer sheets composite over it.
    for _, fragments, mesh_texture in sorted(transparent_layers, key=lambda layer: layer[0], reverse=True):
        for x, y, sample_depth, color, u, v, opacity, _camera_z in fragments:
            offset = y * image.width + x
            # Opaque geometry occludes translucent surfaces behind it.
            if sample_depth >= depth[offset]:
                continue
            texture_alpha = 1.0
            if mesh_texture is not None:
                texel = mesh_texture.sample(u, v)
                color = (
                    color[0] * texel[0] / 255.0,
                    color[1] * texel[1] / 255.0,
                    color[2] * texel[2] / 255.0,
                )
                texture_alpha = texel[3] / 255.0
            alpha = max(0.0, min(1.0, opacity * texture_alpha))
            if alpha <= 0.0:
                continue
            destination = image.get_pixel(x, y)
            blended = tuple(
                alpha * source_channel + (1.0 - alpha) * destination_channel
                for source_channel, destination_channel in zip(color, destination)
            )
            image.set_pixel(x, y, blended)
    return image
