from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .camera import Camera
from .geometry import Mesh
from .image import Image
from .scene import Scene


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


def _project_vertex(vertex: ClipVertex, width: int, height: int) -> Tuple[float, float, float, float, Tuple[float, float, float]]:
    inv_w = 1.0 / vertex[3]
    ndc_x = vertex[0] * inv_w
    ndc_y = vertex[1] * inv_w
    ndc_z = vertex[2] * inv_w
    pixel_x = (ndc_x + 1.0) * 0.5 * width
    pixel_y = (1.0 - ndc_y) * 0.5 * height
    color = (vertex[5] * inv_w, vertex[6] * inv_w, vertex[7] * inv_w)
    return pixel_x, pixel_y, ndc_z, inv_w, color


def _edge(a: Tuple[float, float], b: Tuple[float, float], p: Tuple[float, float]) -> float:
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def _raster_triangle(
    projected: Sequence[Tuple[float, float, float, float, Tuple[float, float, float]]],
    image: Image,
    depth: List[float],
) -> None:
    points = [(vertex[0], vertex[1]) for vertex in projected]
    area = _edge(points[0], points[1], points[2])
    if abs(area) < 1e-10:
        return
    orientation = 1.0 if area > 0.0 else -1.0

    min_x = max(0, int(math.floor(min(point[0] for point in points) - 0.5)))
    max_x = min(image.width - 1, int(math.ceil(max(point[0] for point in points) - 0.5)))
    min_y = max(0, int(math.floor(min(point[1] for point in points) - 0.5)))
    max_y = min(image.height - 1, int(math.ceil(max(point[1] for point in points) - 0.5)))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            sample = (x + 0.5, y + 0.5)
            w0 = _edge(points[1], points[2], sample) * orientation
            w1 = _edge(points[2], points[0], sample) * orientation
            w2 = _edge(points[0], points[1], sample) * orientation
            if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                continue
            b0 = w0 / (area * orientation)
            b1 = w1 / (area * orientation)
            b2 = w2 / (area * orientation)
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
            image.set_pixel(x, y, color)
            depth[offset] = sample_depth


def render(scene: Scene, options: RenderOptions = RenderOptions()) -> Image:
    camera = scene.camera(options.camera_name)
    image = Image(options.width, options.height, scene.background)
    depth = [1.0 for _ in range(options.width * options.height)]
    aspect_ratio = options.width / options.height

    for mesh in scene.meshes:
        vertex_cache = [
            _clip_vertex(camera, vertex, (0.0, 0.0, 0.0), aspect_ratio)
            for vertex in mesh.vertices
        ]
        for triangle in mesh.triangles:
            source_vertices = []
            for index in triangle.indices:
                vertex = list(vertex_cache[index])
                vertex[5] = float(triangle.color[0])
                vertex[6] = float(triangle.color[1])
                vertex[7] = float(triangle.color[2])
                source_vertices.append(vertex)
            clipped = clip_triangle(source_vertices)
            for clipped_triangle in clipped:
                projected = [_project_vertex(vertex, options.width, options.height) for vertex in clipped_triangle]
                _raster_triangle(projected, image, depth)
    return image
