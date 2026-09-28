from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .camera import Camera, view_projection_matrix
from .geometry import Color, Scene
from .math3d import Vec4, transform_point


@dataclass(frozen=True)
class RenderSettings:
    width: int = 320
    height: int = 240
    background: Color = (18, 20, 26)
    camera: str | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if len(self.background) != 3:
            raise ValueError("background must be RGB")


@dataclass(frozen=True)
class ClipVertex:
    position: Vec4


ClipPlane = tuple[float, float, float, float]

CLIP_PLANES: tuple[ClipPlane, ...] = (
    (1.0, 0.0, 0.0, 1.0),   # x >= -w
    (-1.0, 0.0, 0.0, 1.0),  # x <=  w
    (0.0, 1.0, 0.0, 1.0),   # y >= -w
    (0.0, -1.0, 0.0, 1.0),  # y <=  w
    (0.0, 0.0, 1.0, 1.0),   # z >= -w (near)
    (0.0, 0.0, -1.0, 1.0),  # z <=  w (far)
)


def clip_distance(vertex: ClipVertex, plane: ClipPlane) -> float:
    point = vertex.position
    return (
        plane[0] * point[0]
        + plane[1] * point[1]
        + plane[2] * point[2]
        + plane[3] * point[3]
    )


def interpolate_clip_vertex(a: ClipVertex, b: ClipVertex, t: float) -> ClipVertex:
    pa = a.position
    pb = b.position
    return ClipVertex(
        tuple(pa[index] + (pb[index] - pa[index]) * t for index in range(4))  # type: ignore[arg-type]
    )


def clip_polygon(
    polygon: tuple[ClipVertex, ...], plane: ClipPlane
) -> tuple[ClipVertex, ...]:
    if not polygon:
        return ()

    output: list[ClipVertex] = []
    previous = polygon[-1]
    previous_distance = clip_distance(previous, plane)

    for current in polygon:
        current_distance = clip_distance(current, plane)
        if current_distance >= 0.0:
            if previous_distance < 0.0:
                denominator = previous_distance - current_distance
                t = previous_distance / denominator if denominator != 0.0 else 0.0
                output.append(interpolate_clip_vertex(previous, current, t))
            output.append(current)
        elif previous_distance >= 0.0:
            denominator = previous_distance - current_distance
            t = previous_distance / denominator if denominator != 0.0 else 0.0
            output.append(interpolate_clip_vertex(previous, current, t))

        previous = current
        previous_distance = current_distance

    return tuple(output)


def clip_triangle(
    vertices: tuple[ClipVertex, ClipVertex, ClipVertex]
) -> tuple[tuple[ClipVertex, ClipVertex, ClipVertex], ...]:
    polygon: tuple[ClipVertex, ...] = vertices
    for plane in CLIP_PLANES:
        polygon = clip_polygon(polygon, plane)
        if not polygon:
            return ()
    return tuple(
        (polygon[0], polygon[index], polygon[index + 1])
        for index in range(1, len(polygon) - 1)
    )


@dataclass(frozen=True)
class ScreenVertex:
    x: float
    y: float
    ndc_z: float
    inv_w: float


def to_screen_vertex(vertex: ClipVertex, width: int, height: int) -> ScreenVertex:
    x, y, z, w = vertex.position
    if w <= 0.0 or not all(isfinite(value) for value in vertex.position):
        raise ValueError("clipping produced an invalid vertex")
    ndc_x = x / w
    ndc_y = y / w
    return ScreenVertex(
        x=(ndc_x * 0.5 + 0.5) * width,
        y=(1.0 - (ndc_y * 0.5 + 0.5)) * height,
        ndc_z=z / w,
        inv_w=1.0 / w,
    )


def resolve_camera(scene: Scene, camera: Camera | str | None) -> Camera:
    if isinstance(camera, Camera):
        return camera
    name = camera or scene.default_camera
    try:
        return scene.cameras[name]
    except KeyError as error:
        raise ValueError(f"scene has no camera named {name!r}") from error


def edge_value(a: ScreenVertex, b: ScreenVertex, x: float, y: float) -> float:
    return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x)


def is_top_left(a: ScreenVertex, b: ScreenVertex) -> bool:
    return (b.y < a.y) or (b.y == a.y and b.x > a.x)


def rasterize_triangle(
    vertices: tuple[ScreenVertex, ScreenVertex, ScreenVertex],
    color: Color,
    width: int,
    height: int,
    framebuffer: bytearray,
    depthbuffer: list[float],
) -> None:
    a, b, c = vertices
    area = edge_value(a, b, c.x, c.y)
    if abs(area) < 1e-12:
        return
    if area < 0.0:
        a, b, c = c, b, a
        area = -area

    min_x = max(0, int(min(a.x, b.x, c.x) // 1.0))
    max_x = min(width - 1, int(max(a.x, b.x, c.x) // 1.0))
    min_y = max(0, int(min(a.y, b.y, c.y) // 1.0))
    max_y = min(height - 1, int(max(a.y, b.y, c.y) // 1.0))

    edges = ((a, b), (b, c), (c, a))
    for pixel_y in range(min_y, max_y + 1):
        center_y = pixel_y + 0.5
        for pixel_x in range(min_x, max_x + 1):
            center_x = pixel_x + 0.5
            inside = True
            edge_weights: list[float] = []
            for edge_index, (edge_a, edge_b) in enumerate(edges):
                value = edge_value(edge_a, edge_b, center_x, center_y)
                if value < 0.0 or (value == 0.0 and not is_top_left(edge_a, edge_b)):
                    inside = False
                    break
                edge_weights.append(value)
            if not inside:
                continue

            weight_c, weight_a, weight_b = (weight / area for weight in edge_weights)
            wa, wb, wc = weight_a, weight_b, weight_c
            inv_w = wa * a.inv_w + wb * b.inv_w + wc * c.inv_w
            depth = (
                wa * a.ndc_z * a.inv_w
                + wb * b.ndc_z * b.inv_w
                + wc * c.ndc_z * c.inv_w
            ) / inv_w
            pixel_index = pixel_y * width + pixel_x
            if depth < depthbuffer[pixel_index]:
                depthbuffer[pixel_index] = depth
                offset = pixel_index * 3
                framebuffer[offset:offset + 3] = bytes(color)


def render(
    scene: Scene,
    settings: RenderSettings | None = None,
    camera: Camera | str | None = None,
) -> bytes:
    settings = settings or RenderSettings()
    active_camera = resolve_camera(scene, settings.camera or camera)
    matrix = view_projection_matrix(
        active_camera, settings.width / settings.height
    )

    framebuffer = bytearray(settings.background * (settings.width * settings.height))
    depthbuffer = [float("inf")] * (settings.width * settings.height)

    for mesh in scene.meshes:
        for face in mesh.faces:
            world_vertices = tuple(
                ClipVertex(transform_point(matrix, mesh.vertices[index]))
                for index in face
            )
            for clipped in clip_triangle(world_vertices):
                screen_vertices = tuple(
                    to_screen_vertex(vertex, settings.width, settings.height)
                    for vertex in clipped
                )
                rasterize_triangle(
                    screen_vertices,  # type: ignore[arg-type]
                    mesh.color,
                    settings.width,
                    settings.height,
                    framebuffer,
                    depthbuffer,
                )

    return bytes(framebuffer)
