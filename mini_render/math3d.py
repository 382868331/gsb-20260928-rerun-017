from __future__ import annotations

from math import sqrt
from typing import Sequence

Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
Mat4 = tuple[Vec4, Vec4, Vec4, Vec4]


def vec3(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError("a 3D vector requires exactly three components")
    return float(value[0]), float(value[1]), float(value[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def scale(value: Vec3, factor: float) -> Vec3:
    return value[0] * factor, value[1] * factor, value[2] * factor


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(value: Vec3) -> Vec3:
    length = sqrt(dot(value, value))
    if length <= 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return scale(value, 1.0 / length)


def mat4_mul(a: Mat4, b: Mat4) -> Mat4:
    return tuple(
        tuple(sum(a[row][k] * b[k][col] for k in range(4)) for col in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def mat4_vec4(matrix: Mat4, vector: Vec4) -> Vec4:
    return tuple(
        sum(matrix[row][col] * vector[col] for col in range(4))
        for row in range(4)
    )  # type: ignore[return-value]


def transform_point(matrix: Mat4, point: Vec3) -> Vec4:
    return mat4_vec4(matrix, (point[0], point[1], point[2], 1.0))
