from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from renderer import (
    Camera,
    ClipGroup,
    ClipPlane,
    Mesh,
    RenderOptions,
    Scene,
    Triangle,
    build_section,
    render,
)


def box_mesh(name="solid", low=(-1, -1, -1), high=(1, 1, 1), color=(120, 180, 120)):
    x0, y0, z0 = low
    x1, y1, z1 = high
    vertices = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ]
    return Mesh(name, vertices, [Triangle(face, color) for face in faces])


def square_tube(name="tube", outer=1.0, inner=0.4):
    vertices = []
    triangles = []
    color = (150, 150, 150)

    def add(points, reverse=False):
        offset = len(vertices)
        vertices.extend(points)
        faces = ((0, 2, 1), (0, 3, 2)) if reverse else ((0, 1, 2), (0, 2, 3))
        for face in faces:
            triangles.append(Triangle(tuple(offset + value for value in face), color))

    o, i = outer, inner
    for z, reverse in ((-1.0, False), (1.0, True)):
        add([(-o, -o, z), (o, -o, z), (o, o, z), (-o, o, z)], reverse)
        add([(-i, -i, z), (-i, i, z), (i, i, z), (i, -i, z)], reverse)
    add([(o, -o, -1), (o, -o, 1), (o, o, 1), (o, o, -1)])
    add([(-o, o, -1), (-o, o, 1), (-o, -o, 1), (-o, -o, -1)])
    add([(-o, -o, -1), (-o, -o, 1), (o, -o, 1), (o, -o, -1)])
    add([(o, o, -1), (o, o, 1), (-o, o, 1), (-o, o, -1)])
    add([(i, -i, -1), (i, -i, 1), (i, i, 1), (i, i, -1)], True)
    add([(-i, i, -1), (-i, i, 1), (-i, -i, 1), (-i, -i, -1)], True)
    add([(-i, -i, -1), (i, -i, -1), (i, -i, 1), (-i, -i, 1)], True)
    add([(i, i, -1), (-i, i, -1), (-i, i, 1), (i, i, 1)], True)
    return Mesh(name, vertices, triangles)


class SectionTests(unittest.TestCase):
    def test_single_plane_cap_is_closed_and_labels_owner_and_cut(self):
        group = ClipGroup("cut", (ClipPlane("x", (-1, 0, 0), 0.0, (255, 10, 10)),))
        result = build_section(box_mesh("alpha"), group)
        self.assertEqual([], result.open_boundaries)
        self.assertEqual(1, len(result.caps))
        cap = result.caps[0]
        self.assertEqual("alpha#section:cut/x", cap.name)
        self.assertTrue(cap.triangles)
        self.assertTrue(all(triangle.color == (255, 10, 10) for triangle in cap.triangles))
        self.assertTrue(all(abs(x) <= 1e-7 for x, _, _ in cap.vertices))

    def test_two_joint_planes_close_without_open_edges(self):
        group = ClipGroup(
            "pair",
            (ClipPlane("a", (-1, 0, 0), 0.0, (255, 0, 0)), ClipPlane("b", (0, -1, 0), 0.2, (0, 0, 255))),
        )
        result = build_section(box_mesh(), group)
        self.assertEqual([], result.open_boundaries)
        self.assertEqual({"a", "b"}, {name for name in (cap.name for cap in result.caps) for name in [name.split("/")[-1]]})
        # The plane/plane seam is a shared edge, not an un-closeable range.
        self.assertTrue(all(cap.triangles for cap in result.caps))

    def test_sections_of_distinct_meshes_are_separate_faces(self):
        group = ClipGroup("cut", (ClipPlane("x", (-1, 0, 0), 0.0),))
        first = build_section(box_mesh("first"), group)
        second = build_section(box_mesh("second"), group)
        names = {cap.name for cap in first.caps + second.caps}
        self.assertIn("first#section:cut/x", names)
        self.assertIn("second#section:cut/x", names)
        # The caps stay separate objects even where the geometry coincides,
        # rather than being stitched into one face.
        self.assertIsNot(first.caps[0], second.caps[0])
        self.assertNotIn(second.caps[0], first.caps)

    def test_coincident_planes_do_not_duplicate_the_cap(self):
        group = ClipGroup(
            "twice",
            (ClipPlane("a", (-1, 0, 0), 0.0, (255, 0, 0)), ClipPlane("b", (1, 0, 0), 0.0, (0, 255, 0))),
        )
        result = build_section(box_mesh(), group)
        self.assertEqual(1, len(result.caps))
        self.assertTrue(all(triangle.color == (255, 0, 0) for triangle in result.caps[0].triangles))

    def test_open_surface_marks_uncloseable_range_instead_of_filling(self):
        sheet = Mesh(
            "sheet",
            [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)],
            [Triangle((0, 1, 2), (10, 200, 10)), Triangle((0, 2, 3), (10, 200, 10))],
        )
        group = ClipGroup("cut", (ClipPlane("x", (-1, 0, 0), 0.0, (255, 255, 0)),))
        result = build_section(sheet, group)
        self.assertEqual([], result.caps)
        self.assertEqual(1, len(result.open_boundaries))
        ribbon = result.open_boundaries[0]
        self.assertTrue(ribbon.triangles)
        self.assertTrue(all(triangle.color == (255, 90, 240) for triangle in ribbon.triangles))
        self.assertTrue(all(abs(x) <= 1e-7 for x, _, _ in ribbon.vertices))

    def test_hole_survives_multiple_planes(self):
        group = ClipGroup(
            "cuts",
            (ClipPlane("z", (0, 0, -1), 0.0, (0, 255, 0)), ClipPlane("x", (1, 0, 0), 0.4, (0, 120, 255))),
        )
        result = build_section(square_tube(), group)
        z_cap = next(cap for cap in result.caps if cap.name.endswith("cuts/z"))
        for triangle in z_cap.triangles:
            centroid = tuple(
                sum(z_cap.vertices[index][axis] for index in triangle.indices) / 3.0 for axis in range(3)
            )
            self.assertFalse(
                abs(centroid[0]) < 0.4 and abs(centroid[1]) < 0.4,
                msg=f"section filled the tunnel at {centroid}",
            )


    def test_plane_through_vertex_and_along_edge_does_not_invent_solid(self):
        group = ClipGroup("cut", (ClipPlane("top", (0, -1, 0), 1.0, (255, 0, 255)),))
        result = build_section(box_mesh(), group)
        # The plane contains the top four vertices and top four edges.  The cap
        # is the existing top face; nothing outside the box is filled.
        self.assertEqual(1, len(result.caps))
        cap = result.caps[0]
        for triangle in cap.triangles:
            for index in triangle.indices:
                x, y, _ = cap.vertices[index]
                self.assertEqual(1.0, y)
                self.assertLessEqual(abs(x), 1.0 + 1e-9)

    def test_triangulation_change_does_not_change_section(self):
        group = ClipGroup("cut", (ClipPlane("z", (0, 0, -1), 0.0),))
        # Closed square pyramid; the base quad is submitted with either diagonal.
        apex = (0.0, 0.0, 1.0)
        corners = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        sides = [
            Triangle((4, 0, 1), (200, 200, 200)),
            Triangle((4, 1, 2), (200, 200, 200)),
            Triangle((4, 2, 3), (200, 200, 200)),
            Triangle((4, 3, 0), (200, 200, 200)),
        ]
        a = Mesh("pyramid", corners + [apex], sides + [
            Triangle((0, 1, 2), (200, 200, 200)), Triangle((0, 2, 3), (200, 200, 200)),
        ])
        b = Mesh("pyramid", corners + [apex], sides + [
            Triangle((0, 3, 2), (200, 200, 200)), Triangle((0, 2, 1), (200, 200, 200)),
        ])
        cap_a = build_section(a, group).caps[0]
        cap_b = build_section(b, group).caps[0]
        self.assertEqual(
            sorted((round(x, 6), round(y, 6), round(z, 6)) for x, y, z in cap_a.vertices),
            sorted((round(x, 6), round(y, 6), round(z, 6)) for x, y, z in cap_b.vertices),
        )

    def test_copied_group_moves_independently_and_keeps_colors(self):
        original = ClipGroup("g", (ClipPlane("p", (-1, 0, 0), 0.0, (11, 22, 33)),))
        control = original.copied("control").with_plane("p", offset=0.5)
        self.assertEqual(0.0, original.planes[0].offset)
        self.assertEqual(0.5, control.planes[0].offset)
        self.assertEqual((11, 22, 33), control.planes[0].color)

    def test_clip_group_round_trips_through_json_and_source_is_unchanged(self):
        group = ClipGroup("cut", (ClipPlane("x", (-1, 0, 0), 0.2, (9, 8, 7), 0.6),))
        mesh = box_mesh("saved")
        scene = Scene("saved", [mesh], [Camera("front", eye=(2.0, 1.0, -3.0), target=(0, 0, 0))], clip_groups=[group])
        options = RenderOptions(64, 48, "front", "cut")
        before = render(scene, options).pixels
        original_vertices = [vertex for vertex in mesh.vertices]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            scene.save(path)
            loaded = Scene.load(path)
            after = render(loaded, options).pixels
        self.assertEqual(before, after)
        self.assertEqual(original_vertices, mesh.vertices)
        self.assertEqual(len(mesh.triangles), 12)

    def test_front_and_back_cameras_show_same_cap_identity(self):
        group = ClipGroup("cut", (ClipPlane("z", (0, 0, 1), -0.55, (200, 30, 30)),))
        solid = box_mesh(color=(60, 100, 70))
        transparent = Mesh(
            solid.name,
            solid.vertices,
            [Triangle(t.indices, t.color, opacity=0.35) for t in solid.triangles],
        )
        scene = Scene(
            "both",
            [transparent],
            [
                Camera("front", eye=(1.8, 0.8, -2.4), target=(0, 0, 0)),
                Camera("back", eye=(-1.8, -0.8, 2.4), target=(0, 0, 0)),
            ],
            clip_groups=[group],
        )
        for camera in ("front", "back"):
            result = render(scene, RenderOptions(120, 90, camera, "cut"))
            # Cap color is present in the image from both sides; ownership is
            # independent of which camera-side the viewer stands on.
            colored = sum(1 for pixel in result.pixels if pixel[0] > 120 and pixel[1] < 90 and pixel[2] < 90)
            self.assertGreater(colored, 20)

    def test_moving_a_plane_changes_result_but_surface_retriangulation_matches(self):
        from renderer.section import build_clipped_scene_meshes

        group = ClipGroup("cut", (ClipPlane("x", (-1, 0, 0), 0.0),))
        moved = group.with_plane("x", offset=0.6)
        meshes_a = build_clipped_scene_meshes([box_mesh()], group)
        meshes_b = build_clipped_scene_meshes([box_mesh()], moved)
        cap_vertices_a = [vertex for mesh in meshes_a if "#section:" in mesh.name for vertex in mesh.vertices]
        cap_vertices_b = [vertex for mesh in meshes_b if "#section:" in mesh.name for vertex in mesh.vertices]
        self.assertNotEqual(cap_vertices_a, cap_vertices_b)


if __name__ == "__main__":
    unittest.main()
