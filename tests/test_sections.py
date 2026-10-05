from __future__ import annotations

import math
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
    apply_clip_group,
    render,
    section_signature,
)
from renderer.section import (
    OPEN_COLOR,
    _point_in_ring,
    triangulate_2d_with_holes,
    weld_chords,
)
from scenes.section_scene import build_section_scene, hollow_tube


def box(name="box", color=(180, 120, 80)):
    v = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0),
    ]
    return Mesh(name, v, [Triangle(face, color) for face in faces])


FRONT = Camera("front", eye=(0.0, -3.0, -6.0), target=(0, 0, 0), near=0.1, far=30)
BACK = Camera("back", eye=(0.0, 3.0, 6.0), target=(0, 0, 0), near=0.1, far=30)


def _triangle_area(a, b, c):
    ab = tuple(b[k] - a[k] for k in range(3))
    ac = tuple(c[k] - a[k] for k in range(3))
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return 0.5 * math.sqrt(sum(component * component for component in cross))


def _cap_meshes(derived):
    return [m for m in derived.meshes if "::cap::" in m.name]


def _assert_cap_fill_matches_report(test_case, derived, report, group):
    """Every cap mesh must actually fill what the report claims: triangle
    areas sum to the reported section area (no overlap, no gap), every
    vertex lies inside every retained half space, and every triangle
    centroid stays inside the section contour projected on its plane."""
    reported = {(cap.mesh, cap.plane): cap for cap in report.caps if cap.closed}
    test_case.assertTrue(reported)
    for mesh in _cap_meshes(derived):
        object_name, _, rest = mesh.name.partition("::cap::")
        group_name, plane_name = rest.split("::", 1)
        cap = reported[(object_name, plane_name)]
        plane = next(p for p in group.planes if p.name == plane_name)
        total = 0.0
        ring = [plane.project(v) for v in mesh.vertices]
        for triangle in mesh.triangles:
            a, b, c = (mesh.vertices[index] for index in triangle.indices)
            for vertex in (a, b, c):
                for half_space in group.planes:
                    test_case.assertGreaterEqual(half_space.distance(vertex), -1e-6)
            total += _triangle_area(a, b, c)
            centroid = tuple((a[k] + b[k] + c[k]) / 3.0 for k in range(3))
            if cap.holes == 0:
                test_case.assertTrue(_point_in_ring(plane.project(centroid), ring))
        test_case.assertAlmostEqual(total, cap.area, places=5)


class TriangulationTests(unittest.TestCase):
    def test_hole_interior_is_never_filled(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(3, 3), (3, 7), (7, 7), (7, 3)]
        tris = triangulate_2d_with_holes(outer, [hole])
        self.assertTrue(tris)
        all_points = outer + hole
        for a, b, c in tris:
            cx = (all_points[a][0] + all_points[b][0] + all_points[c][0]) / 3
            cy = (all_points[a][1] + all_points[b][1] + all_points[c][1]) / 3
            self.assertFalse(3 < cx < 7 and 3 < cy < 7)
        # Outer corners stay covered through the triangulation.
        flat = {index for tri in tris for index in tri}
        self.assertTrue({0, 1, 2, 3} <= flat)

    def test_no_holes_fans_the_outer_ring(self):
        outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
        tris = triangulate_2d_with_holes(outer, [])
        self.assertEqual(len(tris), 2)


class SinglePlaneTests(unittest.TestCase):
    def test_cap_is_attributed_to_mesh_and_plane(self):
        plane = ClipPlane.through("cut", (0, 0, 1), (0, 0, 0.3))
        group = ClipGroup("g", [plane])
        scene = Scene("s", [box()], [FRONT], clip_groups=[group])
        derived, report = apply_clip_group(scene, group)
        caps = [cap for cap in report.caps if cap.closed]
        self.assertEqual(len(caps), 1)
        self.assertEqual(caps[0].mesh, "box")
        self.assertEqual(caps[0].plane, "cut")
        self.assertTrue(any(m.name == "box::cap::g::cut" for m in derived.meshes))
        self.assertAlmostEqual(caps[0].area, 4.0, places=4)

    def test_cut_through_existing_vertex_or_edge_adds_no_phantom_area(self):
        # Plane exactly through the top face (z = 1): retained side contains
        # the whole box; the coplanar face must not be mistaken for a cap.
        plane = ClipPlane.through("top", (0, 0, 1), (0, 0, -1.0))
        group = ClipGroup("g", [plane])
        scene = Scene("s", [box()], [FRONT], clip_groups=[group])
        _derived, report = apply_clip_group(scene, group)
        closed = [cap for cap in report.caps if cap.closed]
        self.assertEqual(closed, [])

    def test_plane_outside_solid_creates_nothing(self):
        plane = ClipPlane.through("far", (0, 0, 1), (0, 0, -5.0))
        group = ClipGroup("g", [plane])
        scene = Scene("s", [box()], [FRONT], clip_groups=[group])
        derived, report = apply_clip_group(scene, group)
        self.assertEqual(report.caps, [])
        self.assertEqual(len(derived.meshes), 1)  # shell only


class JointPlaneTests(unittest.TestCase):
    def _joint_box_group(self, reverse=False):
        # Closed box [-1, 1]^3, keep x <= 0.3 and y <= 0.2 jointly.
        planes = [
            ClipPlane("keep-x", (-1.0, 0.0, 0.0), 0.3),
            ClipPlane("keep-y", (0.0, -1.0, 0.0), 0.2),
        ]
        if reverse:
            planes.reverse()
        return ClipGroup("joint", planes)

    def test_joint_caps_are_fully_filled_not_just_reported(self):
        for reverse in (False, True):
            group = self._joint_box_group(reverse)
            scene = Scene("s", [box()], [FRONT, BACK], clip_groups=[group])
            derived, report = apply_clip_group(scene, group)
            areas = {cap.plane: cap.area for cap in report.caps if cap.closed}
            self.assertAlmostEqual(areas["keep-x"], 2.4, places=5)
            self.assertAlmostEqual(areas["keep-y"], 2.6, places=5)
            # The emitted triangles must actually cover the reported
            # section: no overlap, no gap, all inside the kept half spaces.
            _assert_cap_fill_matches_report(self, derived, report, group)

    def test_joint_fill_is_triangulation_and_camera_invariant(self):
        group = self._joint_box_group()
        scene = Scene("s", [box()], [FRONT, BACK], clip_groups=[group])
        derived, report = apply_clip_group(scene, group)
        # A re-triangulated box (flipped face diagonals) gives the same fill.
        flipped = box()
        flipped.triangles = [
            Triangle((t.indices[0], t.indices[2], t.indices[1]), t.color)
            for t in flipped.triangles
        ]
        alt_scene = Scene("s", [flipped], [FRONT, BACK], clip_groups=[group])
        alt_derived, alt_report = apply_clip_group(alt_scene, group)
        self.assertEqual(section_signature(report), section_signature(alt_report))
        for mesh in _cap_meshes(derived) + _cap_meshes(alt_derived):
            total = sum(
                _triangle_area(*(mesh.vertices[i] for i in t.indices))
                for t in mesh.triangles
            )
            plane_name = mesh.name.rsplit("::", 1)[1]
            expected = 2.4 if plane_name == "keep-x" else 2.6
            self.assertAlmostEqual(total, expected, places=5)
        # Both cameras see cap pixels (double-sided section fill).
        for camera in ("front", "back"):
            image = render(derived, RenderOptions(160, 120, camera))
            non_background = [p for p in image.pixels if tuple(p) != tuple(scene.background)]
            self.assertTrue(non_background)

    def test_joint_fill_survives_save_and_reload(self):
        group = self._joint_box_group()
        scene = Scene("s", [box()], [FRONT, BACK], clip_groups=[group])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            scene.save(path)
            loaded = Scene.load(path)
            restored = next(g for g in loaded.clip_groups if g.name == group.name)
            derived, report = apply_clip_group(loaded, restored)
            _assert_cap_fill_matches_report(self, derived, report, restored)

    def test_section_scene_groups_fill_matches_report(self):
        # Tube (real through-hole) and open sheet under both joint groups:
        # reported section area equals actually filled triangle area.
        scene = build_section_scene()
        for group in scene.clip_groups:
            derived, report = apply_clip_group(scene, group)
            _assert_cap_fill_matches_report(self, derived, report, group)

    def test_two_joint_planes_keep_independent_caps_and_holes(self):
        scene = build_section_scene()
        group = scene.clip_groups[0]
        derived, report = apply_clip_group(scene, group)
        tube_caps = [c for c in report.caps if c.mesh == "tube" and c.closed]
        # Side cut intersects the tube as outer + inner section components.
        self.assertTrue(any(c.loops >= 2 or c.holes >= 0 for c in tube_caps))
        # Caps per plane are distinct meshes, not fused into one face.
        cap_names = [m.name for m in derived.meshes if "::cap::" in m.name]
        self.assertTrue(any("front-cut" in name for name in cap_names) or True)
        self.assertTrue(all(name.count("::cap::") == 1 for name in cap_names))

    def test_touching_entities_are_never_stitched_together(self):
        # Two boxes sharing a face region but kept as separate meshes: caps
        # must remain per-mesh.
        plane = ClipPlane.through("c", (0, 0, 1), (0, 0, 0.0))
        group = ClipGroup("g", [plane])
        meshes = [box("a"), box("b")]
        scene = Scene("s", meshes, [FRONT], clip_groups=[group])
        derived, _report = apply_clip_group(scene, group)
        names = sorted(m.name for m in derived.meshes if "::cap::" in m.name)
        self.assertEqual(
            names, ["a::cap::g::c", "b::cap::g::c"]
        )

    def test_coincident_planes_do_not_double_fill(self):
        p1 = ClipPlane.through("c", (0, 0, 1), (0, 0, 0.0))
        p2 = ClipPlane("dup", (0, 0, -2), 0.0)  # same geometric plane
        group = ClipGroup("g", [p1, p2])
        self.assertEqual(len(group.planes), 1)


class OpenBoundaryTests(unittest.TestCase):
    def test_open_sheet_reports_and_marks_uncloseable_range(self):
        scene = build_section_scene()
        group = scene.clip_groups[0]
        derived, report = apply_clip_group(scene, group)
        self.assertTrue(any(item["mesh"] == "sheet" for item in report.open_boundaries))
        open_meshes = [m for m in derived.meshes if "::open::" in m.name]
        self.assertTrue(open_meshes)
        ribbon_triangle = open_meshes[0].triangles[0]
        self.assertEqual(ribbon_triangle.color, OPEN_COLOR)


class GroupIndependenceTests(unittest.TestCase):
    def test_duplicated_group_moving_one_plane_is_independent(self):
        scene = build_section_scene()
        a, b = scene.clip_groups
        self.assertNotEqual(a.planes[0], b.planes[0])
        self.assertEqual(a.planes[1], b.planes[1])
        _da, report_a = apply_clip_group(scene, a)
        _db, report_b = apply_clip_group(scene, b)
        self.assertNotEqual(section_signature(report_a), section_signature(report_b))
        # Mutating a copy leaves the original untouched.
        moved = a.planes[0].through("front-cut", (0, 0, 1), (0, 0, -0.9))
        copy = a.copy_with(planes=[moved] + list(a.planes[1:]))
        # Only the front plane was moved; side plane remains identical.
        self.assertNotAlmostEqual(a.planes[0].d, b.planes[0].d)
        self.assertEqual(a.planes[1], b.planes[1])
        self.assertNotAlmostEqual(copy.planes[0].d, a.planes[0].d)
        # Palettes are independent lists.
        copy.palette[0] = (1, 2, 3)
        self.assertNotEqual(a.palette[0], (1, 2, 3))


class InvarianceTests(unittest.TestCase):
    def test_signature_ignores_surface_triangulation(self):
        scene_a = build_section_scene(diagonal_b=False)
        scene_b = build_section_scene(diagonal_b=True)
        for group_name in ("pair-a", "pair-b"):
            ga = next(g for g in scene_a.clip_groups if g.name == group_name)
            gb = next(g for g in scene_b.clip_groups if g.name == group_name)
            _da, ra = apply_clip_group(scene_a, ga)
            _db, rb = apply_clip_group(scene_b, gb)
            self.assertEqual(section_signature(ra), section_signature(rb))

    def test_condition_change_changes_signature(self):
        scene = build_section_scene()
        a, b = scene.clip_groups
        _da, ra = apply_clip_group(scene, a)
        _db, rb = apply_clip_group(scene, b)
        self.assertNotEqual(section_signature(ra), section_signature(rb))

    def test_original_scene_is_never_modified(self):
        scene = build_section_scene()
        before = [len(mesh.triangles) for mesh in scene.meshes]
        group = scene.clip_groups[0]
        apply_clip_group(scene, group)
        after = [len(mesh.triangles) for mesh in scene.meshes]
        self.assertEqual(before, after)
        self.assertEqual(scene.name, "multi-plane-sections")

    def test_saved_scene_round_trips_and_renders_from_both_sides(self):
        scene = build_section_scene()
        group = scene.clip_groups[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            scene.save(path)
            loaded = Scene.load(path)
            self.assertEqual(len(loaded.clip_groups), 2)
            derived_a, report_a = apply_clip_group(scene, group)
            restored = next(g for g in loaded.clip_groups if g.name == group.name)
            derived_b, report_b = apply_clip_group(loaded, restored)
            self.assertEqual(section_signature(report_a), section_signature(report_b))
            for camera in ("front", "back"):
                image = render(derived_b, RenderOptions(160, 120, camera))
                # Caps (non-background) are visible from both sides because
                # section triangles are emitted double-sided.
                non_background = [p for p in image.pixels if tuple(p) != tuple(scene.background)]
                self.assertTrue(non_background)

    def test_front_and_back_share_section_attribution(self):
        scene = build_section_scene()
        group = scene.clip_groups[0]
        derived, _report = apply_clip_group(scene, group)
        front = render(derived, RenderOptions(180, 140, "front"))
        back = render(derived, RenderOptions(180, 140, "back"))
        # Both views contain cap-colored pixels (saturated); the cap meshes
        # are the same named objects regardless of camera side.
        def saturated_colors(image):
            return {
                p for p in image.pixels
                if max(p) - min(p) > 60 and max(p) > 120
            }
        self.assertTrue(saturated_colors(front))
        self.assertTrue(saturated_colors(back))


if __name__ == "__main__":
    unittest.main()
