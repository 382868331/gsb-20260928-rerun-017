from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from renderer import Camera, ClipPlane, Mesh, RenderOptions, Scene, Triangle, render, render_with_report
from renderer.rasterizer import clip_triangle
from scenes.clip_scene import box_mesh, square_tube_mesh
from scenes.texture_scene import (
    near_textured_scene,
    transparent_overlap_scene,
    triangulation_scene,
)


class RendererTests(unittest.TestCase):
    def test_camera_axes_are_right_handed(self):
        camera = Camera("front", eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0))
        self.assertAlmostEqual(camera.view_position((1.0, 0.0, 0.0))[0], 1.0)
        self.assertAlmostEqual(camera.view_position((0.0, 1.0, 0.0))[1], 1.0)
        self.assertAlmostEqual(camera.view_position((0.0, 0.0, 1.0))[2], 1.0)

    def test_depth_occlusion(self):
        scene = Scene(
            "depth",
            [
                Mesh.quad("back", [(-1, -1, 3), (1, -1, 3), (1, 1, 3), (-1, 1, 3)], (20, 40, 200)),
                Mesh.quad("front", [(-.4, -.4, 2), (.4, -.4, 2), (.4, .4, 2), (-.4, .4, 2)], (220, 30, 30)),
            ],
            [Camera("front", near=.1, far=10)],
            background=(0, 0, 0),
        )
        image = render(scene, RenderOptions(41, 41))
        self.assertEqual(image.get_pixel(20, 20), (220, 30, 30))
        self.assertEqual(image.get_pixel(8, 20), (20, 40, 200))

    def test_near_clipping_retains_piece_and_seam(self):
        crossing_quad = Mesh(
            "crossing",
            [(-5, -5, .1), (5, -5, .1), (5, 5, 3), (-5, 5, 3)],
            [
                Triangle((0, 1, 2), (10, 180, 90)),
                Triangle((0, 2, 3), (10, 180, 90)),
            ],
        )
        scene = Scene(
            "near",
            [crossing_quad],
            [Camera("front", near=.5, far=10, fov_y_degrees=45)],
            background=(0, 0, 0),
        )
        image = render(scene, RenderOptions(64, 48))
        background_count = sum(1 for pixel in image.pixels if pixel == (0, 0, 0))
        self.assertLess(background_count, 5)

    def test_triangle_completely_behinds_near_is_empty(self):
        camera = Camera("front", near=.5, far=10)
        from renderer.rasterizer import _clip_vertex

        triangle = [
            _clip_vertex(camera, (-1, -1, .1), (1, 2, 3), 1.0),
            _clip_vertex(camera, (1, -1, .1), (1, 2, 3), 1.0),
            _clip_vertex(camera, (0, 1, .1), (1, 2, 3), 1.0),
        ]
        self.assertEqual(clip_triangle(triangle), [])

    def test_partly_offscreen_triangle_still_covers_edge(self):
        triangle = Mesh.quad(
            "offscreen",
            [(1.4, -1, 3), (8, -1, 3), (8, 1, 3), (1.4, 1, 3)],
            (50, 220, 50),
        )
        scene = Scene("offscreen", [triangle], [Camera("front", near=.1, far=10)], background=(0, 0, 0))
        image = render(scene, RenderOptions(80, 60))
        self.assertIn((50, 220, 50), image.pixels)
        self.assertIn((0, 0, 0), image.pixels)

    def test_saved_scene_reproduces_image_and_camera_changes_view(self):
        back = Mesh.quad("back", [(-1, -1, 3), (1, -1, 3), (1, 1, 3), (-1, 1, 3)], (30, 120, 200))
        offset = Mesh.quad("offset", [(-.4, -.6, 2), (.8, -.6, 2), (.8, .7, 2), (-.4, .7, 2)], (220, 120, 30))
        cameras = [
            Camera("front"),
            Camera("moved", eye=(.6, .1, -1), target=(0, 0, 1)),
        ]
        scene = Scene("saved", [back, offset], cameras, background=(5, 5, 5))
        options = RenderOptions(57, 43, "front")
        before = render(scene, options).pixels
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.json"
            scene.save(path)
            loaded = Scene.load(path)
            after = render(loaded, options).pixels
        self.assertEqual(before, after)
        moved = render(scene, RenderOptions(57, 43, "moved")).pixels
        self.assertNotEqual(before, moved)

    def test_textured_surface_is_independent_of_diagonal_split(self):
        options = RenderOptions(96, 72, "front")
        image_a = render(triangulation_scene("a"), options)
        image_b = render(triangulation_scene("b"), options)
        differing = sum(1 for a, b in zip(image_a.pixels, image_b.pixels) if a != b)
        self.assertLess(differing / len(image_a.pixels), 0.002)
        # The checker texture actually reaches the image rather than flat color.
        self.assertGreaterEqual(len(set(image_a.pixels)), 4)

    def test_texture_coordinates_survive_near_plane_clipping(self):
        image = render(near_textured_scene(), RenderOptions(96, 72, "front"))
        colors = set(image.pixels)
        self.assertIn((70, 130, 220), colors)
        self.assertIn((230, 220, 120), colors)
        self.assertIn((200, 80, 80), colors)

    def test_transparency_blends_back_to_front(self):
        options = RenderOptions(64, 48, "front")
        image = render(transparent_overlap_scene(), options)
        center = image.get_pixel(32, 24)
        for channel in center:
            self.assertNotIn(channel, (0, 255))
        # Alpha order changes when the camera moves.
        moved = render(transparent_overlap_scene(), RenderOptions(64, 48, "moved"))
        self.assertNotEqual(image.pixels, moved.pixels)

    def test_textured_transparent_scene_round_trip_through_json(self):
        scenes = [triangulation_scene("a"), near_textured_scene(), transparent_overlap_scene()]
        options = RenderOptions(73, 57, "front")
        for scene in scenes:
            before = render(scene, options).pixels
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "scene.json"
                scene.save(path)
                after = render(Scene.load(path), options).pixels
            self.assertEqual(before, after)

    def _green_surface_variants(self):
        corners = [
            (-1.4, -1.1, 0.3),
            (1.9, -1.1, 2.8),
            (1.1, 1.2, 2.1),
            (-1.9, 1.2, 1.1),
        ]
        green = (0, 210, 0)
        opacity = 0.5
        variants = {
            "diagonal_a": [Mesh(
                "surface", corners,
                [Triangle((0, 1, 2), green, opacity=opacity),
                 Triangle((0, 2, 3), green, opacity=opacity)])],
            "diagonal_b": [Mesh(
                "surface", corners,
                [Triangle((0, 3, 2), green, opacity=opacity),
                 Triangle((0, 2, 1), green, opacity=opacity)])],
            "reordered": [Mesh(
                "surface", corners,
                [Triangle((0, 2, 3), green, opacity=opacity),
                 Triangle((0, 1, 2), green, opacity=opacity)])],
            # Shared-vertex split: same surface submitted as two meshes.
            "split_meshes": [
                Mesh("surface_a", corners, [Triangle((0, 1, 2), green, opacity=opacity)]),
                Mesh("surface_b", corners, [Triangle((0, 2, 3), green, opacity=opacity)])],
            "split_reversed": [
                Mesh("surface_a", corners, [Triangle((2, 3, 0), green, opacity=opacity)]),
                Mesh("surface_b", corners, [Triangle((2, 0, 1), green, opacity=opacity)])],
        }
        return corners, variants

    def test_transparent_surface_is_independent_of_split_and_submission(self):
        corners, variants = self._green_surface_variants()
        cameras = [
            Camera("front", near=1.5, far=20, fov_y_degrees=55),
            Camera("moved", eye=(0.9, 0.25, -0.8), target=(0, 0, 1), near=0.5, far=20),
        ]
        for camera in cameras:
            for width, height in ((160, 120), (97, 73)):
                options = RenderOptions(width, height, camera.name)
                images = {
                    name: render(Scene("seam", meshes, [camera], background=(255, 255, 255)), options).pixels
                    for name, meshes in variants.items()
                }
                reference = images["diagonal_a"]
                for name, pixels in images.items():
                    self.assertEqual(
                        reference, pixels,
                        msg=f"transparent seam differs for {name} with camera {camera.name}",
                    )

    def test_clip_boundary_has_no_gap_or_double_blend(self):
        # The near-plane cut runs diagonally across the surface; pixels just
        # on either side of the boundary must carry the same single
        # source-over contribution (green present exactly once).
        corners, variants = self._green_surface_variants()
        camera = Camera("front", near=1.5, far=20, fov_y_degrees=55)
        options = RenderOptions(320, 240, "front")
        scene = Scene("seam", variants["split_meshes"], [camera], background=(0, 0, 0))
        image = render(scene, options)
        covered = [p for p in image.pixels if p != (0, 0, 0)]
        self.assertTrue(covered)
        green_values = {pixel[1] for pixel in covered}
        # A single source-over blend of (0,210) at 0.5 over black is 105.
        # A missed seam stays 0 (green lost); a double blend becomes 157.
        self.assertTrue(green_values)
        self.assertTrue(all(value == 105 for value in green_values), green_values)

    def test_multi_layer_transparent_split_round_trip(self):
        corners, variants = self._green_surface_variants()
        red_corners = [(-2.8, -2.2, 1.4), (3.1, -2.2, 2.6), (3.1, 2.4, 2.6), (-2.8, 2.4, 1.4)]
        red = [
            Mesh("red_a", red_corners, [Triangle((0, 1, 2), (220, 40, 40), opacity=0.4)]),
            Mesh("red_b", red_corners, [Triangle((0, 2, 3), (220, 40, 40), opacity=0.4)]),
        ]
        camera = Camera("front", near=1.5, far=20, fov_y_degrees=55)
        options = RenderOptions(120, 90, "front")
        for green_key in ("diagonal_a", "split_meshes", "split_reversed"):
            scene = Scene("layers", variants[green_key] + red, [camera], background=(10, 10, 24))
            before = render(scene, options).pixels
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "layers.json"
                scene.save(path)
                after = render(Scene.load(path), options).pixels
            self.assertEqual(before, after, msg=f"round trip differs for {green_key}")
        reference = render(Scene("layers", variants["diagonal_a"] + red, [camera], background=(10, 10, 24)), options).pixels
        split = render(Scene("layers", variants["split_meshes"] + red, [camera], background=(10, 10, 24)), options).pixels
        self.assertEqual(reference, split)


class ClipPlaneTests(unittest.TestCase):
    SECTION = (250, 205, 60)
    SECTION_B = (90, 220, 235)

    def _scene(self, meshes, planes, cameras=None):
        cameras = cameras or [Camera("front", eye=(-2.0, 0.6, 0.5), target=(0, 0, 3), near=0.1, far=25)]
        return Scene("clip", meshes, cameras, background=(0, 0, 0), clip_planes=planes)

    def _color_pixels(self, image, color):
        return {i for i, pixel in enumerate(image.pixels) if pixel == color}

    def test_cut_shows_distinct_traceable_section(self):
        box = box_mesh("box-a", (0, 0, 3), (1, 1, 1), (80, 150, 220), opacity=0.5)
        plane = ClipPlane("cut", (0, 0, 0), (1, 0, 0), self.SECTION)
        image, report = render_with_report(self._scene([box], [plane]), RenderOptions(60, 60))
        self.assertIn(self.SECTION, image.pixels)
        self.assertEqual(
            [(s.mesh_name, s.plane_name) for s in report.sections],
            [("box-a", "cut")],
        )
        self.assertEqual(report.gaps, [])

    def test_section_identical_under_alternative_triangulation(self):
        plane = ClipPlane("cut", (0.1, 0.05, 0), (1, 0.4, 0.2), self.SECTION)
        options = RenderOptions(80, 60)
        caps = []
        reports = []
        for diagonal in (0, 1):
            box = box_mesh("box", (0, 0, 3), (1.2, 1.0, 0.8), (80, 150, 220), 0.5, diagonal)
            image, report = render_with_report(self._scene([box], [plane]), options)
            caps.append(self._color_pixels(image, self.SECTION))
            reports.append(report)
        self.assertTrue(caps[0])
        # The section face itself is bit-identical across triangulations.
        self.assertEqual(caps[0], caps[1])
        summary = [
            [(s.mesh_name, s.plane_name, s.loop_count, s.color) for s in report.sections]
            for report in reports
        ]
        self.assertEqual(summary[0], summary[1])
        self.assertEqual(reports[0].gaps, reports[1].gaps)

    def test_plane_on_face_and_vertex_does_not_flicker(self):
        options = RenderOptions(60, 60)
        counts = []
        renders = []
        for offset in (0.0, -1e-7, -2e-7):
            box = box_mesh("box", (0, 0, 3), (1, 1, 1), (80, 150, 220), 0.5)
            plane = ClipPlane("cut", (0.5 + offset, 0, 0), (1, 0, 0), self.SECTION)
            image, report = render_with_report(self._scene([box], [plane]), options)
            self.assertEqual(report.gaps, [])
            counts.append(len(self._color_pixels(image, self.SECTION)))
            renders.append(image.pixels)
        # Plane resting exactly on the box face renders the same cap as an
        # infinitesimally inward plane: no sudden fill, no flicker.
        self.assertEqual(renders[0], renders[1])
        self.assertEqual(renders[1], renders[2])
        self.assertGreater(counts[0], 0)

    def test_plane_through_corner_vertex_stays_stable(self):
        options = RenderOptions(60, 60)
        normal = (1.0, 1.0, 1.0)
        corner = (0.5, 0.5, 3.5)
        counts = []
        for delta in (0.0, -1e-7, 1e-7):
            box = box_mesh("box", (0, 0, 3), (1, 1, 1), (80, 150, 220), 0.5)
            point = tuple(corner[k] + delta for k in range(3))
            plane = ClipPlane("cut", point, normal, self.SECTION)
            _, report = render_with_report(self._scene([box], [plane]), options)
            image, _ = render_with_report(self._scene([box], [plane]), options)
            self.assertEqual(report.gaps, [])
            counts.append(len(self._color_pixels(image, self.SECTION)))
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_internal_hole_stays_empty(self):
        tube = square_tube_mesh("tube", (0, 0, 3), 0.7, 0.5, 0.25, (230, 170, 90), 0.6)
        plane = ClipPlane("cut", (0, 0, 3), (0, 0, 1), self.SECTION)
        camera = Camera("front", eye=(0, 0, 0), target=(0, 0, 1), near=0.1, far=20)
        image, report = render_with_report(self._scene([tube], [plane], [camera]), RenderOptions(80, 80))
        self.assertEqual(report.gaps, [])
        self.assertEqual(len(report.sections), 1)
        self.assertEqual(report.sections[0].loop_count, 2)  # outer + inner boundary
        self.assertEqual(image.get_pixel(40, 40), (0, 0, 0))  # bore stays empty
        self.assertIn(self.SECTION, image.pixels)  # annulus ring is filled

    def test_open_surface_reports_gap_instead_of_filling(self):
        panel = Mesh.quad(
            "panel",
            [(-1, -1, 3), (1, -1, 3), (1, 1, 3), (-1, 1, 3)],
            (200, 120, 220),
            opacity=0.5,
        )
        plane = ClipPlane("cut", (0, 0, 0), (1, 0, 0), self.SECTION)
        image, report = render_with_report(self._scene([panel], [plane]), RenderOptions(60, 60))
        self.assertEqual(report.sections, [])
        self.assertEqual(len(report.gaps), 1)
        gap = report.gaps[0]
        self.assertEqual((gap.mesh_name, gap.plane_name), ("panel", "cut"))
        self.assertAlmostEqual(gap.length, 2.0, places=3)
        self.assertNotIn(self.SECTION, image.pixels)

    def test_intersecting_transparent_objects_keep_separate_sections(self):
        box_a = box_mesh("box-a", (-0.3, 0, 3), (1, 1, 1), (80, 150, 220), 0.5)
        box_b = box_mesh("box-b", (0.3, 0, 3), (1, 1, 1), (120, 220, 140), 0.5)
        plane = ClipPlane("cut", (0, 0, 0), (1, 0, 0), self.SECTION)
        image, report = render_with_report(self._scene([box_a, box_b], [plane]), RenderOptions(60, 60))
        sources = {(s.mesh_name, s.plane_name) for s in report.sections}
        self.assertEqual(sources, {("box-a", "cut"), ("box-b", "cut")})
        self.assertIn(self.SECTION, image.pixels)

    def test_two_planes_have_independent_colors(self):
        def build(color_a):
            box = box_mesh("box", (0, 0, 3), (1.2, 1.2, 1.2), (80, 150, 220), 0.5)
            planes = [
                ClipPlane("cut-x", (0, 0, 0), (1, 0, 0), color_a),
                ClipPlane("cut-y", (0, -0.2, 0), (0, 1, 0), self.SECTION_B),
            ]
            return self._scene([box], planes)

        options = RenderOptions(80, 80)
        image_a, report_a = render_with_report(build(self.SECTION), options)
        recolored = (240, 80, 80)
        image_b, report_b = render_with_report(build(recolored), options)
        # Recoloring cut-x leaves every cut-y section pixel untouched.
        self.assertEqual(
            self._color_pixels(image_a, self.SECTION_B),
            self._color_pixels(image_b, self.SECTION_B),
        )
        self.assertTrue(self._color_pixels(image_a, self.SECTION))
        self.assertEqual(
            self._color_pixels(image_a, self.SECTION),
            self._color_pixels(image_b, recolored),
        )
        self.assertEqual(
            {(s.mesh_name, s.plane_name) for s in report_a.sections},
            {(s.mesh_name, s.plane_name) for s in report_b.sections},
        )

    def test_section_visible_from_both_sides_and_after_reload(self):
        cameras = [
            Camera("west", eye=(-4, 0.3, 3), target=(0, 0, 3), near=0.1, far=20),
            Camera("east", eye=(4, 0.3, 3), target=(0, 0, 3), near=0.1, far=20),
        ]
        box = box_mesh("box", (0, 0, 3), (1, 1, 1), (80, 150, 220), 0.5)
        plane = ClipPlane("cut", (0, 0, 0), (1, 0, 0), self.SECTION)
        scene = self._scene([box], [plane], cameras)
        west = render(scene, RenderOptions(80, 80, "west"))
        east = render(scene, RenderOptions(80, 80, "east"))
        # From the removed side the opaque cap is seen directly; from the
        # kept side the remaining transparent shell blends over it. Both are
        # the same depth order evaluated from two cameras.
        self.assertEqual(west.get_pixel(40, 40), self.SECTION)
        self.assertIn(self.SECTION, west.pixels)
        self.assertNotIn((0, 0, 0), [east.get_pixel(40, 40)])
        self.assertNotEqual(east.get_pixel(40, 40), self.SECTION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.json"
            scene.save(path)
            reloaded = Scene.load(path)
            self.assertEqual(len(reloaded.clip_planes), 1)
            again = render(reloaded, RenderOptions(80, 80, "west"))
        self.assertEqual(west.pixels, again.pixels)


if __name__ == "__main__":
    unittest.main()
