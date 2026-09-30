from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from renderer import Camera, Mesh, RenderOptions, Scene, Triangle, render
from renderer.rasterizer import clip_triangle
from scenes.texture_scene import (
    SEAM_BLENDED,
    SEAM_BACKDROP,
    transparent_layers_scene,
    transparent_seam_scene,
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

    def test_transparent_shared_edge_is_covered_exactly_once(self):
        # The translucent plane over a flat backdrop has exactly two legal
        # colours: backdrop (uncovered) and one source-over blend. A darker
        # third colour means the seam was blended twice; a backdrop pixel
        # inside the surface means a gap.
        options = RenderOptions(96, 72, "front")
        image = render(transparent_seam_scene("a"), options)
        legal = {SEAM_BACKDROP, SEAM_BLENDED}
        illegal = [pixel for pixel in image.pixels if pixel not in legal]
        self.assertEqual(illegal, [])
        self.assertIn(SEAM_BLENDED, image.pixels)

    def test_transparent_seam_is_independent_of_triangulation_and_order(self):
        options = RenderOptions(96, 72, "front")
        variants = [
            transparent_seam_scene("a"),
            transparent_seam_scene("b"),
            transparent_seam_scene("a", reverse=True),
            transparent_seam_scene("b", reverse=True),
        ]
        images = [render(scene, options).pixels for scene in variants]
        for pixels in images[1:]:
            self.assertEqual(images[0], pixels)

    def test_transparent_layers_are_independent_of_split_and_submit_order(self):
        options = RenderOptions(104, 80, "front")
        reference = render(transparent_layers_scene("a"), options).pixels
        for scene in (
            transparent_layers_scene("b"),
            transparent_layers_scene("a", reverse_layers=True),
            transparent_layers_scene("b", reverse_layers=True),
        ):
            self.assertEqual(reference, render(scene, options).pixels)

    def test_transparent_layers_stay_consistent_when_camera_moves(self):
        # Moving the camera changes the blend, but triangulation and submit
        # order must still produce identical coverage and colour.
        for camera_name in ("front", "moved"):
            options = RenderOptions(104, 80, camera_name)
            reference = render(transparent_layers_scene("a"), options).pixels
            changed_split = render(transparent_layers_scene("b"), options).pixels
            changed_order = render(transparent_layers_scene("a", reverse_layers=True), options).pixels
            self.assertEqual(reference, changed_split)
            self.assertEqual(reference, changed_order)

    def test_transparent_clip_and_screen_edge_zoom_inspection(self):
        # Local magnification of the strip where the translucent material
        # crosses the near clip boundary and a screen edge. Upscaling each
        # pixel keeps the values readable; both sides of the boundary are
        # compared for the two triangulations, so a bright gap, a dark
        # double-blend line, or a missing green channel would fail the exact
        # pixel equality and legal-colour assertions.
        from renderer.image import Image

        size = 64
        options = RenderOptions(size, size, "front")
        rendered = {
            (diagonal, camera_name): render(
                transparent_seam_scene(diagonal), RenderOptions(size, size, camera_name)
            )
            for diagonal in ("a", "b")
            for camera_name in ("front", "moved")
        }
        for image in rendered.values():
            for pixel in image.pixels:
                self.assertIn(pixel, (SEAM_BACKDROP, SEAM_BLENDED))

        def magnify(image, x0, y0, w, h, scale=4):
            zoom = Image(w * scale, h * scale, (0, 0, 0))
            for row in range(h):
                for col in range(w):
                    colour = image.get_pixel(x0 + col, y0 + row)
                    for dy in range(scale):
                        for dx in range(scale):
                            zoom.set_pixel(col * scale + dx, row * scale + dy, colour)
            return zoom

        inspection_regions = {
            # Front camera: the near clip edge runs across this band, so the
            # magnified strip shows covered and uncovered rows back to back.
            "front": (20, 18, 24, 10),
            # Moved camera: near boundary is slanted and the material also
            # meets a screen edge, so individual rows show both sides of the
            # boundary side by side.
            "moved": (0, 36, 64, 12),
        }
        for camera_name, region in inspection_regions.items():
            x0, y0, w, h = region
            image_a = rendered[("a", camera_name)]
            image_b = rendered[("b", camera_name)]
            seen = set()
            for y in range(y0, y0 + h):
                for x in range(x0, x0 + w):
                    seen.add(image_a.get_pixel(x, y))
            self.assertEqual(seen, {SEAM_BACKDROP, SEAM_BLENDED})
            self.assertEqual(magnify(image_a, *region).pixels, magnify(image_b, *region).pixels)
        # Green is present in the single blend (60, 200, 120 over backdrop)
        # and dominates red/blue there; the reported lost-green seam cannot
        # reappear without an illegal third colour above.
        self.assertGreater(SEAM_BLENDED[1], SEAM_BLENDED[0])

    def test_transparent_seam_round_trip_through_json_at_both_cameras(self):
        for camera_name in ("front", "moved"):
            options = RenderOptions(87, 61, camera_name)
            scenes = [
                transparent_seam_scene("a"),
                transparent_seam_scene("b", reverse=True),
                transparent_layers_scene("b", reverse_layers=True),
            ]
            for scene in scenes:
                before = render(scene, options).pixels
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "seam.json"
                    scene.save(path)
                    after = render(Scene.load(path), options).pixels
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
