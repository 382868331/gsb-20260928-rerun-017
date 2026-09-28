from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from renderer import Camera, Mesh, RenderOptions, Scene, Triangle, render
from renderer.rasterizer import clip_triangle


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


if __name__ == "__main__":
    unittest.main()
