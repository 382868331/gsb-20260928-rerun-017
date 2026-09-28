# Mini Offline Renderer

A small dependency-free Python rasterizer. It converts triangle meshes through a movable perspective camera, homogeneous frustum clipping, depth-buffered rasterization, and direct PNG output.

## Layout

- `renderer/geometry.py`: vertices, triangles, meshes, and quad helpers.
- `renderer/camera.py`: eye/target/up perspective camera and camera-space transform.
- `renderer/scene.py`: scene composition and versioned JSON persistence.
- `renderer/rasterizer.py`: projection, near/far/side clipping, perspective-correct interpolation, and depth occlusion.
- `renderer/image.py`: in-memory RGB image and PNG encoder.
- `scenes/demo_scene.py`: overlap, camera-motion, partial off-screen, and near-crossing geometry.
- `tests/test_renderer.py`: pixel and clipping regression tests.

## Usage

```bash
python make_demo_assets.py
python render_scene.py output/demo_scene.json output/custom.png --width 320 --height 240 --camera moved
python -m unittest discover -s tests
```

The demo writes a reusable JSON scene plus `front` and `moved` camera images in `output/`. Rendering uses only the Python standard library; no external renderer is used in the core pipeline.
