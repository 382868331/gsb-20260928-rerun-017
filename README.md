# Mini Offline Renderer

A small dependency-free Python rasterizer. It converts triangle meshes through a movable perspective camera, homogeneous frustum clipping, depth-buffered rasterization, and direct PNG output.

## Layout

- `renderer/geometry.py`: vertices, triangles, meshes, and quad helpers.
- `renderer/camera.py`: eye/target/up perspective camera and camera-space transform.
- `renderer/texture.py`: procedurally generated RGBA textures with repeat wrapping and nearest sampling.
- `renderer/scene.py`: scene composition (meshes, cameras, textures) and versioned JSON persistence.
- `renderer/rasterizer.py`: projection, near/far/side clipping, perspective-correct color/UV/opacity interpolation, top-left fill rule, depth occlusion, and back-to-front alpha blending.
- `renderer/image.py`: in-memory RGB image and PNG encoder.
- `scenes/demo_scene.py`: overlap, camera-motion, partial off-screen, and near-crossing geometry.
- `scenes/texture_demo_scene.py`: triangulation-invariant textured panels, a near-crossing textured quad, and overlapping translucent sheets.
- `tests/test_renderer.py`: pixel, clipping, texture, seam, and transparency regression tests.

## Textured surfaces

- UVs are stored per vertex and carried through homogeneous clipping; after a near-plane split the new vertices interpolate `u/w` and `v/w`, so texture continuity is preserved across the clip boundary.
- Screen interpolation is perspective-correct (`(u/w)/z_inv`), so a surface looks identical no matter how it is triangulated. A regression test renders the same quad as a 2x2 and 4x4 mesh (collinear diagonals) and requires byte-identical pixels.
- Rasterization uses a Direct3D-style top/left tie-break: a pixel whose center lies exactly on a shared edge belongs to exactly one adjacent triangle, eliminating background seams.
- Textures are computed in code (checker/stripe generators); their RGBA texels are embedded in the saved JSON. No external/pre-rendered images are used.

## Translucency

Render order is explicit and happens in two passes:

1. All opaque triangles (`opacity == 1`) are depth-tested and write the depth buffer.
2. Fragments from triangles with `opacity < 1` are depth-tested against that buffer (opaque occluders win), grouped per clipped triangle, sorted by farthest camera-space depth first, and composited with the "over" operator `dst = src*a + dst*(1-a)`. Triangle opacity is multiplied with texture alpha.

Supported scope: convex/non-interpenetrating translucent layers composited over opaque geometry, which is the standard painter-ordered alpha pipeline. Per-triangle order is sufficient for the demo sheets; intersecting or cyclic transparent geometry is not order-resolvable without per-pixel fragment sorting and is out of scope. A sheet in front of an opaque surface tints it; a sheet behind an opaque surface is hidden.

## Usage

```bash
python make_demo_assets.py
python render_scene.py output/demo_scene.json output/custom.png --width 320 --height 240 --camera moved
python render_scene.py output/texture_scene.json output/textured.png --camera moved
python -m unittest discover -s tests
```

The demos write reusable JSON scenes plus `front`/`moved` camera images and a save/reload re-render in `output/` (`demo_*` and `texture_*`). Moving the camera or saving and reloading the scene reproduces the same computed pixels. Rendering uses only the Python standard library; no external renderer is used in the core pipeline.
