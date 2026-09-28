# Mini Offline Renderer

A small Python software-rendering pipeline. The scene, camera, triangle processing, rasterizer, and PNG output are separate modules, and the rasterizer core uses only Python rather than an existing renderer.

## Pipeline

- `mini_render/geometry.py` defines serializable meshes and scenes.
- `mini_render/camera.py` builds view and perspective-projection matrices.
- `mini_render/rasterizer.py` performs homogeneous six-plane clipping, near-plane interpolation, perspective-correct depth, shared-edge fill rules, and per-pixel occlusion.
- `mini_render/image.py` encodes RGB data as PNG with the standard library.

## Validation Scene

```bash
python scripts/create_validation_scene.py
python -m mini_render.cli scenes/validation.json outputs/main.png --width 320 --height 240 --camera main
python -m mini_render.cli scenes/validation.json outputs/side.png --width 320 --height 240 --camera side
```

The scene contains a large sloped cyan quad that crosses the near plane and extends beyond the frustum, a closer red quad for depth overlap, and a yellow quad partly outside the right edge. Run tests with:

```bash
python -m pytest
```
