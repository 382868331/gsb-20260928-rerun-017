# Mini Offline Renderer

A small dependency-free Python rasterizer. It converts triangle meshes through a movable perspective camera, homogeneous frustum clipping, depth-buffered rasterization, and direct PNG output.

## Layout

- `renderer/geometry.py`: vertices, triangles, meshes, and quad helpers.
- `renderer/camera.py`: eye/target/up perspective camera and camera-space transform.
- `renderer/scene.py`: scene composition and versioned JSON persistence (v2 adds clip groups).
- `renderer/section.py`: joint clipping planes, per-mesh/per-plane section caps, holes, and open-boundary markers.
- `renderer/rasterizer.py`: projection, near/far/side clipping, perspective-correct interpolation, and depth occlusion.
- `renderer/image.py`: in-memory RGB image and PNG encoder.
- `scenes/demo_scene.py`: overlap, camera-motion, partial off-screen, and near-crossing geometry.
- `renderer/texture.py`: procedural textures with repeat-wrap nearest sampling.
- `scenes/texture_scene.py`: alternative-diagonal, near-crossing textured, and overlapping translucent reference scenes.
- `scenes/section_scene.py`: two transparent solids, a square-tube (internal hole) body, and an open sheet cut by two joint planes, plus a moved control group.
- `make_feature_comparisons.py`: renders the texture/transparency comparison contact sheet entirely from scene math.
- `make_seam_inspection.py`: local zoom across a transparent material/near-clip boundary for alternative diagonal and split-mesh submissions.
- `tests/test_renderer.py`: pixel and clipping regression tests.

## Usage

```bash
python make_demo_assets.py
python make_feature_comparisons.py
python make_section_comparisons.py
python make_seam_inspection.py
python render_scene.py output/demo_scene.json output/custom.png --width 320 --height 240 --camera moved
python render_scene.py output/section_scene.json output/section.png --camera front --clip-group paired
python -m unittest discover -s tests
```

The demo writes a reusable JSON scene plus `front` and `moved` camera images in `output/`. Rendering uses only the Python standard library; no external renderer is used in the core pipeline.

## Textures and transparency

- Each triangle optionally carries three UV pairs plus a named mesh texture; UVs and opacity are carried through all six clip planes as attribute / w and recovered with perspective-correct interpolation, so a surface looks identical under either diagonal triangulation and stays continuous where it crosses the near plane.
- A top/left pixel-ownership rule assigns every shared edge to exactly one triangle, preventing seams and double alpha contribution.
- Clip intersections are computed from a canonical inside→outside endpoint, clipped polygons are decomposed with a canonical fan pivot, and projected corners are rasterized in screen-position order, so reversing a triangle or changing the diagonal never changes fragment math. Coincident transparent fragments (same pixel, depth, and material response — e.g. a surface submitted as two shared-vertex meshes) are coalesced before the source-over pass, so a seam pixel is blended exactly once; genuinely stacked layers stay separate.
- Opaque triangles are depth-tested first. Translucent fragments (per-triangle `opacity` in `[0, 1]`) are then composited back-to-front with straight source-over alpha blending; each fragment is hidden behind nearer opaque geometry.
- Supported scope is single-pass alpha compositing with per-fragment depth sorting (intersecting layers are therefore ordered correctly at each pixel). Whole-surface order-independent transparency, transmission/refraction, and texture alpha cutouts are not implemented.
- Textures are embedded procedurally generated pixel data in the versioned scene JSON; saving a scene and re-rendering it is bit-for-bit reproducible.

## Joint clipping planes and sections

- A scene may carry one or more `ClipGroup`s; each group is an ordered set of `ClipPlane` half-spaces applied together (`ClipPlane(name, normal, offset, color, opacity)`, inside is `dot(normal, p) + offset >= 0`). Select one with `RenderOptions(..., clip_group_name=...)` or `render_scene.py --clip-group`.
- Every original mesh is clipped by the group without mutating the source mesh. The kept surface keeps the original color/opacity/texture; the generated section mesh is named `<mesh>#section:<group>/<plane>` and painted with that plane's section color, so every filled cross-section states which object and which cut produced it. Section meshes are generated per source mesh, so two entities are never stitched into one face.
- Plane/plane intersection edges are produced by clipping each plane's section polygon with the other planes' half-spaces; coincident planes in a group coalesce to the earlier plane's cap instead of painting the fill twice. Cap polygons triangulate outer loops and internal holes together (even-odd ear clipping across hole bridges), so a square tube keeps its hole under multiple planes, and cuts through vertices or along existing edges cannot fill nonexistent solid.
- Open surfaces cannot be capped: their un-closeable cut ranges are emitted as a magenta in-plane boundary ribbon (`<mesh>#open:<group>/<plane>`) rather than an invented filled region.
- `group.copied(name)` duplicates the conditions with independent plane/section colors; `.with_plane(name, offset=...)` moves one plane without touching the original. A section built from the same solid with either diagonal triangulation produces the same cap, while moving a plane changes it; sections persist in scene JSON (format version 2, version 1 files still load).
