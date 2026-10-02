# Mini Offline Renderer

A small dependency-free Python rasterizer. It converts triangle meshes through a movable perspective camera, homogeneous frustum clipping, depth-buffered rasterization, and direct PNG output.

## Layout

- `renderer/geometry.py`: vertices, triangles, meshes, and quad helpers.
- `renderer/camera.py`: eye/target/up perspective camera and camera-space transform.
- `renderer/scene.py`: scene composition and versioned JSON persistence.
- `renderer/rasterizer.py`: projection, near/far/side clipping, perspective-correct interpolation, and depth occlusion.
- `renderer/image.py`: in-memory RGB image and PNG encoder.
- `scenes/demo_scene.py`: overlap, camera-motion, partial off-screen, and near-crossing geometry.
- `renderer/texture.py`: procedural textures with repeat-wrap nearest sampling.
- `scenes/texture_scene.py`: alternative-diagonal, near-crossing textured, and overlapping translucent reference scenes.
- `make_feature_comparisons.py`: renders the texture/transparency comparison contact sheet entirely from scene math.
- `make_seam_inspection.py`: local zoom across a transparent material/near-clip boundary for alternative diagonal and split-mesh submissions.
- `renderer/section.py`: joint multi-plane section cuts (`ClipPlane`/`ClipGroup`), per-object/per-plane cap attribution, hole-preserving section triangulation, plane-junction seams, open-boundary markers, and a triangulation-independent section signature.
- `scenes/section_scene.py`: hollow tube (real through-hole), solid box, and an open sheet, with front/back cameras and two independent control cut groups.
- `make_sections.py`: renders every group from both camera sides and writes `section_summary.json` (cap report + signatures across two triangulations).
- `tests/test_renderer.py`: pixel and clipping regression tests.

## Usage

```bash
python make_demo_assets.py
python make_feature_comparisons.py
python make_seam_inspection.py
python make_sections.py
python render_scene.py output/demo_scene.json output/custom.png --width 320 --height 240 --camera moved
python render_scene.py output/section_scene.json output/sections.png --group pair-a --camera front
python -m unittest discover -s tests
```

The demo writes a reusable JSON scene plus `front` and `moved` camera images in `output/`. Rendering uses only the Python standard library; no external renderer is used in the core pipeline.

## Joint multi-plane section cuts

- A `ClipGroup` holds a set of `ClipPlane` half spaces that act jointly (`n.p + d >= 0` is retained). `apply_clip_group(scene, group)` returns a *new* scene plus a `SectionReport`; the source scene and its meshes are never modified. Coincident planes in one group are de-duplicated so identical cuts never double-fill.
- Each section face is emitted as its own mesh named `<object>::cap::<group>::<plane>`, so every cap says which object and which cut produced it; caps from different objects are never stitched into one face even when they touch. Cap color derives from the group palette plus a per-object hue offset.
- Cut chords are welded into planar loops; loops are classified into outer contours and interior holes and triangulated with a hole-aware ear clipper that rejects any triangle whose centroid falls in a hole, so a model with an interior hole keeps that hole under multiple planes.
- Where two cut planes meet, open chains whose ends both terminate on the other plane are closed with a seam segment. Only chains that actually terminate on a second cut plane are sealed; a cut through an open surface leaves genuinely open endpoints, rendered as magenta `::open::` ribbon segments and listed under the report's `open_boundaries` (the range that cannot be capped).
- Vertices that merely lie on a cut plane (the cut passes through an existing vertex or slides along an existing edge, or a face is coplanar with the cut) are tagged separately from true in/out intersections and never fabricate section area.
- Groups are independently copyable: `copy_with` duplicates planes/palette/opacity so moving one plane in one control group leaves the other group unchanged.
- `section_signature(report)` hashes only object/plane attribution, closed-loop/hole counts, and section area - never the original surface triangulation or the cap fan. Re-triangulating the same solid keeps the signature; changing a cut condition changes it. Scenes with cut groups persist as format version 2 (version 1 still loads), so saved sections re-render bit-for-bit. Section triangles are double-sided, giving consistent attribution and transparent-occlusion reading from either camera side.

## Textures and transparency

- Each triangle optionally carries three UV pairs plus a named mesh texture; UVs and opacity are carried through all six clip planes as attribute / w and recovered with perspective-correct interpolation, so a surface looks identical under either diagonal triangulation and stays continuous where it crosses the near plane.
- A top/left pixel-ownership rule assigns every shared edge to exactly one triangle, preventing seams and double alpha contribution.
- Clip intersections are computed from a canonical inside→outside endpoint, clipped polygons are decomposed with a canonical fan pivot, and projected corners are rasterized in screen-position order, so reversing a triangle or changing the diagonal never changes fragment math. Coincident transparent fragments (same pixel, depth, and material response — e.g. a surface submitted as two shared-vertex meshes) are coalesced before the source-over pass, so a seam pixel is blended exactly once; genuinely stacked layers stay separate.
- Opaque triangles are depth-tested first. Translucent fragments (per-triangle `opacity` in `[0, 1]`) are then composited back-to-front with straight source-over alpha blending; each fragment is hidden behind nearer opaque geometry.
- Supported scope is single-pass alpha compositing with per-fragment depth sorting (intersecting layers are therefore ordered correctly at each pixel). Whole-surface order-independent transparency, transmission/refraction, and texture alpha cutouts are not implemented.
- Textures are embedded procedurally generated pixel data in the versioned scene JSON; saving a scene and re-rendering it is bit-for-bit reproducible.
