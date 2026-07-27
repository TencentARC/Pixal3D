# Fix Camera Scene Axis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export ByteSize camera frustums into the existing Pixal3D GLB coordinate frame without rotating the generated shoe mesh.

**Architecture:** Treat `views_N.glb` as the final Y-up mesh frame. Compose the ByteSize Blender-to-GLB conversion into camera geometry only, keep the loaded source scene transform unchanged, and regenerate `scene_1.glb`, `scene_4.glb`, and `scene_8.glb`.

**Tech Stack:** Python 3.10, NumPy, trimesh, unittest, Pixal3D `pixal3d` conda environment.

## Global Constraints

- Preserve every `views_N.glb` byte-for-byte.
- Preserve the generated mesh coordinates and material in each `scene_N.glb`.
- `scene_N.glb` must contain exactly N camera paths.
- `scene_1.glb` must use `upper[0]` and project the shoe as a side view.

---

### Task 1: Lock the GLB coordinate contract

**Files:**
- Modify: `tests/test_multiview_cli.py`
- Modify: `run_multiview_pixal3d.py`

**Interfaces:**
- Consumes: `BytesizeView.transform_matrix` in Blender Z-up coordinates.
- Produces: `_camera_frustum_segments(view, scale)` in the source GLB Y-up frame and identity `_camera_scene_alignment(scene, first_view)`.

- [ ] **Step 1: Write the failing tests**

Assert that `_camera_scene_alignment` is identity, a Blender camera at `(0, -2, 0)` becomes a GLB camera at `(0, 0, -2)`, and destination mesh vertices remain equal to source mesh vertices.

- [ ] **Step 2: Verify RED**

Run:

```bash
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_multiview_cli.MultiViewCliTests.test_camera_scene_alignment_preserves_source_y_up_mesh tests.test_multiview_cli.MultiViewCliTests.test_export_scene_with_cameras_preserves_source_and_adds_paths -v
```

Expected: failures because the current exporter rotates the source scene.

- [ ] **Step 3: Implement the minimal transform fix**

Compose `SOURCE_GLB_TO_Y_UP` into `BLENDER_WORLD_TO_GLB`, return identity from `_camera_scene_alignment`, and keep `_export_scene_with_cameras` otherwise unchanged.

- [ ] **Step 4: Verify GREEN**

Run the command from Step 2 and expect both tests to pass.

### Task 2: Regenerate and validate camera scenes

**Files:**
- Replace: `outputs/bytesize_multiview_huarache_58/scene_1.glb`
- Replace: `outputs/bytesize_multiview_huarache_58/scene_4.glb`
- Replace: `outputs/bytesize_multiview_huarache_58/scene_8.glb`

**Interfaces:**
- Consumes: existing `views_1.glb`, `views_4.glb`, `views_8.glb` and ByteSize reconstruction poses.
- Produces: scene GLBs containing the untouched source mesh plus 1, 4, or 8 camera paths.

- [ ] **Step 1: Record source GLB hashes**

Run `sha256sum` for all three `views_N.glb` files.

- [ ] **Step 2: Regenerate scenes**

Load the ByteSize reconstruction with `max_views=8` and call `_export_scene_with_cameras` for counts 1, 4, and 8.

- [ ] **Step 3: Validate outputs**

Confirm mesh bounds match each source, camera path counts are 1/4/8, `scene_1` uses `upper[0]`, and its projected silhouette matches the side-view condition.

- [ ] **Step 4: Run regression tests**

```bash
/opt/conda/envs/pixal3d/bin/python -m py_compile run_multiview_pixal3d.py tests/test_multiview_cli.py
/opt/conda/envs/pixal3d/bin/python -m unittest tests.test_bytesize_reconstruction tests.test_multiview_cli -v
```

Expected: compilation succeeds and all CPU coordinate/export tests pass.

## Self-Review

- The plan changes only camera-frame conversion and generated scene artifacts.
- Source GLBs remain immutable and no inference rerun is required.
- Test names and production interfaces are consistent.
