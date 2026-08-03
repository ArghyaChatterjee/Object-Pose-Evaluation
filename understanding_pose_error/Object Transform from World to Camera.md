# Object Transform from World to Camera

This document describes how object poses, view matrices, and projection matrices are handled during data export. It clarifies which matrices are used for object pose computation and which are provided only for rendering and reprojection purposes.

## Coordinate Transformation Pipeline

The standard graphics pipeline transforms object coordinates into image pixels as follows:

$$ \mathbf{p}_{\text{img}} =\underbrace{P}_{\text{projection}}\* \underbrace{V}_{\text{view}}\ * \underbrace{T_{\text{world}\rightarrow\text{obj}}}_{\text{pose}}\ * \mathbf{p}_{\text{obj}} $$

where:

- **Object Transform (`T`)** — Object pose in the world
- **View Matrix (`V`)** — World-to-camera transformation
- **Projection Matrix (`P`)** — Camera-to-image projection

The projection matrix is only required when converting 3D geometry into image pixels.

---

# Object Pose Export

The exported object pose is obtained directly from the actor transformation:

```python
M_world_obj = utils.get_actor_user_matrix(mesh_model)
```

This matrix represents the object's pose in world coordinates and is manipulated by:

- User interaction
- Pose editing
- PnP updates
- Reset and undo operations

The exported pose is:

- Independent of the camera
- Independent of rendering
- Independent of the projection matrix

Example export:

```json
"transform_matrix": [
    [...],
    [...],
    [...],
    [0, 0, 0, 1]
]
```

This transformation represents the object pose exactly as placed in the scene.

---

# Camera Projection Matrix

The projection matrix is computed during export but is **not applied** to the object pose.

It is included for downstream applications such as:

- Rendering
- Reprojection
- Dataset generation
- Evaluation pipelines

Example:

```json
"camera_projection_matrix": [
    [...],
    [...],
    [...],
    [...]
]
```

The projection matrix does **not** modify:

- Object translation
- Object rotation
- Object scale
- Exported transform matrix

---

# Camera View Matrix

The exported view matrix represents the camera pose relative to the world.

Example:

```json
"camera_view_matrix": [
    [...],
    [...],
    [...],
    [...]
]
```

It is derived from the camera's:

- Position
- Focal point
- View-up vector

Like the projection matrix, it is exported for downstream processing and is **not** applied to the exported object pose.

---

# Export Summary

| Data | Exported | Applied to Object Pose |
|-------|:--------:|:----------------------:|
| Object Transform | ✓ | ✓ |
| Camera View Matrix | ✓ | ✗ |
| Camera Projection Matrix | ✓ | ✗ |

The exported `transform_matrix` always represents the object's pose in world coordinates and remains independent of the camera.

---

# Applications

## Pose Estimation Datasets

Projection and view matrices are commonly used for:

- Rendering segmentation masks
- Rendering depth maps
- Generating bounding boxes
- Computing reprojection error
- ADD / ADD-S evaluation
- Synthetic dataset generation

Typical transformation:

```python
p_cam = T_cam_obj @ p_obj
p_img = K @ p_cam
```

or

```python
p_clip = P @ V @ T @ p_obj
```

---

## Differentiable Rendering

Rendering frameworks such as:

- NeRF
- D-NeRF
- Gaussian Splatting

use the projection matrix for:

- Ray generation
- Camera frustum construction
- Pixel-to-ray mapping
- Volume rendering

---

## Mesh Rendering

Rendering pipelines use the projection and view matrices to generate:

- RGB images
- Depth images
- Segmentation masks

These matrices are used only during rasterization and visualization.

---

## Reprojection-Based Optimization

Pose refinement and optimization methods use projection matrices to minimize image-space errors.

Typical objective:

$$ \sum_i\left\lVert\pi\\left(PVTX_i\right)-x_i^{\text{gt}}\right\rVert $$ 

where:

- $\(P\)$ — Projection matrix
- $\(V\)$ — View matrix
- $\(T\)$ — Object pose
- $\(\pi\)$ — Perspective projection

---

## Graphics Pipelines

Graphics engines (OpenGL, Vulkan, Unreal Engine, Unity, etc.) use:

```text
Model → World → View → Projection → Screen
```

for:

- Perspective rendering
- Depth testing
- View-frustum culling

---

## SLAM and Visual Odometry

SLAM and SfM systems generally use camera intrinsics rather than an explicit projection matrix.

Typical projection:

```python
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

which is mathematically equivalent to applying the camera projection matrix.

---

# Usage Within This Tool

## Projection and View Matrices

Used for:

- Mesh rendering
- Image overlays
- Segmentation mask generation
- Reprojection consistency

Not used for:

- Computing `transform_matrix`
- Computing object position
- Computing object orientation
- Updating actor pose
- Exporting object pose

All pose computation is based on:

```python
actor.user_matrix
```

---

# Summary

| Matrix | Purpose |
|----------|---------|
| **Object Transform** | Defines the object's pose in world coordinates |
| **View Matrix** | Defines the camera pose relative to the world |
| **Projection Matrix** | Projects 3D points into image pixels |

The exported object pose is independent of both the view and projection matrices. Camera matrices are included solely to support rendering, reprojection, visualization, and downstream dataset consumers.
