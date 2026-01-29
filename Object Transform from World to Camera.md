# Object Transform from World to Camera
## Big picture

A **projection matrix** is used **only when you want to go from 3D → 2D pixels**.

It *is* needed when:

* images are involved
* supervision or evaluation is in pixel space
* rendering must match a real camera

---

## Canonical pipeline

$$
\mathbf{p}_{\text{img}} =\underbrace{P}_{\text{projection}}\* \underbrace{V}_{\text{view}}\ * \underbrace{T_{\text{world}\rightarrow\text{obj}}}_{\text{pose}}\ * \mathbf{p}_{\text{obj}}
$$


Projection is **last**, and only touches **pixels**.

## 1️⃣ Object pose — ❌ NOT using the projection matrix

### Source of truth

The object pose comes from:

```python
M_world_obj = utils.get_actor_user_matrix(mesh_model)
```

That matrix is:

* purely **actor.user_matrix**
* manipulated by UI controls, PnP, reset, undo, etc.
* **already in world coordinates**

👉 **No camera projection matrix is used here.**
👉 **No view matrix is applied here.**

This pose is *independent of the camera*.

So when you export:

```json
"transform_matrix": [[R, t], [0,0,0,1]]
```

that matrix is exactly what the user placed in the scene.

✔️ Safe
✔️ Metric (mm → m conversion aside)
✔️ View-independent

---

## 2️⃣ Camera projection matrix — ✅ exported, but NOT applied to poses

The projection matrix is **computed and saved**, but **never used to modify object pose**.

It is used only for:

* downstream reprojection
* rendering consistency
* dataset consumers (e.g. BOP, NeRF, OpenGL-style pipelines)

Conceptually:

```text
object pose:   World → Object
view matrix:   World → Camera
projection:    Camera → Image
```

You are exporting **all three**, but only the **pose** is used to compute object transforms.

---

## 3️⃣ Where the projection matrix comes from

Inside `utils.build_label_dict_for_image`, the projection matrix is derived from:

* `vtk_camera`
* `fx, fy, cx, cy`
* image width / height
* VTK’s OpenGL-style clip-space

This is roughly:

$$
P = K \cdot [I|0]
$$

converted into a **4×4 OpenGL projection matrix**.

That’s why you see values like:

```json
"camera_projection_matrix": [
  [ 0.765, 0, -0.016, 0 ],
  [ 0, 1.224, -0.040, 0 ],
  [ 0, 0, -2066, -774241 ],
  [ 0, 0, -1, 0 ]
]
```

That matrix is **not used to compute**:

* object translation
* object rotation
* object scale

It’s exported **for completeness**.

---

## 4️⃣ View matrix — also exported, not applied

Same story for:

```json
"camera_view_matrix": [
  [ 1,  0,  0, 0 ],
  [ 0, -1,  0, 0 ],
  [ 0,  0, -1, 0 ],
  [ 0,  0,  0, 1 ]
]
```

This comes from:

* camera position
* focal point
* view-up vector
* VTK’s coordinate conventions

Again:

* ❌ not applied to object pose
* ❌ not baked into transform_matrix
* ✅ exported so downstream code can reproject

---

## 5️⃣ The important guarantee 
> *“Is the view / projection matrix already applied internally when exporting object pose?”*

### ✅ Answer: **NO**

Your exported `transform_matrix` is:

* **world → object**
* **camera-agnostic**
* **not projected**
* **not view-transformed**

---

## Conclusion

* **Object pose export:** ❌ does NOT use projection or view matrices
* **Camera matrices export:** ✅ computed & saved, but **not applied**
* **Rendering only:** projection/view affect visualization, not pose data


## 1️⃣ Pose-estimation datasets (BOP, LINEMOD, YCB-V)

### Used for:

* Generating **2D keypoints**
* Generating **segmentation masks**
* Computing **ADD-S / reprojection error**

Example:

```python
p_cam = T_cam_obj @ p_obj
p_img = K @ p_cam
```

Or OpenGL-style:

```python
p_clip = P @ V @ T @ p_obj
```

### Typical uses

* GT mask rendering
* GT bounding boxes
* Reprojection error evaluation
* Synthetic dataset generation

✔️ **Very common**
✔️ **Exactly why BOP stores P and V**

---

## 2️⃣ Differentiable rendering (NeRF, D-NeRF, Gaussian Splatting)

### Used for:

* Ray generation
* Volume rendering
* Image-based optimization

Projection matrix defines:

* ray directions
* camera frustum
* pixel-to-ray mapping

Example:

```text
pixel → camera ray → world ray → volume integral
```

✔️ **Essential**
✔️ Cannot run without it

---

## 3️⃣ Rendering meshes for visualization / dataset creation

### Used for:

* Rendering RGB images
* Rendering depth maps
* Rendering masks

Code uses it here:

```python
render_mesh(camera=self.plotter.camera.copy())
```

Internally:

* VTK uses **projection + view** to rasterize triangles

✔️ **Used internally**
✔️ But **not baked into pose**

---

## 4️⃣ Reprojection-based loss functions (training time)

Common in:

* Pose refinement networks
* Bundle adjustment
* Tracking

Loss:

$$
\sum_i\left\lVert\pi\\left(PVTX_i\right)-x_i^{\text{gt}}\right\rVert
$$

Where:

* $\pi$ = perspective divide
* $P$ = projection matrix

✔️ **Training-time only**
✔️ Never stored in final pose

---

## 5️⃣ AR / VR / OpenGL / Game engines

### Used for:

* Correct perspective
* Depth testing
* View frustum culling

Standard graphics pipeline:

```text
model → world → view → projection → clip → screen
```

✔️ Mandatory for GPU rendering
✔️ Irrelevant for robotics math

---

## 6️⃣ SLAM / VIO / SfM (indirect use)

Projection matrix isn’t always stored explicitly, but **intrinsics are**.

Used for:

* Feature reprojection
* Epipolar constraints
* Jacobians

Equivalent math:

```python
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

Which is just a **factored projection matrix**.

✔️ Implicit usage
✔️ Intrinsics ≈ projection

---

## In tool specifically

### Projection matrix is used for:

* mesh rendering
* image overlay
* mask export
* reprojection correctness

### Projection matrix is **not used** for:

* computing `transform_matrix`
* computing `location`
* computing `quaternion`
* updating actor pose
* exporting object pose

Pose math lives here:

```python
actor.user_matrix
```

---

## Mental model

* **Pose matrix** → “Where is the object in 3D?”
* **View matrix** → “Where is the camera in 3D?”
* **Projection matrix** → “How does 3D become pixels?”

Only the **last one touches images**.

---

## Conclusion

> The projection matrix is used whenever **pixels matter** (rendering, reprojection, evaluation), and is **never used** when only **3D pose** or **robot motion** matters.
