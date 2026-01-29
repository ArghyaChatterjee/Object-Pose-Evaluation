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
[\mathbf{p}_{img}=\underbrace{P}*{\text{projection}}*\underbrace{V}*{\text{view}}*\underbrace{T_{world \to obj}}*{\text{pose}}*\mathbf{p}*{obj}]
$$

Projection is **last**, and only touches **pixels**.

---

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
[\sum_i |\pi(P V T X_i) - x_i^{gt}|]
$$

Where:

* ( \pi ) = perspective divide
* ( P ) = projection matrix

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

## In *your* tool specifically

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

## Mental model (best takeaway)

* **Pose matrix** → “Where is the object in 3D?”
* **View matrix** → “Where is the camera in 3D?”
* **Projection matrix** → “How does 3D become pixels?”

Only the **last one touches images**.

---

## Final one-line answer

> The projection matrix is used whenever **pixels matter** (rendering, reprojection, evaluation), and is **never used** when only **3D pose** or **robot motion** matters.
