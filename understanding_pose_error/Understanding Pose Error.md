# Understanding Pose Error
This script is an **evaluation / sanity-check tool**: it compares a **ground-truth object pose** (from BOP/LMO JSON: `cam_R_m2c`, `cam_t_m2c`) against **estimated pose** (a 4×4 matrix you typed in), and reports:

* **rotation error (degrees)**
* **translation error (Euclidean distance, same units as `t` — here mm)**

That’s why at the end, you see:

* a printed GT 4×4 matrix
* `4.6193...` (rotation error printed once)
* then the final formatted line with rotation + translation error

---

## What each part is doing

### 1) Printing settings

```py
np.set_printoptions(suppress=True, precision=4)
```

Just makes numpy print matrices nicely:

* no scientific notation
* 4 decimals

---

### 2) Pose parsing: JSON → 4×4 transform

You have two very similar functions:

#### `convert_to_4x4(cam_R_m2c, cam_t_m2c, scale=1)`

Takes a flattened 3×3 `cam_R_m2c` and a 3-vector `cam_t_m2c` and builds:

$$
T =\begin{bmatrix}R & t\0 & 1\end{bmatrix}
$$

(But note: you don’t actually use this function later.)

#### `compute_matrix(data)`

Reads `cam_R_m2c`, `cam_t_m2c`, `obj_id` from a dict and returns `(T, obj_id)`.

---

### 3) `process_scene_gt(json_data)`

```py
data = json.loads(json_data)
T, obj_id = compute_matrix(data)
print(T)
return T
```

So this takes a JSON string for one object pose (GT) and returns the GT 4×4 pose matrix.

---

## 4) Pose error computation

### `rotation_angle_deg(R1, R2)`

Computes the **geodesic angle** between two rotation matrices:

* Computes relative rotation: `R_delta = R2 @ R1.T`
* Converts to angle using:

$$
\theta = \cos^{-1}\left(\frac{\mathrm{trace}(R_\Delta) - 1}{2}\right)
$$

Returns **degrees**.

### `pose_errors(T1, T2)`

This compares two 4×4 poses:

* `T1` = your `annotated_pose`
* `T2` = GT pose from JSON

It computes rotation error **two ways** (redundantly):

1. using `rotation_angle_deg(...)` and prints it

```py
ang_deg1 = rotation_angle_deg(R1, R2)
print(ang_deg1)
```

2. using SciPy rotations:

```py
R_delta = R2 * R1.inv()
ang_deg = np.degrees(R_delta.magnitude())
```

Then translation error:

```py
eucli_dist = ||t2 - t1||
```

Returns: `(rotation_error_deg, translation_error_distance)`

So your final numbers are:

* **Angular distance ~ 4.64°**
* **Translation error ~ 14.03 mm**

---

## 5) The `image_000008_obj_0000XX()` functions

Each of these is just a **hardcoded test case**:

* `json_data` = GT pose for a specific object in image 000008
* `annotated_pose` = your estimated pose matrix
* compare them and print errors

At the bottom:

```py
image_000008_obj_000010()
```

So it runs only object **10**.

---

## Why your output looks like that

You printed the GT matrix:

```text
Object ID 10:
[[0.0194, 0.9854, 0.1690, 189.8814],
 ...
]
```

Then `pose_errors` prints the first rotation error method:

```text
4.619308...
```

Then you print the final summary using the SciPy rotation angle:

```text
Angular distance (deg): 4.64, Euclidean distance (mm): 14.03
```

(4.6193 vs 4.64 differs only due to rounding + two different computations.)

---

## A couple of “gotchas” in this script

1. **Units:** Your `cam_t_m2c` values look like **mm** (typical BOP). Your `estimated_pose` translations are also mm, so distance comes out in **mm**. If you ever convert one to meters, you must convert the other too.

2. **Redundant / confusing naming:** you import `Rotation as R`, but also use `R` as a matrix variable name in other functions. It works because scopes differ, but it’s easy to mess up.

3. **Bug / incomplete case:**
   `image_000008_obj_000006()` calls:

```py
gt_pose = process_scene_gt()
```

but `process_scene_gt(json_data)` requires an argument — that function would crash if you uncomment it.

---

## Summary

It’s a **pose error checker**: for a given object instance, it converts GT pose JSON into a 4×4 transform and compares it against your annotated 4×4 pose, outputting **rotation error (deg)** and **translation error (mm)**.