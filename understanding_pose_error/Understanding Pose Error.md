# Understanding Pose Error

This script evaluates the accuracy of an estimated 6D object pose by comparing it against the ground-truth pose from a BOP/LMO annotation. It reports:

- **Rotation error** (degrees)
- **Translation error** (Euclidean distance, typically in millimeters)

The script is intended as a simple sanity-check tool for validating manually annotated or estimated object poses.

## Overview

The evaluation pipeline consists of the following steps:

1. Read the ground-truth pose (`cam_R_m2c`, `cam_t_m2c`) from a BOP-format JSON annotation.
2. Convert the ground-truth pose into a 4×4 homogeneous transformation matrix.
3. Compare it against an estimated 4×4 pose matrix.
4. Compute:
   - Angular difference between the two rotations.
   - Euclidean distance between the two translations.
5. Print the ground-truth pose and the resulting errors.

## Functions

### `convert_to_4x4(cam_R_m2c, cam_t_m2c, scale=1)`

Constructs a homogeneous transformation matrix

\[
T =
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\]

from a flattened rotation matrix and translation vector.

---

### `compute_matrix(data)`

Extracts:

- `cam_R_m2c`
- `cam_t_m2c`
- `obj_id`

from a BOP annotation and returns the corresponding 4×4 pose matrix.

---

### `process_scene_gt(json_data)`

Parses a single object annotation, generates the ground-truth pose matrix, prints it, and returns the matrix.

---

### `rotation_angle_deg(R1, R2)`

Computes the geodesic angular distance between two rotation matrices using

\[
\theta =
\cos^{-1}
\left(
\frac{\operatorname{trace}(R_\Delta)-1}{2}
\right)
\]

where

\[
R_\Delta = R_2 R_1^T.
\]

The returned value is in **degrees**.

---

### `pose_errors(T1, T2)`

Computes:

- Rotation error (degrees)
- Translation error (Euclidean distance)

between two 4×4 transformation matrices.

Returns:

```python
(rotation_error_deg, translation_error)
```

## Test Cases

Functions named like

```python
image_000008_obj_000010()
```

contain hardcoded evaluation examples for individual object instances. Each function:

1. Defines the ground-truth pose.
2. Defines the estimated pose.
3. Computes the pose error.
4. Prints the evaluation results.

The script executes one of these test cases from the main section.

## Example Output

```text
Object ID 10:

[[ ... ground truth pose ... ]]

Angular distance (deg): 4.64
Euclidean distance (mm): 14.03
```

## Notes

- Translation values follow the units used in the dataset (typically **millimeters** for BOP datasets).
- The estimated pose should use the same coordinate frame and units as the ground truth.
- Rotation error is reported in degrees, while translation error is reported as Euclidean distance.
