# Object Pose Evaluation
A repository for evaluating object pose w.r.t ground truth pose. This repo uses the [bop toolkit](https://github.com/thodan/bop_toolkit) for pose accuracy evaluation. 

## Requirements

This work was tested on Ubuntu 22.04 and Python versions 3.8-3.12.

## Clone the repo
```
git clone https://github.com/ArghyaChatterjee/Object-Pose-Evaluation.git
cd Object-Pose-Evaluation
```

## Installation

### Using [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv python install 3.12
uv sync --python 3.12
uv python pin 3.12
uv sync
uv add --editable ./bop_toolkit
source .venv/bin/activate
uv add torch
uv add "hand_tracking_toolkit @ git+https://github.com/facebookresearch/hand_tracking_toolkit"
```
This commands sets up a local venv, installs necessary dependencies and bop_toolkit_lib. You may provide additional flags such as:

- `--python 3.10`: specify the venv python version
- `--extra eval_coco`: install dependencies for coco evaluation
- `--extra eval_gpu`: install dependencies for gpu evaluation
- `--extra eval_hot3d`: install dependencies for hot3d evaluation
- `--extra scripts`: install dependencies for utility scripts (e.g. `annotation_tools.py`)

### Using pip
```bash
pip install .  # bop_toolkit_lib with core dependencies only
# with additional dependencies
pip install .[eval_coco]  # install dependencies for coco evaluation
pip install .[eval_gpu]  # install dependencies for gpu evaluation
pip install .[eval_hot3d]  # install dependencies for hot3d evaluation
uv pip install .[scripts]  # install dependencies for utility scripts (e.g. `annotation_tools.py`)
```

### Unittests
```
export BOP_PATH=$HOME/bop_datasets
python -m unittest discover bop_toolkit/bop_toolkit_lib/tests
```

### Compute evaluation metrics

Compute AUC of ADD:
```
python3 scripts/compute_auc_add.py
```
Compute AUC of ADDS:
```
python3 scripts/compute_auc_adds.py
```
Compute ADD>0.1D:
```
python3 scripts/compute_add_01d.py
```
Compute Pose Error in Degree / CM:
```
python3 scripts/compute_degree_cm.py
```

# Description of Evaluation Metrics 

This repository evaluates rigid object pose estimation results using several commonly reported metrics from the 6D object pose estimation literature.

The following metrics are implemented:

1. AUC of ADD
2. AUC of ADD-S
3. ADD-0.1d Recall
4. 5° / 5 cm Accuracy
5. 10° / 10 cm Accuracy

The evaluation requires:

* Ground-truth object poses
* Estimated object poses
* Object mesh

All pose errors are computed using the BOP Toolkit implementation.

---

## Pose Representation

A rigid object pose is represented by:

`T = [R | t]`

where:

* `R` is a 3×3 rotation matrix.
* `t` is a 3×1 translation vector.

For a model point `x`, the transformed point is:

`x' = R x + t`

Given:

* Ground-truth pose `(R_gt, t_gt)`
* Estimated pose `(R_est, t_est)`

the following metrics are computed.

---

## ADD (Average Distance of Model Points)

ADD measures the average distance between mesh vertices transformed by the estimated pose and the ground-truth pose.

`ADD = (1 / |M|) * Σ || (R_est x + t_est) - (R_gt x + t_gt) ||`

where:

* `M` is the set of mesh vertices.
* `|M|` is the number of mesh vertices.

ADD is typically used for asymmetric objects.

Lower values indicate better pose estimates.

The error is reported in the same units as the mesh, typically meters.

---

## ADD-S (Average Distance of Model Points for Symmetric Objects)

For symmetric objects, multiple poses may produce identical visual appearances. ADD-S replaces direct point correspondences with nearest-neighbor matching.

`ADD-S = (1 / |M|) * Σ min || (R_est x1 + t_est) - (R_gt x2 + t_gt) ||`

where the minimum is taken over all model points in the ground-truth pose.

ADD-S is commonly used for:

* bottles
* bowls
* cans
* cylinders
* rotationally symmetric objects

Lower values indicate better pose estimates.

---

## AUC of ADD / ADD-S

Many pose estimation papers report the Area Under the Recall Curve (AUC) rather than a single threshold.

For a threshold `t`:

`Recall(t) = (# poses with error <= t) / N`

where `N` is the total number of evaluated poses.

The AUC is computed as:

`AUC = (Integral Recall(t) dt from 0 to t_max) / t_max`

and reported as a percentage.

For the YCB-Video evaluation protocol:

`t_max = 0.1 m`

Higher values indicate better performance.

A perfect pose estimator would achieve:

`AUC = 100%`

---

## ADD-0.1d Recall

ADD-0.1d evaluates whether the ADD error is less than 10% of the object diameter.

A pose is considered correct if:

`ADD < 0.1 * d`

where:

`d = object diameter`

The recall is then:

`Recall = (# correct poses) / N`

and is reported as a percentage.

This metric is commonly reported by:

* PoseCNN
* DenseFusion
* FoundationPose
* MegaPose
* SAM-6D
* OnePose++

Higher values indicate better performance.

---

## Rotation Error

Rotation error is computed as the geodesic distance between the estimated and ground-truth rotations.

First:

`R_err = R_est * R_gt^T`

Then:

`rotation_error = acos((trace(R_err) - 1) / 2)`

The result is reported in degrees.

Lower values indicate better pose estimates.

---

## Translation Error

Translation error is the Euclidean distance between estimated and ground-truth translations.

`translation_error = || t_est - t_gt ||`

The result may be reported in:

* meters
* centimeters

depending on the benchmark protocol.

Lower values indicate better pose estimates.

---

## 5° / 5 cm Accuracy

A pose estimate is considered correct if:

`rotation_error < 5 degrees`

and

`translation_error < 5 cm`

Accuracy is computed as:

`Accuracy = (# correct poses) / N`

and reported as a percentage.

---

## 10° / 10 cm Accuracy

A pose estimate is considered correct if:

`rotation_error < 10 degrees`

and

`translation_error < 10 cm`

Accuracy is computed as:

`Accuracy = (# correct poses) / N`

and reported as a percentage.

---

# Additional BOP Metrics

The BOP benchmark primarily evaluates pose estimation using three additional metrics.

---

## VSD (Visible Surface Discrepancy)

VSD compares only the visible surfaces of the object rendered from the estimated and ground-truth poses.

The metric evaluates depth consistency between visible object pixels while accounting for occlusions and pose ambiguities.

VSD is particularly useful for symmetric objects and heavily occluded scenes.

Lower values indicate better pose estimates.

---

## MSSD (Maximum Symmetry-Aware Surface Distance)

MSSD measures the maximum distance between transformed model points while accounting for valid object symmetries.

`MSSD = min_S max || T_est x - T_gt S x ||`

where:

* `S` is a valid symmetry transformation.
* `T_est` is the estimated pose.
* `T_gt` is the ground-truth pose.

Lower values indicate better pose estimates.

---

## MSPD (Maximum Symmetry-Aware Projection Distance)

MSPD measures the maximum 2D reprojection error while accounting for object symmetries.

`MSPD = min_S max || proj(T_est x) - proj(T_gt S x) ||`

where:

* `proj(.)` denotes projection into the image plane.
* `S` is a valid symmetry transformation.

MSPD is reported in pixels.

Lower values indicate better pose estimates.




