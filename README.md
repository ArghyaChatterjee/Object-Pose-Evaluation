# Object Pose Evaluation
A repository for evaluating object pose w.r.t ground truth.

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
python scripts/compute_auc_adds.py
```
Compute ADD>0.1D:
```
python scripts/compute_add_01d.py
```
Compute Pose Error in Degree / CM:
```
python scripts/compute_degree_cm.py
```

# Object Pose Estimation Evaluation Metrics

This repository evaluates rigid object pose estimation results using several commonly reported metrics from the 6D object pose estimation literature, including ADD, ADD-S, ADD-0.1d, AUC, rotation/translation thresholds, and BOP challenge metrics.

---

# Pose Representation

A rigid pose is represented as:

[
\mathbf{T} =
\begin{bmatrix}
\mathbf{R} & \mathbf{t} \
0 & 1
\end{bmatrix}
]

where:

* (\mathbf{R}\in SO(3)) is the rotation matrix.
* (\mathbf{t}\in \mathbb{R}^{3}) is the translation vector.

For a model point (\mathbf{x}), the transformed point is:

[
\mathbf{x}' = \mathbf{R}\mathbf{x} + \mathbf{t}
]

Given:

* Ground-truth pose ((\mathbf{R}*{gt}, \mathbf{t}*{gt}))
* Estimated pose ((\mathbf{R}*{est}, \mathbf{t}*{est}))

the following metrics are computed.

---

# Average Distance of Model Points (ADD)

ADD measures the average distance between model points transformed by the estimated pose and the ground-truth pose.

[
\text{ADD}
==========

\frac{1}{|\mathcal{M}|}
\sum_{\mathbf{x}\in\mathcal{M}}
\left|
(\mathbf{R}*{est}\mathbf{x}+\mathbf{t}*{est})
---------------------------------------------

(\mathbf{R}*{gt}\mathbf{x}+\mathbf{t}*{gt})
\right|_2
]

where:

* (\mathcal{M}) is the set of mesh vertices.
* (|\mathcal{M}|) is the number of vertices.

ADD is typically used for asymmetric objects.

Smaller values indicate better pose estimates.

Units are the same as the mesh units, typically meters.

---

# Average Distance of Model Points for Symmetric Objects (ADD-S)

For symmetric objects, multiple rotations may produce identical appearances.

ADD-S replaces the direct point correspondence with nearest-neighbor matching:

[
\text{ADD-S}
============

\frac{1}{|\mathcal{M}|}
\sum_{\mathbf{x}*1\in\mathcal{M}}
\min*{\mathbf{x}*2\in\mathcal{M}}
\left|
(\mathbf{R}*{est}\mathbf{x}*1+\mathbf{t}*{est})
-----------------------------------------------

(\mathbf{R}_{gt}\mathbf{x}*2+\mathbf{t}*{gt})
\right|_2
]

ADD-S is commonly used for:

* bottles
* bowls
* cans
* cylinders
* rotationally symmetric objects

---

# Area Under the Curve (AUC)

Instead of evaluating a single threshold, many papers report the Area Under the ADD or ADD-S Recall Curve.

For a threshold (t):

[
\text{Recall}(t)
================

\frac{
#{\text{frames with error} \le t}
}{
N
}
]

The AUC is:

[
\text{AUC}
==========

\frac{
\int_{0}^{t_{max}}
\text{Recall}(t),dt
}{
t_{max}
}
\times 100
]

where:

[
t_{max}=0.1\ \text{m}
]

for the standard YCB-Video PoseCNN evaluation protocol.

Reported metrics:

* AUC ADD
* AUC ADD-S

Higher values are better.

Perfect performance corresponds to:

[
\text{AUC}=100
]

---

# ADD-0.1d

ADD-0.1d evaluates whether the ADD error is below 10% of the object diameter.

Object diameter:

[
d
=

\max_{\mathbf{x}_i,\mathbf{x}_j\in\mathcal{M}}
|\mathbf{x}_i-\mathbf{x}_j|_2
]

Pose estimate is considered correct if:

[
\text{ADD}
<
0.1d
]

Recall is then:

[
\text{Recall}
=============

\frac{
#{\text{correct poses}}
}{
N
}
\times 100
]

This metric is widely reported in:

* FoundationPose
* MegaPose
* SAM-6D
* OnePose++
* CNOS

---

# Rotation Error

Rotation error is the geodesic distance between two rotations.

Let:

[
\mathbf{R}_{err}
================

\mathbf{R}*{est}
\mathbf{R}*{gt}^{T}
]

Then:

[
e_R
===

\cos^{-1}
\left(
\frac{
\mathrm{trace}(\mathbf{R}_{err})-1
}{2}
\right)
]

reported in degrees.

[
e_R [^\circ]
============

\frac{180}{\pi}
e_R
]

---

# Translation Error

Translation error is the Euclidean distance between translations:

[
e_t
===

|
\mathbf{t}_{est}
----------------

\mathbf{t}_{gt}
|_2
]

Usually reported in:

* meters
* centimeters

depending on the benchmark.

---

# 5° / 5 cm Metric

A pose estimate is considered correct if:

[
e_R < 5^\circ
]

and

[
e_t < 5\ \text{cm}
]

Accuracy is:

[
\frac{
#{\text{correct poses}}
}{
N
}
\times100
]

---

# 10° / 10 cm Metric

A pose estimate is considered correct if:

[
e_R < 10^\circ
]

and

[
e_t < 10\ \text{cm}
]

Accuracy is:

[
\frac{
#{\text{correct poses}}
}{
N
}
\times100
]

---

# BOP Challenge Metrics

Recent BOP benchmarks primarily use:

* VSD
* MSSD
* MSPD

instead of ADD.

These metrics are symmetry-aware and designed to evaluate real-world pose ambiguity.

---

## VSD (Visible Surface Discrepancy)

VSD compares only the visible surfaces of the rendered object.

For a visibility mask (V):

[
\text{VSD}
==========

\frac{1}{|V|}
\sum_{p\in V}
c(p)
]

where

[
c(p)
====

\begin{cases}
0 & \text{if } |D_{est}(p)-D_{gt}(p)| < \tau \
1 & \text{otherwise}
\end{cases}
]

and:

* (D_{est}) is the rendered depth from the estimated pose.
* (D_{gt}) is the rendered depth from the ground-truth pose.
* (\tau) is a depth tolerance.

VSD ignores ambiguities caused by object symmetries and occlusions.

---

## MSSD (Maximum Symmetry-Aware Surface Distance)

MSSD measures the largest surface point deviation after considering all valid symmetry transformations.

[
\text{MSSD}
===========

\min_{S\in\mathcal{S}}
\max_{\mathbf{x}\in\mathcal{M}}
|
\mathbf{T}_{est}\mathbf{x}
--------------------------

\mathbf{T}_{gt}S\mathbf{x}
|
]

where:

* (\mathcal{S}) is the symmetry group.

Lower values are better.

---

## MSPD (Maximum Symmetry-Aware Projection Distance)

MSPD measures the largest 2D reprojection error while accounting for object symmetries.

[
\text{MSPD}
===========

\min_{S\in\mathcal{S}}
\max_{\mathbf{x}\in\mathcal{M}}
|
\pi(\mathbf{T}_{est}\mathbf{x})
-------------------------------

\pi(\mathbf{T}_{gt}S\mathbf{x})
|
]

where:

[
\pi(\cdot)
]

denotes projection into the image plane.

MSPD is measured in pixels.

Lower values indicate better alignment.

---

# Metrics Implemented in This Repository

The provided scripts compute:

1. AUC ADD
2. AUC ADD-S
3. ADD-0.1d Recall
4. 5° / 5 cm Accuracy
5. 10° / 10 cm Accuracy

using ground-truth and estimated poses stored as JSON files and object meshes stored as OBJ files.




