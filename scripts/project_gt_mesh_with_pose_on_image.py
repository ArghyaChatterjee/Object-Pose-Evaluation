#!/usr/bin/env python3

# Copyright (c) IHMC and affiliates.

import json
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path


def project_mesh_on_image(image_path, mesh_vertices, pose_4x4,
                          fx, fy, cx, cy, out_path):

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] Could not read {image_path}")
        return

    V = mesh_vertices

    V_h = np.hstack([V, np.ones((V.shape[0], 1))])

    # Object -> Camera
    V_cam = (pose_4x4 @ V_h.T).T[:, :3]

    valid = V_cam[:, 2] > 0
    V_cam = V_cam[valid]

    X = V_cam[:, 0]
    Y = V_cam[:, 1]
    Z = V_cam[:, 2]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    pts = np.stack([u, v], axis=1).astype(np.int32)

    h, w = image.shape[:2]

    for x, y in pts[::10]:      # draw every 10th vertex
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    cv2.imwrite(str(out_path), image)


# ------------------------------------------------------------------

root = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper Recordings/"
    "SupervisePose/bottle_1"
)

image_dir = root / "image"
label_dir = root / "labels"
mesh_dir = root / "mesh"
output_dir = root / "projected_labels"

output_dir.mkdir(exist_ok=True)

# Load mesh only once
mesh_path = next(mesh_dir.glob("*.obj"))

mesh = o3d.io.read_triangle_mesh(str(mesh_path))
mesh_vertices = np.asarray(mesh.vertices, dtype=np.float64)

# ------------------------------------------------------------------

json_files = sorted(label_dir.glob("*.json"))

for json_path in json_files:

    timestamp = json_path.stem

    image_path = image_dir / f"{timestamp}.png"

    if not image_path.exists():
        image_path = image_dir / f"{timestamp}.jpg"

    if not image_path.exists():
        print(f"[WARN] Missing image for {timestamp}")
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    intr = data["camera_data"]["intrinsics"]

    pose = np.array(
        data["objects"][0]["transform_matrix"],
        dtype=np.float64,
    )

    project_mesh_on_image(
        image_path=image_path,
        mesh_vertices=mesh_vertices,
        pose_4x4=pose,
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"],
        out_path=output_dir / f"{timestamp}.png",
    )

    print(f"Done {timestamp}")

print("\nFinished projecting all labels.")