#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
from bop_toolkit_lib import pose_error


GT_DIR = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper_Recordings/GroundTruthPose/trash_can_2/filtered/labels")
EST_DIR = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper_Recordings/SupervisePose/trash_can_2/filtered/labels")
MESH_PATH = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper_Recordings/GroundTruthPose/trash_can_2/filtered/mesh/trash_can_2.obj")


def load_obj_vertices(path):
    verts = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])

    return np.asarray(verts, dtype=np.float64)


def object_diameter(points):
    extent = points.max(axis=0) - points.min(axis=0)
    return np.linalg.norm(extent)


def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)

    T = np.asarray(data["objects"][0]["transform_matrix"], dtype=np.float64)

    R = T[:3, :3]
    t = T[:3, 3]

    return R, t


def main():
    pts = load_obj_vertices(MESH_PATH)

    diameter = object_diameter(pts)
    threshold = 0.1 * diameter

    errors = []
    successes = []

    for gt_path in sorted(GT_DIR.glob("*.json")):
        est_path = EST_DIR / gt_path.name

        if not est_path.exists():
            print(f"[SKIP] Missing estimated pose: {est_path.name}")
            continue

        R_gt, t_gt = load_pose(gt_path)
        R_est, t_est = load_pose(est_path)

        err = pose_error.adi(R_est, t_est, R_gt, t_gt, pts)

        errors.append(err)
        successes.append(err < threshold)

    if len(errors) == 0:
        print("[ERROR] No matching JSON files were evaluated.")
        return

    print(f"Frames evaluated: {len(errors)}")
    print(f"Object diameter: {diameter:.6f} m")
    print(f"ADD-S-0.1d threshold: {threshold:.6f} m")
    print(f"ADD-S-0.1d Recall: {np.mean(successes) * 100.0:.4f} %")
    print(f"Mean ADD-S: {np.mean(errors):.6f} m")


if __name__ == "__main__":
    main()