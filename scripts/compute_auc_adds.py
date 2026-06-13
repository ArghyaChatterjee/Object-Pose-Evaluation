#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from bop_toolkit_lib import pose_error

GT_DIR = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper_Recordings/GroundTruthPose/trash_can_2/filtered/labels")
EST_DIR = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper_Recordings/SupervisePose/trash_can_2/filtered/labels")
MESH_PATH = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper_Recordings/GroundTruthPose/trash_can_2/filtered/mesh/trash_can_2.obj")

AUC_MAX_THRESHOLD_M = 0.1


def load_obj_vertices(path):
    verts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.asarray(verts, dtype=np.float64)


def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["objects"][0]["transform_matrix"], dtype=np.float64)
    return T[:3, :3], T[:3, 3]


def auc(errors, max_threshold=AUC_MAX_THRESHOLD_M):
    errors = np.asarray(errors)
    thresholds = np.linspace(0.0, max_threshold, 1000)
    recalls = np.array([(errors <= th).mean() for th in thresholds])
    return np.trapz(recalls, thresholds) / max_threshold * 100.0


def main():
    pts = load_obj_vertices(MESH_PATH)
    errors = []

    for gt_path in sorted(GT_DIR.glob("*.json")):
        est_path = EST_DIR / gt_path.name
        if not est_path.exists():
            print(f"[SKIP] Missing estimated pose: {est_path.name}")
            continue

        R_gt, t_gt = load_pose(gt_path)
        R_est, t_est = load_pose(est_path)

        err = pose_error.adi(R_est, t_est, R_gt, t_gt, pts)
        errors.append(err)

    print(f"Frames evaluated: {len(errors)}")
    print(f"AUC ADD-S: {auc(errors):.4f} %")
    print(f"Mean ADD-S: {np.mean(errors):.6f} m")


if __name__ == "__main__":
    main()