#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from bop_toolkit_lib import pose_error

GT_DIR = Path("/home/arghya/Object-Pose-Evaluation/data/ground_truth")
EST_DIR = Path("/home/arghya/Object-Pose-Evaluation/data/estimated")


def load_pose(path):
    with open(path, "r") as f:
        data = json.load(f)
    T = np.asarray(data["objects"][0]["transform_matrix"], dtype=np.float64)
    return T[:3, :3], T[:3, 3]


def main():
    rot_errors = []
    trans_errors_cm = []
    success_5_5 = []
    success_10_10 = []

    for gt_path in sorted(GT_DIR.glob("*.json")):
        est_path = EST_DIR / gt_path.name
        if not est_path.exists():
            print(f"[SKIP] Missing estimated pose: {est_path.name}")
            continue

        R_gt, t_gt = load_pose(gt_path)
        R_est, t_est = load_pose(est_path)

        rot_err_deg = pose_error.re(R_est, R_gt)
        trans_err_cm = pose_error.te(t_est, t_gt) * 100.0

        rot_errors.append(rot_err_deg)
        trans_errors_cm.append(trans_err_cm)

        success_5_5.append(rot_err_deg < 5.0 and trans_err_cm < 5.0)
        success_10_10.append(rot_err_deg < 10.0 and trans_err_cm < 10.0)

    print(f"Frames evaluated: {len(rot_errors)}")
    print(f"Mean rotation error: {np.mean(rot_errors):.4f} deg")
    print(f"Mean translation error: {np.mean(trans_errors_cm):.4f} cm")
    print(f"5 deg / 5 cm Accuracy: {np.mean(success_5_5) * 100.0:.4f} %")
    print(f"10 deg / 10 cm Accuracy: {np.mean(success_10_10) * 100.0:.4f} %")


if __name__ == "__main__":
    main()