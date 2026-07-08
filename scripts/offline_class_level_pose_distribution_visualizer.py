#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


# Example:
# CATEGORY_ROOT/
# ├── bottle_1/labels/*.json
# ├── bottle_2/labels/*.json
# └── bottle_3/labels/*.json
CATEGORY_ROOT = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper Recordings/SupervisePose"
)

CLASS_NAME = "trash_can"
LABEL_FOLDER_NAME = "labels"


def load_pose_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    T = np.asarray(data["objects"][0]["transform_matrix"], dtype=np.float64)

    position = T[:3, 3]
    rot_mat = T[:3, :3]

    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    euler_xyz_deg = R.from_matrix(rot_mat).as_euler("xyz", degrees=True)

    return position, quat_xyzw, euler_xyz_deg


def plot_histograms(data_dict, title, output_path):
    keys = list(data_dict.keys())
    fig, ax = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4))

    if len(keys) == 1:
        ax = [ax]

    for i, key in enumerate(keys):
        ax[i].hist(data_dict[key], bins=30)
        ax[i].set_title(key)
        ax[i].set_xlabel(key)
        ax[i].set_ylabel("count")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] {output_path}")


def find_instance_pose_dirs(category_root, class_name):
    pose_dirs = []

    for instance_dir in sorted(category_root.iterdir()):
        if not instance_dir.is_dir():
            continue

        if not instance_dir.name.startswith(f"{class_name}_"):
            continue

        labels_dir = instance_dir / LABEL_FOLDER_NAME

        if labels_dir.exists():
            pose_dirs.append(labels_dir)
        else:
            print(f"[WARN] Missing labels folder: {labels_dir}")

    return pose_dirs


def main():
    positions = {"x": [], "y": [], "z": []}
    orientations = {"qx": [], "qy": [], "qz": [], "qw": []}
    rotations_deg = {
        "roll_x_deg": [],
        "pitch_y_deg": [],
        "yaw_z_deg": [],
    }

    pose_dirs = find_instance_pose_dirs(CATEGORY_ROOT, CLASS_NAME)

    if len(pose_dirs) == 0:
        raise RuntimeError(f"No instance label folders found for class: {CLASS_NAME}")

    total_files = 0

    print(f"[INFO] Class: {CLASS_NAME}")
    print(f"[INFO] Found {len(pose_dirs)} instance folders")

    for pose_dir in pose_dirs:
        json_files = sorted(pose_dir.glob("*.json"))
        print(f"[INFO] {pose_dir.parent.name}: {len(json_files)} poses")

        for path in json_files:
            position, quat_xyzw, euler_xyz_deg = load_pose_json(path)

            positions["x"].append(position[0])
            positions["y"].append(position[1])
            positions["z"].append(position[2])

            orientations["qx"].append(quat_xyzw[0])
            orientations["qy"].append(quat_xyzw[1])
            orientations["qz"].append(quat_xyzw[2])
            orientations["qw"].append(quat_xyzw[3])

            rotations_deg["roll_x_deg"].append(euler_xyz_deg[0])
            rotations_deg["pitch_y_deg"].append(euler_xyz_deg[1])
            rotations_deg["yaw_z_deg"].append(euler_xyz_deg[2])

            total_files += 1

    print(f"[INFO] Total poses loaded for {CLASS_NAME}: {total_files}")

    output_dir = CATEGORY_ROOT / f"{CLASS_NAME}_pose_distribution"
    output_dir.mkdir(exist_ok=True)

    plot_histograms(
        positions,
        f"{CLASS_NAME}: Position Distribution",
        output_dir / f"{CLASS_NAME}_position_distribution.png",
    )

    plot_histograms(
        orientations,
        f"{CLASS_NAME}: Orientation Quaternion Distribution",
        output_dir / f"{CLASS_NAME}_orientation_quaternion_distribution.png",
    )

    plot_histograms(
        rotations_deg,
        f"{CLASS_NAME}: Rotation Distribution in Degrees",
        output_dir / f"{CLASS_NAME}_rotation_degree_distribution.png",
    )


if __name__ == "__main__":
    main()