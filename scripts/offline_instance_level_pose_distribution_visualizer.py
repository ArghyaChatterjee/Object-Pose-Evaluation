#!/usr/bin/env python3

# Copyright (c) IHMC and affiliates.

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


POSE_DIR = Path("/home/arghya/ihmc-repos/ihmc-humanoid-labeler/humanoid_data/2d_3d_data/Paper Recordings/SupervisePose/bottle_1/labels")


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


def main():
    positions = {
        "x": [],
        "y": [],
        "z": [],
    }

    orientations = {
        "qx": [],
        "qy": [],
        "qz": [],
        "qw": [],
    }

    rotations_deg = {
        "roll_x_deg": [],
        "pitch_y_deg": [],
        "yaw_z_deg": [],
    }

    json_files = sorted(POSE_DIR.glob("*.json"))

    if len(json_files) == 0:
        raise RuntimeError(f"No JSON pose files found in {POSE_DIR}")

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

    print(f"[INFO] Loaded {len(json_files)} poses")

    output_dir = POSE_DIR / "pose_distribution"
    output_dir.mkdir(exist_ok=True)

    plot_histograms(
        positions,
        "Position Distribution",
        output_dir / "position_distribution.png",
    )

    plot_histograms(
        orientations,
        "Orientation Quaternion Distribution",
        output_dir / "orientation_quaternion_distribution.png",
    )

    plot_histograms(
        rotations_deg,
        "Rotation Distribution in Degrees",
        output_dir / "rotation_degree_distribution.png",
    )


if __name__ == "__main__":
    main()