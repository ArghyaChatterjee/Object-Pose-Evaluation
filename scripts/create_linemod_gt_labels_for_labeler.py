#!/usr/bin/env python3

# Copyright (c) IHMC and affiliates.

import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# ============================================================
# Hardcoded paths
# ============================================================

ROOT = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper Recordings/"
    "Linemod"
)

GROUND_TRUTH_PATH = ROOT / "ground_truth.txt"
INTRINSICS_PATH = ROOT / "intrinsics.txt"
IMAGE_DIR = ROOT / "image"
LABEL_DIR = ROOT / "labels"

LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Metadata
# ============================================================

PROVENANCE = "linemod"

# Fill manually if you want exact mesh extents.
DEFAULT_SCALE = [1.0, 1.0, 1.0]


# ============================================================
# Create intrinsics.txt from ground_truth.txt
# ============================================================

def write_intrinsics_from_ground_truth(gt_path: Path, intrinsics_path: Path):
    text = gt_path.read_text()

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Could not find camera intrinsics JSON block in {gt_path}")

    cam = json.loads(match.group(0))

    fx = float(cam["fx"])
    fy = float(cam["fy"])
    cx = float(cam["cx"])
    cy = float(cam["cy"])
    width = int(cam["width"])
    height = int(cam["height"])

    distortion = " ".join(["0.000000e+00"] * 12)

    intrinsics_text = (
        f"fx {fx:.18e} fy {fy:.18e}\n"
        f"cx {cx:.18e} cy {cy:.18e}\n"
        f"distortion {distortion}\n"
        f"resolution {width}x{height}\n"
    )

    intrinsics_path.write_text(intrinsics_text)
    print(f"[OK] Wrote intrinsics.txt: {intrinsics_path}")

    return {
        "height": height,
        "width": width,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        },
    }


# ============================================================
# Pose parser
# ============================================================

def parse_matrix(matrix_text: str):
    rows = re.findall(r"\[([^\[\]]+)\]", matrix_text)
    matrix = []

    for row in rows:
        nums = re.findall(
            r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
            row,
        )
        matrix.append([float(x) for x in nums])

    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise RuntimeError(f"Invalid 4x4 matrix:\n{matrix_text}")

    return matrix


def read_ground_truth_poses(gt_path: Path):
    """
    Expected format:

    {
      "cx": ...,
      "cy": ...,
      "fx": ...,
      "fy": ...,
      "height": 480,
      "width": 640
    }

    obj_000001 <-> 000019
    [[...]]
    """

    text = gt_path.read_text()

    # Remove camera intrinsics block
    text = re.sub(r"\{.*?\}", "", text, count=1, flags=re.DOTALL)

    pattern = re.compile(
        r"(obj_\d+)\s*<->\s*(\d+)\s*"
        r"(\[\s*\[.*?\]\s*\])",
        re.DOTALL,
    )

    entries = []

    for match in pattern.finditer(text):
        mesh_name = match.group(1)
        image_name = match.group(2)
        matrix_text = match.group(3)

        pose = parse_matrix(matrix_text)

        entries.append(
            {
                "mesh_name": mesh_name,
                "image_name": image_name,
                "pose": pose,
            }
        )

    return entries


# ============================================================
# Write label JSON
# ============================================================

def write_label_json(image_name: str, mesh_name: str, pose, camera_data):
    pose_np = np.array(pose, dtype=float)

    # IMPORTANT:
    # LINEMOD translation is in millimeters.
    # Label JSON expects meters.
    pose_np[:3, 3] /= 1000.0

    rotation_matrix = pose_np[:3, :3]
    translation = pose_np[:3, 3]

    quat_xyzw = R.from_matrix(rotation_matrix).as_quat().tolist()

    label_data = {
        "camera_data": camera_data,
        "objects": [
            {
                "class": mesh_name,
                "name": mesh_name,
                "provenance": PROVENANCE,
                "transform_matrix": pose_np.tolist(),
                "location": translation.tolist(),
                "quaternion_xyzw": quat_xyzw,
                "scale": DEFAULT_SCALE,
            }
        ],
    }

    out_path = LABEL_DIR / f"{image_name}.json"

    with open(out_path, "w") as f:
        json.dump(label_data, f, indent=4)

    return out_path


# ============================================================
# Main
# ============================================================

def main():
    if not ROOT.exists():
        raise RuntimeError(f"ROOT does not exist: {ROOT}")

    if not GROUND_TRUTH_PATH.exists():
        raise RuntimeError(f"Missing ground_truth.txt: {GROUND_TRUTH_PATH}")

    if not IMAGE_DIR.exists():
        raise RuntimeError(f"Missing image directory: {IMAGE_DIR}")

    camera_data = write_intrinsics_from_ground_truth(
        GROUND_TRUTH_PATH,
        INTRINSICS_PATH,
    )

    entries = read_ground_truth_poses(GROUND_TRUTH_PATH)

    print(f"[INFO] Found {len(entries)} object-image pose entries")

    written = 0
    skipped = 0

    for entry in entries:
        mesh_name = entry["mesh_name"]
        image_name = entry["image_name"]
        pose = entry["pose"]

        image_path = IMAGE_DIR / f"{image_name}.png"

        if not image_path.exists():
            print(f"[WARN] Missing image for {image_name}: {image_path}")
            skipped += 1
            continue

        out_path = write_label_json(
            image_name=image_name,
            mesh_name=mesh_name,
            pose=pose,
            camera_data=camera_data,
        )

        print(f"[OK] Wrote {out_path}")
        written += 1

    print()
    print(f"[DONE] Written labels: {written}")
    print(f"[DONE] Skipped labels: {skipped}")
    print(f"[DONE] Intrinsics: {INTRINSICS_PATH}")
    print(f"[DONE] Labels dir: {LABEL_DIR}")


if __name__ == "__main__":
    main()