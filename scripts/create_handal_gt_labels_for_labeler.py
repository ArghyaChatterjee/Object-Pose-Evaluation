#!/usr/bin/env python3

# Copyright (c) IHMC and affiliates.

import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# ============================================================
# Hardcoded paths
# ============================================================

ROOT = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper Recordings/"
    "handal_dataset_pots_pans_no_depth/031000/"
)

RGB_DIR = ROOT / "rgb"
SCENE_GT_PATH = ROOT / "scene_gt.json"
SCENE_CAMERA_PATH = ROOT / "scene_camera.json"

IMAGE_DIR = ROOT / "image"
LABEL_DIR = ROOT / "labels"
INTRINSICS_PATH = ROOT / "intrinsics.txt"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Metadata
# ============================================================

PROVENANCE = "handal"
DEFAULT_SCALE = [1.0, 1.0, 1.0]


# ============================================================
# Helpers
# ============================================================

def obj_id_to_mesh_name(obj_id: int) -> str:
    return f"obj_{obj_id:06d}"


def image_id_to_name(image_id: str) -> str:
    return f"{int(image_id):06d}"


def make_transform_matrix(cam_R_m2c, cam_t_m2c):
    R_mat = np.array(cam_R_m2c, dtype=float).reshape(3, 3)
    t_mm = np.array(cam_t_m2c, dtype=float)

    # BOP/LINEMOD translation is mm.
    # Our labeler expects meters.
    t_m = t_mm / 1000.0

    T = np.eye(4, dtype=float)
    T[:3, :3] = R_mat
    T[:3, 3] = t_m

    return T


def camera_data_from_scene_camera(camera_entry):
    K = camera_entry["cam_K"]

    fx = float(K[0])
    fy = float(K[4])
    cx = float(K[2])
    cy = float(K[5])

    width = int(camera_entry["width"])
    height = int(camera_entry["height"])

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


def write_intrinsics_txt(camera_data):
    intr = camera_data["intrinsics"]

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    width = camera_data["width"]
    height = camera_data["height"]

    distortion = " ".join(["0.000000e+00"] * 12)

    text = (
        f"fx {fx:.18e} fy {fy:.18e}\n"
        f"cx {cx:.18e} cy {cy:.18e}\n"
        f"distortion {distortion}\n"
        f"resolution {width}x{height}\n"
    )

    INTRINSICS_PATH.write_text(text)
    print(f"[OK] Wrote {INTRINSICS_PATH}")


def copy_sorted_images(scene_gt):
    copied = 0
    skipped = 0

    for image_id in sorted(scene_gt.keys(), key=lambda x: int(x)):
        image_name = image_id_to_name(image_id)

        src_png = RGB_DIR / f"{image_name}.png"
        src_jpg = RGB_DIR / f"{image_name}.jpg"
        src_jpeg = RGB_DIR / f"{image_name}.jpeg"

        if src_png.exists():
            src = src_png
            dst = IMAGE_DIR / f"{image_name}.png"
        elif src_jpg.exists():
            src = src_jpg
            dst = IMAGE_DIR / f"{image_name}.jpg"
        elif src_jpeg.exists():
            src = src_jpeg
            dst = IMAGE_DIR / f"{image_name}.jpeg"
        else:
            print(f"[WARN] Missing RGB image for frame {image_name}")
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1

    print(f"[OK] Copied images: {copied}")
    print(f"[OK] Skipped images: {skipped}")


def write_label_json(image_id, objects_gt, scene_camera):
    image_name = image_id_to_name(image_id)

    camera_data = camera_data_from_scene_camera(scene_camera[image_id])

    objects = []

    for obj in objects_gt:
        obj_id = int(obj["obj_id"])
        mesh_name = obj_id_to_mesh_name(obj_id)

        T = make_transform_matrix(
            obj["cam_R_m2c"],
            obj["cam_t_m2c"],
        )

        rotation_matrix = T[:3, :3]
        translation = T[:3, 3]

        quat_xyzw = R.from_matrix(rotation_matrix).as_quat().tolist()

        objects.append(
            {
                "class": mesh_name,
                "name": mesh_name,
                "provenance": PROVENANCE,
                "transform_matrix": T.tolist(),
                "location": translation.tolist(),
                "quaternion_xyzw": quat_xyzw,
                "scale": DEFAULT_SCALE,
            }
        )

    label_data = {
        "camera_data": camera_data,
        "objects": objects,
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

    if not RGB_DIR.exists():
        raise RuntimeError(f"Missing rgb folder: {RGB_DIR}")

    if not SCENE_GT_PATH.exists():
        raise RuntimeError(f"Missing scene_gt.json: {SCENE_GT_PATH}")

    if not SCENE_CAMERA_PATH.exists():
        raise RuntimeError(f"Missing scene_camera.json: {SCENE_CAMERA_PATH}")

    with open(SCENE_GT_PATH, "r") as f:
        scene_gt = json.load(f)

    with open(SCENE_CAMERA_PATH, "r") as f:
        scene_camera = json.load(f)

    # Write intrinsics.txt using the first sorted image.
    first_image_id = sorted(scene_camera.keys(), key=lambda x: int(x))[0]
    first_camera_data = camera_data_from_scene_camera(scene_camera[first_image_id])
    write_intrinsics_txt(first_camera_data)

    # Copy rgb images into labeler image/ folder.
    copy_sorted_images(scene_gt)

    written = 0
    skipped = 0

    for image_id in sorted(scene_gt.keys(), key=lambda x: int(x)):
        if image_id not in scene_camera:
            print(f"[WARN] Missing camera entry for image {image_id}")
            skipped += 1
            continue

        out_path = write_label_json(
            image_id=image_id,
            objects_gt=scene_gt[image_id],
            scene_camera=scene_camera,
        )

        print(f"[OK] Wrote {out_path}")
        written += 1

    print()
    print(f"[DONE] Labels written: {written}")
    print(f"[DONE] Labels skipped: {skipped}")
    print(f"[DONE] Image dir: {IMAGE_DIR}")
    print(f"[DONE] Label dir: {LABEL_DIR}")
    print(f"[DONE] Intrinsics: {INTRINSICS_PATH}")


if __name__ == "__main__":
    main()