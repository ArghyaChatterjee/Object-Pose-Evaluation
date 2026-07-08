#!/usr/bin/env python3

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

import json
from pathlib import Path

import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import cKDTree


# =========================
# OBJECTS
# =========================

OBJECT_NAMES = [
    "bottle_1",
    "bottle_2",
    "bottle_3",
    "charge_1",
    "charge_2",
    "door_lever_1",
    "door_lever_2",
    "storage_container_1",
    "storage_container_2",
    "trash_can_3",
    "trash_can_2",
    "trash_can_1",
]

ADDS_OBJECTS = {"trash_can_2"}

DEFAULT_OBJECT_VIS = {
    "rot_x": 0.0,
    "rot_y": -90.0,
    "rot_z": 180.0,
    "scale": 1.0,
    "dx": 0.0,
    "dy": 0.0,
    "dz": 0.0,
}

OBJECT_VIS_OVERRIDES = {
    # Example overrides. Tune these manually.
    "door_lever_1": {
        "rot_x": 0.0,
        "rot_y": -105.0,
        "rot_z": 0.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
    "door_lever_2": {
        "rot_x": 0.0,
        "rot_y": -70.0,
        "rot_z": 0.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
    "storage_container_2": {
        "rot_x": -20.0,
        "rot_y": 0.0,
        "rot_z": 180.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
    "charge_1": {
        "rot_x": 0.0,
        "rot_y": 180.0,
        "rot_z": 180.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
    "trash_can_1": {
        "rot_x": 0.0,
        "rot_y": -70.0,
        "rot_z": 180.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
    "trash_can_2": {
        "rot_x": 0.0,
        "rot_y": 100.0,
        "rot_z": 180.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
    "trash_can_3": {
        "rot_x": 210.0,
        "rot_y": -5.0,
        "rot_z": 0.0,
        "scale": 1.0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
    },
}


# =========================
# PATHS
# =========================

DATA_ROOT = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper_Recordings"
)

OUTPUT_DIR = DATA_ROOT / "results" / "overall_all_instances_pose_error_scene"
OUTPUT_IMAGE = OUTPUT_DIR / "overall_all_instances_pose_error_scene.png"
OUTPUT_SUMMARY = OUTPUT_DIR / "overall_all_instances_pose_error_scene_summary.txt"
OUTPUT_CSV = OUTPUT_DIR / "overall_all_instances_pose_error_scene_summary.csv"


# =========================
# SETTINGS
# =========================

MESH_UNIT = "m"
ACCUMULATION_MODE = "mean"
NORMALIZE_FROM_ZERO = False

RENDER_WIDTH = 2200
RENDER_HEIGHT = 1200

# Larger value = objects farther apart.
X_SPACING = 0.70
Y_SPACING = 0.65

# Manual visual scale only. Does NOT affect ADD calculation.
VISUAL_SCALE = 1.0

# Camera / scene view.
CAMERA_DISTANCE = 3.9
CAMERA_FOCAL_LENGTH = 1800.0

# Rotate objects visually for better 3D look.
VIEW_ROT_X_DEG = 0.0
VIEW_ROT_Y_DEG = -90.0
VIEW_ROT_Z_DEG = 180.0

EPS = 1e-12


# =========================
# PATH HELPERS
# =========================

def gt_dir_for(obj_name):
    return DATA_ROOT / "GroundTruthPose" / obj_name / "filtered" / "labels"


def est_dir_for(obj_name):
    return DATA_ROOT / "SupervisePose" / obj_name / "filtered" / "labels"


def mesh_dir_candidates_for(obj_name):
    return [
        DATA_ROOT / "GroundTruthPose" / obj_name / "filtered" / "mesh",
        DATA_ROOT / "SupervisePose" / obj_name / "filtered" / "mesh",
    ]


def find_mesh_path(obj_name):
    for mesh_dir in mesh_dir_candidates_for(obj_name):
        mesh_files = sorted(mesh_dir.glob("*.obj"))
        if len(mesh_files) == 1:
            return mesh_files[0]

    searched = "\n".join(str(p) for p in mesh_dir_candidates_for(obj_name))
    raise RuntimeError(f"Could not find exactly one OBJ for {obj_name}. Searched:\n{searched}")


def get_object_vis(obj_name):
    vis = DEFAULT_OBJECT_VIS.copy()
    vis.update(OBJECT_VIS_OVERRIDES.get(obj_name, {}))
    return vis

# =========================
# POSE / GEOMETRY
# =========================

def load_pose_rt(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    T = np.asarray(data["objects"][0]["transform_matrix"], dtype=np.float64)
    return T[:3, :3], T[:3, 3]


def transform_points(R, t, points):
    return (R @ points.T).T + t


def mesh_vertices_in_m(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if MESH_UNIT == "mm":
        return vertices / 1000.0
    if MESH_UNIT == "m":
        return vertices.copy()

    raise ValueError(f"Unknown MESH_UNIT: {MESH_UNIT}")


def object_diameter(points):
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extent))


def rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)],
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)],
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1],
    ])

    return Rz @ Ry @ Rx


# =========================
# ERROR COMPUTATION
# =========================

def compute_frame_errors(metric_name, vertices_m, R_gt, t_gt, R_est, t_est):
    pts_gt = transform_points(R_gt, t_gt, vertices_m)
    pts_est = transform_points(R_est, t_est, vertices_m)

    if metric_name == "ADD":
        return np.linalg.norm(pts_est - pts_gt, axis=1)

    if metric_name == "ADD-S":
        tree = cKDTree(pts_est)
        errors, _ = tree.query(pts_gt, k=1)
        return errors

    raise ValueError(metric_name)


def compute_sequence_vertex_errors(obj_name, vertices_m):
    metric_name = "ADD-S" if obj_name in ADDS_OBJECTS else "ADD"

    gt_paths = sorted(gt_dir_for(obj_name).glob("*.json"))
    if len(gt_paths) == 0:
        raise RuntimeError(f"No GT JSONs found for {obj_name}")

    sum_vertex_errors = np.zeros(len(vertices_m), dtype=np.float64)

    frame_values = []
    valid_frames = 0
    skipped_frames = 0

    for gt_path in gt_paths:
        est_path = est_dir_for(obj_name) / gt_path.name

        if not est_path.exists():
            skipped_frames += 1
            continue

        R_gt, t_gt = load_pose_rt(gt_path)
        R_est, t_est = load_pose_rt(est_path)

        per_vertex_errors = compute_frame_errors(
            metric_name,
            vertices_m,
            R_gt,
            t_gt,
            R_est,
            t_est,
        )

        sum_vertex_errors += per_vertex_errors
        frame_values.append(float(np.mean(per_vertex_errors)))
        valid_frames += 1

    if valid_frames == 0:
        raise RuntimeError(f"No valid frames for {obj_name}")

    mean_vertex_errors = sum_vertex_errors / valid_frames

    if ACCUMULATION_MODE == "mean":
        final_vertex_errors = mean_vertex_errors
    elif ACCUMULATION_MODE == "sum":
        final_vertex_errors = sum_vertex_errors
    else:
        raise ValueError(ACCUMULATION_MODE)

    return {
        "metric_name": metric_name,
        "final_vertex_errors": final_vertex_errors,
        "frame_values": np.asarray(frame_values),
        "valid_frames": valid_frames,
        "skipped_frames": skipped_frames,
    }


# =========================
# COLORING
# =========================

def normalize_errors_to_colors(errors, global_min, global_max):
    denom = max(global_max - global_min, EPS)
    normalized = (errors - global_min) / denom
    normalized = np.clip(normalized, 0.0, 1.0)

    cmap = plt.cm.get_cmap("turbo")
    return (cmap(normalized) * 255).astype(np.uint8)


def stats_for_errors(errors):
    return {
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
    }


# =========================
# OBJECT PROCESSING
# =========================

def process_object(obj_name):
    print(f"[INFO] Processing {obj_name}")

    mesh_path = find_mesh_path(obj_name)
    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices_m = mesh_vertices_in_m(mesh)

    result = compute_sequence_vertex_errors(obj_name, vertices_m)

    errors = result["final_vertex_errors"]
    stats = stats_for_errors(errors)

    diameter = object_diameter(vertices_m)
    threshold = 0.1 * diameter
    recall = float(np.mean(result["frame_values"] < threshold)) * 100.0
    seq_mean = float(np.mean(result["frame_values"]))

    print(
        f"  {result['metric_name']}: "
        f"mean={seq_mean * 100.0:.3f} cm, "
        f"range={stats['min'] * 100.0:.3f}-{stats['max'] * 100.0:.3f} cm"
    )

    return {
        "object_name": obj_name,
        "metric_name": result["metric_name"],
        "mesh_path": mesh_path,
        "mesh": mesh,
        "vertices_m": vertices_m,
        "errors": errors,
        "stats": stats,
        "diameter": diameter,
        "threshold": threshold,
        "recall": recall,
        "sequence_mean": seq_mean,
        "valid_frames": result["valid_frames"],
        "skipped_frames": result["skipped_frames"],
    }


# =========================
# SCENE LAYOUT / RENDER
# =========================


def make_scene_mesh(obj_data, colors_rgba, translation):
    mesh = obj_data["mesh"].copy()
    vertices_m = obj_data["vertices_m"].copy()
    obj_name = obj_data["object_name"]

    vis = get_object_vis(obj_name)

    center = 0.5 * (vertices_m.min(axis=0) + vertices_m.max(axis=0))
    vertices = vertices_m - center

    vertices *= VISUAL_SCALE
    vertices *= vis["scale"]

    # Convert to OpenGL camera convention first.
    vertices_gl = vertices.copy()
    vertices_gl[:, 1] *= -1.0
    vertices_gl[:, 2] *= -1.0

    # Apply per-object rotation.
    R_obj = rotation_matrix_xyz(
        vis["rot_x"],
        vis["rot_y"],
        vis["rot_z"],
    )
    vertices_gl = (R_obj @ vertices_gl.T).T

    # Place into shared layout, including per-object offset.
    vertices_gl[:, 0] += translation[0] + vis["dx"]
    vertices_gl[:, 1] += translation[1] + vis["dy"]
    vertices_gl[:, 2] += translation[2] + vis["dz"]

    mesh.vertices = vertices_gl
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors_rgba)

    return mesh


def layout_positions(n_objects):
    n_cols = 6
    n_rows = int(np.ceil(n_objects / n_cols))

    positions = []
    for i in range(n_objects):
        row = i // n_cols
        col = i % n_cols

        x = (col - (n_cols - 1) / 2.0) * X_SPACING
        y = ((n_rows - 1) / 2.0 - row) * Y_SPACING

        # OpenGL camera looks along negative Z.
        z = -CAMERA_DISTANCE

        positions.append((x, y, z))

    return positions


def render_all_objects(processed_objects, global_min, global_max):
    scene = pyrender.Scene(
        bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
        ambient_light=np.array([0.85, 0.85, 0.85, 1.0]),
    )

    positions = layout_positions(len(processed_objects))

    for obj_data, pos in zip(processed_objects, positions):
        colors = normalize_errors_to_colors(
            obj_data["errors"],
            global_min,
            global_max,
        )

        mesh_scene = make_scene_mesh(
            obj_data=obj_data,
            colors_rgba=colors,
            translation=pos,
        )

        scene.add(pyrender.Mesh.from_trimesh(mesh_scene, smooth=True))

    camera = pyrender.IntrinsicsCamera(
        fx=CAMERA_FOCAL_LENGTH,
        fy=CAMERA_FOCAL_LENGTH,
        cx=RENDER_WIDTH / 2.0,
        cy=RENDER_HEIGHT / 2.0,
        znear=0.01,
        zfar=100.0,
    )

    camera_pose = np.eye(4)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(RENDER_WIDTH, RENDER_HEIGHT)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color


# =========================
# SAVE FIGURE
# =========================

def save_final_figure(render_rgb, processed_objects, global_min, global_max):
    norm_min_cm = global_min * 100.0
    norm_max_cm = global_max * 100.0

    cmap = plt.cm.get_cmap("turbo")
    norm = mpl.colors.Normalize(vmin=norm_min_cm, vmax=norm_max_cm)

    fig = plt.figure(figsize=(15, 8), dpi=180)

    ax_img = fig.add_axes([0.02, 0.08, 0.82, 0.84])
    ax_img.imshow(render_rgb)
    ax_img.axis("off")

    ax_cb = fig.add_axes([0.87, 0.25, 0.035, 0.50])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cb)

    cbar.set_label(
        "Mean per-vertex pose error over sequence (cm)\nADD for all objects except trash_can_2 uses ADD-S",
        rotation=90,
        labelpad=18,
    )

    fig.suptitle(
        "Overall Sequence Pose Error Heatmaps Across All Instances\n"
        f"Global color range: {norm_min_cm:.3f} cm to {norm_max_cm:.3f} cm",
        fontsize=14,
        y=0.97,
    )

    # Add compact text under image.
    text_lines = []
    for d in processed_objects:
        text_lines.append(
            f"{d['object_name']} ({d['metric_name']}): "
            f"{d['stats']['mean'] * 100.0:.2f} cm"
        )

    text = " | ".join(text_lines)
    fig.text(0.02, 0.025, text, fontsize=7)

    fig.savefig(str(OUTPUT_IMAGE), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_summary(processed_objects, global_min, global_max):
    with open(OUTPUT_CSV, "w") as f:
        f.write(
            "object,metric,frames,skipped,sequence_mean_cm,"
            "vertex_min_cm,vertex_max_cm,vertex_mean_cm,recall_01d_percent,mesh\n"
        )

        for d in processed_objects:
            f.write(
                f"{d['object_name']},{d['metric_name']},"
                f"{d['valid_frames']},{d['skipped_frames']},"
                f"{d['sequence_mean'] * 100.0:.10f},"
                f"{d['stats']['min'] * 100.0:.10f},"
                f"{d['stats']['max'] * 100.0:.10f},"
                f"{d['stats']['mean'] * 100.0:.10f},"
                f"{d['recall']:.6f},"
                f"{d['mesh_path']}\n"
            )

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(f"output_image: {OUTPUT_IMAGE}\n")
        f.write(f"global_min_cm: {global_min * 100.0:.10f}\n")
        f.write(f"global_max_cm: {global_max * 100.0:.10f}\n")
        f.write(f"adds_objects: {sorted(ADDS_OBJECTS)}\n")
        f.write(f"objects: {OBJECT_NAMES}\n")


# =========================
# MAIN
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    processed_objects = [process_object(name) for name in OBJECT_NAMES]

    all_errors = np.concatenate([d["errors"] for d in processed_objects])

    global_min = 0.0 if NORMALIZE_FROM_ZERO else float(np.min(all_errors))
    global_max = float(np.max(all_errors))

    print(f"\n[INFO] Global range: {global_min * 100.0:.3f} cm to {global_max * 100.0:.3f} cm")

    render_rgb = render_all_objects(
        processed_objects=processed_objects,
        global_min=global_min,
        global_max=global_max,
    )

    save_final_figure(
        render_rgb=render_rgb,
        processed_objects=processed_objects,
        global_min=global_min,
        global_max=global_max,
    )

    save_summary(processed_objects, global_min, global_max)

    print(f"[INFO] Saved: {OUTPUT_IMAGE}")
    print(f"[INFO] CSV: {OUTPUT_CSV}")
    print(f"[INFO] Summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()