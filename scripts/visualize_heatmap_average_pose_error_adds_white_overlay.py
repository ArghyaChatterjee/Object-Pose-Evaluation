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
# CHANGE ONLY THIS
# =========================

OBJECT_NAME = "trash_can_2"


# =========================
# PATH SETTINGS
# =========================

DATA_ROOT = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper_Recordings"
)

GT_DIR = DATA_ROOT / "GroundTruthPose" / OBJECT_NAME / "filtered" / "labels"
EST_DIR = DATA_ROOT / "SupervisePose" / OBJECT_NAME / "filtered" / "labels"

MESH_DIR_CANDIDATES = [
    DATA_ROOT / "GroundTruthPose" / OBJECT_NAME / "filtered" / "mesh",
    DATA_ROOT / "SupervisePose" / OBJECT_NAME / "filtered" / "mesh",
]

OUTPUT_DIR = DATA_ROOT / "results" / "overall_sequence_adds_heatmap" / OBJECT_NAME
OUTPUT_IMAGE = OUTPUT_DIR / f"{OBJECT_NAME}_overall_sequence_adds_heatmap_first_gt_view_colorbar.png"
OUTPUT_SUMMARY = OUTPUT_DIR / f"{OBJECT_NAME}_overall_sequence_adds_summary.txt"


# =========================
# SETTINGS
# =========================

MESH_UNIT = "m"

# "mean" = average per-vertex ADD-S error over sequence
# "sum"  = total accumulated per-vertex ADD-S error over sequence
ACCUMULATION_MODE = "mean"

# False: actual min error -> lowest color, actual max error -> highest color
# True:  0 error -> lowest color, actual max error -> highest color
NORMALIZE_FROM_ZERO = False

NUM_COLOR_BINS = 10
EPS = 1e-12


# =========================
# HELPERS
# =========================

def find_mesh_path():
    for mesh_dir in MESH_DIR_CANDIDATES:
        mesh_files = sorted(mesh_dir.glob("*.obj"))
        if len(mesh_files) == 1:
            return mesh_files[0]

    searched = "\n".join(str(p) for p in MESH_DIR_CANDIDATES)
    raise RuntimeError(f"Could not find exactly one OBJ mesh. Searched:\n{searched}")


def load_pose_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    T = np.asarray(data["objects"][0]["transform_matrix"], dtype=np.float64)

    cam = data["camera_data"]
    intr = cam["intrinsics"]

    K = np.array(
        [
            [intr["fx"], 0.0, intr["cx"]],
            [0.0, intr["fy"], intr["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    height = int(cam["height"])
    width = int(cam["width"])

    return T, K, height, width


def load_pose_rt(path: Path):
    T, _, _, _ = load_pose_json(path)
    return T[:3, :3], T[:3, 3]


def transform_points(R, t, points):
    return (R @ points.T).T + t


def object_diameter(points):
    extent = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(extent))


def mesh_vertices_in_m(mesh):
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if MESH_UNIT == "mm":
        return vertices / 1000.0

    if MESH_UNIT == "m":
        return vertices.copy()

    raise ValueError(f"Unknown MESH_UNIT: {MESH_UNIT}")


def get_first_gt_json():
    gt_jsons = sorted(GT_DIR.glob("*.json"))
    if len(gt_jsons) == 0:
        raise RuntimeError(f"No GT JSON files found in: {GT_DIR}")
    return gt_jsons[0]


# =========================
# SEQUENCE ADD-S
# =========================

def compute_sequence_vertex_errors_adds(vertices_m):
    sum_vertex_errors = np.zeros(len(vertices_m), dtype=np.float64)

    frame_adds_values = []
    valid_frames = 0
    skipped_frames = 0

    gt_paths = sorted(GT_DIR.glob("*.json"))

    if len(gt_paths) == 0:
        raise RuntimeError(f"No GT JSON files found in: {GT_DIR}")

    for gt_path in gt_paths:
        est_path = EST_DIR / gt_path.name

        if not est_path.exists():
            print(f"[SKIP] Missing estimated pose: {est_path.name}")
            skipped_frames += 1
            continue

        R_gt, t_gt = load_pose_rt(gt_path)
        R_est, t_est = load_pose_rt(est_path)

        pts_gt = transform_points(R_gt, t_gt, vertices_m)
        pts_est = transform_points(R_est, t_est, vertices_m)

        # ADD-S: for every GT vertex, find the closest predicted vertex.
        tree = cKDTree(pts_est)
        per_vertex_errors, nearest_indices = tree.query(pts_gt, k=1)

        sum_vertex_errors += per_vertex_errors

        frame_adds = float(np.mean(per_vertex_errors))
        frame_adds_values.append(frame_adds)
        valid_frames += 1

        print(
            f"[FRAME] {gt_path.stem}: "
            f"ADD-S = {frame_adds:.8f} m = {frame_adds * 100.0:.4f} cm"
        )

    if valid_frames == 0:
        raise RuntimeError("No valid frames were evaluated.")

    mean_vertex_errors = sum_vertex_errors / valid_frames

    if ACCUMULATION_MODE == "mean":
        final_vertex_errors = mean_vertex_errors
    elif ACCUMULATION_MODE == "sum":
        final_vertex_errors = sum_vertex_errors
    else:
        raise ValueError(f"Unknown ACCUMULATION_MODE: {ACCUMULATION_MODE}")

    return {
        "final_vertex_errors": final_vertex_errors,
        "frame_adds_values": np.asarray(frame_adds_values, dtype=np.float64),
        "valid_frames": valid_frames,
        "skipped_frames": skipped_frames,
    }


# =========================
# COLORING
# =========================

def error_to_vertex_colors(errors):
    err_min = float(np.min(errors))
    err_max = float(np.max(errors))
    err_mean = float(np.mean(errors))
    err_median = float(np.median(errors))

    norm_min = 0.0 if NORMALIZE_FROM_ZERO else err_min
    norm_max = err_max
    norm_range = max(norm_max - norm_min, EPS)

    normalized = (errors - norm_min) / norm_range
    normalized = np.clip(normalized, 0.0, 1.0)

    cmap = plt.cm.get_cmap("turbo")
    colors = (cmap(normalized) * 255).astype(np.uint8)

    hist_counts, hist_edges = np.histogram(
        errors,
        bins=NUM_COLOR_BINS,
        range=(norm_min, norm_max if norm_max > norm_min else norm_min + EPS),
    )

    stats = {
        "err_min": err_min,
        "err_max": err_max,
        "err_mean": err_mean,
        "err_median": err_median,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "norm_range": norm_range,
        "color_gap": norm_range / NUM_COLOR_BINS,
        "hist_counts": hist_counts,
        "hist_edges": hist_edges,
    }

    return colors, stats


def print_distribution(stats):
    print("\n[OVERALL SEQUENCE ADD-S COLOR DISTRIBUTION]")
    print(f"  object            : {OBJECT_NAME}")
    print(f"  accumulation mode : {ACCUMULATION_MODE}")
    print(f"  actual min error  : {stats['err_min']:.8f} m = {stats['err_min'] * 100.0:.4f} cm")
    print(f"  actual max error  : {stats['err_max']:.8f} m = {stats['err_max'] * 100.0:.4f} cm")
    print(f"  mean error        : {stats['err_mean']:.8f} m = {stats['err_mean'] * 100.0:.4f} cm")
    print(f"  median error      : {stats['err_median']:.8f} m = {stats['err_median'] * 100.0:.4f} cm")
    print(f"  color min         : {stats['norm_min']:.8f} m = {stats['norm_min'] * 100.0:.4f} cm")
    print(f"  color max         : {stats['norm_max']:.8f} m = {stats['norm_max'] * 100.0:.4f} cm")
    print(f"  color range       : {stats['norm_range']:.8f} m = {stats['norm_range'] * 100.0:.4f} cm")
    print(f"  color gap/bin     : {stats['color_gap']:.8f} m = {stats['color_gap'] * 100.0:.4f} cm")


# =========================
# RENDERING USING FIRST GT VIEW
# =========================

def make_mesh_in_first_gt_pose(mesh_template, vertices_m, colors_rgba, first_gt_json):
    T_first, K_first, height_first, width_first = load_pose_json(first_gt_json)

    R_first = T_first[:3, :3]
    t_first = T_first[:3, 3]

    pts_first_cv = transform_points(R_first, t_first, vertices_m)

    # OpenCV camera frame -> OpenGL camera frame.
    pts_first_gl = pts_first_cv.copy()
    pts_first_gl[:, 1] *= -1.0
    pts_first_gl[:, 2] *= -1.0

    mesh = mesh_template.copy()
    mesh.vertices = pts_first_gl
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=colors_rgba)

    return mesh, K_first, width_first, height_first, first_gt_json.stem


def render_white_background(mesh_cam_gl, K, width, height):
    scene = pyrender.Scene(
        bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
        ambient_light=np.array([0.8, 0.8, 0.8, 1.0]),
    )

    scene.add(pyrender.Mesh.from_trimesh(mesh_cam_gl, smooth=True))

    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=0.01,
        zfar=100.0,
    )

    camera_pose = np.eye(4, dtype=np.float64)
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color


def save_with_colorbar(render_rgb, stats, out_path, first_frame_name):
    norm_min_cm = stats["norm_min"] * 100.0
    norm_max_cm = stats["norm_max"] * 100.0

    cmap = plt.cm.get_cmap("turbo")
    norm = mpl.colors.Normalize(vmin=norm_min_cm, vmax=norm_max_cm)

    fig = plt.figure(figsize=(12, 7), dpi=150)

    ax_img = fig.add_axes([0.02, 0.05, 0.80, 0.90])
    ax_img.imshow(render_rgb)
    ax_img.axis("off")

    ax_cb = fig.add_axes([0.86, 0.15, 0.035, 0.70])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cb)

    if ACCUMULATION_MODE == "mean":
        cbar_label = "Mean per-vertex ADD-S over sequence (cm)"
        title_metric = "Overall mean ADD-S heatmap"
    else:
        cbar_label = "Summed per-vertex ADD-S over sequence (cm)"
        title_metric = "Overall summed ADD-S heatmap"

    cbar.set_label(cbar_label, rotation=90, labelpad=15)

    title = (
        f"{OBJECT_NAME} | {title_metric}\n"
        f"Rendered using first GT view: {first_frame_name}\n"
        f"Color range: {norm_min_cm:.3f} cm to {norm_max_cm:.3f} cm | "
        f"Mesh mean: {stats['err_mean'] * 100.0:.3f} cm"
    )

    fig.suptitle(title, fontsize=12)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# =========================
# MAIN
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_path = find_mesh_path()
    first_gt_json = get_first_gt_json()

    print(f"[INFO] Object: {OBJECT_NAME}")
    print(f"[INFO] Metric: ADD-S")
    print(f"[INFO] Mesh: {mesh_path}")
    print(f"[INFO] GT dir: {GT_DIR}")
    print(f"[INFO] EST dir: {EST_DIR}")
    print(f"[INFO] First GT view: {first_gt_json}")
    print(f"[INFO] Output: {OUTPUT_IMAGE}")
    print(f"[INFO] Mesh unit: {MESH_UNIT}")
    print(f"[INFO] Accumulation mode: {ACCUMULATION_MODE}")
    print(f"[INFO] Normalize from zero: {NORMALIZE_FROM_ZERO}")

    mesh_template = trimesh.load(str(mesh_path), force="mesh")
    vertices_m = mesh_vertices_in_m(mesh_template)

    diameter = object_diameter(vertices_m)
    threshold = 0.1 * diameter

    result = compute_sequence_vertex_errors_adds(vertices_m)

    final_vertex_errors = result["final_vertex_errors"]
    frame_adds_values = result["frame_adds_values"]
    valid_frames = result["valid_frames"]
    skipped_frames = result["skipped_frames"]

    colors_rgba, stats = error_to_vertex_colors(final_vertex_errors)
    print_distribution(stats)

    heatmap_mesh, K_first, width_first, height_first, first_frame_name = make_mesh_in_first_gt_pose(
        mesh_template=mesh_template,
        vertices_m=vertices_m,
        colors_rgba=colors_rgba,
        first_gt_json=first_gt_json,
    )

    render_rgb = render_white_background(
        mesh_cam_gl=heatmap_mesh,
        K=K_first,
        width=width_first,
        height=height_first,
    )

    save_with_colorbar(
        render_rgb=render_rgb,
        stats=stats,
        out_path=OUTPUT_IMAGE,
        first_frame_name=first_frame_name,
    )

    adds_recall = float(np.mean(frame_adds_values < threshold)) * 100.0
    sequence_mean_adds = float(np.mean(frame_adds_values))

    with open(OUTPUT_SUMMARY, "w") as f:
        f.write(f"object: {OBJECT_NAME}\n")
        f.write("metric: ADD-S\n")
        f.write(f"mesh: {mesh_path}\n")
        f.write(f"gt_dir: {GT_DIR}\n")
        f.write(f"est_dir: {EST_DIR}\n")
        f.write(f"first_gt_view: {first_gt_json}\n")
        f.write(f"frames_evaluated: {valid_frames}\n")
        f.write(f"frames_skipped: {skipped_frames}\n")
        f.write(f"mesh_unit: {MESH_UNIT}\n")
        f.write(f"accumulation_mode: {ACCUMULATION_MODE}\n")
        f.write(f"normalize_from_zero: {NORMALIZE_FROM_ZERO}\n")
        f.write(f"object_diameter_m: {diameter:.10f}\n")
        f.write(f"adds_0.1d_threshold_m: {threshold:.10f}\n")
        f.write(f"adds_0.1d_recall_percent: {adds_recall:.6f}\n")
        f.write(f"sequence_mean_adds_m: {sequence_mean_adds:.10f}\n")
        f.write(f"sequence_mean_adds_cm: {sequence_mean_adds * 100.0:.10f}\n")
        f.write(f"vertex_error_min_m: {stats['err_min']:.10f}\n")
        f.write(f"vertex_error_min_cm: {stats['err_min'] * 100.0:.10f}\n")
        f.write(f"vertex_error_max_m: {stats['err_max']:.10f}\n")
        f.write(f"vertex_error_max_cm: {stats['err_max'] * 100.0:.10f}\n")
        f.write(f"vertex_error_mean_m: {stats['err_mean']:.10f}\n")
        f.write(f"vertex_error_mean_cm: {stats['err_mean'] * 100.0:.10f}\n")
        f.write(f"output_image: {OUTPUT_IMAGE}\n")

    print(f"\n[INFO] Frames evaluated: {valid_frames}")
    print(f"[INFO] Frames skipped: {skipped_frames}")
    print(f"[INFO] Object diameter: {diameter:.8f} m")
    print(f"[INFO] ADD-S-0.1d threshold: {threshold:.8f} m")
    print(f"[INFO] ADD-S-0.1d Recall: {adds_recall:.4f} %")
    print(
        f"[INFO] Sequence mean ADD-S: "
        f"{sequence_mean_adds:.8f} m = {sequence_mean_adds * 100.0:.4f} cm"
    )
    print(f"[INFO] Saved heatmap: {OUTPUT_IMAGE}")
    print(f"[INFO] Saved summary: {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()