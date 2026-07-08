#!/usr/bin/env python3

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"

import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial import cKDTree


# =========================
# HARD-CODED SETTINGS
# =========================

DATA_ROOT = Path(
    "/home/arghya/ihmc-repos/ihmc-humanoid-labeler/"
    "humanoid_data/2d_3d_data/Paper Recordings"
)

DATASET_NAME = "trash_can_2"

GT_DIR = DATA_ROOT / "GroundTruthPose" / DATASET_NAME / "filtered" / "labels"
EST_DIR = DATA_ROOT / "SupervisePose" / DATASET_NAME / "filtered" / "labels"
MESH_DIR = DATA_ROOT / "SupervisePose" / DATASET_NAME / "filtered" / "mesh"

USE_ADDS = True
METRIC_NAME = "ADD-S" if USE_ADDS else "ADD"

OUTPUT_DIR = DATA_ROOT / "results" / f"{METRIC_NAME.lower().replace('-', '')}_heatmap_white_dynamic" / DATASET_NAME

MESH_UNIT = "m"

NUM_COLOR_BINS = 10
EPS = 1e-12

NORMALIZE_FROM_ZERO = False

SAVE_RAW_RENDER = False
SAVE_WITH_COLORBAR = True


# =========================
# RENDERER
# =========================

class WhiteBackgroundPyrender:
    def __init__(self):
        self.renderer = None
        self.size = None

    def render(self, mesh_cam_gl_m, K, width, height):
        if self.renderer is None:
            self.renderer = pyrender.OffscreenRenderer(width, height)
            self.size = (width, height)

        if self.size != (width, height):
            raise RuntimeError("Image size changed between frames.")

        scene = pyrender.Scene(
            bg_color=np.array([1.0, 1.0, 1.0, 1.0]),
            ambient_light=np.array([0.7, 0.7, 0.7, 1.0]),
        )

        scene.add(pyrender.Mesh.from_trimesh(mesh_cam_gl_m, smooth=True))

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

        light = pyrender.DirectionalLight(
            color=np.ones(3),
            intensity=3.0,
        )
        scene.add(light, pose=camera_pose)

        color, depth = self.renderer.render(scene)
        return color, depth


# =========================
# HELPERS
# =========================

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


def transform_points(T, points):
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1,
    )
    return (T @ points_h.T).T[:, :3]


def get_mesh_path():
    mesh_files = sorted(MESH_DIR.glob("*.obj"))
    if len(mesh_files) != 1:
        raise RuntimeError(f"Expected exactly one OBJ in {MESH_DIR}, found {len(mesh_files)}")
    return mesh_files[0]


def compute_vertex_errors(pts_gt_cv_m, pts_est_cv_m):
    if USE_ADDS:
        tree = cKDTree(pts_est_cv_m)
        errors_m, nearest_indices = tree.query(pts_gt_cv_m, k=1)
        return errors_m
    else:
        return np.linalg.norm(pts_gt_cv_m - pts_est_cv_m, axis=1)


def error_to_colors_dynamic(errors_m):
    err_min = float(np.min(errors_m))
    err_max = float(np.max(errors_m))
    err_mean = float(np.mean(errors_m))
    err_median = float(np.median(errors_m))

    if NORMALIZE_FROM_ZERO:
        norm_min = 0.0
    else:
        norm_min = err_min

    norm_max = err_max
    norm_range = max(norm_max - norm_min, EPS)

    normalized = (errors_m - norm_min) / norm_range
    normalized = np.clip(normalized, 0.0, 1.0)

    cmap = plt.cm.get_cmap("turbo")
    colors = (cmap(normalized) * 255).astype(np.uint8)

    hist_counts, hist_edges = np.histogram(
        errors_m,
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


def print_stats(stem, stats):
    print(f"\n[COLOR DISTRIBUTION] {stem}")
    print(f"  metric           : {METRIC_NAME}")
    print(f"  actual min error : {stats['err_min']:.8f} m = {stats['err_min'] * 100:.4f} cm")
    print(f"  actual max error : {stats['err_max']:.8f} m = {stats['err_max'] * 100:.4f} cm")
    print(f"  mean {METRIC_NAME:<8}: {stats['err_mean']:.8f} m = {stats['err_mean'] * 100:.4f} cm")
    print(f"  median error     : {stats['err_median']:.8f} m = {stats['err_median'] * 100:.4f} cm")
    print(f"  color min        : {stats['norm_min']:.8f} m = {stats['norm_min'] * 100:.4f} cm")
    print(f"  color max        : {stats['norm_max']:.8f} m = {stats['norm_max'] * 100:.4f} cm")
    print(f"  color range      : {stats['norm_range']:.8f} m = {stats['norm_range'] * 100:.4f} cm")
    print(f"  color gap/bin    : {stats['color_gap']:.8f} m = {stats['color_gap'] * 100:.4f} cm")

    edges = stats["hist_edges"]
    counts = stats["hist_counts"]

    print("  bins:")
    for i, count in enumerate(counts):
        lo = edges[i]
        hi = edges[i + 1]
        print(
            f"    bin {i:02d}: "
            f"{lo:.8f} m - {hi:.8f} m "
            f"({lo * 100:.4f} cm - {hi * 100:.4f} cm), "
            f"vertices: {int(count)}"
        )


def save_with_colorbar(render_rgb, stats, out_path):
    norm_min_cm = stats["norm_min"] * 100.0
    norm_max_cm = stats["norm_max"] * 100.0

    cmap = plt.cm.get_cmap("turbo")
    norm = mpl.colors.Normalize(vmin=norm_min_cm, vmax=norm_max_cm)

    fig = plt.figure(figsize=(12, 7), dpi=150)

    ax_img = fig.add_axes([0.02, 0.05, 0.83, 0.90])
    ax_img.imshow(render_rgb)
    ax_img.axis("off")

    ax_cb = fig.add_axes([0.88, 0.15, 0.035, 0.70])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label(f"{METRIC_NAME} vertex error (cm)", rotation=90, labelpad=15)

    title = (
        f"Dynamic {METRIC_NAME} heatmap: "
        f"{norm_min_cm:.3f} cm to {norm_max_cm:.3f} cm\n"
        f"Mean {METRIC_NAME}: {stats['err_mean'] * 100.0:.3f} cm"
    )
    fig.suptitle(title, fontsize=12)

    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_one_frame(rasterizer, mesh_template, gt_json, est_json, colorbar_out_path):
    stem = gt_json.stem

    T_gt, K_gt, height, width = load_pose_json(gt_json)
    T_est, K_est, height_est, width_est = load_pose_json(est_json)

    if height != height_est or width != width_est:
        raise RuntimeError(f"Image size mismatch for {gt_json.name}")

    if not np.allclose(K_gt, K_est):
        print(f"[WARN] Intrinsics differ for {gt_json.name}. Using GT intrinsics.")

    vertices = np.asarray(mesh_template.vertices, dtype=np.float64)

    if MESH_UNIT == "mm":
        vertices_m = vertices / 1000.0
    elif MESH_UNIT == "m":
        vertices_m = vertices.copy()
    else:
        raise ValueError(f"Unknown MESH_UNIT: {MESH_UNIT}")

    pts_gt_cv_m = transform_points(T_gt, vertices_m)
    pts_est_cv_m = transform_points(T_est, vertices_m)

    errors_m = compute_vertex_errors(pts_gt_cv_m, pts_est_cv_m)

    colors_rgba, stats = error_to_colors_dynamic(errors_m)
    print_stats(stem, stats)

    pts_gt_gl_m = pts_gt_cv_m.copy()
    pts_gt_gl_m[:, 1] *= -1.0
    pts_gt_gl_m[:, 2] *= -1.0

    heatmap_mesh = mesh_template.copy()
    heatmap_mesh.vertices = pts_gt_gl_m
    heatmap_mesh.visual = trimesh.visual.ColorVisuals(
        heatmap_mesh,
        vertex_colors=colors_rgba,
    )

    color, depth = rasterizer.render(
        mesh_cam_gl_m=heatmap_mesh,
        K=K_gt,
        width=width,
        height=height,
    )

    save_with_colorbar(color, stats, colorbar_out_path)

    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_path = get_mesh_path()
    print(f"[INFO] Mesh: {mesh_path}")
    print(f"[INFO] Mesh unit: {MESH_UNIT}")
    print(f"[INFO] Metric: {METRIC_NAME}")
    print(f"[INFO] Background: white")
    print(f"[INFO] Normalize from zero: {NORMALIZE_FROM_ZERO}")
    print(f"[INFO] Color style: turbo")
    print(f"[INFO] Color bins: {NUM_COLOR_BINS}")

    mesh_template = trimesh.load(str(mesh_path), force="mesh")
    print(f"[INFO] Mesh extents: {mesh_template.extents}")

    gt_jsons = sorted(GT_DIR.glob("*.json"))
    if len(gt_jsons) == 0:
        raise RuntimeError(f"No GT JSON files found in {GT_DIR}")

    rasterizer = WhiteBackgroundPyrender()

    mean_values = []
    min_values = []
    max_values = []

    csv_path = OUTPUT_DIR / f"per_frame_{METRIC_NAME.lower().replace('-', '')}_distribution.csv"

    with open(csv_path, "w") as f:
        f.write(
            "frame,metric,"
            "actual_min_m,actual_min_cm,"
            "actual_max_m,actual_max_cm,"
            "mean_m,mean_cm,"
            "median_m,median_cm,"
            "color_min_m,color_min_cm,"
            "color_max_m,color_max_cm,"
            "color_gap_m,color_gap_cm\n"
        )

        for gt_json in gt_jsons:
            stem = gt_json.stem
            est_json = EST_DIR / gt_json.name
            colorbar_out_path = OUTPUT_DIR / f"{stem}_{METRIC_NAME.lower().replace('-', '')}_white_dynamic_heatmap_colorbar.png"

            if not est_json.exists():
                print(f"[SKIP] Missing prediction: {est_json}")
                continue

            print(f"\n[INFO] Rendering {stem}")

            stats = render_one_frame(
                rasterizer=rasterizer,
                mesh_template=mesh_template,
                gt_json=gt_json,
                est_json=est_json,
                colorbar_out_path=colorbar_out_path,
            )

            mean_values.append(stats["err_mean"])
            min_values.append(stats["err_min"])
            max_values.append(stats["err_max"])

            f.write(
                f"{stem},{METRIC_NAME},"
                f"{stats['err_min']:.10f},{stats['err_min'] * 100:.10f},"
                f"{stats['err_max']:.10f},{stats['err_max'] * 100:.10f},"
                f"{stats['err_mean']:.10f},{stats['err_mean'] * 100:.10f},"
                f"{stats['err_median']:.10f},{stats['err_median'] * 100:.10f},"
                f"{stats['norm_min']:.10f},{stats['norm_min'] * 100:.10f},"
                f"{stats['norm_max']:.10f},{stats['norm_max'] * 100:.10f},"
                f"{stats['color_gap']:.10f},{stats['color_gap'] * 100:.10f}\n"
            )

            print(f"[INFO] Saved: {colorbar_out_path}")

    if len(mean_values) > 0:
        summary_path = OUTPUT_DIR / "summary.txt"

        with open(summary_path, "w") as f:
            f.write(f"dataset: {DATASET_NAME}\n")
            f.write(f"frames_rendered: {len(mean_values)}\n")
            f.write(f"metric: {METRIC_NAME}\n")
            f.write(f"mesh: {mesh_path}\n")
            f.write(f"mesh_unit: {MESH_UNIT}\n")
            f.write("background: white\n")
            f.write("color_style: turbo\n")
            f.write(f"normalize_from_zero: {NORMALIZE_FROM_ZERO}\n")
            f.write("normalization: per_frame_dynamic_min_to_max\n")
            f.write(f"sequence_mean_{METRIC_NAME.lower().replace('-', '')}_m: {np.mean(mean_values):.10f}\n")
            f.write(f"sequence_mean_{METRIC_NAME.lower().replace('-', '')}_cm: {np.mean(mean_values) * 100:.10f}\n")
            f.write(f"sequence_global_min_m: {np.min(min_values):.10f}\n")
            f.write(f"sequence_global_min_cm: {np.min(min_values) * 100:.10f}\n")
            f.write(f"sequence_global_max_m: {np.max(max_values):.10f}\n")
            f.write(f"sequence_global_max_cm: {np.max(max_values) * 100:.10f}\n")
            f.write(f"csv: {csv_path}\n")

        print(f"\n[INFO] Rendered frames: {len(mean_values)}")
        print(f"[INFO] Sequence mean {METRIC_NAME}: {np.mean(mean_values):.8f} m = {np.mean(mean_values) * 100:.4f} cm")
        print(f"[INFO] Sequence global min: {np.min(min_values):.8f} m = {np.min(min_values) * 100:.4f} cm")
        print(f"[INFO] Sequence global max: {np.max(max_values):.8f} m = {np.max(max_values) * 100:.4f} cm")
        print(f"[INFO] Summary saved: {summary_path}")
        print(f"[INFO] CSV saved: {csv_path}")


if __name__ == "__main__":
    main()