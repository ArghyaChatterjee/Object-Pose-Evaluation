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
IMAGE_DIR = DATA_ROOT / "SupervisePose" / DATASET_NAME / "filtered" / "image"
MESH_DIR = DATA_ROOT / "SupervisePose" / DATASET_NAME / "filtered" / "mesh"

OUTPUT_DIR = DATA_ROOT / "results" / "add_heatmap_pyrender_dynamic" / DATASET_NAME

OVERLAY = True
OPACITY = 0.70

MESH_UNIT = "m"

# Number of printed color/error intervals.
NUM_COLOR_BINS = 10

# Numerical safety if all vertices have almost identical error.
EPS = 1e-12


# =========================
# SMALL SELF-CONTAINED RENDERER
# =========================

def opencv_to_opengl_transform():
    T = np.eye(4, dtype=np.float64)
    T[1, 1] = -1.0
    T[2, 2] = -1.0
    return T


class SimplePyrenderRasterizer:
    def __init__(self):
        self.renderer = None
        self.im_size = None

    def render_meshes(self, meshes_in_cam_mm, K, width, height):
        if self.renderer is None:
            self.im_size = (width, height)
            self.renderer = pyrender.OffscreenRenderer(width, height)
        else:
            if self.im_size != (width, height):
                raise RuntimeError("Renderer image size changed between frames.")

        scene = pyrender.Scene(
            bg_color=np.zeros(4),
            ambient_light=np.array([0.02, 0.02, 0.02, 1.0]),
        )

        for mesh_mm in meshes_in_cam_mm:
            mesh_m = mesh_mm.copy()
            mesh_m.vertices = np.asarray(mesh_m.vertices, dtype=np.float64) / 1000.0
            scene.add(pyrender.Mesh.from_trimesh(mesh_m, smooth=False))

        camera = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            znear=0.1,
            zfar=3000.0,
        )

        # camera_pose = opencv_to_opengl_transform()
        camera_pose = np.eye(4, dtype=np.float64)

        scene.add(camera, pose=camera_pose)

        light = pyrender.SpotLight(
            color=np.ones(3),
            intensity=2.4,
            innerConeAngle=np.pi / 16.0,
            outerConeAngle=np.pi / 6.0,
        )
        scene.add(light, pose=camera_pose)

        color, depth_m = self.renderer.render(scene)

        color = color.astype(np.float32) / 255.0
        depth_mm = depth_m.astype(np.float32) * 1000.0
        mask = depth_mm > 0

        return color, depth_mm, mask


# =========================
# HELPERS
# =========================

def load_pose_json(path: Path):
    with open(path, "r") as f:
        data = json.load(f)

    obj = data["objects"][0]
    T = np.asarray(obj["transform_matrix"], dtype=np.float64)

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
        raise RuntimeError(
            f"Expected exactly one OBJ in {MESH_DIR}, found {len(mesh_files)}"
        )
    return mesh_files[0]


def dynamic_error_to_vertex_colors(errors_m):
    err_min_m = float(np.min(errors_m))
    err_max_m = float(np.max(errors_m))
    err_mean_m = float(np.mean(errors_m))
    err_median_m = float(np.median(errors_m))

    error_range_m = err_max_m - err_min_m

    if error_range_m < EPS:
        normalized = np.zeros_like(errors_m, dtype=np.float64)
        color_gap_m = 0.0
    else:
        normalized = (errors_m - err_min_m) / error_range_m
        color_gap_m = error_range_m / NUM_COLOR_BINS

    cmap = plt.cm.get_cmap("turbo")
    colors_rgba = cmap(normalized)
    colors_rgba_u8 = (colors_rgba * 255).astype(np.uint8)

    hist_counts, hist_edges = np.histogram(
        errors_m,
        bins=NUM_COLOR_BINS,
        range=(err_min_m, err_max_m if err_max_m > err_min_m else err_min_m + EPS),
    )

    stats = {
        "err_min_m": err_min_m,
        "err_max_m": err_max_m,
        "err_mean_m": err_mean_m,
        "err_median_m": err_median_m,
        "error_range_m": error_range_m,
        "color_gap_m": color_gap_m,
        "hist_counts": hist_counts,
        "hist_edges": hist_edges,
    }

    return colors_rgba_u8, stats


def print_color_distribution(stem, stats):
    print(f"\n[COLOR DISTRIBUTION] {stem}")
    print(
        f"  min error      : {stats['err_min_m']:.8f} m "
        f"= {stats['err_min_m'] * 100.0:.4f} cm"
    )
    print(
        f"  max error      : {stats['err_max_m']:.8f} m "
        f"= {stats['err_max_m'] * 100.0:.4f} cm"
    )
    print(
        f"  mean ADD       : {stats['err_mean_m']:.8f} m "
        f"= {stats['err_mean_m'] * 100.0:.4f} cm"
    )
    print(
        f"  median error   : {stats['err_median_m']:.8f} m "
        f"= {stats['err_median_m'] * 100.0:.4f} cm"
    )
    print(
        f"  error range    : {stats['error_range_m']:.8f} m "
        f"= {stats['error_range_m'] * 100.0:.4f} cm"
    )
    print(
        f"  color gap/bin  : {stats['color_gap_m']:.8f} m "
        f"= {stats['color_gap_m'] * 100.0:.4f} cm"
    )
    print(f"  number of bins : {NUM_COLOR_BINS}")

    edges = stats["hist_edges"]
    counts = stats["hist_counts"]

    print("  bins:")
    for i, count in enumerate(counts):
        lo_m = edges[i]
        hi_m = edges[i + 1]
        print(
            f"    bin {i:02d}: "
            f"{lo_m:.8f} m - {hi_m:.8f} m "
            f"({lo_m * 100.0:.4f} cm - {hi_m * 100.0:.4f} cm), "
            f"vertices: {int(count)}"
        )


def overlay_render(base_rgb, render_rgb, render_mask, opacity):
    base = base_rgb.astype(np.float32)

    if render_rgb.max() <= 1.0:
        rendered = render_rgb * 255.0
    else:
        rendered = render_rgb.astype(np.float32)

    mask = render_mask.astype(bool)

    output = base.copy()
    output[mask] = (1.0 - opacity) * base[mask] + opacity * rendered[mask]

    return np.clip(output, 0, 255).astype(np.uint8)


def render_one_frame(rast, mesh_template, gt_json, est_json, image_path, out_path):
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

    pts_gt_m = transform_points(T_gt, vertices_m)
    pts_est_m = transform_points(T_est, vertices_m)

    per_vertex_error_m = np.linalg.norm(pts_gt_m - pts_est_m, axis=1)

    colors_rgba, stats = dynamic_error_to_vertex_colors(per_vertex_error_m)
    print_color_distribution(stem, stats)

    heatmap_mesh = mesh_template.copy()

    # Mesh is now placed in GT camera coordinates.
    # Renderer expects input mesh vertices in millimeters.
    # heatmap_mesh.vertices = pts_gt_m * 1000.0

    pts_gt_render_m = pts_gt_m.copy()
    pts_gt_render_m[:, 1] *= -1.0
    pts_gt_render_m[:, 2] *= -1.0

    heatmap_mesh.vertices = pts_gt_render_m * 1000.0

    heatmap_mesh.visual = trimesh.visual.ColorVisuals(
        heatmap_mesh,
        vertex_colors=colors_rgba,
    )

    render_rgb, render_depth_mm, render_mask = rast.render_meshes(
        meshes_in_cam_mm=[heatmap_mesh],
        K=K_gt,
        width=width,
        height=height,
    )

    if OVERLAY:
        base_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if base_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)

        if base_rgb.shape[:2] != (height, width):
            raise RuntimeError(
                f"RGB image size {base_rgb.shape[:2]} does not match JSON size {(height, width)}"
            )

        output_rgb = overlay_render(
            base_rgb=base_rgb,
            render_rgb=render_rgb,
            render_mask=render_mask,
            opacity=OPACITY,
        )
    else:
        output_rgb = (render_rgb * 255.0).astype(np.uint8)

    output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), output_bgr)

    return stats


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_path = get_mesh_path()
    print(f"[INFO] Mesh: {mesh_path}")
    print(f"[INFO] Mesh unit: {MESH_UNIT}")
    print(f"[INFO] Dynamic normalization: per-frame min-to-max")
    print(f"[INFO] Color bins printed: {NUM_COLOR_BINS}")

    mesh_template = trimesh.load(str(mesh_path), force="mesh")

    gt_jsons = sorted(GT_DIR.glob("*.json"))
    if len(gt_jsons) == 0:
        raise RuntimeError(f"No GT JSON files found in {GT_DIR}")

    rast = SimplePyrenderRasterizer()

    num_rendered = 0
    mean_add_values = []
    max_error_values = []
    min_error_values = []

    summary_csv_path = OUTPUT_DIR / "per_frame_error_distribution.csv"

    with open(summary_csv_path, "w") as csv_file:
        csv_file.write(
            "frame,"
            "min_error_m,min_error_cm,"
            "max_error_m,max_error_cm,"
            "mean_add_m,mean_add_cm,"
            "median_error_m,median_error_cm,"
            "error_range_m,error_range_cm,"
            "color_gap_m,color_gap_cm\n"
        )

        for gt_json in gt_jsons:
            stem = gt_json.stem

            est_json = EST_DIR / gt_json.name
            image_path = IMAGE_DIR / f"{stem}.png"
            out_path = OUTPUT_DIR / f"{stem}_add_heatmap_dynamic.png"

            if not est_json.exists():
                print(f"[SKIP] Missing prediction: {est_json}")
                continue

            if OVERLAY and not image_path.exists():
                print(f"[SKIP] Missing RGB image: {image_path}")
                continue

            print(f"\n[INFO] Rendering {stem}")

            stats = render_one_frame(
                rast=rast,
                mesh_template=mesh_template,
                gt_json=gt_json,
                est_json=est_json,
                image_path=image_path,
                out_path=out_path,
            )

            num_rendered += 1
            mean_add_values.append(stats["err_mean_m"])
            max_error_values.append(stats["err_max_m"])
            min_error_values.append(stats["err_min_m"])

            csv_file.write(
                f"{stem},"
                f"{stats['err_min_m']:.10f},{stats['err_min_m'] * 100.0:.10f},"
                f"{stats['err_max_m']:.10f},{stats['err_max_m'] * 100.0:.10f},"
                f"{stats['err_mean_m']:.10f},{stats['err_mean_m'] * 100.0:.10f},"
                f"{stats['err_median_m']:.10f},{stats['err_median_m'] * 100.0:.10f},"
                f"{stats['error_range_m']:.10f},{stats['error_range_m'] * 100.0:.10f},"
                f"{stats['color_gap_m']:.10f},{stats['color_gap_m'] * 100.0:.10f}\n"
            )

            print(f"[INFO] Saved: {out_path}")

    if num_rendered > 0:
        sequence_mean_add_m = float(np.mean(mean_add_values))
        sequence_mean_max_error_m = float(np.mean(max_error_values))
        sequence_global_max_error_m = float(np.max(max_error_values))
        sequence_global_min_error_m = float(np.min(min_error_values))

        summary_path = OUTPUT_DIR / "summary.txt"

        with open(summary_path, "w") as f:
            f.write(f"dataset: {DATASET_NAME}\n")
            f.write(f"frames_rendered: {num_rendered}\n")
            f.write(f"mesh: {mesh_path}\n")
            f.write(f"mesh_unit: {MESH_UNIT}\n")
            f.write("normalization: dynamic_per_frame_min_to_max\n")
            f.write(f"num_color_bins: {NUM_COLOR_BINS}\n")
            f.write(f"overlay: {OVERLAY}\n")
            f.write(f"opacity: {OPACITY}\n")
            f.write(f"sequence_mean_add_m: {sequence_mean_add_m:.10f}\n")
            f.write(f"sequence_mean_add_cm: {sequence_mean_add_m * 100.0:.10f}\n")
            f.write(f"sequence_mean_max_error_m: {sequence_mean_max_error_m:.10f}\n")
            f.write(f"sequence_mean_max_error_cm: {sequence_mean_max_error_m * 100.0:.10f}\n")
            f.write(f"sequence_global_min_error_m: {sequence_global_min_error_m:.10f}\n")
            f.write(f"sequence_global_min_error_cm: {sequence_global_min_error_m * 100.0:.10f}\n")
            f.write(f"sequence_global_max_error_m: {sequence_global_max_error_m:.10f}\n")
            f.write(f"sequence_global_max_error_cm: {sequence_global_max_error_m * 100.0:.10f}\n")
            f.write(f"per_frame_csv: {summary_csv_path}\n")

        print(f"\n[INFO] Rendered frames: {num_rendered}")
        print(
            f"[INFO] Sequence mean ADD: "
            f"{sequence_mean_add_m:.8f} m = {sequence_mean_add_m * 100.0:.4f} cm"
        )
        print(
            f"[INFO] Sequence global min error: "
            f"{sequence_global_min_error_m:.8f} m = {sequence_global_min_error_m * 100.0:.4f} cm"
        )
        print(
            f"[INFO] Sequence global max error: "
            f"{sequence_global_max_error_m:.8f} m = {sequence_global_max_error_m * 100.0:.4f} cm"
        )
        print(f"[INFO] Summary saved: {summary_path}")
        print(f"[INFO] Per-frame CSV saved: {summary_csv_path}")
    else:
        print("[WARN] No frames were rendered.")


if __name__ == "__main__":
    main()