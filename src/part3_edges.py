"""
CS 898BA - Homework 1 - Part 3
Subset partitioning and edge/boundary detection.

Usage:
    python src/part3_edges.py --group 0
(group is 0-3, selecting which of the 4 partitions to analyze)
"""

import argparse
import glob
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

STAGE2_DIR = "outputs/stage2"
OUT_ROOT = "outputs/stage3"
PLOT_DIR = "outputs/stage3/figures"
GROUP_SIZE = 42
GROUP_COUNT = 4
SEED = 137  # arbitrary fixed seed, different from any other student's choice


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def collect_stage2_outputs():
    search_paths = [
        os.path.join(STAGE2_DIR, "*.png"),
        os.path.join(STAGE2_DIR, "warped", "*.png"),
        os.path.join(STAGE2_DIR, "smoothed", "*.png"),
    ]
    found = []
    for pattern in search_paths:
        found.extend(glob.glob(pattern))
    return sorted(found)


def partition_into_groups(file_paths):
    expected = GROUP_SIZE * GROUP_COUNT
    if len(file_paths) != expected:
        raise RuntimeError(
            f"Expected {expected} images from stage 2, found {len(file_paths)}. "
            "Run part2_processing.py first."
        )
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(file_paths))
    shuffled = [file_paths[i] for i in order]
    return [shuffled[i:i + GROUP_SIZE] for i in range(0, expected, GROUP_SIZE)]


# --- boundary detection operators ------------------------------------------

def sobel_boundary(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    combined = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    return combined


def laplacian_boundary(gray):
    smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
    lap = cv2.Laplacian(smoothed, cv2.CV_32F)
    return cv2.convertScaleAbs(lap)


def auto_canny_boundary(gray, sigma_factor=0.33):
    """
    Canny with thresholds derived from the image's median intensity rather
    than a fixed pair of constants.
    """
    median_val = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma_factor) * median_val))
    upper = int(min(255, (1.0 + sigma_factor) * median_val))
    return cv2.Canny(gray, lower, upper)


def prewitt_boundary(gray):
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray.astype(np.float32), -1, kx)
    gy = cv2.filter2D(gray.astype(np.float32), -1, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


OPERATORS = {
    "sobel": sobel_boundary,
    "laplacian": laplacian_boundary,
    "canny": auto_canny_boundary,
    "prewitt": prewitt_boundary,
}


def process_one_image(path, out_dir):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    tag = os.path.splitext(os.path.basename(path))[0]

    make_dirs(out_dir)
    cv2.imwrite(os.path.join(out_dir, f"{tag}__input.png"), gray)

    results = {"input": gray}
    for op_name, op_fn in OPERATORS.items():
        edge_img = op_fn(gray)
        results[op_name] = edge_img
        cv2.imwrite(os.path.join(out_dir, f"{tag}__{op_name}.png"), edge_img)

    return tag, results


def render_grid(tag, results, save_path):
    order = ["input", "sobel", "laplacian", "canny", "prewitt"]
    headers = ["Original", "Sobel", "Laplacian", "Auto-Canny", "Prewitt"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    flat_axes = axes.flatten()

    for ax, key, header in zip(flat_axes, order, headers):
        ax.imshow(results[key], cmap="gray")
        ax.set_title(header, fontsize=10)
        ax.axis("off")

    flat_axes[-1].axis("off")  # unused 6th cell in the 2x3 grid
    fig.suptitle(tag, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()

    files = collect_stage2_outputs()
    print(f"Located {len(files)} images from stage 2.")

    groups = partition_into_groups(files)
    for i, g in enumerate(groups):
        print(f"Group {i}: {len(g)} images")

    active_group = groups[args.group]
    print(f"\nAnalyzing group {args.group} ({len(active_group)} images).\n")

    make_dirs(OUT_ROOT, PLOT_DIR)
    edge_dir = os.path.join(OUT_ROOT, "boundaries")
    processed = []

    for path in active_group:
        tag, results = process_one_image(path, edge_dir)
        processed.append((tag, results))

    total_written = len(active_group) + len(active_group) * 4
    print(f"Boundary detection complete. Count check: {total_written} (expect 210).")

    rng = random.Random(SEED)
    sample_for_plots = rng.sample(processed, 6)

    for tag, results in sample_for_plots:
        out_path = os.path.join(PLOT_DIR, f"{tag}__grid.png")
        render_grid(tag, results, out_path)
        print(f"Saved figure: {out_path}")

    print(f"\nFinished. {len(sample_for_plots)} figures saved to {PLOT_DIR}")


if __name__ == "__main__":
    main()
