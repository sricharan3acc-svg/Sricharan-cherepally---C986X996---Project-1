"""
CS 898BA - Homework 2 - Part 5
Quantitative evaluation (IoU / Dice) of the three segmentation methods
against a manually-traced ground truth mask, plus a side-by-side comparison
figure for the README.

Usage:
    python src/part5_evaluation.py --original data/input/HW1_IMG_CS898BA.png ^
        --normalized outputs/stage2_segmentation/normalized_color.png ^
        --otsu_mask outputs/stage3_segmentation/otsu_mask.png ^
        --adaptive_mask outputs/stage3_segmentation/adaptive_mask.png ^
        --kmeans_mask outputs/stage4_segmentation/kmeans_mask.png ^
        --ground_truth outputs/stage4_segmentation/ground_truth_mask.png
"""

import argparse
import os

import cv2
import numpy as np

OUT_ROOT = "outputs/stage5_evaluation"


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_binary_mask(path, target_shape):
    """
    Loads a mask image as a strict 0/255 binary array, resizing it to match
    target_shape if its dimensions differ (e.g. if the ground truth was
    exported at a slightly different size than the pipeline outputs).
    """
    raw = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"Unable to read mask: {path}")

    if raw.shape != target_shape:
        raw = cv2.resize(raw, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
    return binary


def compute_iou(mask_a, mask_b):
    """Intersection over Union / Jaccard index between two binary masks."""
    a_bool = mask_a > 0
    b_bool = mask_b > 0
    intersection = np.logical_and(a_bool, b_bool).sum()
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 0.0
    return intersection / union


def compute_dice(mask_a, mask_b):
    """Dice (Sorensen-Dice) coefficient between two binary masks."""
    a_bool = mask_a > 0
    b_bool = mask_b > 0
    intersection = np.logical_and(a_bool, b_bool).sum()
    total = a_bool.sum() + b_bool.sum()
    if total == 0:
        return 0.0
    return (2.0 * intersection) / total


def label_panel(img, text):
    """Adds a small black label strip with white text under a panel image."""
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    strip = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(strip, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([img, strip])


def build_comparison_grid(panels_with_labels, out_path, panel_width=420):
    """Resizes all panels to a common width and tiles them in a single row."""
    resized = []
    for img, label in panels_with_labels:
        h, w = img.shape[:2]
        scale = panel_width / w
        resized_img = cv2.resize(img, (panel_width, int(h * scale)))
        resized.append(label_panel(resized_img, label))

    max_height = max(p.shape[0] for p in resized)
    padded = []
    for p in resized:
        if p.shape[0] < max_height:
            pad = np.zeros((max_height - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    grid = np.hstack(padded)
    cv2.imwrite(out_path, grid)
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--normalized", required=True)
    parser.add_argument("--otsu_mask", required=True)
    parser.add_argument("--adaptive_mask", required=True)
    parser.add_argument("--kmeans_mask", required=True)
    parser.add_argument("--ground_truth", required=True)
    args = parser.parse_args()

    make_dirs(OUT_ROOT)

    original = cv2.imread(args.original)
    normalized = cv2.imread(args.normalized)
    if original is None or normalized is None:
        raise FileNotFoundError("Unable to read original or normalized image.")

    target_shape = normalized.shape[:2]
    otsu_mask = load_binary_mask(args.otsu_mask, target_shape)
    adaptive_mask = load_binary_mask(args.adaptive_mask, target_shape)
    kmeans_mask = load_binary_mask(args.kmeans_mask, target_shape)
    ground_truth = load_binary_mask(args.ground_truth, target_shape)

    results = {}
    for name, mask in [
        ("Otsu", otsu_mask),
        ("Adaptive", adaptive_mask),
        ("K-Means", kmeans_mask),
    ]:
        iou = compute_iou(mask, ground_truth)
        dice = compute_dice(mask, ground_truth)
        results[name] = {"iou": iou, "dice": dice}
        print(f"{name:10s} | IoU: {iou:.4f} | Dice: {dice:.4f}")

    metrics_path = os.path.join(OUT_ROOT, "metrics_summary.txt")
    with open(metrics_path, "w") as f:
        f.write("Method     | IoU    | Dice\n")
        f.write("-----------|--------|-------\n")
        for name, vals in results.items():
            f.write(f"{name:10s} | {vals['iou']:.4f} | {vals['dice']:.4f}\n")
    print(f"Metrics written to {metrics_path}")

    panels = [
        (original, "Original"),
        (normalized, "LAB Normalized"),
        (otsu_mask, "Otsu"),
        (adaptive_mask, "Adaptive"),
        (kmeans_mask, "K-Means"),
        (ground_truth, "Ground Truth"),
    ]
    grid_path = os.path.join(OUT_ROOT, "comparison_grid.png")
    build_comparison_grid(panels, grid_path)
    print(f"Comparison grid written to {grid_path}")


if __name__ == "__main__":
    main()