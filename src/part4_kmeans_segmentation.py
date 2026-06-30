"""
CS 898BA - Homework 2 - Part 4
Color-space clustering segmentation using K-Means in HSV space, applied to
the LAB-normalized color image produced in Part 2.

Rather than picking K by eye, this script tests K = 3, 4, and 5 and selects
the best one using the silhouette score - a measure of how well-separated
the resulting clusters are - so the choice of K is backed by a number
instead of a visual guess.

Usage:
    python src/part4_kmeans_segmentation.py --image outputs/stage2_segmentation/normalized_color.png
"""

import argparse
import os

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT_ROOT = "outputs/stage4_segmentation"
CANDIDATE_K_VALUES = [3, 4, 5]
SILHOUETTE_SAMPLE_SIZE = 2000  # subsample for speed - full pixel grid is too slow for silhouette


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def score_candidate_k(pixel_samples, k, seed):
    """
    Fits K-Means for a single K value and returns both the fitted model and
    its silhouette score, computed on a random subsample of pixels since
    silhouette scoring is too slow to run on every pixel in the image.
    """
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = model.fit_predict(pixel_samples)

    rng = np.random.default_rng(seed)
    sample_count = min(SILHOUETTE_SAMPLE_SIZE, len(pixel_samples))
    sample_idx = rng.choice(len(pixel_samples), size=sample_count, replace=False)

    score = silhouette_score(pixel_samples[sample_idx], labels[sample_idx])
    return model, labels, score


def select_best_k(hsv_pixels, seed=42):
    """
    Runs K-Means for each candidate K and returns the model/labels/K that
    achieved the highest silhouette score.
    """
    best_score = -1.0
    best_k = None
    best_model = None
    best_labels = None

    for k in CANDIDATE_K_VALUES:
        model, labels, score = score_candidate_k(hsv_pixels, k, seed)
        print(f"K={k} -> silhouette score: {score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels

    print(f"Selected K={best_k} (silhouette score: {best_score:.4f})")
    return best_model, best_labels, best_k


def identify_figure_cluster(labels, image_shape):
    """
    Heuristic for picking which cluster corresponds to the foreground
    figure: assumes the figure occupies a contiguous region away from the
    image border, so the cluster with the smallest mean distance-to-center
    among pixels that are NOT the single largest (background) cluster is
    chosen as the figure.
    """
    rows, cols = image_shape[:2]
    label_grid = labels.reshape(rows, cols)

    cluster_ids, counts = np.unique(label_grid, return_counts=True)
    largest_cluster = cluster_ids[np.argmax(counts)]

    center_y, center_x = rows / 2, cols / 2
    yy, xx = np.mgrid[0:rows, 0:cols]
    dist_from_center = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    best_cluster = None
    best_mean_dist = np.inf
    for cluster_id in cluster_ids:
        if cluster_id == largest_cluster:
            continue
        mask = label_grid == cluster_id
        mean_dist = dist_from_center[mask].mean()
        if mean_dist < best_mean_dist:
            best_mean_dist = mean_dist
            best_cluster = cluster_id

    return label_grid, best_cluster


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    normalized_color = cv2.imread(args.image)
    if normalized_color is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    make_dirs(OUT_ROOT)

    hsv_img = cv2.cvtColor(normalized_color, cv2.COLOR_BGR2HSV)
    rows, cols = hsv_img.shape[:2]
    pixel_samples = hsv_img.reshape(-1, 3).astype(np.float32)

    model, labels, best_k = select_best_k(pixel_samples)

    # Visualize all clusters with each cluster's mean color, for inspection
    cluster_visual = model.cluster_centers_[labels].reshape(rows, cols, 3).astype(np.uint8)
    cluster_visual_bgr = cv2.cvtColor(cluster_visual, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(OUT_ROOT, f"kmeans_clusters_k{best_k}.png"), cluster_visual_bgr)

    label_grid, figure_cluster = identify_figure_cluster(labels, (rows, cols))
    figure_mask = np.where(label_grid == figure_cluster, 255, 0).astype(np.uint8)

    figure_foreground = cv2.bitwise_and(normalized_color, normalized_color, mask=figure_mask)

    cv2.imwrite(os.path.join(OUT_ROOT, "kmeans_mask.png"), figure_mask)
    cv2.imwrite(os.path.join(OUT_ROOT, "kmeans_foreground.png"), figure_foreground)

    print(f"K-Means segmentation outputs written to {OUT_ROOT} (checkpoint: expect 3 files).")


if __name__ == "__main__":
    main()