"""
CS 898BA - Homework 1 - Part 3
Random subset selection and edge detection (Sobel, Laplacian, Canny, Prewitt).

Run this AFTER part2_processing.py has finished, from the project root:
    python src/part3_edges.py --subset 0
(subset can be 0, 1, 2, or 3 - which of the 4 random subsets to use)
"""

import argparse
import os
import glob
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

PART2_DIR = "data/part2_outputs"
OUTPUT_DIR = "data/part3_outputs"
PLOTS_DIR = "data/part3_outputs/plots"
SUBSET_SIZE = 42
NUM_SUBSETS = 4
RANDOM_SEED = 42  # fixed seed so results are reproducible run to run


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1-2: gather all Part 2 images and split into 4 random subsets of 42
# ---------------------------------------------------------------------------

def gather_all_part2_images():
    """Collects every image produced in Part 2 (should be 168 total)."""
    patterns = [
        os.path.join(PART2_DIR, "*.png"),
        os.path.join(PART2_DIR, "affine", "*.png"),
        os.path.join(PART2_DIR, "blurred", "*.png"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def split_into_subsets(file_list):
    assert len(file_list) == SUBSET_SIZE * NUM_SUBSETS, (
        f"Expected {SUBSET_SIZE * NUM_SUBSETS} images, found {len(file_list)}. "
        "Make sure part2_processing.py ran successfully first."
    )
    rng = random.Random(RANDOM_SEED)
    shuffled = file_list.copy()
    rng.shuffle(shuffled)

    subsets = [
        shuffled[i * SUBSET_SIZE:(i + 1) * SUBSET_SIZE]
        for i in range(NUM_SUBSETS)
    ]
    return subsets


# ---------------------------------------------------------------------------
# Step 4: edge detection techniques
# ---------------------------------------------------------------------------

def apply_sobel(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sx, sy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_canny(gray):
    return cv2.Canny(gray, 100, 200)


def apply_prewitt(gray):
    # OpenCV has no built-in Prewitt, so the kernels are defined manually
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    gx = cv2.filter2D(gray.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(gray.astype(np.float32), -1, kernel_y)
    magnitude = cv2.magnitude(gx, gy)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


EDGE_METHODS = {
    "sobel": apply_sobel,
    "laplacian": apply_laplacian,
    "canny": apply_canny,
    "prewitt": apply_prewitt,
}


def run_edge_detection(image_path, output_dir):
    """
    Loads one image, applies all 4 edge detectors, and saves the original
    plus each edge-detected version. Returns a dict {method_name: edge_image}
    plus the grayscale 'before' image, for use in the comparison plots.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ensure_dir(output_dir)

    # save the "before" image
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_before.png"), gray)

    edge_results = {"before": gray}
    for method_name, method_fn in EDGE_METHODS.items():
        edge_img = method_fn(gray)
        edge_results[method_name] = edge_img
        out_path = os.path.join(output_dir, f"{base_name}_{method_name}.png")
        cv2.imwrite(out_path, edge_img)

    return base_name, edge_results


# ---------------------------------------------------------------------------
# Step 8: 5-image comparison plots, 6 random ones saved for the README
# ---------------------------------------------------------------------------

def make_comparison_plot(base_name, edge_results, save_path):
    methods_order = ["before", "sobel", "laplacian", "canny", "prewitt"]
    titles = ["Input", "Sobel", "Laplacian", "Canny", "Prewitt"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, method, title in zip(axes, methods_order, titles):
        ax.imshow(edge_results[method], cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(base_name, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=0, choices=[0, 1, 2, 3],
                         help="Which of the 4 random subsets (0-3) to use")
    args = parser.parse_args()

    all_images = gather_all_part2_images()
    print(f"Found {len(all_images)} images from Part 2.")

    subsets = split_into_subsets(all_images)
    for i, s in enumerate(subsets):
        print(f"Subset {i}: {len(s)} images")

    chosen_subset = subsets[args.subset]
    print(f"\nUsing subset {args.subset} ({len(chosen_subset)} images) for Part 3.\n")

    ensure_dir(OUTPUT_DIR)
    ensure_dir(PLOTS_DIR)

    edge_dir = os.path.join(OUTPUT_DIR, "edges")
    all_plot_candidates = []

    for image_path in chosen_subset:
        base_name, edge_results = run_edge_detection(image_path, edge_dir)
        all_plot_candidates.append((base_name, edge_results))

    print(f"Edge detection complete. Images saved to {edge_dir}")
    expected_total = len(chosen_subset) + len(chosen_subset) * 4
    print(f"Image count check: {expected_total} (expected 210)")

    # randomly choose 6 of the 42 to actually save as plots for the README
    rng = random.Random(RANDOM_SEED)
    chosen_for_plots = rng.sample(all_plot_candidates, 6)

    for base_name, edge_results in chosen_for_plots:
        save_path = os.path.join(PLOTS_DIR, f"{base_name}_comparison.png")
        make_comparison_plot(base_name, edge_results, save_path)
        print(f"Saved comparison plot: {save_path}")

    print(f"\nDone. {len(chosen_for_plots)} comparison plots saved to {PLOTS_DIR}")
    print("Copy these into your README along with a description of the processing chain.")


if __name__ == "__main__":
    main()
