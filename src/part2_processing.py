"""
CS 898BA - Homework 1 - Part 2
Basic image statistics, color space conversions, histogram equalization,
affine transformations, and Gaussian blurring.

Run this from the project root, e.g.:
    python src/part2_processing.py --image data/input/your_image.jpg
"""

import argparse
import os
import csv
import math
import numpy as np
import cv2
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUTPUT_DIR = "data/part2_outputs"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Basic per-channel statistics
# ---------------------------------------------------------------------------

def compute_channel_stats(image, channel_names):
    """
    Computes min, max, mean, median, mode, skew, range, std, variance
    for each channel of the image. Returns a list of dicts (one per channel).
    """
    results = []
    for i, name in enumerate(channel_names):
        channel = image[:, :, i].flatten()

        mode_result = scipy_stats.mode(channel, keepdims=True)
        mode_value = mode_result.mode[0]

        stats_dict = {
            "channel": name,
            "min": int(channel.min()),
            "max": int(channel.max()),
            "mean": float(channel.mean()),
            "median": float(np.median(channel)),
            "mode": float(mode_value),
            "skew": float(scipy_stats.skew(channel)),
            "range": int(channel.max() - channel.min()),
            "std_dev": float(channel.std()),
            "variance": float(channel.var()),
        }
        results.append(stats_dict)
    return results


def print_and_save_stats(stats_list, out_path):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats_list[0].keys()))
        writer.writeheader()
        for row in stats_list:
            writer.writerow(row)
            print(row)
    print(f"Saved stats to {out_path}")


# ---------------------------------------------------------------------------
# Step 2: Color space conversions
# ---------------------------------------------------------------------------

def make_base_images(original_bgr):
    """
    Returns a dict of {name: image} for the 7 images required by step 5:
    original, grayscale, binary, HSV, LAB, HLS, and the histogram-equalized
    (V-channel) image converted back to RGB/BGR.
    """
    images = {}

    images["original"] = original_bgr

    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    images["grayscale"] = gray

    # Otsu's method picks a good threshold automatically for the binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    images["binary"] = binary

    hsv = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    images["hsv"] = hsv

    lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    images["lab"] = lab

    hls = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HLS)
    images["hls"] = hls

    # Step 3: histogram-equalize the V (value) channel of the HSV image
    h, s, v = cv2.split(hsv)
    v_equalized = cv2.equalizeHist(v)
    hsv_equalized = cv2.merge([h, s, v_equalized])

    # Step 4: convert the normalized image back to RGB (BGR for OpenCV saving)
    equalized_bgr = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)
    images["equalized_rgb"] = equalized_bgr

    return images


# ---------------------------------------------------------------------------
# Step 6: Affine transformations (14 total, 2 unique per image)
# ---------------------------------------------------------------------------

def get_affine_matrix(transform_type, value, image_shape):
    """
    Builds a 2x3 affine transform matrix based on transform_type:
    'rotate', 'translate', 'scale', or 'shear'.
    """
    h, w = image_shape[:2]
    center = (w / 2, h / 2)

    if transform_type == "rotate":
        return cv2.getRotationMatrix2D(center, value, 1.0)

    if transform_type == "scale":
        return cv2.getRotationMatrix2D(center, 0, value)

    if transform_type == "translate":
        tx, ty = value
        return np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    if transform_type == "shear":
        shx, shy = value
        # shear about the image center
        M = np.array([[1, shx, -shx * center[1]],
                       [shy, 1, -shy * center[0]]], dtype=np.float32)
        return M

    raise ValueError(f"Unknown transform type: {transform_type}")


# 14 unique transform specs (type, value, short label for filenames)
AFFINE_SPECS = [
    ("rotate", 30, "rot30"),
    ("rotate", 95, "rot95"),
    ("rotate", 186, "rot186"),
    ("rotate", 275, "rot275"),
    ("translate", (40, 20), "trans_40_20"),
    ("translate", (-35, 55), "trans_n35_55"),
    ("translate", (60, -45), "trans_60_n45"),
    ("scale", 0.6, "scale0_6"),
    ("scale", 1.4, "scale1_4"),
    ("scale", 0.8, "scale0_8"),
    ("shear", (0.3, 0.0), "shearx0_3"),
    ("shear", (0.0, 0.25), "sheary0_25"),
    ("shear", (0.2, 0.15), "shearxy"),
    ("rotate", 150, "rot150"),
]


def apply_affine_transforms(images_dict, output_dir):
    """
    Applies exactly 2 unique affine transforms to each of the 7 base images
    (14 total transforms across the set, no two identical).
    Returns a dict of {name: image} for the 14 new transformed images.
    """
    ensure_dir(output_dir)
    transformed = {}
    base_names = list(images_dict.keys())
    spec_idx = 0

    for base_name in base_names:
        img = images_dict[base_name]
        for _ in range(2):
            t_type, value, label = AFFINE_SPECS[spec_idx]
            spec_idx += 1
            M = get_affine_matrix(t_type, value, img.shape)
            h, w = img.shape[:2]
            warped = cv2.warpAffine(img, M, (w, h))

            new_name = f"{base_name}_{label}"
            transformed[new_name] = warped

            out_path = os.path.join(output_dir, f"{new_name}.png")
            cv2.imwrite(out_path, warped)

    print(f"Saved {len(transformed)} affine-transformed images to {output_dir}")
    return transformed


# ---------------------------------------------------------------------------
# Step 8: Gaussian blur at 7 sigma levels, applied to all 21 images
# ---------------------------------------------------------------------------

SIGMA_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def sigma_to_ksize(sigma):
    """OpenCV needs an odd kernel size; this picks one proportional to sigma."""
    k = int(2 * math.ceil(3 * sigma) + 1)
    return k


def apply_gaussian_blurs(images_dict, output_dir):
    """
    Applies all 7 sigma levels to every image in images_dict.
    Returns a dict of {name: image} for the new blurred images.
    """
    ensure_dir(output_dir)
    blurred = {}

    for name, img in images_dict.items():
        for sigma in SIGMA_LEVELS:
            ksize = sigma_to_ksize(sigma)
            blurred_img = cv2.GaussianBlur(img, (ksize, ksize), sigma)

            sigma_label = str(sigma).replace(".", "_")
            new_name = f"{name}_blur_s{sigma_label}"
            blurred[new_name] = blurred_img

            out_path = os.path.join(output_dir, f"{new_name}.png")
            cv2.imwrite(out_path, blurred_img)

    print(f"Saved {len(blurred)} blurred images to {output_dir}")
    return blurred


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    original = cv2.imread(args.image)
    if original is None:
        raise FileNotFoundError(f"Could not read image at {args.image}")

    ensure_dir(OUTPUT_DIR)

    # --- Step 1: stats on the ORIGINAL image (BGR channels, as loaded) ---
    stats = compute_channel_stats(original, ["Blue", "Green", "Red"])
    print_and_save_stats(stats, os.path.join(OUTPUT_DIR, "original_stats.csv"))

    # --- Steps 2-5: build the 7 base images ---
    base_images = make_base_images(original)
    for name, img in base_images.items():
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}.png"), img)
    print(f"Saved {len(base_images)} base images (step 5 checkpoint).")

    # --- Steps 6-7: 14 affine transforms -> 21 images total ---
    affine_dir = os.path.join(OUTPUT_DIR, "affine")
    affine_images = apply_affine_transforms(base_images, affine_dir)

    all_21 = {**base_images, **affine_images}
    print(f"Total image count after Part 2 step 7: {len(all_21)} (expected 21)")

    # --- Steps 8-9: Gaussian blur all 21 -> 168 images total ---
    blur_dir = os.path.join(OUTPUT_DIR, "blurred")
    blurred_images = apply_gaussian_blurs(all_21, blur_dir)

    grand_total = len(all_21) + len(blurred_images)
    print(f"Total image count after Part 2 step 9: {grand_total} (expected 168)")


if __name__ == "__main__":
    main()
