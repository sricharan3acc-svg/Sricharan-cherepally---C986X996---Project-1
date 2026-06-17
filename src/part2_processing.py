"""
CS 898BA - Homework 1 - Part 2
Pixel statistics, color space conversions, contrast normalization,
geometric (affine) warps, and Gaussian smoothing.

Usage:
    python src/part2_processing.py --image data/input/your_image.jpg
"""

import argparse
import json
import math
import os
from collections import Counter

import cv2
import numpy as np
from scipy import stats as sci_stats

OUT_ROOT = "outputs/stage2"


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


class ChannelStats:
    """Computes descriptive statistics for a single image channel."""

    def __init__(self, channel_label, pixel_values):
        self.label = channel_label
        self.values = pixel_values.flatten()

    def summarize(self):
        vals = self.values
        # mode computed via Counter rather than scipy.stats.mode
        most_common_value, _ = Counter(vals.tolist()).most_common(1)[0]

        return {
            "channel": self.label,
            "minimum": int(vals.min()),
            "maximum": int(vals.max()),
            "average": float(np.mean(vals)),
            "median_value": float(np.median(vals)),
            "mode_value": float(most_common_value),
            "skewness": float(sci_stats.skew(vals)),
            "value_range": int(vals.max() - vals.min()),
            "std_deviation": float(np.std(vals)),
            "variance": float(np.var(vals)),
        }


def report_statistics(bgr_image, save_path):
    labels = ["Blue", "Green", "Red"]
    summary = []
    for idx, label in enumerate(labels):
        stat_block = ChannelStats(label, bgr_image[:, :, idx]).summarize()
        summary.append(stat_block)
        print(f"[{label}] " + ", ".join(f"{k}={v}" for k, v in stat_block.items() if k != "channel"))

    make_dirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Statistics written to {save_path}")
    return summary


def manual_histogram_equalization(channel):
    """
    Implements histogram equalization by hand (build the histogram, derive
    the cumulative distribution, and remap pixel values) rather than calling
    cv2.equalizeHist directly.
    """
    histogram, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    return cdf_normalized[channel]


def build_color_variants(source_bgr):
    """Produces the seven required representations of the source image."""
    variants = {"source": source_bgr}

    mono = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    variants["mono"] = mono

    # adaptive thresholding instead of a single global (Otsu) threshold
    bilevel = cv2.adaptiveThreshold(
        mono, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )
    variants["bilevel"] = bilevel

    hsv_img = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
    variants["hsv_space"] = hsv_img

    lab_img = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    variants["lab_space"] = lab_img

    hls_img = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HLS)
    variants["hls_space"] = hls_img

    hue, sat, val = cv2.split(hsv_img)
    val_normalized = manual_histogram_equalization(val)
    hsv_normalized = cv2.merge([hue, sat, val_normalized])
    rgb_normalized = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)
    variants["normalized_rgb"] = rgb_normalized

    return variants


# A different set of 14 warp definitions (type, params, tag)
WARP_PLAN = [
    ("rotate", 18, "r18"),
    ("rotate", 72, "r72"),
    ("rotate", 161, "r161"),
    ("rotate", 290, "r290"),
    ("shift", (-25, 35), "shA"),
    ("shift", (50, -10), "shB"),
    ("shift", (15, 70), "shC"),
    ("resize", 0.55, "zoomDown1"),
    ("resize", 1.25, "zoomUp1"),
    ("resize", 1.7, "zoomUp2"),
    ("skew", (0.22, 0.0), "skewA"),
    ("skew", (0.0, 0.35), "skewB"),
    ("skew", (-0.18, 0.12), "skewC"),
    ("rotate", 230, "r230"),
]


def build_warp_matrix(kind, param, frame_shape):
    rows, cols = frame_shape[:2]
    midpoint = (cols / 2, rows / 2)

    if kind == "rotate":
        return cv2.getRotationMatrix2D(midpoint, param, 1.0)
    if kind == "resize":
        return cv2.getRotationMatrix2D(midpoint, 0, param)
    if kind == "shift":
        dx, dy = param
        return np.float32([[1, 0, dx], [0, 1, dy]])
    if kind == "skew":
        kx, ky = param
        return np.float32([
            [1, kx, -kx * midpoint[1]],
            [ky, 1, -ky * midpoint[0]],
        ])
    raise ValueError(f"unsupported warp kind: {kind}")


def apply_warps(variant_dict, out_dir):
    make_dirs(out_dir)
    warped = {}
    plan_pointer = 0

    for base_label, base_img in variant_dict.items():
        for _ in range(2):
            kind, param, tag = WARP_PLAN[plan_pointer]
            plan_pointer += 1
            matrix = build_warp_matrix(kind, param, base_img.shape)
            h, w = base_img.shape[:2]
            result = cv2.warpAffine(base_img, matrix, (w, h))

            key = f"{base_label}__{tag}"
            warped[key] = result
            cv2.imwrite(os.path.join(out_dir, f"{key}.png"), result)

    print(f"{len(warped)} geometrically warped images written to {out_dir}")
    return warped


SMOOTHING_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def apply_smoothing(image_dict, out_dir):
    """
    Applies Gaussian smoothing at each sigma level. Kernel size is left at
    (0, 0) so OpenCV derives an appropriate kernel from sigma automatically,
    rather than computing it manually.
    """
    make_dirs(out_dir)
    smoothed = {}

    for label, img in image_dict.items():
        for sigma in SMOOTHING_LEVELS:
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            tag = f"{label}__sigma{str(sigma).replace('.', 'p')}"
            smoothed[tag] = blurred
            cv2.imwrite(os.path.join(out_dir, f"{tag}.png"), blurred)

    print(f"{len(smoothed)} smoothed images written to {out_dir}")
    return smoothed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image)
    if source is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    make_dirs(OUT_ROOT)

    report_statistics(source, os.path.join(OUT_ROOT, "source_statistics.json"))

    variants = build_color_variants(source)
    for label, img in variants.items():
        cv2.imwrite(os.path.join(OUT_ROOT, f"{label}.png"), img)
    print(f"{len(variants)} base representations saved (checkpoint: expect 7).")

    warp_dir = os.path.join(OUT_ROOT, "warped")
    warped_images = apply_warps(variants, warp_dir)

    combined_21 = {**variants, **warped_images}
    print(f"Running total after warps: {len(combined_21)} (expect 21).")

    smooth_dir = os.path.join(OUT_ROOT, "smoothed")
    smoothed_images = apply_smoothing(combined_21, smooth_dir)

    grand_total = len(combined_21) + len(smoothed_images)
    print(f"Running total after smoothing: {grand_total} (expect 168).")


if __name__ == "__main__":
    main()
