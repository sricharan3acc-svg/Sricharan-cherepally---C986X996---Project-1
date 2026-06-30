"""
CS 898BA - Homework 2 - Part 3
Threshold-based segmentation: Otsu's global thresholding and adaptive
(Gaussian-window) thresholding, applied to the LAB-normalized color image
produced in Part 2.

Usage:
    python src/part3_threshold_segmentation.py --image outputs/stage2_segmentation/normalized_color.png
"""

import argparse
import os

import cv2

OUT_ROOT = "outputs/stage3_segmentation"


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def otsu_segmentation(gray_img):
    """
    Otsu's method picks a single global threshold by minimizing intra-class
    intensity variance between the foreground and background pixel groups.
    cv2.threshold handles the variance search internally when THRESH_OTSU
    is passed alongside a normal binary threshold flag.
    """
    chosen_value, mask = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"Otsu's automatically selected threshold value: {chosen_value}")
    return mask


def adaptive_segmentation(gray_img):
    """
    Adaptive thresholding recomputes a local threshold for each pixel
    neighborhood (here, a Gaussian-weighted 25x25 window with an offset of
    5) rather than one fixed cutoff for the whole image - this is the same
    block size/offset pairing used for the bilevel image back in Homework
    One's part2_processing.py, kept consistent here for comparability.
    """
    mask = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 5
    )
    return mask


def extract_foreground(color_img, binary_mask):
    """Applies a binary mask to the color image to pull out the foreground."""
    return cv2.bitwise_and(color_img, color_img, mask=binary_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    normalized_color = cv2.imread(args.image)
    if normalized_color is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    make_dirs(OUT_ROOT)

    gray = cv2.cvtColor(normalized_color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(OUT_ROOT, "grayscale_input.png"), gray)

    otsu_mask = otsu_segmentation(gray)
    otsu_foreground = extract_foreground(normalized_color, otsu_mask)
    cv2.imwrite(os.path.join(OUT_ROOT, "otsu_mask.png"), otsu_mask)
    cv2.imwrite(os.path.join(OUT_ROOT, "otsu_foreground.png"), otsu_foreground)

    adaptive_mask = adaptive_segmentation(gray)
    adaptive_foreground = extract_foreground(normalized_color, adaptive_mask)
    cv2.imwrite(os.path.join(OUT_ROOT, "adaptive_mask.png"), adaptive_mask)
    cv2.imwrite(os.path.join(OUT_ROOT, "adaptive_foreground.png"), adaptive_foreground)

    print(f"Threshold segmentation outputs written to {OUT_ROOT} (checkpoint: expect 5 files).")


if __name__ == "__main__":
    main()