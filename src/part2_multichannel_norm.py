"""
CS 898BA - Homework 2 - Part 2
Multi-channel color normalization in LAB space.

Extends the single-channel (V-only) equalization approach from Homework 1's
part2_processing.py to all three channels of the LAB color space, so that
both lightness and the two color-opponent channels get independently
contrast-normalized before segmentation.

Usage:
    python src/part2_multichannel_norm.py --image data/input/your_image.jpg
"""

import argparse
import os

import cv2
import numpy as np

OUT_ROOT = "outputs/stage2_segmentation"


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def manual_histogram_equalization(channel):
    """
    Same manual implementation used in Homework 1: build the histogram,
    derive the cumulative distribution, and remap pixel values by hand
    rather than calling cv2.equalizeHist directly.
    """
    histogram, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    return cdf_normalized[channel]


def normalize_lab_channels(source_bgr):
    """
    Splits the source image into L, A, and B channels, equalizes each one
    independently using the manual equalization routine, then merges them
    back into a single normalized color image.

    LAB was chosen over a straight R/G/B split because L isolates lightness
    from color information, which matters for a doorbell-camera image shot
    under mixed/uneven lighting - equalizing L alone fixes brightness
    contrast, while independently equalizing A and B additionally pulls out
    faint color-channel detail that a lightness-only pass (like Homework 1's
    V-channel approach) would leave untouched.
    """
    lab_img = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab_img)

    l_equalized = manual_histogram_equalization(l_chan)
    a_equalized = manual_histogram_equalization(a_chan)
    b_equalized = manual_histogram_equalization(b_chan)

    lab_equalized = cv2.merge([l_equalized, a_equalized, b_equalized])
    bgr_normalized = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    return {
        "l_channel_raw": l_chan,
        "a_channel_raw": a_chan,
        "b_channel_raw": b_chan,
        "l_channel_equalized": l_equalized,
        "a_channel_equalized": a_equalized,
        "b_channel_equalized": b_equalized,
        "lab_normalized": lab_equalized,
        "normalized_color": bgr_normalized,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    source = cv2.imread(args.image)
    if source is None:
        raise FileNotFoundError(f"Unable to read image: {args.image}")

    make_dirs(OUT_ROOT)

    results = normalize_lab_channels(source)
    for label, img in results.items():
        cv2.imwrite(os.path.join(OUT_ROOT, f"{label}.png"), img)
    print(f"{len(results)} images written to {OUT_ROOT} (checkpoint: expect 8).")

    primary_output = os.path.join(OUT_ROOT, "normalized_color.png")
    print(f"Primary normalized output for downstream segmentation: {primary_output}")


if __name__ == "__main__":
    main()