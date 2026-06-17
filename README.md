# CS 898BA - Homework One

**Author:** (your name)

## Overview

This project applies basic image analysis and processing techniques to a
single input image using Python and OpenCV: channel statistics, color space
conversions, histogram equalization, affine transformations, Gaussian
blurring, and edge detection (Sobel, Laplacian, Canny, Prewitt).

## Setup

1. Install Python 3.10+.
2. Clone this repository and `cd` into it.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Place the input image in `data/input/` (this repo does not include the
   original image; download it from the link provided in the assignment).

## How to run

Run Part 2 first (it generates the 168 images Part 3 depends on):

```
python src/part2_processing.py --image data/input/YOUR_IMAGE_NAME.jpg
```

This creates `data/part2_outputs/`, containing:
- `original_stats.csv` - per-channel statistics of the original image
- 7 base images (original, grayscale, binary, HSV, LAB, HLS, equalized_rgb)
- `affine/` - 14 affine-transformed versions of those 7 images
- `blurred/` - 147 Gaussian-blurred versions (7 sigma levels x 21 images)

Then run Part 3:

```
python src/part3_edges.py --subset 0
```

`--subset` can be 0, 1, 2, or 3 (the 4 random, equally-sized 42-image subsets
of the 168 images from Part 2). This creates `data/part3_outputs/`, containing:
- `edges/` - the "before" image and 4 edge-detected versions for each of the
  42 images in the chosen subset (210 images total)
- `plots/` - 6 randomly chosen 5-panel comparison plots (input + the 4 edge
  outputs side by side)

## Code explanation

`src/part2_processing.py`
- `compute_channel_stats()` calculates min, max, mean, median, mode, skew,
  range, standard deviation, and variance for each color channel.
- `make_base_images()` produces the 7 required images, including histogram
  equalizing the V channel of the HSV image and converting it back to RGB.
- `apply_affine_transforms()` applies 14 unique rotation/translation/scale/
  shear transforms (2 per base image, no two identical).
- `apply_gaussian_blurs()` blurs every one of the 21 images at 7 sigma levels.

`src/part3_edges.py`
- Gathers all 168 images from Part 2, shuffles them with a fixed random seed
  (for reproducibility), and splits them into 4 subsets of 42.
- Runs Sobel, Laplacian, Canny, and Prewitt edge detection (Prewitt is
  implemented manually with `cv2.filter2D` since OpenCV has no built-in
  version) on the chosen subset.
- Builds 5-panel comparison plots and saves 6 random ones for this README.

## Results

### Original image statistics

(Paste the contents of `data/part2_outputs/original_stats.csv` here, or a
formatted table of it, once you've run the script on your real image.)

### Effect of Gaussian blur sigma

(Discuss here how the image changes as sigma increases from 0.5 to 3.5 -
e.g. at what point does the figure in the image become hard to make out,
does higher sigma help or hurt the "is it an alien" investigation, etc.)

### Edge detection comparison

(Insert the 6 comparison plots from `data/part3_outputs/plots/` here, e.g.:)

```
![comparison](data/part3_outputs/plots/example_comparison.png)
```

Discuss the pros and cons of each technique (Sobel, Laplacian, Canny,
Prewitt) and which one performed best for this specific image set, with
reasoning tied to your actual results.

## Discussion

(Any additional observations, limitations, or notes on the process.)
