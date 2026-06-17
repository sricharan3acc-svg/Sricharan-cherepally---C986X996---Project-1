# CS 898BA - Homework One

**Author:** (your name)

## What this project does

Given one input image, this codebase runs a sequence of image analysis
operations using Python and OpenCV: per-channel pixel statistics, color
space conversions (HSV / LAB / HLS), contrast normalization via histogram
equalization, geometric warps (rotation / translation / scaling / shear),
Gaussian smoothing across several sigma levels, and four boundary/edge
detection operators (Sobel, Laplacian, an auto-thresholded Canny, and a
manually implemented Prewitt operator).

## Getting set up

```
pip install -r requirements.txt
```

Place your input image inside `data/input/`. The original assignment image
is not checked into this repository - it must be downloaded separately from
the link provided in the assignment instructions.

## Running the pipeline

Stage 2 (must run first):
```
python src/part2_processing.py --image data/input/YOUR_IMAGE.jpg
```
Produces `outputs/stage2/`:
- `source_statistics.json` - per-channel statistics for the original image
- 7 base representations (source, mono, bilevel, hsv_space, lab_space, hls_space, normalized_rgb)
- `warped/` - 14 geometrically transformed images
- `smoothed/` - 147 Gaussian-smoothed images (7 sigma levels x 21 images)

Stage 3:
```
python src/part3_edges.py --group 0
```
`--group` selects which of the 4 random 42-image partitions to analyze (0-3).
Produces `outputs/stage3/`:
- `boundaries/` - the grayscale input plus 4 boundary-detected versions for
  each of the 42 images in the chosen group (210 images total)
- `figures/` - 6 randomly selected comparison grids for inclusion below

## Implementation notes

`src/part2_processing.py`
- Statistics are computed per-channel (mode found via frequency counting
  rather than a statistics library call) and saved as JSON.
- The binary image uses adaptive (local) thresholding rather than a single
  global threshold, so it responds to local contrast rather than one
  whole-image cutoff.
- Histogram equalization on the V channel is implemented manually (building
  the histogram, computing the cumulative distribution, and remapping pixel
  values), rather than calling a built-in equalization function.
- Gaussian smoothing lets OpenCV derive the kernel size automatically from
  each sigma value.

`src/part3_edges.py`
- Partitioning uses a NumPy random permutation with a fixed seed, so the
  4 groups are reproducible from run to run.
- The Canny step automatically derives its lower/upper thresholds from the
  image's median intensity instead of using fixed constants.
- Prewitt is implemented from scratch with manual convolution kernels, since
  OpenCV doesn't provide one natively.

## Results

### Original image statistics

(Paste the contents of `outputs/stage2/source_statistics.json` here once
you've run this on your real image.)

### Effect of increasing the smoothing sigma

(Your own discussion of what changes visually as sigma goes from 0.5 up to
3.5, and what that means for trying to make out the figure in the photo.)

### Boundary detection comparison

(Insert the 6 figures from `outputs/stage3/figures/` here, e.g.:)

```
![figure](outputs/stage3/figures/example__grid.png)
```

Discuss the pros and cons of Sobel, Laplacian, Canny, and Prewitt for this
specific image set, and state which one you found most useful and why,
based on your actual results rather than general reputation.

## Additional discussion

(Anything else worth noting about your process, limitations, or surprises.)
