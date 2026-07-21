# CS 898BA - Homework One

Sricharan Cherepally – C986X996

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
| Channel | Min | Max | Average | Median | Mode | Skewness | Range | Std Dev | Variance | 
|---|---|---|---|---|---|---|---|---|---| 
| Blue | 0 | 255 | 21.83 | 10 | 4 | 1.68 | 255 | 26.23 | 687.99 | 
| Green | 0 | 255 | 24.64 | 16 | 10 | 1.76 | 255 | 22.23 | 493.96 | 
| Red | 0 | 255 | 20.61 | 12 | 4 | 2.11 | 255 | 22.46 | 504.26 | 

All three channels span the full 0-255 range but have averages and medians well below the midpoint (around 20-25 out of 255), with strongly positive skewness (1.68-2.11). This is consistent with a dark, low-light photo where most pixels are concentrated in the shadows, with a smaller number of much brighter pixels (highlights, lights, reflections) pulling the distribution's tail to the right. The Red channel is the most skewed and has the lowest average, while Green has the highest average of the three, suggesting the image's brighter regions lean slightly toward green/neutral tones rather than red. 

### Effect of increasing the smoothing sigma 

Looking across the sampled images at different sigma levels, the lower values (1.5-2.0) preserve enough structure that the person and background houses/cars remain clearly identifiable, with edges still sharp enough for Sobel, Prewitt, and even Canny to pick up real detail (see the sigma1p5 and sigma2p0 examples below, which look nearly identical to each other - at this range the extra half-step of blur barely changes what's detectable). By sigma 2.5, fine texture starts to disappear and edge detectors begin returning noticeably sparser results, particularly Canny. At sigma 3.5, the blurring is heavy enough that even Sobel and Prewitt only recover a rough silhouette rather than real detail, and Canny's output becomes almost entirely empty in several cases. Overall, sigma in this dataset acts as a fairly sharp cutoff for Canny's usefulness specifically - it degrades faster than the other three methods as blur increases. 


### Edge detection comparison 

![figure](outputs/stage3/figures/normalized_rgb__skewC__sigma1p5__grid.png)

![figure](outputs/stage3/figures/normalized_rgb__skewC__sigma2p0__grid.png)

![figure](outputs/stage3/figures/hls_space__skewB__sigma2p5__grid.png)

![figure](outputs/stage3/figures/bilevel__shA__sigma2p5__grid.png)

![figure](outputs/stage3/figures/hsv_space__zoomDown1__sigma3p5__grid.png)

![figure](outputs/stage3/figures/normalized_rgb__r230__sigma3p5__grid.png)

Across these six examples, Sobel and Prewitt were consistently the most reliable. They produced clear, recognizable outlines of the person and background houses in every single case, including the heavily blurred and geometrically distorted versions (rotated, sheared, scaled down), and the two operators look almost identical to each other throughout, which makes sense given how similar their underlying gradient kernels are. 

Laplacian performed noticeably worse on the more heavily preprocessed images. On the scaled-down (zoomDown1, sigma 3.5) and heavily rotated (r230, sigma 3.5) examples, its output is mostly fine-grained noise rather than a clean outline, since Laplacian's second-derivative calculation is inherently more sensitive to noise than a first-derivative method like Sobel. On the least-blurred examples (sigma 1.5-2.0), it still produced a usable but visibly dimmer and grainier result than Sobel or Prewitt. 

Canny was the most inconsistent of the four. On the lightly blurred, contrast-normalized images (sigma 1.5 and 2.0), it actually performed quite well, returning clean and fairly complete edges for the person, houses, and cars. But on every more heavily processed example - the binarized image, the heavily blurred scaled-down image, and the heavily rotated image at sigma 3.5 - Canny's output collapsed to almost nothing, often just a faint fragment of the strongest edge (typically the image's own warp boundary) with the rest of the frame coming back completely black. This lines up with how Canny works: it depends on a clear gradient peak to cross its threshold, and once an image has been blurred and warped enough, that peak gets washed out faster than it does for the gradient-magnitude approach Sobel and Prewitt use. 

Taking all of this together, Sobel (tied closely with Prewitt) was the most useful and consistent edge detector for this specific image set, precisely because this dataset includes so many heavily preprocessed variants (blurred, warped, binarized, rescaled). Canny's reputation as the "best" general-purpose edge detector did not hold up here - its strong performance was limited to the least-altered images, and it broke down the most under the kind of heavy preprocessing this assignment specifically generates.
























































## Homework Two: Image Segmentation

### Multi-channel normalization

Rather than equalizing a single channel as in Homework One (which normalized
only the V channel of HSV), this assignment splits the source image into its
LAB components and equalizes the L, A, and B channels independently before
merging them back into a color image. LAB was chosen specifically because L
isolates lightness from color, which matters for a doorbell-camera image
shot under uneven, low-light conditions - equalizing L alone corrects
brightness contrast, while independently equalizing A and B additionally
recovers faint color-channel detail that a lightness-only pass would leave
untouched.

### Quantitative comparison (IoU / Dice against ground truth)

| Method   | IoU    | Dice   |
|----------|--------|--------|
| Otsu     | 0.0368 | 0.0710 |
| Adaptive | 0.0916 | 0.1678 |
| K-Means  | 0.0230 | 0.0450 |

All three methods score low against the manually-traced ground truth, which
is itself a meaningful result given how dark and low-contrast the source
image is - but the relative ordering is informative. Adaptive thresholding
came out ahead of both Otsu and K-Means by a clear margin, which lines up
with what the masks show visually.

### Qualitative analysis

**Otsu's global thresholding** performed the worst of the three. Otsu
assumes the image splits cleanly into one bright group and one dark group,
but in this image the figure and the surrounding shadowed lawn/background
are both dark - so Otsu's single global cutoff lumped the person in with
the background shadow instead of separating him out. The resulting
foreground extraction kept the houses, sky, and lawn rather than the
figure, which is the opposite of what was needed.

**Adaptive thresholding** handled the same low-light conditions better
because it recomputes a threshold locally for each neighborhood rather than
using one global cutoff. This let it pick up on local contrast around the
figure's outline even where the overall scene was dark. The tradeoff is
noise: because adaptive thresholding reacts to small local intensity
variations, the output is considerably grainier than Otsu's, with the
sensor noise in the original low-light shot showing up as scattered
speckling across the whole mask, especially in the lawn and sky regions.
Even with that noise, it preserved more of the figure's actual silhouette
than either of the other two methods, which is reflected in its higher
IoU and Dice scores.

**K-Means clustering** in HSV space, with K selected automatically via
silhouette score, chose K=3 (silhouette score 0.5184, beating K=4 at 0.4827
and K=5 at 0.5046). However, the resulting clusters grouped the figure's
dark clothing together with the shadowed rooftops and architectural
features in the background, rather than isolating the person as his own
cluster - both share similar low-saturation, dark-value HSV characteristics
under these lighting conditions. This produced the lowest IoU of the three
methods, since the selected "figure" cluster mask captured house structure
in addition to (and in places instead of) the person.

**Effect of color normalization compared to Homework One's raw results:**
Equalizing all three LAB channels independently visibly brightened and
added contrast to the normalized image compared to the original, and
compared to Homework One's single-channel (V-only) normalization, the
LAB version preserves more separation between the figure and the
background in terms of raw pixel values. However, this normalization alone
was not enough to overcome the fundamental challenge for all three
segmentation methods: the figure and the background shadow/architecture
occupy a genuinely overlapping range of intensity and color values in this
particular photo. This suggests that intensity- and color-based
segmentation alone is insufficient for this image, and that a spatial or
edge-aware refinement step (e.g. combining the boundary detection from
Homework One with these masks, or adding a connected-component filter)
would likely be necessary to cleanly isolate the figure in future work.

### Comparison figure

![Segmentation comparison grid](outputs/stage5_evaluation/comparison_grid.png)




## Homework Three: Deep Learning for Fish Classification

### Dataset

The provided dataset contains 1,016 images across 6 fish species: Bete (194), Cray (80), Discuss (201), Gold (207), Guppy (189), and Oscar (145). All images are 800x600 RGB. There is a moderate class imbalance - Cray has less than half as many images as Gold - which is preserved proportionally across the train/val/test split via stratified sampling, and is worth keeping in mind when reading the per-class metrics below (Cray is the class most likely to be data-starved).

### Pipeline

Run in order:

```
python src/part2_data_pipeline.py
python src/part3_baseline_cnn.py
python src/part4_hyperparameter_tuning.py
python src/part5_evaluation.py
```

`part2` builds a stratified 70/15/15 split (saved to `outputs/stage2_classification/dataset_split.csv` so every later script trains/evaluates on the exact same split), resizes all images to 128x128, and applies horizontal flip / rotation / brightness jitter augmentation to the training set only. A sanity-check grid comparing raw vs. augmented images is saved to confirm augmentation isn't distorting the fish or scrambling labels before it feeds into training.

`part3` trains the baseline CNN (3 conv blocks: 32/64/128 filters, ReLU, MaxPool, followed by a 256-unit dense layer and dropout) with the assignment's specified starting hyperparameters (Adam, lr=0.001, batch size=32).

`part4` runs a grid search over 3 learning rates x 2 batch sizes x 2 dropout rates (12 configurations total), each trained for a shorter epoch budget to rank them by validation loss, then retrains the winning configuration for the full epoch budget.

`part5` evaluates both the baseline and optimized models on the held-out test set.

### Results

*(To be filled in after running the pipeline - grid search results, best hyperparameter configuration, classification report, and the comparison/confusion matrix figure below.)*

**Grid search results:**

| Learning Rate | Batch Size | Dropout | Best Val Loss | Best Val Acc |
|---|---|---|---|---|
| _pending_ | | | | |

**Winning configuration:** _pending_

**Baseline vs. optimized - test set metrics:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Baseline | | | | |
| Optimized | | | | |

### Qualitative Analysis

*(To be filled in after running - discuss the effect of augmentation on training stability, which hyperparameter(s) had the largest effect on overfitting/convergence, and what the confusion matrix reveals about which species get confused with each other - e.g. whether visually similar species, or the underrepresented Cray class, drive most of the errors.)*

### Comparison Figure

![comparison grid](outputs/stage5_classification/comparison_grid.png)




## Homework Three: Deep Learning for Fish Classification

### Dataset

The provided dataset contains 1,016 images across 6 fish species: Bete (194), Cray (80), Discuss (201), Gold (207), Guppy (189), and Oscar (145). All images are 800x600 RGB. There is a moderate class imbalance - Cray has less than half as many images as Gold - which is preserved proportionally across the train/val/test split via stratified sampling (711 train / 152 val / 153 test), and matters directly for the results below, since Cray is the smallest class by a wide margin.

### Pipeline

Run in order:

python src/part2_data_pipeline.py
python src/part3_baseline_cnn.py
python src/part4_hyperparameter_tuning.py
python src/part5_evaluation.py


`part2` builds a stratified 70/15/15 split (saved to `outputs/stage2_classification/dataset_split.csv` so every later script trains/evaluates on the exact same split), resizes all images to 128x128, and applies horizontal flip / rotation / brightness jitter augmentation to the training set only. A sanity-check grid comparing raw vs. augmented images confirmed augmentation preserved species identity before it was trusted for training.

`part3` trains the baseline CNN (3 conv blocks: 32/64/128 filters, ReLU, MaxPool, followed by a 256-unit dense layer and dropout=0.4) with the assignment's specified starting hyperparameters (Adam, lr=0.001, batch size=32), for 25 epochs.

`part4` runs a grid search over 3 learning rates (0.01, 0.001, 0.0001) x 2 batch sizes (32, 64) x 2 dropout rates (0.3, 0.5) - 12 configurations, each trained for a 10-epoch search budget and ranked by validation loss - then retrains the winning configuration for the full 25-epoch budget.

`part5` evaluates both the baseline and optimized models on the held-out test set.

### Results

**Grid search - winning configuration:**

| Learning Rate | Batch Size | Dropout | Best Val Loss (10-epoch search) | Best Val Acc |
|---|---|---|---|---|
| 0.001 | 32 | 0.3 | 0.7041 | 0.7632 |

**Baseline vs. optimized - test set metrics:**

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| Baseline | 0.88 | 0.85 | 0.86 | 0.85 |
| Optimized | 0.82 | 0.81 | 0.78 | 0.78 |

Per-class F1 (baseline / optimized):

| Species | Baseline F1 | Optimized F1 |
|---|---|---|
| Bete | 0.91 | 0.84 |
| Cray | 0.64 | 0.55 |
| Discuss | 0.97 | 0.91 |
| Gold | 0.90 | 0.85 |
| Guppy | 0.92 | 0.89 |
| Oscar | 0.76 | 0.67 |

### Qualitative Analysis

Contrary to what the assignment structure might suggest, hyperparameter tuning did not improve on the baseline here - the optimized model scored lower across every aggregate metric (accuracy 0.82 vs 0.88, macro F1 0.78 vs 0.85) and lower per-class F1 in every single species, with no exceptions. This is a meaningful result in its own right and is worth explaining rather than treating as a failed experiment.

The winning grid search configuration (lr=0.001, batch_size=32, dropout=0.3) is nearly identical to the baseline's own hyperparameters (lr=0.001, batch_size=32, dropout=0.4) - the only real difference is a slightly lower dropout rate. Given how close the two configurations are, the gap in final test performance is more likely explained by training variance than by a genuine hyperparameter effect: the grid search ranked configurations using validation loss after only a 10-epoch search budget, but the winning config was then retrained for the full 25 epochs, and the loss/accuracy curves (see comparison figure) show validation loss for both the baseline and optimized model bottoming out early - around epoch 5-8 - before climbing back up while training loss keeps falling. That's classic overfitting, and it happens in both models to a similar degree; the optimized model's curve is not meaningfully more stable than the baseline's, despite dropout ostensibly regularizing against exactly this pattern. With a dataset this small (1,016 images total), the difference between hyperparameter configurations is easily within the noise introduced by which specific images land in the training batches and where training happens to be when overfitting sets in.

The confusion matrix for the optimized model shows the errors are concentrated almost entirely in two classes: Cray and Oscar. Cray - the smallest class in the dataset at only 80 total images (56 for training) - was correctly classified in just 6 of 12 test images, with 5 of the 6 errors predicted as Guppy. Oscar fared similarly poorly, correct on only 12 of 22, with its errors spread across Bete, Gold, and Cray rather than concentrated on one confusable species. By contrast, Gold, Guppy, and Discuss - the three largest classes - were classified correctly in the vast majority of cases for both models. This pattern lines up directly with the class imbalance noted in the dataset section: species with fewer training examples produced measurably weaker per-class performance in both models, which is the expected outcome of a class-imbalanced dataset trained without any class-weighting or oversampling correction.

Taken together, the results suggest two changes to try next rather than either the baseline or optimized configuration as-is: address the class imbalance directly (e.g. weighted loss or oversampling Cray), and use a validation-based early-stopping criterion during the final 25-epoch retrain, since both models' validation loss curves show overfitting setting in well before epoch 25 while the checkpoint-saving logic already correctly captures the best epoch regardless.

### Comparison Figure

![comparison grid](outputs/stage5_classification/comparison_grid.png)

