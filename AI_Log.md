# AI Log
## Entry 1

**Date and Time:** 15th June – 9:30 Pm
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to check if it could open and read the GitHub assignment repository instructions and give a step-by-step breakdown.
**Response Synopsis:** Confirmed the README was readable and broke the assignment down into four parts: project setup, a basic image-processing pipeline producing 168 images, edge detection on a 42-image subset producing 210 images, and final submission.
**Changes Made:** None yet - used to understand the assignment scope before writing any code.

## Entry 2

**Date and Time:** 15th June – 10:00 Pm
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked for a beginner-friendly, step-by-step process.
**Response Synopsis:** Provided setup guidance and wrote two Python scripts: one for image statistics, color space conversions, affine warps, and Gaussian blur (Part 2), and one for edge detection and comparison plots (Part 3), along with README, AI_Log, requirements.txt, and hello_world.py.
**Changes Made:** Added the initial src/ scripts and supporting project files.

## Entry 3

**Date and Time:** 15th June – 10:30 Pm
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked if the project could be set up and version-controlled without installing Git directly.
**Response Synopsis:** Explained GitHub's browser-based file upload feature and GitHub Desktop as alternatives to the Git command line.
**Changes Made:** Decided to use GitHub Desktop instead of the Git CLI.


## Entry 4

**Date and Time:** 16th June – 12:00 Am
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked for next steps after installing GitHub Desktop, including a screenshot of the GitHub Desktop start screen.
**Response Synopsis:** Walked through cloning the repository through GitHub Desktop and copying the starter project files into the cloned folder.
**Changes Made:** Repository cloned locally; initial commit made containing hello_world.py, AI_Log.md, README.md, requirements.txt, and the src/ folder.

## Entry 5

**Date and Time:** 16th June – 12:15 Am
**AI Tool:** Claude (Anthropic)
**Prompt:** A series of prompts working through environment setup and errors, a "python command not found" Microsoft Store alias issue, discovering the "py" launcher, whether a PNG image was an acceptable input format, numpy/scipy version conflicts, numpy/matplotlib version conflicts, where to place the input image, and a recurring ".vs folder permission denied" commit error in GitHub Desktop.
**Response Synopsis:** Diagnosed each issue (PATH and execution-alias problems, package version mismatches between the existing Anaconda installation and pip, Visual Studio locking its own metadata folder) and provided fixes for each, including using the "py" launcher, force-reinstalling matching package versions, and closing Visual Studio and ignoring the .vs folder in Git.
**Changes Made:** Got the local Python environment working correctly; successfully ran part2_processing.py and part3_edges.py, then committed and pushed the generated outputs.

## Entry 7

**Date and Time:** 16th June – 1 Am
**AI Tool:** Claude (Anthropic)
**Prompt:** Uploaded a generated comparison figure and asked whether it looked correct.
**Response Synopsis:** Explained why the figure looked the way it did based on its specific processing chain (binarization, shear, heavy blur), and pointed out the near-empty Auto-Canny result as a genuine, useful observation for the discussion section.
**Changes Made:** Used this interpretation as a starting point for the edge-detector discussion section of the README.

## Entry 9

**Date and Time:** 16th June – 2:30 Am
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to review the README for formatting issues and overall correctness.
**Response Synopsis:** Identified a markdown table line-break issue, a text-encoding glitch in the author line, and image links that needed to be placed on separate lines, while confirming the rest of the content was accurate.
**Changes Made:** Fixed the formatting issues in README.md.





















##Homework-2

## Entry 10

**Date and Time:** 29th June – 6:00 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to review the Homework Two assignment README and break down the requirements into an actionable plan.
**Response Synopsis:** Outlined the six required components: branch management, multi-channel color normalization, threshold-based segmentation, clustering-based segmentation, quantitative evaluation, and submission.
**Changes Made:** Created `Feature-Segmentation` branch from the existing repository.

## Entry 11

**Date and Time:** 29th June – 6:30 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude for guidance on selecting a color space for normalization and a method for choosing K in K-Means clustering.
**Response Synopsis:** Recommended LAB color space for channel-wise histogram equalization, since it separates lightness from chrominance and is well suited to the unevenly lit source image, and recommended selecting K via silhouette score rather than visual judgment.
**Changes Made:** Adopted LAB-based normalization and silhouette-driven K selection as the implementation approach for Parts 2 and 4.

## Entry 12

**Date and Time:** 29th June – 7:00 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Provided the existing Part 2 script from Homework One for reference and asked Claude to implement multi-channel normalization for Homework Two.
**Response Synopsis:** Implemented `part2_multichannel_norm.py`, extending the manual histogram equalization routine from Homework One to operate independently on the L, A, and B channels before merging them back into a color image.
**Changes Made:** Added `src/part2_multichannel_norm.py`; ran it on the Homework One source image, generating 8 output images in `outputs/stage2_segmentation/`.

## Entry 13

**Date and Time:** 29th June – 7:45 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to implement Otsu's global thresholding and adaptive thresholding for the normalized image.
**Response Synopsis:** Implemented `part3_threshold_segmentation.py` using `cv2.threshold` with `THRESH_OTSU` and `cv2.adaptiveThreshold` with a Gaussian-weighted window, using the same block size and offset as the bilevel conversion in Homework One.
**Changes Made:** Added `src/part3_threshold_segmentation.py`; generated binary masks and foreground extractions in `outputs/stage3_segmentation/`.


## Entry 14

**Date and Time:** 29th June – 8:15 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to implement K-Means clustering segmentation with a systematic method for selecting K.
**Response Synopsis:** Implemented `part4_kmeans_segmentation.py`, performing K-Means clustering in HSV space across K=3 to 5 and selecting the optimal K using silhouette score on a pixel subsample, then identifying the foreground cluster using a distance-from-center heuristic.
**Changes Made:** Added `src/part4_kmeans_segmentation.py`; K=3 was selected (silhouette score 0.5184); generated a cluster visualization and binary mask in `outputs/stage4_segmentation/`.

## Entry 15

**Date and Time:** 29th June – 8:45 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Provided a manually-traced ground truth mask of the figure and asked Claude to implement the evaluation script.
**Response Synopsis:** Implemented `part5_evaluation.py` to compute Intersection over Union (IoU) and Dice coefficients for each of the three segmentation methods against the ground truth mask, and to assemble a six-panel side-by-side comparison figure.
**Changes Made:** Added `src/part5_evaluation.py`; generated `metrics_summary.txt` and `comparison_grid.png` in `outputs/stage5_evaluation/`.

## Entry 16

**Date and Time:** 29th June – 9:00 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to draft a qualitative and quantitative analysis section for the README based on the actual IoU/Dice results and observed mask outputs.
**Response Synopsis:** Drafted an analysis explaining why Otsu's and K-Means' results misclassified the figure against shadow and background regions, why adaptive thresholding scored comparatively higher, and how LAB-based normalization compared to Homework One's single-channel approach.
**Changes Made:** Added a "Homework Two: Image Segmentation" section to `README.md`, including the results table and written analysis.


##Homework-3

## Entry 17

**Date and Time:** 19th July  9:00 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to review the Homework Three assignment README and break down the requirements into an actionable plan.
**Response Synopsis:** Outlined the six required parts: branch/logging setup, data preprocessing and augmentation, a baseline CNN, hyperparameter tuning, evaluation, and submission. Also flagged that a prior review comment on Homework Two (K-Means/Otsu masks were inverted relative to the true foreground) should translate into explicit sanity checks throughout this assignment - specifically around class-index-to-species-name mapping and best-checkpoint selection.
**Changes Made:** Created `Feature-Classification` branch from the existing repository, without modifying the `Feature-Segmentation` branch's code.

## Entry 18

**Date and Time:** 19th July  9:20 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Uploaded the Fish.7z dataset and asked Claude to confirm the class structure before writing any pipeline code.
**Response Synopsis:** Extracted and inspected the archive - 1,016 images across 6 species folders (Bete, Cray, Discuss, Gold, Guppy, Oscar), all 800x600 RGB. Flagged a moderate class imbalance (Cray at 80 images vs. Gold at 207) as worth noting in the Part 5 analysis, and confirmed a 70/15/15 stratified split preserves per-class proportions across train/val/test.
**Changes Made:** None yet - used to confirm dataset structure before implementation.

## Entry 19

**Date and Time:** 19th July  9:35 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude for a framework and image-size recommendation given the dataset size, deferring the choice to Claude's judgment.
**Response Synopsis:** Recommended PyTorch, 128x128 image size (224x224 judged likely to overfit further given only ~1,000 images total), and grid search over the required hyperparameters (learning rate, batch size, dropout) since the search space is only 12 configurations - small enough that grid search is fully tractable without needing random or Bayesian sampling.
**Changes Made:** Adopted PyTorch / 128x128 / grid search as the implementation approach for Parts 2-4.

## Entry 20

**Date and Time:** 19th July  9:50 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to implement the data pipeline: stratified split, resize/normalize transforms, and training-set-only augmentation.
**Response Synopsis:** Implemented `part2_data_pipeline.py` - a `collect_dataset` walk over the species folders, a two-step stratified `train_test_split` for 70/15/15, a saved `dataset_split.csv` so later scripts reuse the identical split, torchvision transform pipelines (train gets flip/rotation/brightness jitter, val/test only resize+normalize), a shared `FishDataset` class with a single alphabetically-ordered `class_to_idx` mapping, and a sanity-check image grid comparing raw vs. augmented samples before trusting the pipeline downstream.
**Changes Made:** Added `src/part2_data_pipeline.py`. Verified `collect_dataset` and the stratified split against the actual extracted dataset (1,016 images, proportions preserved across splits) before moving on.

## Entry 21

**Date and Time:** 19th July  10:10 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to implement the baseline CNN architecture and training loop per the assignment spec (3 conv blocks, dense hidden layer, Adam/lr=0.001/batch=32).
**Response Synopsis:** Implemented `part3_baseline_cnn.py` - a `FishCNN` class with 3 conv blocks (32/64/128 filters, ReLU, MaxPool) feeding a flatten, 256-unit dense layer, dropout, and a final linear layer to 6 classes. Training loop checkpoints weights only when validation loss improves (not the last epoch), and saves loss/accuracy curves.
**Changes Made:** Added `src/part3_baseline_cnn.py`.

## Entry 22

**Date and Time:** 19th July  10:25 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to implement the Part 4 hyperparameter tuning step.
**Response Synopsis:** Implemented `part4_hyperparameter_tuning.py` - grid search across 3 learning rates x 2 batch sizes x 2 dropout rates (12 configs), each trained for a reduced epoch budget to rank configs by validation loss, then the winning config gets retrained for the full epoch budget and saved as the optimized model. Added a printed gap check between the best and runner-up configs so a near-tie doesn't get over-interpreted as a real hyperparameter effect in the writeup.
**Changes Made:** Added `src/part4_hyperparameter_tuning.py`.

## Entry 23

**Date and Time:** 19th July  10:40 PM
**AI Tool:** Claude (Anthropic)
**Prompt:** Asked Claude to implement the Part 5 evaluation and comparison script.
**Response Synopsis:** Implemented `part5_evaluation.py` - loads both saved models, evaluates on the held-out test split, and explicitly passes an ordered `labels=`/`target_names=` list built from the same `class_to_idx` mapping used in training to sklearn's `classification_report` and `confusion_matrix`, rather than letting sklearn infer an order. Also added a 5-example spot-check printing true vs. predicted species side by side before trusting the aggregate metrics, directly following the sanity-check habit raised in the Homework Two review.
**Changes Made:** Added `src/part5_evaluation.py`. Full pipeline written and syntax-verified; actual training/results to be generated after running locally with GPU/full compute.

