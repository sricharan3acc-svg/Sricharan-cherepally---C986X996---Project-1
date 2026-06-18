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
