# Tasks 1–6 image processing

`tasks.ipynb` is the main notebook. It includes the saved figures and observations for all six tasks. `preprocessing.py` implements the image operations, and `utils.py` supplies the image-and-histogram plotting function.

## Run

From this folder,

```bash
conda env create -f environment.yml
conda activate robot_vision
python -m notebook tasks.ipynb
```

Open the notebook, select the environment's Python kernel and use Run All. The notebook reads `Test_Image_2026.bmp` from this folder. The handwritten spatial filters can take several minutes to run.

## Requirement map

| Task | Notebook section | Processing |
|---|---|---|
| 1 | Display Image | Original image, grayscale, contrast stretching, thresholding and histogram equalization |
| 2 | Apply Average filter | Required 5×5 mean filter, window-size comparisons, square/cross medians and processing-order experiments |
| 3 | Create Sub-image | Crop `(x, y, width, height) = (36, 202, 912, 136)` |
| 4 | Binary conversion after the crop | Threshold 110 and cross-median comparisons with widths 23 and 13 |
| 5 | Determine Character Outlines | Inner boundary of the Task 4 width-23 cross-median binary mask |
| 6 | Connected Components and Bounding Boxes | Grayscale crop, local threshold, two-pass 4/8-connectivity labeling, area filtering and boxes |

Task 6 uses a 3×3 mean, a 31×31 local mean with offset 4, and a 5×5 binary median. A 500-pixel component-area cutoff retains ten characters under both connectivity definitions. These are the reviewed Task 6 settings.

Task 7 contains its own copy of the relevant segmentation routines for standalone use. Its image-processing study additionally refines masks within established character windows. Details and the selected recognition parameters are in [the Task 7 guide](../task7/README.md).

## Files

| File | Purpose |
|---|---|
| `tasks.ipynb` | Tasks 1–6 implementation and saved results |
| `preprocessing.py` | Contrast, filtering, binary masks, outlines and connected components |
| `utils.py` | Figure and histogram display |
| `Test_Image_2026.bmp` | Supplied chip image |
| `filter_test.ipynb`, `noise.png` | Supporting filter experiments from the original repository |
| `environment.yml` | Conda environment, including the notebook interface |

The report exporter reads the saved outputs in `tasks.ipynb`. After rerunning and saving the notebook, run `python report/export_figures.py` from the repository root to refresh its figures.
