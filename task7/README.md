# Task 7 character recognition

This folder implements the complete Task 7 workflow, from dataset auditing and hyperparameter comparisons to selected CNN/SVM models and chip recognition. The supplied dataset, code, trained models and experiment records are included so the folder can be copied and used independently.

## Run

Create the environment from inside `task7/`.

```bash
conda env create -f environment.yml
conda activate robot-vision-task7
```

Recognize the chip using the selected models.

```bash
python recognize_chip.py
```

Run the complete experiment and regenerate its result tables.

```bash
python prepare_data.py
python train.py
```

`train.py` runs the search defined in `search_space.json`, selects the best configuration for each method using seed 5411, refits the selected CNN and SVM on the full development partition, evaluates them and exports the reports. Completed trials are reused. Default paths are relative to the Python files, so scripts can also be invoked by absolute path from another working directory.

## Selected models and results

<!-- results:start -->
The registry contains **95 trial records**, including 1 numerically failed trial(s).

| Selected model | Validation accuracy | Test accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|---:|
| CNN | 99.91% | 99.72% | 0.99719 | 70,599 |
| SVM | 99.72% | 99.33% | 0.99324 | 12,355 |

| Parameter | CNN | SVM |
|---|---|---|
| Representation | 2 conv layers, FC 2048→32→7 | HOG cell 4×4 |
| Optimizer | adam | adam |
| Learning rate | 0.003 | 0.03 |
| Preprocessing | stretch | stretch |
| Regularization / loss | Dropout 0.0, WD 0.0 | squared_hinge, C=100 |
| Refit epochs | 9 | 9 |
| Batch size | 32 | 128 |

| Connectivity | CNN prediction | SVM prediction |
|---|---|---|
| 4 | `HD44780A00` (100.00%) | `HD44780A00` (100.00%) |
| 8 | `HD44780A00` (100.00%) | `HD44780A00` (100.00%) |

The chip reference is `HD44780A00`. Position scores describe this development image.

Read the [full experiment tables](results/all_experiment_tables.md), [single-seed selection](results/validation_selection.md) and [final measurements](results/task7_results.md).

Image processing uses mild filtering to locate characters and bounded grayscale-median refinement for recognition. Read the [image-processing experiments](results/chip_preprocessing_experiments.md). Those choices used feedback from this chip image with the selected classifiers held fixed.
<!-- results:end -->

The inference entry point loads `results/cnn_model.pt` and `results/svm_model.pt`. Each checkpoint includes its class order, preprocessing and architecture configuration. `results/selection.json` records the selected configurations and their single-seed validation evidence.

## Data and evaluation protocol

The seven supported classes are `0`, `4`, `7`, `8`, `A`, `D` and `H`. There are 7,112 supplied class images, with 1,016 images per class. Class PNGs are 128×128. The chip bitmap at the dataset root is excluded from classifier training.

The fixed grouped split contains 4,270 inner-training images, 1,064 validation images and 1,778 test images. The inner-training and validation portions together form the required 75% development partition of 5,334 images. Final selected models are initialized again and fitted on all development images.

Decoded grayscale dimensions and pixel bytes identify exact duplicate groups through direct equality. Whole groups stay within one partition. The split records 5,850 unique pixel arrays and zero cross-partition exact-duplicate groups. Duplicate copies retain their weight within the assigned partition. Related fonts and near-duplicate glyphs may span partitions.

All search and final fitting use seed 5411. Each method selects the configuration with the highest validation accuracy, with ties resolved by fewer learned parameters and then an earlier best epoch. That best epoch determines the development-refit budget. These selection settings are recorded in `search_space.json` and `results/selection.json`.

Test results and the chip example had already been observed during project development. This implementation chooses parameters using validation data and freezes the selection before final evaluation. Its reported scores cover the supplied split and one chip image.

## CNN implementation

`models.py` defines `CharacterCNN`. All convolutions use 3×3 kernels with padding one. The first two stages have 16 and 32 channels, ReLU and 2×2 maximum pooling. A third 64-channel convolution is configurable. Adaptive average pooling produces a 4×4 or 8×8 grid. The final convolutional channel count multiplied by the pooling area determines the fully connected input size.

The hidden dense layer supports widths 32, 64 and 128. Dropout and weight decay are configurable. The final layer produces seven logits. Cross-entropy trains the network from random initialization. The search compares Adam, AdamW and SGD with momentum, together with appropriate learning-rate candidates.

## Handwritten HOG and SVM implementation

`hog_features()` uses NumPy gradients, unsigned orientations, nine orientation bins and interpolation between neighboring orientation bins. Overlapping 2×2-cell blocks use L2 normalization, clipping at 0.2 and renormalization. Spatial interpolation between cells is omitted. Cell sizes 8 and 4 produce 324 and 1,764 features, respectively. Flattened 1,024-dimensional pixels provide an additional feature baseline.

The classifier contains seven linear one-vs-rest margin functions. The local objective implements ordinary hinge or squared-hinge loss, summed over classes and averaged over images, plus half the squared weight norm. Biases are unregularized. The search compares C values 0.1, 1, 10 and 100. HOG extraction and margin losses are implemented in this repository.

SVM optimizer comparisons include Adam, SGD with momentum and LBFGS. Adam and SGD use minibatches. LBFGS uses the complete training partition, with up to ten quasi-Newton iterations and a strong-Wolfe line search per recorded outer epoch. Epoch counts therefore have optimizer-specific computational meaning, and the records also include fitting time. Numerically divergent trials are recorded as failed and excluded from selection.

## Preprocessing and chip segmentation

Both methods use float32, 32×32 inputs in [0, 1] with bright foreground. `resize` scales the whole image. `pad` crops inverted intensities above 51, centers the crop on a square canvas with a 12% per-side margin of at least two pixels, then resizes. `stretch` uses the same crop and directly resizes it. Bilinear interpolation is shared by these comparisons.

Chip recognition uses the Task 6 filtering and handwritten two-pass connected-component functions in `segmentation.py`. The fixed ROI is `(36, 202, 912, 136)`. `chip_processing.py` applies two processing stages, with parameters stored in `chip_preprocessing.json`.

1. Locate characters using a 3×3 mean filter, a 31×31 local mean with offset 2 and a 5×5 binary median filter. Label connected components, retain regions of at least 500 foreground pixels and order their bounding boxes from left to right.
2. Prepare a smoother recognition mask from the grayscale ROI using a 13×13 grayscale median, a 3×3 mean, a 31×31 local mean with offset 2 and a 5×5 binary median. Expand each located box by two pixels, clip it to the ROI and extract the largest component within that window from the smoother mask. Both labeling stages use the requested 4-connectivity or 8-connectivity.

Each extracted mask becomes a dark-on-white glyph before the checkpoint's saved normalization. The same processing parameters apply to all positions and both classifiers. The classifier scores determine each character, and the reference string `HD44780A00` then scores the predicted sequence. The original bitmap is read directly for every run.

### Chip recognition comparison

The comparison contains two cases using the saved CNN and SVM classifiers. Results agree under 4-connectivity and 8-connectivity.

| Character input | CNN | SVM |
|---|---|---|
| Task 6 binary character masks, classified directly | `H044780A00` (90%) | `8D44780A00` (90%) |
| Grayscale median filtering, local thresholding and character-window component refinement | `HD44780A00` (100%) | `HD44780A00` (100%) |

The Task 6 case uses its reviewed 3×3 mean, 31×31 local mean with offset 4, 5×5 binary median and minimum component area 500. Each retained component supplies a binary glyph to the classifier. CNN misclassifies D as 0, and SVM misclassifies H as 8.

Further processing uses the two-stage method described above, with grayscale median 13, local threshold offset 2 and box expansion 2. It produces a smoother binary mask inside each character window. Both classifiers correctly recognize all ten positions. The same normalization and model files are used in the two cases.

The chip image supplied development feedback for these image-processing settings. The 100% score covers its ten characters. Classifier test measurements are CNN 99.72% and SVM 99.33% on the 1,778-image character test set.

Run the two-case comparison using

```bash
python compare_chip_preprocessing.py
```

Run the applied recognition pipeline using

```bash
python recognize_chip.py --preprocessing chip_preprocessing.json
```

The [comparison table](results/chip_preprocessing_experiments.md) records both input cases and both connectivity definitions. JSON and CSV provide their complete predictions and component boxes. The before/after figure is `results/chip_preprocessing_comparison.png`.

## Change hyperparameters

Candidate values are declared in `search_space.json`. Each group contains a method, a starting configuration and parameter grids. `base: "fixed"` starts from the method defaults plus the group's fixed settings. `base: "best"` starts from the strongest recorded single-seed validation configuration for that method. Selection compares the recorded configurations across the entire registry using seed 5411.

For example, the CNN optimizer group contains candidates such as

```json
{"optimizer": ["adamw"], "weight_decay": [0.0001], "lr": [0.001, 0.003]}
```

The program expands each grid into individual configuration dictionaries. Parameters flow through `training.fit()` into model construction and optimization. Every trial records its complete settings, seed, best epoch, parameter count, duration and learning history. Matching configurations and random seeds reuse their recorded results. New trials receive sequential identifiers. Each result directory represents one dataset and split, so use a fresh output directory when starting experiments with a different dataset or split.

| Argument | Purpose |
|---|---|
| `--config` | Path to the search-space JSON |
| `--phase search` | Execute parameter comparisons |
| `--phase confirm` | Select the best configuration for each method |
| `--phase final` | Refit and evaluate the confirmed pair |
| `--group` | Run a named search group with `--phase search` |
| `--epochs` | Default CNN search budget, 20 epochs |
| `--data` | Dataset directory |
| `--output` | Results and experiment-registry directory |

Individual execution examples are

```bash
python train.py --phase search --group cnn_optimizers
python train.py --phase search --group svm_optimizers
python train.py --phase confirm
python train.py --phase final
python summarize_results.py
```

The normal `python train.py` command executes these steps in order for all groups. A copied folder with a fresh results directory can reproduce the full search. The included records allow reviewing and resuming the completed experiment.

## Output files

| Path | Contents |
|---|---|
| `results/cnn_model.pt`, `results/svm_model.pt` | Selected inference models |
| `results/selection.json` | Selection rule, confirmed settings and trial IDs |
| `results/study.json` | Shared trial registry and search progress |
| `results/experiments/` | Per-trial records and supporting evaluated-model evidence |
| `results/experiments.csv` | One row per validation trial |
| `results/all_experiment_tables.md` | All candidate settings and validation results |
| `results/validation_selection.md` | Single-seed selected configurations |
| `results/classification_results.json` | Test metrics, sizes and timings |
| `results/task7_results.md`, `results/task7_results.tex` | Summary and includable LaTeX tables |
| `results/chip_results.json` | 4/8-connectivity predictions and boxes |
| `chip_preprocessing.json` | Selected chip filtering and extraction parameters |
| `results/chip_preprocessing_experiments.*` | Two chip input cases, per-connectivity results and LaTeX table |
| `results/chip_processing_4.png`, `results/chip_processing_8.png` | Grayscale ROI, locating mask and extracted recognition masks |
| `results/*_confusion.png`, `results/*_comparison.png` | Report figures |

The LaTeX result table requires `booktabs`. Timing measurements use a four-thread CPU, a warm-up and the median of five full-test passes. Model-only timing excludes file reading, normalization and segmentation. CNN model time includes learned feature extraction. SVM feature-plus-model timing includes the handwritten descriptor computation.

## Sources

The comparison factors incorporate the complementary CNN and preprocessing investigations in [Task 7 PR #3](https://github.com/lohxinzhi/robot_vision_project/pull/3), alongside this project's local HOG, margin-loss implementation and Task 6 segmentation. All experiments in the shared search use the same dataset manifest and implementation interfaces.

The project uses Python, NumPy, Pillow, Matplotlib, OpenCV and PyTorch. `environment.yml` supplies the dependencies. The course project brief and lecture materials define the task. The group should develop its own discussion from the recorded evidence and follow the assignment's rules for report authorship and assistance disclosure.

The complete LaTeX experiment table is `results/experiments.tex` and requires `booktabs` and `longtable`. `results/task7_results.tex` contains the selected-model comparison table and requires `booktabs`.
