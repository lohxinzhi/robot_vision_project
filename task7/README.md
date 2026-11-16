# ME5411 Robot Vision — Task 7: Character Classification

**Team handoff: reproducible code, saved outputs, and the chip-character case study.**

This module implements **HOG + Linear SVM** and a **two-convolution-layer CNN** for classifying seven symbols: `0`, `4`, `7`, `8`, `A`, `D`, and `H`. It is designed to be placed as a subfolder named `task7` at the root of the team's repository, without changing the existing Task 1–6 work.

> **Status:** Original notebook outputs and saved models are included, but this archive **does not contain the instructor's `dataset_2026` images**. Rerunning full training requires the original dataset. The two sets of ten chip-character crops are included for inference checks.

## What to look at first

- [Results summary](RESULTS.md) — headline numbers and interpretation.
- [SVM notebook](notebooks/task7_svm.ipynb) — pixel and HOG baselines, kernel comparison, cropping study, C tuning, final test, real-chip inference.
- [CNN notebook](notebooks/task7_cnn.ipynb) — CNN design, baseline, learning-rate comparison, final test, real-chip inference.
- [Final 4-pipeline comparison CSV](results/task7/final_svm_cnn_comparison.csv).
- [CNN chip comparison figure](results/task7/cnn_chip_comparison.png).
- [Task 6 segmentation side-by-side figure](results/task7/task6_segmentation_comparison.png).

## Summary of reported results

The same **7,112** images were used in a fixed split: training **4,536**, validation **798**, independent test **1,778**. Hyperparameters were chosen on validation data; final models were retrained using **5,334** train+validation samples and evaluated on **1,778** test samples.

| Metric | HOG + Linear SVM (C=10) | CNN (2 Conv; Adam LR=0.001, 10 epochs) |
|---|---:|---:|
| Independent test accuracy | **99.38%** (1,767/1,778) | **99.16%** (1,763/1,778) |
| Independent test macro F1 | **0.9938** | **0.9916** |
| Incorrect test predictions | 11 | 15 |
| Chip Pipeline A: own Task 6 | 8/10 (80%) | 9/10 (90%) |
| Chip Pipeline B: teammate Task 6 | 8/10 (80%) | 10/10 (100%) |

The chip case study uses **one image containing ten characters**, so its percentages must not be generalized to all chip images. Standard test and target-chip results have different characteristics. Training times are not strictly comparable because SVM timing excludes HOG extraction.

## Directory layout

```text
robot_vision_project/                # existing group repository, not shipped here
├── tasks.ipynb                       # existing Tasks 1–6, unchanged
├── dataset_2026/                     # instructor-provided dataset (supply locally)
└── task7/                            # copy this module here
    ├── README.md
    ├── RESULTS.md
    ├── notebooks/
    │   ├── task7_svm.ipynb
    │   └── task7_cnn.ipynb
    ├── requirements-svm.txt
    ├── requirements-cnn.txt
    └── results/
        ├── task6_characters_cleaned/       # 10 Pipeline A PNGs
        ├── task6_teammate_characters/      # 10 Pipeline B PNGs
        └── task7/                          # CSV, figures, checkpoints
```

The module may also be used outside the group repository by placing `dataset_2026` next to `task7`. Both notebooks automatically locate `task7/results/task7/task7_split_manifest.csv` from the working directory and resolve dataset image paths against `../dataset_2026` or `task7/dataset_2026`. If you keep the dataset elsewhere, set the environment variable **`ME5411_DATASET_DIR`** to the full path to the `dataset_2026` directory before starting VS Code/Jupyter.

The included manifest contains relative image paths such as `dataset_2026/0/img001-00001.png`. Do not change its `split` labels. The `pixel_hash` values were checked for overlap between splits, but cannot be independently checked against source pixels until the original dataset is available.

## Environment setup (Windows + Anaconda + VS Code)

**Important:** The original Windows experiments used separate SVM and CNN environments to avoid a PyTorch / OpenMP DLL initialization conflict. Do not install all packages into the original SVM environment just to rerun the CNN. Do not rely on `KMP_DUPLICATE_LIB_OK=TRUE` as a permanent fix.

Create and activate one environment for each notebook; example commands in Anaconda Prompt:

```bat
conda create -n me5411_svm_share python=3.11 pip -y
conda activate me5411_svm_share
python -m pip install -r task7\requirements-svm.txt
python -m ipykernel install --user --name me5411_svm_share --display-name "ME5411 SVM"
```

For CNN, start in a **new terminal** (with sufficient disk space):

```bat
conda create -n me5411_cnn_share python=3.11 pip -y
conda activate me5411_cnn_share
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r task7\requirements-cnn.txt
python -m ipykernel install --user --name me5411_cnn_share --display-name "ME5411 CNN"
```

If the Conda installation drive is almost full, use `conda create --prefix C:\ME5411_envs\me5411_cnn_share python=3.11 pip -y` (replace the drive with an available drive). Install commands assume your Anaconda Prompt is at the **repository root**; otherwise adjust the requirements paths. For existing saved `joblib` models, compatible scikit-learn versions are important; if loading raises a version warning or error, recreate the environment or retrain.

The original run environments used Python 3.11.16 (SVM) and 3.11.18 (CNN), with PyTorch 2.15.0+cpu recorded for the CNN. `requirements-*` list needed libraries, **not locked versions**, so exact reproducibility across versions/CPUs is not guaranteed. The original SVM HOG extractor uses **`skimage.feature.hog`**, not `cv2.HOGDescriptor`.

## Reproduce the analyses

1. Obtain the original `dataset_2026` and put it next to `task7/` (or set `ME5411_DATASET_DIR`). Verify classes `0/4/7/8/A/D/H` contain their images.
2. Launch VS Code from the repository folder. Open the corresponding notebook, select the matching Conda kernel, and choose **Restart Kernel → Run All**. All output paths are anchored at `task7/results/task7`, so the notebook may run from the repository root or its own folder.
3. Review the final model metrics, `results/task7/*test_confusion_matrix.png`, and `results/task7/final_svm_cnn_comparison.csv`.
4. The saved models and chip crops support a smaller inference-only verification without the original dataset: `python task7/verify_cnn_chip.py` in the CNN environment or `python task7/verify_svm_chip.py` in the SVM environment.
5. Compare the fresh outputs with the **preserved notebook outputs**, but expect timing and possibly minor training-result differences across dependency versions. Do not retune based on the test set.

## Integration with Task 1–6

Pipeline A: the user's Task 6 crops using global thresholding, contour filtering and vertical-projection splitting; Pipeline B: the teammate's local thresholding / connected-component segmentation. Both produce 10 ordered character crops (`character_01.png` … `character_10.png`). The crops and resulting classification comparison are included. **The actual Task 1–6 source notebooks are not included in this ZIP**; they remain in their respective original project locations. To reproduce segmentation from the raw chip image, the team must also provide the original Task 1–6 notebooks and chip image. The two pipelines differ in preprocessing and segmentation, so do not attribute their entire difference to a single algorithm.

## Contribution / review checklist

- [ ] Verify dataset root and split manifest on a clean checkout.
- [ ] Run SVM notebook with SVM environment (or verify model-only predictions).
- [ ] Run CNN notebook with CNN environment (or verify model-only predictions).
- [ ] Compare `RESULTS.md` with preserved notebook outputs.
- [ ] Discuss Pipeline A/B differences, error cases, and small real-chip sample size.
- [ ] Integrate Task 7 report text and figures into the group report.

**No changes to the existing group repository have been made by this package.** Copy this directory into a Git branch and propose a Pull Request for review.
