# ME5411 Robot Vision and AI

Group 33 project for character recognition on the supplied microchip image.

## Project layout

| Folder | Contents | Start here |
|---|---|---|
| `task1-6/` | Contrast, filtering, cropping, binary conversion, outlines and character segmentation | [Tasks 1–6 guide](task1-6/README.md) and `tasks.ipynb` |
| `task7/` | Dataset, CNN and handwritten HOG–SVM, hyperparameter experiments, selected checkpoints and chip inference | [Task 7 guide](task7/README.md) |
| `report/` | English LaTeX report draft, generated tables, figures and compiled PDF | [Report guide](report/README.md) and `main.pdf` |

## Tasks 1–6

```bash
cd task1-6
conda env create -f environment.yml
conda activate robot_vision
python -m notebook tasks.ipynb
```

Use Run All in the notebook. The folder includes its input image and local Python modules. The [task guide](task1-6/README.md) maps each requirement to the corresponding notebook section.

## Task 7

```bash
cd task7
conda env create -f environment.yml
conda activate robot-vision-task7
python recognize_chip.py
```

The complete dataset and selected checkpoints are included. The folder supports independent use after copying it elsewhere.

| Evaluation | CNN | HOG–SVM |
|---|---:|---:|
| Character test set, 1,778 images | 99.72% | 99.33% |
| Chip image, 4-connectivity | 100% | 100% |
| Chip image, 8-connectivity | 100% | 100% |

Both chip predictions are `HD44780A00`. Classifier selection uses validation accuracy with seed 5411. Chip filtering parameters were developed using feedback from the supplied chip image. Its score describes those ten characters.

Review the [95 training records](task7/results/all_experiment_tables.md), [classifier selection](task7/results/validation_selection.md) and [image-processing comparisons](task7/results/chip_preprocessing_experiments.md). From `task7/`, `python train.py` runs or resumes the classifier study, and `python compare_chip_preprocessing.py` compares Task 6 binary characters with further image processing.

## Report

From the repository root, generate the report's saved figures and numerical tables, then compile it.

```bash
python report/export_figures.py
bash report/build.sh
```

The build uses Arial and Tectonic or XeLaTeX. The report contains editable task sections and the complete classifier experiment table. Member names, student numbers and the group's discussion remain to be completed in the draft.

## Attribution

Tasks 1–4 build on [Xin Zhi's repository](https://github.com/lohxinzhi/robot_vision_project). The unified Task 7 investigation incorporates comparison factors from both group contributions. Source acknowledgments and experimental provenance are documented in the Task 7 guide and report.
