# Group 33 report draft

`main.tex` assembles the English LaTeX draft, and `main.pdf` is its compiled preview. The report covers Tasks 1–6, the unified Task 7 study, image-processing refinement and the recorded final results. It uses 11-point Arial, single spacing and A4 pages.

The assignment requires the group's own ideas, analysis and reasoning. The draft identifies AI assistance and marks group discussion areas. Complete member names and student numbers, review the methods and references, and supply the group's discussion and conclusion before submission. Group 33 follows the existing notebook heading.

## Build

Install Arial and either Tectonic or XeLaTeX. From the repository root, run

```bash
python report/export_figures.py
bash report/build.sh
```

The Python exporter uses the standard library to extract saved notebook figures, copy Task 7 figures and generate numerical tables from experiment JSON files. The build script produces `report/main.pdf` and keeps temporary compilation files outside the repository. The included figures and tables support compiling the report folder after copying it elsewhere. Tectonic may download TeX resources on first use.

## Editable files

| File | Contents |
|---|---|
| `main.tex` | Formatting, cover, contents, introduction and implementation map |
| `sections/task1-6.tex` | Reviewed image-processing methods, saved figures and segmentation results |
| `sections/task7.tex` | Shared search, CNN, HOG–SVM, model comparison and fixed-model image-processing experiments |
| `sections/discussion.tex` | Group discussion and conclusion drafting area |
| `sections/running.tex` | Step-by-step execution guide for all three directories |
| `sections/references.tex` | Course references, source acknowledgments and experiment provenance |
| `tables/` | Generated numerical tables and all 95 classifier records |
| `figures/` | Figures referenced by the report |

Edit the prose in `main.tex` and `sections/`. `export_figures.py` regenerates the files under `tables/` from the recorded results.

## Evidence

Tasks 1–6 use saved outputs from `../task1-6/tasks.ipynb`. The original bitmap is included in `../task1-6/` and the standalone Task 7 dataset. Task 7 uses the experiment registry and selected results under `../task7/results/`.

The classifier study contains 95 trial records, including one numerical failure. All experiments use seed 5411. Each method selects its configuration by validation accuracy, with ties resolved by fewer parameters and then an earlier best epoch. Both selected configurations are refitted for nine epochs. The selected CNN has 70,599 parameters and 99.72% character-test accuracy. The selected HOG–SVM has 12,355 parameters and 99.33% character-test accuracy.

Image-processing experiments hold the selected models fixed. The report compares Task 6 binary character masks (90% for both classifiers) with further grayscale-median filtering, local thresholding and character-window refinement (100% for both). Both connectivity definitions and a before/after mask figure are included. Both classifiers recognize `HD44780A00` with 100% position accuracy under 4-connectivity and 8-connectivity. The chip image supplied feedback for those processing choices, so this score describes its ten characters.

The source PR comparison is retained among the [experiment notes](../task7/results/experiments/notes/pr_comparison.md). The report's numerical tables use the unified registry and selected checkpoints.

Member names, student numbers and the group's discussion and conclusions require completion in the draft. The report identifies AI assistance and the group should review its technical wording and source acknowledgments against the assignment requirements.
