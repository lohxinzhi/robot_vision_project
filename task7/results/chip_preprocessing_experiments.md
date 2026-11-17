# Chip recognition using Task 6 masks and further image processing

Case 1 directly classifies the binary character masks produced by Task 6. Case 2 applies grayscale median filtering, local thresholding and connected-component refinement within character windows. Both cases use the saved CNN/SVM classifiers and their recorded normalization.

| Case | CNN, 4-connectivity | CNN, 8-connectivity | SVM, 4-connectivity | SVM, 8-connectivity |
|---|---|---|---|---|
| Task 6 binary characters | `H044780A00` (90%) | `H044780A00` (90%) | `8D44780A00` (90%) | `8D44780A00` (90%) |
| Further mask refinement | `HD44780A00` (100%) | `HD44780A00` (100%) | `HD44780A00` (100%) | `HD44780A00` (100%) |

## Applied processing

Task 6 uses a 3×3 mean, 31×31 local mean with offset 4, 5×5 binary median and minimum component area 500. Each retained component supplies a binary character mask directly to the classifier.

Character location uses a 3×3 mean, 31×31 local mean with offset 2, 5×5 binary median and minimum component area 500. Each located window is expanded by two pixels. The recognition mask uses a 13×13 grayscale median, 3×3 mean, 31×31 local mean with offset 2 and 5×5 binary median. The largest connected region is retained inside each window. Parameters are in `../chip_preprocessing.json`.

Rows labeled 4 and 8 in the LaTeX table denote the connectivity definitions. The reference `HD44780A00` scores each sequence after prediction. The chip image supplied development feedback, and the reported score covers its ten characters.
