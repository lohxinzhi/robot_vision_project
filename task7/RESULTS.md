# Task 7 — Results and discussion

**Source:** recorded outputs embedded in `notebooks/task7_svm.ipynb` and `notebooks/task7_cnn.ipynb`, plus packaged CSV files. These are *reported run results*; a fresh full-dataset run was not possible because the original `dataset_2026` images were not included in the supplied ZIP.

## Dataset

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 0, 4, 7, 8, A, D, H (each) | 648 | 114 | 254 | 1,016 |
| **Total** | **4,536** | **798** | **1,778** | **7,112** |

`task7_split_manifest.csv` stores `path`, `label`, `pixel_hash`, and `split`. Among these 7,112 entries there are 5,850 distinct pixel-hash values. Repeated hashes occur **within** a split; none occurs in multiple splits or classes. This is an integrity check on the manifest only, not an independent rehash of original images.

## Validation studies

### SVM baselines (same 798-image validation split)

| Feature + kernel | Validation accuracy | Macro F1 | Errors |
|---|---:|---:|---:|
| Raw pixels + Linear | 98.25% | 0.9824 | 14 |
| Raw pixels + RBF | 98.25% | 0.9825 | 14 |
| HOG + Linear | 99.25% | 0.9925 | 6 |
| HOG + RBF | 99.00% | 0.9900 | 8 |

Preprocessing comparison with HOG + Linear, C=1: Original 99.25% (6 errors); Crop + Stretch 99.75% (2); Crop + Pad 99.12% (7). With Crop + Stretch + HOG + Linear, the validation search over C = 0.01, 0.1, 1, 10 gave 97.12%, 99.00%, 99.75%, **99.87%**, respectively. The final SVM used **C=10**.

### CNN validation and learning rate

The baseline SimpleCNN (2 conv + 2 max-pool) achieved a best reported validation accuracy of **99.75%** (796/798) with two `0 → 4` errors. Two subsequent controlled ten-epoch LR experiments (fixed architecture/batch/seed) selected `LR=0.001`: **99.75%** validation accuracy at the minimum-loss checkpoint, `val loss=0.0068` at epoch 10. `LR=0.0003` reached 99.50% at its selected epoch with `val loss=0.0284`.

## Independent test set (1,778 images)

| Metric | HOG + Linear SVM | SimpleCNN |
|---|---:|---:|
| Correct | 1,767 | 1,763 |
| Incorrect | 11 | 15 |
| Accuracy | **99.38%** | **99.16%** |
| Macro F1 | **0.9938** | **0.9916** |

The SVM correctly predicted four more images on this fixed test set, a difference of approximately **0.22 percentage points**. This is a small, single-split difference, not a conclusive claim of superiority across other domains. The CNN confusion matrix shows eleven of fifteen errors from character `H` (nine predicted as `A`).

## Real chip case study (10 characters, one image)

Reference: `HD44780A00`.

| Pipeline | Predicted string | Correct | Accuracy |
|---|---|---:|---:|
| SVM + Pipeline A | `HD4478DAD0` | 8/10 | 80% |
| SVM + Pipeline B | `HD44780ADD` | 8/10 | 80% |
| CNN + Pipeline A | `H044780A00` | 9/10 | 90% |
| CNN + Pipeline B | `HD44780A00` | 10/10 | 100% |

Pipeline A and B are different end-to-end segmentation/preprocessing procedures. The CNN corrected more characters in this one chip image even though the SVM scored slightly higher on the standard test set; this is consistent with a difference in the target domain, but ten characters are not enough to estimate general real-world reliability.

## Useful figures

- [CNN baseline learning curves](results/task7/cnn_baseline_training_curves.png)
- [CNN learning-rate comparison](results/task7/cnn_learning_rate_comparison.png)
- [CNN test confusion matrix](results/task7/cnn_final_test_confusion_matrix.png)
- [SVM test confusion matrix](results/task7/svm_final_test_confusion_matrix.png)
- [CNN chip comparison](results/task7/cnn_chip_comparison.png)
- [Task 6 raw segmentation comparison](results/task7/task6_segmentation_comparison.png)
- [SVM preprocessing comparison](results/task7/svm_preprocessing_accuracy.png)
- [SVM C tuning](results/task7/svm_C_tuning.png)

## Reproducibility cautions

- Fixed pixel hashes were checked for overlap *within manifest* only; raw source files are not part of this handoff.
- Hyperparameters were chosen using the validation set, and final train+validation models were tested on the independent test set.
- The stored `joblib` and PyTorch checkpoints were generated in the author's software environment; loading them requires compatible dependencies and trusted files.
- Comparing `SVM fit()` timing to an end-to-end CNN training loop is not a fair runtime comparison; for meaningful throughput measurement include identical preprocessing/feature extraction/device/batching conditions.
- Do not claim CNN universally outperforms SVM based on the 10-character chip case study.
