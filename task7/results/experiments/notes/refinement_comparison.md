# Sequential SVM refinement

Each round was chosen after observing the preceding result. The dataset split and seed remain fixed.

| Round | HOG cell | C | Validation accuracy | Test accuracy | Chip prediction |
|---|---:|---:|---:|---:|---|
| Baseline | 8 | 10 | 96.52% | 96.51% | `H844780A00` |
| c100 | 8 | 100 | 97.93% | Not evaluated | `H844780A00` |
| c100_hog4 | 4 | 100 | 99.53% | 99.55% | `HD44780A00` |

The first round increased C from 10 to 100. Validation accuracy improved, while the second chip character retained the prediction 8. This motivated a feature change.

The second round held C at 100 and changed HOG cells from 8x8 to 4x4. The descriptor grew from 324 to 1764 entries. This resolved the second character and improved validation accuracy. The validation-selected checkpoint was refitted for eight epochs on the full development partition, then evaluated on the reserved test set.

The chip image participates in this refinement feedback. Its correct transcription is a development result. Each classifier predicts all positions using its learned scores.

The finer descriptor increases learned parameters from 2275 to 12355. HOG-plus-model timing changed from 0.2008 to 0.5532 ms per image on this CPU run.

Original weights, predictions, metrics and plots are retained in baseline/. Each iteration has its own weights, training history and chip outputs under iterations/.
