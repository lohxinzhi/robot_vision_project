# Task 7 selected models

Selection uses validation accuracy with seed 5411. Ties favor fewer parameters and then an earlier best epoch.

| Method | Validation accuracy | Test accuracy | Macro F1 | Errors | Parameters | Refit seconds | Model ms/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN | 99.91% | 99.72% | 0.99719 | 5 | 70599 | 8.186 | 0.07695 |
| SVM | 99.72% | 99.33% | 0.99324 | 12 | 12355 | 0.267 | 0.00105 |

SVM feature-plus-model inference is 0.55789 ms/image. Model-only timings exclude file decoding, input normalization and segmentation. SVM model timing also excludes HOG extraction. Timings use CPU four threads, one warm-up and the median of five passes.

## Selected parameters

### CNN

```json
{
  "mode": "stretch",
  "optimizer": "adam",
  "batch_size": 32,
  "shuffle_seed": 5411,
  "lr": 0.003,
  "epochs": 9,
  "depth": 2,
  "pool_size": 8,
  "hidden_size": 32,
  "dropout": 0.0,
  "weight_decay": 0.0
}
```

### SVM

```json
{
  "mode": "stretch",
  "optimizer": "adam",
  "batch_size": 128,
  "shuffle_seed": 5411,
  "lr": 0.03,
  "epochs": 9,
  "C": 100.0,
  "feature": "hog",
  "hog_cell_size": 4,
  "loss": "squared_hinge"
}
```

## Chip recognition

| Connectivity | Method | Prediction | Position accuracy |
|---|---|---|---:|
| 4 | CNN | `HD44780A00` | 100.00% |
| 4 | SVM | `HD44780A00` | 100.00% |
| 8 | CNN | `HD44780A00` | 100.00% |
| 8 | SVM | `HD44780A00` | 100.00% |

Chip image-processing parameters were selected using the chip development image. Read [the image-processing comparisons](chip_preprocessing_experiments.md) for the baseline, bounded-refinement experiments and fixed-checkpoint evidence.

```json
{
  "segmentation": {
    "gray_median_size": 1,
    "smooth_size": 3,
    "background_size": 31,
    "offset": 2,
    "median_size": 5,
    "min_area": 500
  },
  "glyph_refinement": {
    "gray_median_size": 13,
    "smooth_size": 3,
    "background_size": 31,
    "offset": 2,
    "median_size": 5,
    "box_margin": 2
  }
}
```

The shared test partition and chip image had already been inspected during project development. Parameter selection in this search uses validation data. The final scores describe this dataset split and this chip example.

All trials are in [all_experiment_tables.md](all_experiment_tables.md) and `experiments.csv`. Selected configurations are in [validation_selection.md](validation_selection.md).
