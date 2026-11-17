# Task 7 experiments

One dataset manifest is shared by all trials. Inner training uses 4,270 images, validation 1,064, and testing 1,778.
Each trial is initialized independently using seed 5411. Validation accuracy selects the configuration and epoch before the final development refit.

| Method | Optimizer | Mode | Structure / feature | Regularization / loss | LR | Batch | Seed | Best epoch | Validation |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| CNN | adam | pad | depth 2, pool 4, FC 512→64→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 17 | 99.15% |
| CNN | adam | pad | depth 2, pool 4, FC 512→64→7 | dropout 0.25, WD 0.0001 | 0.003 | 128 | 5411 | 19 | 98.97% |
| CNN | adam | pad | depth 2, pool 8, FC 2048→64→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 18 | 99.34% |
| CNN | adam | pad | depth 2, pool 8, FC 2048→64→7 | dropout 0.25, WD 0.0001 | 0.003 | 128 | 5411 | 20 | 99.06% |
| CNN | adam | pad | depth 3, pool 4, FC 1024→64→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 18 | 99.25% |
| CNN | adam | pad | depth 3, pool 4, FC 1024→64→7 | dropout 0.25, WD 0.0001 | 0.003 | 128 | 5411 | 16 | 99.25% |
| CNN | adam | pad | depth 3, pool 8, FC 4096→64→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 17 | 99.34% |
| CNN | adam | pad | depth 3, pool 8, FC 4096→64→7 | dropout 0.25, WD 0.0001 | 0.003 | 128 | 5411 | 16 | 99.25% |
| CNN | adam | pad | depth 2, pool 8, FC 2048→64→7 | dropout 0, WD 0.0001 | 0.003 | 128 | 5411 | 16 | 99.25% |
| CNN | adam | pad | depth 2, pool 8, FC 2048→64→7 | dropout 0.25, WD 0 | 0.003 | 128 | 5411 | 16 | 99.25% |
| CNN | adam | resize | depth 2, pool 8, FC 2048→64→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 16 | 99.15% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→64→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 14 | 99.72% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→64→7 | dropout 0, WD 0 | 0.001 | 128 | 5411 | 17 | 99.15% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→64→7 | dropout 0, WD 0 | 0.003 | 32 | 5411 | 7 | 99.72% |
| SVM | adam | pad | pixels, cell 8 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 7 | 96.43% |
| SVM | adam | pad | hog, cell 8 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 20 | 96.62% |
| SVM | adam | resize | hog, cell 8 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 11 | 95.96% |
| SVM | adam | stretch | hog, cell 8 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 18 | 97.65% |
| SVM | adam | stretch | hog, cell 4 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 20 | 98.97% |
| SVM | adam | stretch | hog, cell 4 | hinge, C=10 | 0.01 | 128 | 5411 | 9 | 98.12% |
| SVM | adam | stretch | hog, cell 4 | hinge, C=0.1 | 0.01 | 128 | 5411 | 28 | 92.39% |
| SVM | adam | stretch | hog, cell 4 | hinge, C=1 | 0.01 | 128 | 5411 | 1 | 96.62% |
| SVM | adam | stretch | hog, cell 4 | hinge, C=100 | 0.01 | 128 | 5411 | 9 | 99.15% |
| SVM | adam | stretch | hog, cell 4 | squared_hinge, C=0.1 | 0.01 | 128 | 5411 | 1 | 95.30% |
| SVM | adam | stretch | hog, cell 4 | squared_hinge, C=1 | 0.01 | 128 | 5411 | 4 | 97.37% |
| SVM | adam | stretch | hog, cell 4 | squared_hinge, C=100 | 0.01 | 128 | 5411 | 29 | 99.53% |
| SVM | adam | pad | hog, cell 4 | hinge, C=100 | 0.01 | 128 | 5411 | 8 | 98.68% |
| SVM | adam | pad | hog, cell 4 | squared_hinge, C=100 | 0.01 | 128 | 5411 | 9 | 99.34% |
| CNN | adam | resize | depth 3, pool 4, FC 1024→64→7 | dropout 0.25, WD 0.0001 | 0.001 | 128 | 5411 | 17 | 98.21% |
| CNN | adam | resize | depth 3, pool 4, FC 1024→64→7 | dropout 0.25, WD 0.0001 | 0.003 | 128 | 5411 | 19 | 98.78% |
| CNN | adam | pad | depth 3, pool 4, FC 1024→64→7 | dropout 0.25, WD 0.0001 | 0.001 | 128 | 5411 | 18 | 98.31% |
| CNN | adam | pad | depth 2, pool 4, FC 512→64→7 | dropout 0, WD 0.0001 | 0.003 | 128 | 5411 | 19 | 98.97% |
| CNN | adam | pad | depth 2, pool 4, FC 512→64→7 | dropout 0.25, WD 0 | 0.003 | 128 | 5411 | 20 | 98.97% |
| CNN | adam | pad | depth 3, pool 4, FC 1024→64→7 | dropout 0, WD 0.0001 | 0.003 | 128 | 5411 | 19 | 99.25% |
| CNN | adam | pad | depth 3, pool 4, FC 1024→64→7 | dropout 0.25, WD 0 | 0.003 | 128 | 5411 | 17 | 99.25% |
| CNN | adam | pad | depth 3, pool 8, FC 4096→64→7 | dropout 0, WD 0.0001 | 0.003 | 128 | 5411 | 14 | 99.53% |
| CNN | adam | pad | depth 3, pool 8, FC 4096→64→7 | dropout 0.25, WD 0 | 0.003 | 128 | 5411 | 13 | 99.34% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.003 | 32 | 5411 | 9 | 99.91% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→128→7 | dropout 0, WD 0 | 0.003 | 32 | 5411 | 11 | 99.91% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.001 | 32 | 5411 | 18 | 99.53% |
| CNN | adamw | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0.0001 | 0.001 | 32 | 5411 | 18 | 99.53% |
| CNN | adamw | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0.0001 | 0.003 | 32 | 5411 | 7 | 99.81% |
| CNN | sgd | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.01 | 32 | 5411 | 19 | 99.62% |
| CNN | sgd | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.03 | 32 | 5411 | 11 | 99.72% |
| CNN | sgd | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.1 | 32 | 5411 | 9 | 98.59% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0.0001 | 0.003 | 32 | 5411 | 12 | 99.72% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0.25, WD 0 | 0.003 | 32 | 5411 | 15 | 99.81% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0.25, WD 0.0001 | 0.003 | 32 | 5411 | 14 | 99.81% |
| CNN | adam | resize | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.003 | 32 | 5411 | 15 | 99.53% |
| CNN | adam | pad | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.003 | 32 | 5411 | 11 | 99.25% |
| CNN | adam | stretch | depth 2, pool 8, FC 2048→32→7 | dropout 0, WD 0 | 0.003 | 128 | 5411 | 11 | 99.72% |
| SVM | adam | resize | pixels, cell 8 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 7 | 95.86% |
| SVM | adam | stretch | pixels, cell 8 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 12 | 98.21% |
| SVM | adam | resize | hog, cell 8 | hinge, C=0.1 | 0.01 | 128 | 5411 | 4 | 81.39% |
| SVM | adam | resize | hog, cell 8 | hinge, C=1 | 0.01 | 128 | 5411 | 21 | 93.80% |
| SVM | adam | resize | hog, cell 8 | hinge, C=10 | 0.01 | 128 | 5411 | 16 | 94.92% |
| SVM | adam | resize | hog, cell 8 | hinge, C=100 | 0.01 | 128 | 5411 | 22 | 96.80% |
| SVM | adam | resize | hog, cell 8 | squared_hinge, C=0.1 | 0.01 | 128 | 5411 | 7 | 89.10% |
| SVM | adam | resize | hog, cell 8 | squared_hinge, C=1 | 0.01 | 128 | 5411 | 10 | 93.80% |
| SVM | adam | resize | hog, cell 8 | squared_hinge, C=100 | 0.01 | 128 | 5411 | 20 | 97.74% |
| SVM | adam | resize | hog, cell 4 | hinge, C=0.1 | 0.01 | 128 | 5411 | 1 | 88.82% |
| SVM | adam | resize | hog, cell 4 | hinge, C=1 | 0.01 | 128 | 5411 | 4 | 95.30% |
| SVM | adam | resize | hog, cell 4 | hinge, C=10 | 0.01 | 128 | 5411 | 24 | 97.65% |
| SVM | adam | resize | hog, cell 4 | hinge, C=100 | 0.01 | 128 | 5411 | 25 | 98.78% |
| SVM | adam | resize | hog, cell 4 | squared_hinge, C=0.1 | 0.01 | 128 | 5411 | 1 | 94.08% |
| SVM | adam | resize | hog, cell 4 | squared_hinge, C=1 | 0.01 | 128 | 5411 | 2 | 96.15% |
| SVM | adam | resize | hog, cell 4 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 20 | 98.21% |
| SVM | adam | resize | hog, cell 4 | squared_hinge, C=100 | 0.01 | 128 | 5411 | 26 | 99.25% |
| SVM | adam | pad | hog, cell 8 | hinge, C=0.1 | 0.01 | 128 | 5411 | 2 | 87.50% |
| SVM | adam | pad | hog, cell 8 | hinge, C=1 | 0.01 | 128 | 5411 | 20 | 94.27% |
| SVM | adam | pad | hog, cell 8 | hinge, C=10 | 0.01 | 128 | 5411 | 10 | 95.68% |
| SVM | adam | pad | hog, cell 8 | hinge, C=100 | 0.01 | 128 | 5411 | 27 | 97.27% |
| SVM | adam | pad | hog, cell 8 | squared_hinge, C=0.1 | 0.01 | 128 | 5411 | 7 | 89.66% |
| SVM | adam | pad | hog, cell 8 | squared_hinge, C=1 | 0.01 | 128 | 5411 | 10 | 94.64% |
| SVM | adam | pad | hog, cell 8 | squared_hinge, C=100 | 0.01 | 128 | 5411 | 26 | 98.03% |
| SVM | adam | pad | hog, cell 4 | hinge, C=0.1 | 0.01 | 128 | 5411 | 1 | 91.92% |
| SVM | adam | pad | hog, cell 4 | hinge, C=1 | 0.01 | 128 | 5411 | 1 | 95.21% |
| SVM | adam | pad | hog, cell 4 | hinge, C=10 | 0.01 | 128 | 5411 | 24 | 97.93% |
| SVM | adam | pad | hog, cell 4 | squared_hinge, C=0.1 | 0.01 | 128 | 5411 | 1 | 94.36% |
| SVM | adam | pad | hog, cell 4 | squared_hinge, C=1 | 0.01 | 128 | 5411 | 4 | 96.33% |
| SVM | adam | pad | hog, cell 4 | squared_hinge, C=10 | 0.01 | 128 | 5411 | 16 | 98.50% |
| SVM | adam | stretch | hog, cell 8 | hinge, C=0.1 | 0.01 | 128 | 5411 | 2 | 92.39% |
| SVM | adam | stretch | hog, cell 8 | hinge, C=1 | 0.01 | 128 | 5411 | 32 | 96.05% |
| SVM | adam | stretch | hog, cell 8 | hinge, C=10 | 0.01 | 128 | 5411 | 7 | 97.09% |
| SVM | adam | stretch | hog, cell 8 | hinge, C=100 | 0.01 | 128 | 5411 | 26 | 98.21% |
| SVM | adam | stretch | hog, cell 8 | squared_hinge, C=0.1 | 0.01 | 128 | 5411 | 7 | 93.14% |
| SVM | adam | stretch | hog, cell 8 | squared_hinge, C=1 | 0.01 | 128 | 5411 | 31 | 95.96% |
| SVM | adam | stretch | hog, cell 8 | squared_hinge, C=100 | 0.01 | 128 | 5411 | 19 | 98.78% |
| SVM | adam | stretch | hog, cell 4 | squared_hinge, C=100 | 0.003 | 128 | 5411 | 13 | 99.44% |
| SVM | adam | stretch | hog, cell 4 | squared_hinge, C=100 | 0.03 | 128 | 5411 | 9 | 99.72% |
| SVM | sgd | stretch | hog, cell 4 | squared_hinge, C=100 | 0.001 | 128 | 5411 | 9 | 99.53% |
| SVM | sgd | stretch | hog, cell 4 | squared_hinge, C=100 | 1e-05 | 128 | 5411 | 27 | 98.68% |
| SVM | sgd | stretch | hog, cell 4 | squared_hinge, C=100 | 0.0001 | 128 | 5411 | 30 | 99.44% |
| SVM | lbfgs | stretch | hog, cell 4 | squared_hinge, C=100 | 1 | full | 5411 | 3 | 99.44% |
| SVM | sgd | stretch | hog, cell 4 | squared_hinge, C=100 | 0.01 | 128 | 5411 | None | Numerical failure |

LBFGS uses full-batch updates. One recorded outer epoch allows up to ten quasi-Newton iterations with a strong-Wolfe line search. Adam, AdamW and SGD use the recorded minibatch size. Runtime accompanies validation results.
