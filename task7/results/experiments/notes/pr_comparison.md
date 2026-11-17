# Task 7 PR 对比与报告素材

这份记录保留了两份来源 PR 的提交快照对比。统一实现、当前选模证据和最终测量见 `task7/README.md`、`results/selection.json` 与 `results/task7_results.md`。正式报告的数值表格由统一实验记录生成。

比较日期为 2026-11-17。你的工作对应 PR #2，提交 `486459d95e399dae93bb05453e989faf456e691c`。队友的工作对应 PR #3，提交 `d131b4eecce236678e711baed67327ce12afd3f9`。

- [你的 PR #2](https://github.com/lohxinzhi/robot_vision_project/pull/2)
- [队友的 PR #3](https://github.com/lohxinzhi/robot_vision_project/pull/3)

本次核对包括代码、Notebook 保存的输出、CSV、数据清单、CNN 保存权重和字符裁剪图。没有重新训练两边模型。队友 SVM 的准确率和耗时取自其已保存的实验输出，本次没有执行其 SVM 模型。CNN 的权重推理和测试集指标已在本地复核。

## 1. 数据划分有区别

| 项目 | 你的 PR #2 | 队友 PR #3 |
|---|---:|---:|
| 总图像数 | 7,112 | 7,112 |
| 每类总数 | 1,016 | 1,016 |
| 内部训练 | 4,270，每类 610 | 4,536，每类 648 |
| 验证 | 1,064，每类 152 | 798，每类 114 |
| 最终重训的开发集 | 5,334 | 5,334 |
| 测试 | 1,778，每类 254 | 1,778，每类 254 |
| 验证占开发集 | 约 20% | 约 15% |
| 实验随机种子 | 5411 | CNN 为 42 |

两边满足 75% 开发集、25% 测试集。两边使用完全相同的 7,112 个文件名，但划分不同。测试集的交集是 413 张，占各自测试集约 23.23%。因此报告可以并列记录各自结果，跨 PR 的准确率差异需要同时说明划分差异。

两份清单都有 5,850 个不同像素哈希，清单内部均没有重复哈希跨分区。本次已用你的实际数据核验队友清单的全部 7,112 个哈希，全部匹配。你的哈希包含尺寸和灰度模式前缀，她的哈希直接计算灰度像素字节，所以哈希字符串本身不能逐项直接比较。

你的代码包含重新生成分组划分和训练前重算哈希的流程。她的提交读取已有划分清单，便于固定实验条件。

## 2. 预处理

| 项目 | 你的 PR #2 | 队友 PR #3 |
|---|---|---|
| 输入尺寸 | 32×32 | 32×32 |
| 极性 | 亮字符、暗背景 | 暗字符、亮背景 |
| 裁剪阈值 | 反色后 >51，等价于原灰度 <204 | 原灰度 <245 |
| 选择的模式 | 裁剪、方形补边、缩放 | 裁剪、直接拉伸缩放 |
| 补边候选 | 每侧约 12%，至少 2 像素 | 方形边长为原最长边的 1.20 倍 |
| 缩放 | Pillow bilinear | OpenCV INTER_AREA |
| 预处理实验 | resize / pad | original / crop_stretch / crop_pad |

两个 PR 的 pad 模式也包含不同的阈值、边距和插值规则。你的实验偏向保留纵横比，她的 crop_stretch 把紧裁剪后的字符填满方形输入。

报告可使用她的 `preprocessing_comparison.png` 展示形状变化，使用 `svm_preprocessing_accuracy.png` 对照其验证结果。她的同一组验证数据下，C=1 的 HOG+Linear 得到 Original 99.25%、Crop+Stretch 99.75%、Crop+Pad 99.12%。你的 pad 结果应放在自己的实验条件下讨论。

## 3. Subtask 7-1 CNN

| 项目 | 你的 PR #2 | 队友 PR #3 |
|---|---|---|
| 卷积层 | 1→16→32→64，共三层 | 1→16→32，共两层 |
| 池化 | 两次 max pool，之后 adaptive average pool 到 4×4 | 两次 max pool，特征为 32×8×8 |
| 全连接 | 1024→64→7 | 2048→64→7 |
| 参数量 | 89,351 | 136,391，已从模型计算核验 |
| 正则化 | Dropout 0.25，weight decay 1e-4 | 模型无 Dropout，Adam 默认无 weight decay |
| Batch size | 128 | 32 |
| 学习率候选 | 0.001 / 0.003 | 0.001 / 0.0003 |
| 选模规则 | 最佳验证准确率，保留最早达到的 epoch | 最低验证 loss |
| 最终训练 | pad，lr=0.003，19 epochs | crop_stretch，lr=0.001，10 epochs |
| 测试准确率 | 99.38%，11 个错误 | 99.16%，15 个错误 |
| Macro F1 | 0.99382 | 0.9916 |

她的卷积层数较少，但 flatten 后的全连接输入更大，导致总参数量更大。可以在报告的结构表中明确这个关系。两份 PR 同时改变了结构、预处理、种子、训练规模与预算，因此现有结果不足以把跨 PR 的差异归因到卷积层数。

本次使用她的 checkpoint 和她的测试清单复核，正确 1,763/1,778，与保存结果一致。错误中 H 类占 11 个，其中 9 个被预测为 A。你的 CNN 中 H 类有 6 个被预测为 A。两者可以作为各自误分类分析的内容。

可用图表包括你现有的学习曲线和混淆矩阵，以及她的 `cnn_learning_rate_comparison.png`、`cnn_baseline_errors.png` 和 `cnn_final_test_confusion_matrix.png`。基线验证错误图与最终测试混淆矩阵属于不同阶段，图注需要注明分区。

## 4. Subtask 7-2 SVM

| 项目 | 你的 PR #2 | 队友 PR #3 |
|---|---|---|
| HOG | NumPy 手写 | skimage.feature.hog |
| 分类器 | Torch 优化的线性 one-vs-rest squared-hinge SVM | sklearn.svm.SVC |
| 分类器比较 | 固定线性 SVM，研究 C 和 HOG cell | 原始像素/HOG × Linear/RBF |
| C 调查 | 1、10，后续 100 | 0.01、0.1、1、10 |
| HOG cell | 基线 8×8，最终 4×4 | 8×8 |
| HOG 维数 | 324→1,764 | 324 |
| 最终配置 | pad，C=100，cell=4 | crop_stretch，Linear，C=10 |
| 测试准确率 | 99.55%，8 个错误 | 99.38%，11 个错误 |
| Macro F1 | 0.99550 | 0.9938 |

同为线性 SVM，两边具体实现和目标函数存在区别，C 数值不适合直接比较。你的实现能展示损失函数、梯度优化和从 324 到 1,764 维特征的变化。她的实验提供更广的特征与核函数调查。

她在同一验证集上的基线结果为 Raw+Linear 98.25%、Raw+RBF 98.25%、HOG+Linear 99.25%、HOG+RBF 99.00%。加入 crop_stretch 后，C 从 0.01、0.1、1、10 变化时，验证准确率为 97.12%、99.00%、99.75%、99.87%。

你的顺序试验保留了基线 C10/cell8、C100/cell8 和 C100/cell4，验证准确率依次为 96.52%、97.93%、99.53%，芯片第二位的 D 在最后一轮纠正。芯片图参与了这轮修改的反馈，报告中应保留开发阶段结果的说明。

她的 `svm_C_tuning.png`、`svm_preprocessing_accuracy.png`、`svm_all_misclassified_samples.png` 能补充报告中的参数敏感性和误分类实例。你的 `refinement_comparison.md` 与 baseline/iterations 记录能补充针对具体错误逐步修改的过程。

## 5. 芯片识别和 Task 6

她比较的 Pipeline A/B 是两套不同的分割及预处理输出。你的 4/8 是同一分割流程的两种连通定义。这两种比较各自回答不同的问题。

本次已重新生成你的 Task 6 十张字符 mask，并逐像素比较她提交的 `task6_teammate_characters/character_01.png` 至 `character_10.png`，十张全部一致。因此她的 Pipeline B 确实使用了你当前的 Task 6 输出。

| 提交与模型 | Pipeline A | Pipeline B，即你的 Task 6 输出 |
|---|---|---|
| 她的 CNN | H044780A00，90% | HD44780A00，100% |
| 她的 SVM | HD4478DAD0，80% | HD44780ADD，80% |
| 你的 CNN | 未记录此输入实验 | HD44780A00，100% |
| 你的改进 SVM | 未记录此输入实验 | HD44780A00，100% |

你的 4/8 模式均达到上述同一预测。她的 CNN 两套芯片输入的结果已在本次本地推理中重现。她的 SVM 一行取自保存的 Notebook 输出与 CSV。

报告可以利用她的 `task6_segmentation_comparison.png` 展示两套 mask，再使用 `cnn_chip_comparison.png` 和四组合 CSV 呈现分类结果。她的 CNN 在 Pipeline B 中第二位从 0 变成 D，可以作为输入流程变化对应预测变化的观察。由于 A/B 同时改变了多项处理，不能把变化单独归因于某一个分割算子。

你们的 CNN 在自己字符测试集中略低于各自 SVM，但在这张芯片图上呈现不同结果。她的 SVM 尤其保留了较高字符测试成绩与 80% 芯片成绩的差异，可供小组讨论数据域、裁剪和归一化的影响。十个字符只代表这一个开发样例。

## 6. 效率与可复现性

你的记录包含 CPU 四线程、预热后的批量推理、model-only 以及 HOG+model 时间。CNN model-only 为 0.10678 ms/image，最终 SVM model-only 为 0.00123 ms/image，HOG+model 为 0.55322 ms/image。

她的 SVM 记录 predict 为 0.0186 ms/image，特征在计时前已算好。她的 CNN 测试总时间 1.236 s 包含 DataLoader 读取图像及预处理。她的最终 CNN 训练时间为 23.53 s，SVM fit 为 0.129 s，后者未包含 HOG 特征构建。两份 PR 及两类模型的硬件、批大小和计时边界都有区别。

报告中可使用你的测量说明公平计时的边界。她的计时可以作为原实验记录呈现，和你的数据并列时应说明条件。

你的 PR 包含完整 dataset、独立运行脚本、checkpoint、环境 YAML，以及从原始芯片图到结果的调用流程。她的 PR 包含较丰富的 Notebook 叙述、已保存的模型、两套裁剪图和推理校验脚本，完整重训需要另行提供 dataset。她的两套 Task 6 原始代码在这个 PR 中未附带。

## 7. 报告可以如何组织

1. 数据与评估协议。统一说明 7,112 张七类数据，随后分别列两组划分。保留来源和每组实验条件。
2. CNN。把两套结构及参数量放入结构表。各自呈现学习率实验、选择规则、测试混淆矩阵和误分类实例。
3. 非 CNN。加入她的像素/HOG、Linear/RBF、三类预处理与 C 扫描，再呈现你固定线性 SVM 后的 C100 和细网格 HOG 调查。
4. 芯片实验。呈现她的 A/B 两套分割对照，注明 B 与你的 Task 6 mask 完全一致，再列两个模型在这些输入上的已记录输出。
5. 效果与效率。并列四个最终分类器的成绩并注明各自划分，讨论参数量、特征构建和推理时间的测量边界。
6. 小组讨论。基于对应图像和错误记录，解释特征、预处理和输入域如何影响结果。组员核对并撰写自己的分析结论。

如需统一比较，可以先固定双方测试集的 413 张交集，在保持权重和设置固定的情况下重新计算四个模型的成绩，并说明这个共同子集的类别数量。更大范围的统一实验也可以从一致划分重新训练。直接拿其中一方完整测试集评估另一方模型，会遇到后者已经用过其中许多图片的问题。

当前仅新增此对比说明。正式 LaTeX 报告还没有整合队友的实验和图表。

## 原始证据入口

- [PR #2 方法与运行说明](https://github.com/FFraankk/robot_vision_project/blob/486459d95e399dae93bb05453e989faf456e691c/task7/README.md)
- [PR #2 结果](https://github.com/FFraankk/robot_vision_project/blob/486459d95e399dae93bb05453e989faf456e691c/task7/results/task7_results.md)
- [PR #2 SVM 顺序修改](https://github.com/FFraankk/robot_vision_project/blob/486459d95e399dae93bb05453e989faf456e691c/task7/results/refinement_comparison.md)
- [她的结果说明](https://github.com/sisiW-0115/robot_vision_project/blob/d131b4eecce236678e711baed67327ce12afd3f9/task7/RESULTS.md)
- [她的 CNN Notebook](https://github.com/sisiW-0115/robot_vision_project/blob/d131b4eecce236678e711baed67327ce12afd3f9/task7/notebooks/task7_cnn.ipynb)
- [她的 SVM Notebook](https://github.com/sisiW-0115/robot_vision_project/blob/d131b4eecce236678e711baed67327ce12afd3f9/task7/notebooks/task7_svm.ipynb)
