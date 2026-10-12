# BBBC045 四分类测试与标签异常记录

初建日期：2026-10-11；状态：**待核查，不能判定原始标签错误**。CRF022、CRF049 的亮度诊断、三折增强对照及两档压力测试已统一移至[亮度实验汇总](bbbc045_brightness_findings.md)；本文件只记录测试错分、图块来源与标签核查边界。

## 复现依据

- 模型：`runs/resnet18_four_imagenet_flips_20261011_143841/checkpoints/best.ckpt`。
- 测试输出：同一运行目录下的 `test_evaluation/test_metrics.json`、`test_predictions.csv`、`confusion_matrix_counts.png` 和 `misclassified_examples.png`。
- 图块定位：`test_predictions.csv` 中的 `packed_index` 对应 `data/processed/bbbc045_split.csv` 的第 `packed_index` 行（从 0 起）。原始 TIFF 位于项目外的 `../Stained_Montages/`。
- 测试样本 15,290 个；accuracy 0.9629，macro-F1 0.8377。单核细胞 F1 0.4870，召回率 216/588 = 0.3673。

## 集中错分现象

已补存该运行的逐图块错分清单：`runs/resnet18_four_imagenet_flips_20261011_143841/test_evaluation/test_misclassified.csv`（测试集 568 条）和 `runs/resnet18_four_imagenet_flips_20261011_143841/validation_errors/best_misclassified.csv`（最佳权重下验证集 239 条）。两份文件记录原始 TIFF 路径、网格位置、原始类别、模型判断和置信度；验证文件只是最佳权重的一次快照，不能还原旧训练每轮的错分。新训练会另存逐轮验证错分与 `frequency.csv` 重复次数统计。

| 受试者及原始目录类别 | 图块数 | 判对单核 | 判为淋巴 | 判为嗜酸 | 判为中性 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CRF049/monocyte | 376 | 8 | 285 | 80 | 3 |
| CRF066/monocyte | 212 | 208 | 3 | 0 | 1 |

CRF049 的 `monocyte/batch1/ch5_1.tif`：320/320 错分（237 淋巴、80 嗜酸、3 中性）；`monocyte/batch2/ch5_1.tif`：48/56 错分，均判为淋巴。CRF049 错分的 368 个单核图块中，183 个预测类别的置信度至少为 0.9。可从以下高置信度候选开始人工复核；这些是**候选**，不是已确认的错标：

| packed_index | 原始 TIFF 后缀 | row | col | 预测类别 | 置信度 |
| ---: | --- | ---: | ---: | --- | ---: |
| 15140 | `CRF049/monocyte/batch1/ch5_1.tif` | 5 | 8 | eosinophil | 0.9969 |
| 15358 | `CRF049/monocyte/batch1/ch5_1.tif` | 17 | 10 | lymphocyte | 0.9940 |
| 15385 | `CRF049/monocyte/batch2/ch5_1.tif` | 2 | 7 | lymphocyte | 0.9999998 |

## 裁切与索引复核（2026-10-11）

对 `CRF049/monocyte` 的全部 376 条清单记录（batch1 320 条、batch2 56 条）逐一从原始 `ch5` TIFF 按 `row × 55`、`col × 55` 裁出图块，与 `data/processed/bbbc045_tiles.npy[packed_index]` 做逐像素相等比较：**376/376 完全一致，0 个不匹配**。清单与打包数组均有 100,105 条；这 376 条的目录类别、受试者、test 划分均一致，文件与网格位置组合无重复。测试推理代码按同一个 CSV 行号读图，并检查预测顺序和标签对应。因而目前没有发现项目在裁切、打包、CSV 索引或测试预测关联时把这批图块放错位置的证据。该检查不能证明原始数据目录的生物学标签正确，也不能排除图像质量问题。

## `.cif` 逐事件像素复核（2026-10-11）

读取官方 `Stained_Populations/CRF049/monocyte/CRF049_monocyte_{1,2}.cif`。文件元数据所记来源分别是 `CRF049-1-CD14-mono_4.rif` 和 `CRF049-2-CD14-mono_5.rif`，事件数分别为 320、56。参照 Bio-Formats 的 FlowSight 解码规则还原每个事件的 16 位图像，取第 5 通道，以 `像素值 / 65535` 转为 TIFF 使用的 `float32`，按事件顺序与相应 `ch5_1.tif` 的有效格逐一比对。由于拼图对较小图像填充背景、对较大图像裁剪，仅比较保留下来的原始细胞像素；在每个 55×55 格内搜索其精确位置，避免填充偏移造成误报。结果：**batch1 320/320、batch2 56/56，共 376/376 逐像素完全相同；每个事件在对应格中都只有一个精确匹配位置**。其中 2 个事件的原图宽或高超过 55，只比较被保留的裁剪区域；新增的填充背景不计入比对。

这进一步排除了该批 TIFF 与 `.cif` 之间的事件顺序／图像错位，但不能单凭文件名或图像一致性证明官方 CD14 门控对每个事件的生物学判定无误。

## 解释边界与后续核查

CRF049 的两个单核批次都集中出错，但目前只有这一份四分类权重的逐细胞测试预测；`best.ckpt` 与 `last.ckpt` 内容相同，未保存逐轮预测。因此不能说“同一个图块在每一轮都被错分”。跨受试者成像差异、难以从明场辨认的形态、荧光门控标签噪声，以及原始数据中荧光与明场通道的对应关系仍是待排查假设；亮度方面已完成的证据和限制见[亮度实验汇总](bbbc045_brightness_findings.md)。本项目的图块索引／裁切已通过上述逐像素检查。

先核对上述原始 TIFF 的位置和对应的荧光／暗场图像、门控信息；必要时请熟悉细胞形态的人独立复核。不得凭模型预测自动修改标签或删除测试样本。此次测试集已被查看，后续若据此调整模型，应另用未触碰的受试者作最终独立评估。

BBBC045 官方说明其目录标签源于荧光通道门控，并提醒部分图像是不完整细胞：[数据集说明](https://bbbc.broadinstitute.org/BBBC045)。原始论文的四类主类型实验在 13 名受试者上做按受试者交叉验证，使用明场和暗场形态特征、质量筛选及随机欠采样，报告约 97% 平均 F1：[Nassar 等，2019](https://carpenter-singh-lab.broadinstitute.org/files/anne/files/123_Nassar_CytometryA_2019.pdf)。本项目测试 macro-F1 为 83.77%，与论文数字表面相差约 13.2 个百分点，但输入通道、质量筛选、划分和指标汇总方式不同，**不能作为严格的模型优劣差值**。
