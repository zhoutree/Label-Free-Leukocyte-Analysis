# 完整运行与诊断指南

以下命令均从项目根目录执行。先按 [README](../README.md#从零准备环境和数据) 安装环境、下载数据并生成处理文件。本页保留详细参数与历史诊断命令；旧运行目录示例需替换为自己实际保存的结果路径。

## 当前数据

- `data/processed/bbbc045_tiles.npy`：100,105 个 55×55 明场图块；第 `i` 个图块对应清单第 `i` 行。新准备的数据会同时生成 `bbbc045_tiles.npy.meta.json`，记录图块内容及打包时的行顺序指纹。
- `data/processed/bbbc045_manifest.csv`：图块来源、受试者、类别和网格位置。
- `data/processed/bbbc045_split.csv`：在清单基础上增加 train/val/test 划分。清单始终保留原始 B/T 标签；训练加载时才映射为四类，不需重新裁切或打包。
- `CRF101_2014` 和 `CRF101_2015` 共用 CRF101 编号；官方资料尚未明确核实是否同一人，因此保守地视作同一划分组。当前固定划分中两者均在训练集。今后的常规训练即使用自定义 `--split-csv`，启动前也会拒绝把它们分到不同的 train/val/test 侧；交叉验证同样保持同折。
- `data/processed/bbbc045_train_stats.json`：仅由训练集计算的归一化参数；新生成的文件也记录清单和图块包的指纹。新统计文件若依赖图块旁的 `.meta.json`，缺失它会直接报错；旧包没有顺序元数据时仍可使用，但会提示降级。此时仍可核对图块规格、行数和统计文件所记录的当前输入指纹，不能事后证明图块与原清单的打包顺序一致。
- `reports/bbbc045_audit.json`：此前的基础审计；`reports/bbbc045_audit_focus.json`：新版审计，以每类训练图块的清晰度代理分数第 1 百分位标记 train/val 低细节候选，不自动删除样本，也不能确认多细胞或错标；`reports/previews/`：预览图。

## 运行

在项目根目录激活 `label-free-leukocyte` 环境后，先做不保存权重的冒烟测试：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m leukocyte_analysis.train --smoke-test --batch-size 8 --init random
```

确认无误后运行当前默认配置的完整训练（最多 45 轮，可用参数覆盖）：

```powershell
python -m leukocyte_analysis.train
```

完整训练默认加载 torchvision 的 ImageNet-1K ResNet-18 权重，优先复用项目 `data/pretrained/` 中的缓存；缓存不存在时需要联网下载。若已设置 `TORCH_HOME`，则尊重该位置。单通道 3×3 首层按 Kaiming 方式初始化，四分类头使用 PyTorch 默认初始化，其他层保留预训练参数并全部参与微调。常规训练默认学习率为 `0.0001`，最多训练 45 轮，AdamW 的 `weight_decay=0.01`（不是 L1 正则化）。合并 B/T 后，类别权重由四类训练集数量重新计算，不沿用旧五分类权重。

默认 `--augmentation flips_photometric_geometric`：只对训练图块做随机水平／垂直翻转、随机 0°／90°／180°／270° 旋转，并以 50% 概率再做 ±15° 旋转、水平／垂直各最多 4%（55 像素图块约 2 像素）的平移及 0.95–1.05 倍缩放；小角度变换采用双线性插值，不做弹性形变。此前的 torchvision `ColorJitter` 仍对原始灰度施加 0.8–1.25 倍亮度、0.9–1.1 倍对比度；另以各 20% 概率使用轻微 3×3 高斯模糊（σ=0.2–0.6 像素）和高斯噪声（原始像素尺度 σ=0.00002），再按训练集均值和标准差归一化。验证与测试集不增强。旧增强保持原样：`--augmentation flips_photometric` 仅翻转加亮度／对比度，`--augmentation flips` 仅翻转。该组合配置已有三折和原保留测试集结果，但同时改变了多项训练设置，不能将成绩单独归因于某一种增强、学习率调度或标签平滑。默认 `--early-stopping-patience 15` 与 `--early-stopping-min-delta 0`：验证 macro-F1 连续 15 次检查未创新高就停止；耐心值设为 `0` 可禁用早停。余弦方案至少训练 15 轮（若总轮数不足 15 则训练到上限）。早停不保证提高最佳分数。训练循环、Rich 终端进度条、日志和检查点由 Lightning 管理；macro-F1 与各类召回率由 TorchMetrics 计算。每次训练新建 `runs/resnet18_four_初始化方式_增强方式_调度方式_时间戳/`，其中 `csv/metrics.csv` 是逐轮指标，`tensorboard/` 是曲线日志，`checkpoints/best.ckpt` 是按验证集 macro-F1 选出的权重。`last.ckpt` 是最近一次实际保存的检查点，可能与 `best.ckpt` 相同。测试集不参与训练或选择权重。
常规训练还会在该运行目录写入 `data_split.json`，记录所用清单、CRF101 合并分组规则、每组目录标识及 train/val/test 归属和图块数量；这只记录划分，不复制原始图像。

默认 `--lr-scheduler plateau`：验证 macro-F1 停滞时降低 AdamW 学习率，`patience=3`、`cooldown=2`、每次乘以 0.5，最低为初始学习率的 1%；与早停 `patience=15` 搭配，给降学习率后的模型留出观察时间。Lightning 每轮记录实际学习率。`--lr-scheduler cosine` 仍可选择：前 2 轮线性预热，随后按轮次余弦下降到初始值的 1%；`--lr-scheduler none` 可运行固定学习率对照，**不会叠加两种调度器**。标签平滑默认 `0.05`；可用 `--label-smoothing 0` 关闭，验证指标和 `val_loss` 不作平滑。`train_loss` 与 `train_weighted_loss` 保留未平滑口径用于旧实验对照，新增 `train_objective_loss` 显示实际优化目标。此前 Plateau 与余弦三折的折外指标十分接近，余弦并未显示稳定优势；当前命令行默认早停最小改善量为 0，与最近完成的组合配置一致。光度增强范围曾参考旧验证异常，因此同折比较属于探索性分析，不是全新独立测试。

当前只提供四分类训练与评估入口；旧五分类权重即使存在，也不能直接用于四分类。早停主要减少最佳验证轮次之后的无效训练，并不保证提高最佳分数。亮度审计、增强对照与压力测试的结果和解释统一见[亮度实验汇总](bbbc045_brightness_findings.md)；测试错分及标签来源核查见[测试与标签异常记录](bbbc045_test_audit.md)。

终端使用 Lightning 的 `RichProgressBar` 显示批次进度，并按实时窗口宽度逐级收起指标：窄窗口只保留轮次、横条和批次；中等宽度加入剩余时间及简短的 `loss`、`val_f1`；宽窗口再显示 `val_loss`。完整指标始终写入日志。极窄窗口或训练时频繁缩放仍可能造成显示残影。首次使用前，在已激活的环境中执行 `python -m pip install rich`；以后从更新后的 `environment.yml` 重建环境时会自动安装。训练时或训练后可另开终端运行：

```powershell
tensorboard --logdir .\runs
```

然后打开终端显示的本地地址（通常为 `http://localhost:6006/`）查看 `train_loss`、`val_loss`、`val_macro_f1`、各类 `val_recall_*` 和逐轮学习率 `lr-AdamW`。`train_loss` 和 `val_loss` 是未加权交叉熵；`train_weighted_loss` 是未平滑加权交叉熵。启用标签平滑后，实际更新参数的损失见 `train_objective_loss`，不要直接拿它的数值与未平滑的 `val_loss` 比较。指标每轮才更新一次，第一轮结束前不会出现曲线。按 `Ctrl+C` 可以中断；完整训练的检查点包含优化器状态，但当前命令行暂未提供断点续训参数，重新执行会新建运行目录。

若要进行随机初始化的对照实验，使用 `python -m leukocyte_analysis.train --init random`。也可通过 `--weight-decay` 显式改变权重衰减。冒烟测试命令使用随机初始化，以免仅为检查训练流程就下载预训练权重。

`environment.yml` 让 Lightning、TorchMetrics 等普通依赖由 pip 自行选择兼容版本；只固定 Python 3.14 和已验证可使用 GPU 的 PyTorch/torchvision CUDA 13.2 配对，以免重建环境时意外改为 CPU 版本。

## 按受试者交叉验证

交叉验证只使用原 train＋val 的 11 个目录标识，默认分为 3 折。`CRF101_2014` 与 `CRF101_2015` 共用 CRF101 编号；官方资料未明确核实两者是否同一人，因此**今后的分折保守地把两者视为同一组**（可用分组共 10 个），绝不分放训练和验证侧。每折从 ImageNet 权重重新初始化模型，并仅用该折训练图块重算像素均值、标准差和类别权重。CRF022 正常参与交叉验证；**CRF049 和 CRF066 继续保持原测试划分**，只是为了让下一次实验与旧折外基线可比，**不是将它们认定为坏数据或错标而筛除**。折内 CSV 保留全部原行及索引，但测试受试者标为 `excluded`，不会被训练或验证数据加载器读取。旧运行的 `plan.json` 和指标仍按原分折解释，不与新分折成绩直接作受控对比。

当前没有针对 `CRF049/monocyte` 的额外质量屏蔽规则；高错分率和亮度敏感性都不能证明官方标签错误。若今后决定改变受试者划分，可将 CRF049 作为一个完整受试者纳入新训练／验证方案，并从新的训练集重新计算归一化参数和类别权重；但那将是**新的数据划分**，不能与旧 11 人折外成绩直接比较，也不能再把已多次查看的 CRF049 当作未使用过的独立测试集。

先检查第一折的两批训练和验证（不保存权重）：

```powershell
python -m leukocyte_analysis.cross_validate --smoke-test --init random --batch-size 8
```

再运行按新 CRF101 同组规则分折的完整实验；每折默认最多 30 轮，耗时可能明显增加，不会由上述冒烟命令自动启动。若想复现最近完成的组合配置，命令如下；其他历史实验命令仍保留其当时的参数：

```powershell
python -m leukocyte_analysis.cross_validate --folds 3 --init imagenet --lr 0.0001 --epochs 30 --batch-size 128 --augmentation flips_photometric_geometric --lr-scheduler plateau --early-stopping-patience 15 --early-stopping-min-delta 0 --label-smoothing 0.05
```

标签平滑三折实验使用已验证过的翻转＋亮度／对比度方案，不叠加本次第 1 折未带来收益的几何／模糊／噪声增强；`--early-stopping-min-delta 0` 与此前同分折 Plateau 基线保持一致。系数 0.05 是比白细胞染色 RGB 图研究中常见的 0.1 更保守的起点，**不是 BBBC045 上已验证的最优值**：

```powershell
python -m leukocyte_analysis.cross_validate --folds 3 --init imagenet --lr 0.0001 --epochs 30 --batch-size 128 --augmentation flips_photometric --lr-scheduler plateau --early-stopping-patience 15 --early-stopping-min-delta 0 --label-smoothing 0.05
```

不加 `--first-fold-only` 就会一次运行完整三折；`plan.json` 和检查点会保存平滑系数。比较时以三折折外准确率、macro-F1、单核细胞 precision／recall 和各受试者指标为主，不把训练损失变大或预测置信度变低本身当作退步。原测试受试者 CRF049、CRF066 仍不参与本次参数选择。

最近完成的三折组合实验同时启用了统一 `0.05` 标签平滑与前述几何／模糊／噪声增强，命令如下；这也是当前三折入口的默认配置：

```powershell
python -m leukocyte_analysis.cross_validate --folds 3 --init imagenet --lr 0.0001 --epochs 30 --batch-size 128 --augmentation flips_photometric_geometric --lr-scheduler plateau --early-stopping-patience 15 --early-stopping-min-delta 0 --label-smoothing 0.05
```

这同时改变了增强和平滑两个因素，适合检验组合方案的总效果，但无法单独归因于其中一项。

若想先完整评估第 1 折，再决定是否训练剩余两折，在同一命令末尾加 `--first-fold-only`。完成后会输出第 1 折的准确率、macro-F1 与结果目录；这不是三折折外成绩。确定继续时，用它打印的目录运行 `python -m leukocyte_analysis.cross_validate --continue-run "结果目录"`：程序按保存的 `plan.json` 复用同一划分和训练配置，只训练第 2、3 折，最后生成完整 `summary.json`。不要改动第一折的目录或把 `--smoke-test` 当作完整训练；若某折训练中途被打断，程序不会覆盖该折已有文件。

新运行的 `plan.json` 和最佳检查点引用采用相对于运行目录的路径；只要数据与运行目录的相对布局不变，整体迁移后仍能解析。旧运行记录中的绝对路径仍可读取。若迁移后旧路径失效，继续交叉验证时加 `--relocate-inputs --split-csv 新清单 --tiles-npy 新图块`；单折续训可加 `--split-csv`／`--tiles-npy`，亮度压力测试可加 `--tiles-npy`。迁移时必须使用原始同一份清单和图块，不要借此更换实验数据；新生成的统计文件与图块元数据可核对指纹，旧记录若没有文件哈希，程序无法仅凭路径证明新图块内容完全相同。

如需在**同一新分折**下比较固定学习率，其他设置不变、仅关闭调度：

```powershell
python -m leukocyte_analysis.cross_validate --folds 3 --init imagenet --lr 0.0001 --epochs 30 --batch-size 128 --augmentation flips_photometric_geometric --lr-scheduler none
```

每次新建 `runs/cv_resnet18_.../`。`plan.json` 记录受试者分折，`fold_*/data/` 保存该折划分和训练集统计，`fold_*/training/` 保存日志及最佳检查点，`fold_*/validation_predictions.csv` 保存该折预测；`summary.json` 汇总折外准确率、macro-F1、各类指标和各受试者指标。这些验证结果用于比较训练方案；CRF049/CRF066 已经被查看过，不能再把它们当作新的独立最终测试。

若仅要检查本次 Plateau 实验中**第 2 折**是否被 20 轮上限截断，可从其 `last.ckpt` 续训到**总计最多 30 轮**；第 1、3 折不重跑，原运行目录保持不变：

```powershell
python -m leukocyte_analysis.resume_cv --source-run ".\runs\cv_resnet18_imagenet_flips_photometric_plateau_20261012_204944" --fold 2 --epochs 30
```

续训结果进入新建的 `runs/cv_fold02_extended_.../`：保留旧折预测，比较第 2 折旧／新最佳权重的验证 macro-F1，并重新汇总折外指标。`--epochs 30` 是总轮数上限，不是再加 30 轮；早停仍生效。由于 DataLoader 的独立随机生成器未保存在旧检查点，续训不保证与一次性训练 30 轮逐位一致。这个延长轮数的分析仍使用已反复查看的交叉验证数据，属于探索性比较。

两组交叉验证训练完成后，可用已有 `best.ckpt` 做**亮度压力测试**，不重新训练、不读取原测试受试者：

```powershell
python -m leukocyte_analysis.brightness_stress `
  --baseline-run .\runs\cv_resnet18_imagenet_flips_20261011_213201 `
  --augmented-run .\runs\cv_resnet18_imagenet_flips_photometric_20261012_111752 `
  --output-dir .\runs\brightness_stress_20261012
```

默认将每折验证图块的原始灰度值分别乘以 0.8、1.0、1.2，随后使用**该折训练集**统计量归一化；1.0 保持原始推理路径。工具会核对两次运行的受试者分折和其他训练参数一致，并拒绝把 CRF049/CRF066 纳入诊断。输出 `overall.csv`、`per_class_recall.csv`、`by_donor.csv`、两张曲线图和设置记录。报告目录必须不存在，避免覆盖既有结果。此测试只衡量既有验证集上的亮度敏感性：亮度区间和增强方案都曾参考验证异常，结果仍是**探索性证据**，不是全新独立测试；若 1.0 的指标与各次运行的 `summary.json` 明显不同，应先核对数据与检查点版本。

为判断 ±20% 扰动是否贴近实际采集差异，可先做**只读的真实亮度分布审计**。默认只统计交叉验证的 11 名受试者，不读取 CRF049/CRF066：

```powershell
python -m leukocyte_analysis.audit_brightness
```

若要把**原 train、val、test 三组全部纳入同一份描述性报告**，显式使用：

```powershell
python -m leukocyte_analysis.audit_brightness --include-test
```

默认分别新建 `reports/bbbc045_brightness_distribution/` 或 `reports/bbbc045_brightness_distribution_all_splits/`，互不覆盖。`tile_metrics.csv` 保存逐图块整图均值、外侧 5 像素环中位数和 P90−P10；`group_summary.csv` 按原划分／受试者／原始类别／batch 汇总，`split_summary.csv` 按原划分汇总，`within_donor_batch_pairs.csv` 给出同一受试者同一类别的批次比值，`largest_batch_shifts.png` 显示样本充足组中差异最大的批次。**边缘环中位数只是背景亮度代理**：细胞触边时也会受细胞影响；强度差异也可能来自细胞占图比例，不能单凭这些统计判定光照或标签错误。含测试集的输出仅用于事后排查，不改变原有 train/val/test 划分、图块或模型权重；已经查看过 CRF049/CRF066，不能用其统计结果挑选预处理后仍把它们称为全新独立测试集。

## 最终测试评估

以下命令展示如何用新四分类训练的 `best.ckpt` 评估原保留测试受试者；模型配置和权重应在训练／验证侧确定，评估入口会拒绝旧五分类检查点。该保留集已被多次事后分析，再次运行不能称为全新独立测试：

```powershell
$checkpoint = Read-Host "请粘贴选定实验的 best.ckpt 完整路径"
python -m leukocyte_analysis.evaluate --checkpoint $checkpoint
```

测试集批量推理由 Lightning 的 `Trainer.predict()` 管理；分类指标和混淆矩阵交给 scikit-learn。默认在该实验目录下新建 `test_evaluation/`，生成 `test_metrics.json`、各类别 precision/recall/F1 的文本与 CSV 表、逐细胞预测 CSV、**仅含错分的 `test_misclassified.csv`**、计数及按真实类别归一化的混淆矩阵，以及每类最多三张高置信度错分示例图。错分 CSV 包含 `packed_index`、原始 TIFF 相对路径、受试者、网格行列、原始标签、模型预测与置信度；方便回到原图复核，不能直接当作错标证据。示例图的 1–99% 逐图对比度拉伸**只用于观看**，不影响推理。若报告目录已存在，命令会拒绝覆盖，避免误把多次测试当成独立验证。该测试集仅含两名保留受试者，指标不能替代跨设备、跨实验室的外部验证。

新的完整训练会用 Lightning 回调在 `validation_errors/epoch_000.csv` 等文件保存**每一轮**的验证错分；`validation_errors/frequency.csv` 统计同一 `packed_index` 错分的轮数、轮号和被判成的类别。轮号从 0 开始。冒烟测试不写这些文件。旧训练没有逐轮预测，无法事后恢复每轮错分；只能用其 `best.ckpt` 对验证集得到一份最佳权重的快照：

```powershell
$run = ".\runs\resnet18_four_imagenet_flips_20261011_143841"
python -m leukocyte_analysis.record_errors --split val --checkpoint "$run\checkpoints\best.ckpt" --output-csv "$run\validation_errors\best_misclassified.csv"
```

对于旧测试报告，不再次运行测试推理，而是从**已有**的预测 CSV 补出可定位的错分记录：

```powershell
python -m leukocyte_analysis.record_errors --split test --predictions-csv "$run\test_evaluation\test_predictions.csv" --output-csv "$run\test_evaluation\test_misclassified.csv"
```

记录程序拒绝覆盖已有目标文件；人工核查时不要根据测试集错分调整模型后再把同一测试集称为独立最终测试。


## 特征与质量复核

此入口处理原始图块，待定判断只使用 RMD，独立检查图片质量。分类置信度仅作为结果信息保存，不参与拒识。使用当前四分类权重，不重新训练 ResNet。仅运行本文命令才会启动图块推理；训练侧和验证侧的完整前向提取仍需时间。

先安装可选依赖；作者 CUDA 环境的 `environment.yml` 也包含同一固定版本：

```powershell
python -m pip install -e ".[review]"
```

使用常规训练的最佳检查点和它对应的数据三件套。下面默认读取 `data/processed/`；交叉验证检查点须显式传入同折 `--split-csv`、`--stats-json` 及对应 `--tiles-npy`，不能混用其他折的归一化或参考分布。

```powershell
$checkpoint = Read-Host "请粘贴四分类最佳检查点 best.ckpt 路径"
python -m leukocyte_analysis.review_cells fit `
  --checkpoint $checkpoint --known-reject-rate 0.01 `
  --output-dir .\reports\cell_review_reference
```

`0.01` 是示例工作点：在完整已知类验证数据上，以约 1% 为误拒目标。训练侧仅拟合四类特征中心和协方差，验证侧只选择 RMD 上分位数，不更新网络参数；分数越大越偏离，**严格大于阈值**才送复核。并列分数整体保留，实测误拒可低于目标。此目标不是未知类检出率，也不保证部署时的误拒率。验证集已用于模型选择，结果仍属探索性校准。

新目录输出：

- `rmd.pt`：只包含张量统计量、训练归一化、类别顺序、检查点摘要与工作点；不保存额外的分类模型。
- `validation_scores.csv`：完整验证图块的候选类别、置信度、RMD 与清晰度评分，行号可回溯原图。
- `summary.json`：来源摘要、训练／校准数量、总体及各类误拒计数与限制。

质量检查沿用中央区域、固定 `sigma=0.8` 平滑后计算拉普拉斯方差，并新增 Sobel 梯度能量 `tenengrad`。规则使用 `normalized_tenengrad`，即梯度能量除以原始中央区域灰度方差；低值表示细节少，对纯亮度偏移及线性对比度缩放更稳定，但仍受噪声、天然低纹理和成像条件影响。该归一化只用于质量评分，不修改分类模型输入；常数图评分为零，单独标记 `no_contrast`。

快速原型可直接复用上一步完整验证评分校准质量下限，无需重新推理：

```powershell
python -m leukocyte_analysis.review_cells calibrate-quality `
  --validation-scores .\reports\cell_review_reference\validation_scores.csv `
  --split-csv .\data\processed\bbbc045_split.csv `
  --quality-reject-rate 0.01 --output-dir .\reports\cell_quality_threshold
```

输出 `quality_threshold.json`。每类按 `floor(0.01 × 样本数)` 的顺序统计量选择下限，再取四类最小值，作为不依赖推理类别的统一阈值；严格低于阈值才拦截，并列保留。因此每类**校准数据的经验拦截比例**不超过 1%，但没有正式质量标签，不能称为实测误判废率或失焦检出率，也不保证新数据上的比例。阈值不通过最大化剩余图片的四分类 F1 选择；原始成绩不变。报告记录来源摘要、各类拦截数与限制，拒绝覆盖旧目录。

`--split-csv` 必须是生成评分时使用的完整划分清单，默认是上述常规清单；折内评分应显式传入对应折的清单。程序要求评分含 `packed_index`、`donor`、`true_index`、`normalized_tenengrad`，并核对是否完整且不重复地覆盖清单验证集、受试者和标签是否一致；拒绝训练／测试索引及原保留测试受试者参与校准，CRF101 两年份也不得跨组。新参考额外保存来源清单路径和摘要；旧质量参考仍可读取，无需覆盖或重算旧结果。文件名叫“validation”本身不再作为来源证明。

人工质量标注和真实失焦验证可在后续补充；模拟模糊不能代替真实失焦验证，同一原图与变体须保持同组。此工作点只是保守工程筛选，不证明被拦截图片确实不可用。

对无标签图块应用：

```powershell
$incoming = Read-Host "请粘贴待复核的原始灰度图块 NPY 路径"
python -m leukocyte_analysis.review_cells predict `
  --checkpoint $checkpoint --reference .\reports\cell_review_reference\rmd.pt `
  --tiles-npy $incoming --output-dir .\reports\cell_review_predictions `
  --quality-reference .\reports\cell_quality_threshold\quality_threshold.json
```

输入必须为非空 `N×55×55 float32` 数组，保持与 BBBC045 原始灰度相同的数值单位；不能使用截图、显示拉伸图或直接把 0–255 灰度图转成 float32 冒充原输入。程序不自动缩放、裁切或更改灰度范围，跨设备输入仍需另行验证。

上面的命令启用校准质量参考，低于下限标记 `unusable`，原候选类别与原图保留，原因仍是候选性质的 `severe_blur_candidate`。省略 `--quality-reference` 则不启用模糊判废；也可用 `--focus-min` 手动指定，但不能与质量参考同时使用。可选 `--focus-review-min` 是额外的质量存疑下限（此前探索值为 8.67，不是正式合格线）；它不能低于判废下限，否则程序报错。所有下限须非负，等于下限保留。报告区分验证尾部校准和未验证的手动阈值。

输出 `predictions_with_review.csv`、`needs_review.csv`（包括不可用项）和 `summary.json`。所有原始候选类别仍保留；严重模糊／无灰度结构优先输出 `unusable`，特征偏离或质量存疑触发输出 `uncertain`。未设置清晰度阈值时 `quality_status=not_assessed`，不能当成质量合格证明。不可用图片不从输入中删除，也不参与偷偷重算一个更高的原模型准确率。

作者当前已生成的质量参考位于 `reports/cell_quality_threshold_20261028/quality_threshold.json`，下限约 `8.060`；直接使用时，将上面命令的 `--quality-reference` 换成这个路径即可，无需重新训练或再次拟合。RMD 继续使用 `reports/cell_review_reference/rmd.pt` 中与原检查点绑定的工作点，约 `0.00149634`。这些本地产物不随 Git 分发，克隆者需按前述步骤生成自己的参考；不能仅将数值硬编码后用于其他模型或仪器。该命令是独立复核入口，不会自动改变原 `evaluate` 命令的输出。

RMD 使用 [PyTorch-OOD 官方实现](https://pytorch-ood.readthedocs.io/en/stable/detectors/rmd.html)，方法依据 [RMD 原论文](https://arxiv.org/abs/2106.09022)。未知细胞可能与已知类特征重叠，成像变化也可能触发偏离；因此 `feature_outlier` 仅代表待复核原因，不是嗜碱性粒细胞或病理异常诊断。当前未提供真实未知／失焦验证成绩。原保留测试集不用于拟合参考或选阈值；更换检查点必须重新拟合，程序通过摘要拒绝混用。

## 患者 PDF 导出

此入口只读取 `review_cells predict` 已生成的 `predictions_with_review.csv` 与 `summary.json`，不加载网络，不重新推理、训练或修改阈值。不能直接用缺少复核状态的旧四分类预测表代替。安装独立的 PDF 排版依赖（无需为单独导出安装 PyTorch-OOD）：

```powershell
python -m pip install -e ".[pdf]"
```

准备 `patients.csv`，每张图块一行，完整覆盖本次输入 NPY 的全部索引。`packed_index` 是**本次输入图块包的零起始索引**，不是原始 montage 编号；不能把一个含多人的 BBBC045 包直接当成一位患者。示例：

```csv
packed_index,patient_id,sample_id,cell_id
0,P001,S001,C001
1,P001,S001,C002
2,P001,S002,C001
3,P002,S003,C001
```

`patient_id`、`sample_id` 必填，建议使用脱敏编号；`cell_id` 可省略，默认 `tile-索引`。编号最多 64 个可显示字符，不允许换行；同一 `sample_id` 只能属于一位患者，样本内 `cell_id` 不能重复。一位患者生成一份 PDF，同患者多份样本分别汇总和分组展示，不合并计算比例。

```powershell
$incoming = Read-Host "本次复核使用的原始灰度 NPY 路径"
python -m leukocyte_analysis.patient_report `
  --review-dir .\reports\cell_review_predictions `
  --tiles-npy $incoming `
  --patient-manifest .\data\processed\patients.csv `
  --output-dir .\reports\patient_pdf
```

程序核对图块包与原复核摘要一致、患者清单与预测完整且不重复地覆盖同一包、候选编号和类别名称一致，以及决策与原策略一致。患者对应关系本身仍需由上游采集／裁切流程正确提供；哈希不能证明人为填写的患者编号正确。输出目录必须是新目录；来源检查或排版失败不发布半成品，不覆盖已有报告。

输出为安全命名的 `patient_0001_摘要.pdf` 等文件，文件名不直接使用患者编号，并附 `summary.json` 记录各样本计数、比例、原策略、模型摘要和输入文件来源摘要。PDF 严格参考“先总表、再类别栏和十列图册”的结构：黑白窄表、浅灰分区，逐类列名称、数量、占比，下方连续排列 48 点方图，跨页接续；不再使用卡片或逐图长文字。四类按中性、淋巴、单核、嗜酸顺序展示，之后是待复核和质量不合格；零数量项仍列在总表，图册不留空分区。大样本不默认截断或丢弃图片。

逐图明细保存为同目录 `cell_details.csv`，含 PDF 文件、类别分区、分区内顺序、图块索引、患者／样本／细胞编号、原候选、置信度、RMD、清晰度、状态和原因，便于对照图册而不增加图片间距。

PDF 总表与每类栏统一以本样本**全部输入图块数**为分母，四类 accepted 候选、待复核、质量不合格六项计数互斥且合计为输入总数；显示百分比取一位小数，四舍五入后可能不恰好合计 100.0%。JSON 的 `report_rows.proportion` 使用该口径，`classes.proportion` 仍保留四类在 accepted 内部的条件比例（没有 accepted 时为 null），由元数据明确区分，不能混用。两种比例都不是血液绝对浓度或完整五分类血常规。

PDF 为纯结果展示：不追加分母说明、显示处理说明、文件指引或免责声明；页脚仅有患者／样本编号及页码。这些口径和功能边界保留在本指南、`summary.json` 与 `cell_details.csv`，不改变原始评分与决策。未启用质量下限时，明细中的 `not_assessed` 不代表质量合格。RMD 待定不等于病变，`accepted` 不等于人工确认正常；报告不提供幼稚细胞、嗜碱性粒细胞、红细胞形态或血小板分析，也不照搬参考报告的临床高低标记。报告生成时间不是原图采集或模型分析时间。显示图片逐图做 1-99 百分位灰度拉伸及最近邻放大，仅为观看，不参与模型计算。

中文字体会嵌入 PDF。Windows 优先使用参考报告同类的黑体、再查找微软雅黑；Linux/macOS 若无兼容字体，用 `--font-ttf` 指定支持中文、可由 ReportLab 嵌入的 TrueType 字体文件（部分 CFF/OpenType 字体不兼容）；不静默退回缺字字体。生成物保存在 Git 忽略的 `reports/` 或 `output/pdf/`，真实患者报告与映射不得上传公开仓库。

按“从零准备环境和数据”中的 `pip install -e .` 安装后，不必设置 `PYTHONPATH`；也可使用 `leukocyte-prepare-data`、`leukocyte-train`、`leukocyte-cross-validate`、`leukocyte-evaluate`、`leukocyte-record-errors`、`leukocyte-review-cells`、`leukocyte-patient-report`、`leukocyte-brightness-stress` 和 `leukocyte-audit-brightness` 命令。若沿用旧环境而只需注册命令，可执行 `python -m pip install --no-deps --no-build-isolation -e .`。

运行不依赖完整数据集的单元测试：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```
