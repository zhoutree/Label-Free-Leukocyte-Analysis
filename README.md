# Label-Free Leukocyte Analysis

使用 BBBC045 的明场图像训练四分类模型：淋巴细胞（B＋T）、嗜酸性粒细胞、单核细胞、中性粒细胞。模型为适配 55×55 单通道图块的 ResNet-18；训练、验证和测试按受试者分开，完整训练默认从 ImageNet-1K 预训练权重微调。这四类不构成完整的临床血常规分类：数据集中未单列嗜碱性粒细胞等群体。

## 目录

```text
environment.yml                 Conda 环境
pyproject.toml                  Python 包与命令行入口
src/leukocyte_analysis/
  data/                         数据读取、裁切、打包与审计代码
  models/                       模型结构代码
  training/                     数据加载器、Lightning 模块和实验配置
  evaluation/                   独立测试推理、指标和图像报告
  train.py                      命令行入口
  cross_validate.py             按受试者交叉验证入口
  evaluate.py                   训练完成后的测试集评估入口
  record_errors.py              从预测文件或检查点导出可定位的错分记录
data/processed/                 已处理图块、清单、划分和训练集统计
data/pretrained/                官方预训练权重缓存（初次下载后复用）
reports/                        审计结果及预览图
runs/                           每次完整训练生成的权重和指标（初始不存在）
docs/                           对已完成实验的分析记录，不参与训练
scripts/                        针对已完成实验的一次性核查脚本
tests/                          不依赖完整数据集的自动化测试
```

`src/leukocyte_analysis/data/` 存放**处理数据的代码**；项目根目录的 `data/` 存放**数据文件**。原始 TIFF 仍在项目外的 `../Stained_Montages/`，没有复制或修改。

`scripts/` 中的历史核查脚本绑定了特定运行目录，用于复核 [测试集异常记录](docs/bbbc045_test_audit.md) 及训练集错分聚集，不是通用训练入口；其输出保存在已忽略的 `runs/` 或 `reports/` 中。

## 当前数据

- `data/processed/bbbc045_tiles.npy`：100,105 个 55×55 明场图块；第 `i` 个图块对应清单第 `i` 行。
- `data/processed/bbbc045_manifest.csv`：图块来源、受试者、类别和网格位置。
- `data/processed/bbbc045_split.csv`：在清单基础上增加 train/val/test 划分。清单始终保留原始 B/T 标签；训练加载时才映射为四类，不需重新裁切或打包。
- `data/processed/bbbc045_train_stats.json`：仅由训练集计算的归一化参数。
- `reports/bbbc045_audit.json`：此前的基础审计；`reports/bbbc045_audit_focus.json`：新版审计，以每类训练图块的清晰度代理分数第 1 百分位标记 train/val 低细节候选，不自动删除样本，也不能确认多细胞或错标；`reports/previews/`：预览图。

## 运行

在项目根目录激活 `label-free-leukocyte` 环境后，先做不保存权重的冒烟测试：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m leukocyte_analysis.train --smoke-test --batch-size 8 --init random
```

确认无误后运行带轻量增强与早停的下一次完整训练：

```powershell
python -m leukocyte_analysis.train --init imagenet --lr 0.0001 --epochs 20 --batch-size 128 --augmentation flips --early-stopping-patience 5
```

完整训练默认加载 torchvision 的 ImageNet-1K ResNet-18 权重，优先复用项目 `data/pretrained/` 中的缓存；缓存不存在时需要联网下载。若已设置 `TORCH_HOME`，则尊重该位置。单通道 3×3 首层按 Kaiming 方式初始化，四分类头使用 PyTorch 默认初始化，其他层保留预训练参数并全部参与微调。默认学习率为 `0.0001`，最多训练 20 轮，AdamW 的 `weight_decay=0.01`（不是 L1 正则化）。合并 B/T 后，类别权重由四类训练集数量重新计算，不沿用旧五分类权重。

默认 `--augmentation flips` 仅对训练图块做随机水平与垂直翻转；验证与测试集不增强。默认 `--early-stopping-patience 5`：验证 macro-F1 连续 5 轮未提高就提前结束，`0` 表示禁用。训练循环、Rich 终端进度条、日志和检查点由 Lightning 管理；macro-F1 与各类召回率由 TorchMetrics 计算。每次训练新建 `runs/resnet18_four_初始化方式_增强方式_时间戳/`，其中 `csv/metrics.csv` 是逐轮指标，`tensorboard/` 是曲线日志，`checkpoints/best.ckpt` 是按验证集 macro-F1 选出的权重。`last.ckpt` 是最近一次实际保存的检查点，可能与 `best.ckpt` 相同。测试集不参与训练或选择权重。

可选 `--augmentation flips_photometric`：除上述翻转外，使用 torchvision `ColorJitter` 对**训练图块的原始灰度值**随机施加 0.8–1.25 倍亮度、0.9–1.1 倍对比度，再沿用该折训练集的均值和标准差归一化。验证与测试图块仍只做原有归一化。这个范围是在查看既有验证集的亮度异常后选定的，因此下面的同折比较属于探索性分析，不应视为全新独立测试。

当前只提供四分类训练与评估入口；旧五分类权重即使存在，也不能直接用于四分类。早停主要减少最佳验证轮次之后的无效训练，并不保证提高最佳分数。已完成的同折三折对照显示，加入亮度／对比度增强后，折外 macro-F1 从 0.9464 升至 0.9679，但个别受试者和采集批次退步；详情见[实验与异常记录](docs/bbbc045_test_audit.md)。这一对照不等于新的独立测试结果。

终端使用 Lightning 的 `RichProgressBar` 显示批次进度，并按实时窗口宽度逐级收起指标：窄窗口只保留轮次、横条和批次；中等宽度加入剩余时间及简短的 `loss`、`val_f1`；宽窗口再显示 `val_loss`。完整指标始终写入日志。极窄窗口或训练时频繁缩放仍可能造成显示残影。首次使用前，在已激活的环境中执行 `python -m pip install rich`；以后从更新后的 `environment.yml` 重建环境时会自动安装。训练时或训练后可另开终端运行：

```powershell
tensorboard --logdir .\runs
```

然后打开终端显示的本地地址（通常为 `http://localhost:6006/`）查看 `train_loss`、`val_loss`、`val_macro_f1` 和各类 `val_recall_*`。训练与验证损失曲线均为**未加权交叉熵**；用于更新参数的**加权交叉熵**单独记录为 `train_weighted_loss`。指标每轮才更新一次，第一轮结束前不会出现曲线。按 `Ctrl+C` 可以中断；完整训练的检查点包含优化器状态，但当前命令行暂未提供断点续训参数，重新执行会新建运行目录。

若要进行随机初始化的对照实验，使用 `python -m leukocyte_analysis.train --init random`。也可通过 `--weight-decay` 显式改变权重衰减。冒烟测试命令使用随机初始化，以免仅为检查训练流程就下载预训练权重。

`environment.yml` 让 Lightning、TorchMetrics 等普通依赖由 pip 自行选择兼容版本；只固定 Python 3.14 和已验证可使用 GPU 的 PyTorch/torchvision CUDA 13.2 配对，以免重建环境时意外改为 CPU 版本。

## 按受试者交叉验证

交叉验证只使用原 train＋val 的 11 名受试者，默认分为 3 折。每折从 ImageNet 权重重新初始化模型，并仅用该折训练受试者重算像素均值、标准差和类别权重；同一人不会同时出现在训练与验证中。原测试受试者 **CRF049 和 CRF066 全部排除**，包括待核查的 `CRF049/monocyte` 图块；原始数据与正式测试报告不变。折内 CSV 保留全部原行及索引，但测试受试者标为 `excluded`，不会被训练或验证数据加载器读取。

后续若改变受试者划分或使用更多数据重新训练，**不要把 `CRF049/monocyte` 纳入训练或训练统计量**；应先在新划分中排除该受试者的全部单核图块，再重算训练集归一化参数和类别权重。当前普通训练与交叉验证均因 CRF049 不属于训练受试者而满足此约定。高错分率只是排查依据，尚不能证明官方标签错误；保留原始清单和图块供复核。由于已据测试集表现制定排除规则，调整后的同一测试集不能再宣称为未使用过的独立测试集。

先检查第一折的两批训练和验证（不保存权重）：

```powershell
python -m leukocyte_analysis.cross_validate --smoke-test --init random --batch-size 8
```

再运行完整的三折实验；每折最多 20 轮，耗时约为单次训练的三倍，不会由上述冒烟命令自动启动：

```powershell
python -m leukocyte_analysis.cross_validate --folds 3 --init imagenet --lr 0.0001 --epochs 20 --batch-size 128
```

亮度/对比度增强对照只改变训练增强方式，其余参数及受试者分折保持相同：

```powershell
python -m leukocyte_analysis.cross_validate --folds 3 --init imagenet --lr 0.0001 --epochs 20 --batch-size 128 --augmentation flips_photometric
```

每次新建 `runs/cv_resnet18_.../`。`plan.json` 记录受试者分折，`fold_*/data/` 保存该折划分和训练集统计，`fold_*/training/` 保存日志及最佳检查点，`fold_*/validation_predictions.csv` 保存该折预测；`summary.json` 汇总折外准确率、macro-F1、各类指标和各受试者指标。这些验证结果用于比较训练方案；CRF049/CRF066 已经被查看过，不能再把它们当作新的独立最终测试。

## 最终测试评估

先仅用训练/验证集确定实验配置和最佳模型，再用新四分类训练的 `best.ckpt` 对保留测试受试者评估一次。评估入口会拒绝旧五分类检查点：

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

可选：若希望不再设置 `PYTHONPATH`，在已激活环境中执行 `python -m pip install --no-deps --no-build-isolation -e .`，此后可使用 `leukocyte-train`、`leukocyte-cross-validate`、`leukocyte-evaluate` 和 `leukocyte-record-errors` 命令。

运行不依赖完整数据集的单元测试：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```
