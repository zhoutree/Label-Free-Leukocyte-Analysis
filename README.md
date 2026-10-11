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
  evaluate.py                   训练完成后的测试集评估入口
data/processed/                 已处理图块、清单、划分和训练集统计
data/pretrained/                官方预训练权重缓存（初次下载后复用）
reports/                        审计结果及预览图
runs/                           每次完整训练生成的权重和指标（初始不存在）
tests/                          不依赖完整数据集的自动化测试
```

`src/leukocyte_analysis/data/` 存放**处理数据的代码**；项目根目录的 `data/` 存放**数据文件**。原始 TIFF 仍在项目外的 `../Stained_Montages/`，没有复制或修改。

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

此前的五分类运行目录和检查点没有删除，但新代码不再提供五分类训练或评估入口；旧权重不能直接当作四分类权重使用。早停主要减少最佳验证轮次之后的无效训练，并不保证提高最佳分数；轻量增强能否改善泛化，需要与同一四分类任务的基线比较后才能判断。四类与原五类的 macro-F1 分母不同，不能直接以高低判断模型是否改善。

终端使用 Lightning 的 `RichProgressBar` 显示批次进度，并按实时窗口宽度逐级收起指标：窄窗口只保留轮次、横条和批次；中等宽度加入剩余时间及简短的 `loss`、`val_f1`；宽窗口再显示 `val_loss`。完整指标始终写入日志。极窄窗口或训练时频繁缩放仍可能造成显示残影。首次使用前，在已激活的环境中执行 `python -m pip install rich`；以后从更新后的 `environment.yml` 重建环境时会自动安装。训练时或训练后可另开终端运行：

```powershell
tensorboard --logdir .\runs
```

然后打开终端显示的本地地址（通常为 `http://localhost:6006/`）查看 `train_loss`、`val_loss`、`val_macro_f1` 和各类 `val_recall_*`。训练与验证损失曲线均为**未加权交叉熵**；用于更新参数的**加权交叉熵**单独记录为 `train_weighted_loss`。指标每轮才更新一次，第一轮结束前不会出现曲线。按 `Ctrl+C` 可以中断；完整训练的检查点包含优化器状态，但当前命令行暂未提供断点续训参数，重新执行会新建运行目录。

若要进行随机初始化的对照实验，使用 `python -m leukocyte_analysis.train --init random`。也可通过 `--weight-decay` 显式改变权重衰减。冒烟测试命令使用随机初始化，以免仅为检查训练流程就下载预训练权重。

`environment.yml` 让 Lightning、TorchMetrics 等普通依赖由 pip 自行选择兼容版本；只固定 Python 3.14 和已验证可使用 GPU 的 PyTorch/torchvision CUDA 13.2 配对，以免重建环境时意外改为 CPU 版本。

## 最终测试评估

先仅用训练/验证集确定实验配置和最佳模型，再用新四分类训练的 `best.ckpt` 对保留测试受试者评估一次。评估入口会拒绝旧五分类检查点：

```powershell
$checkpoint = Read-Host "请粘贴选定实验的 best.ckpt 完整路径"
python -m leukocyte_analysis.evaluate --checkpoint $checkpoint
```

测试集批量推理由 Lightning 的 `Trainer.predict()` 管理；分类指标和混淆矩阵交给 scikit-learn。默认在该实验目录下新建 `test_evaluation/`，生成 `test_metrics.json`、各类别 precision/recall/F1 的文本与 CSV 表、逐细胞预测 CSV、计数及按真实类别归一化的混淆矩阵，以及每类最多三张高置信度错分示例图。示例图的 1–99% 逐图对比度拉伸**只用于观看**，不影响推理。若报告目录已存在，命令会拒绝覆盖，避免误把多次测试当成独立验证。该测试集仅含两名保留受试者，指标不能替代跨设备、跨实验室的外部验证。

可选：若希望不再设置 `PYTHONPATH`，在已激活环境中执行 `python -m pip install --no-deps --no-build-isolation -e .`，此后可使用 `leukocyte-train` 和 `leukocyte-evaluate` 命令。

运行不依赖完整数据集的单元测试：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```
