# Label-Free Leukocyte Analysis

基于 BBBC045 的 55×55 单通道明场图块，训练四类白细胞分类器：淋巴细胞（B＋T）、嗜酸性粒细胞、单核细胞和中性粒细胞。模型使用适配小图的 ResNet-18，从 ImageNet-1K 权重微调；按受试者划分数据，归一化统计量仅由相应训练侧计算。

[查看结果图](#结果可视化) · [了解实现](#项目实现) · [运行代码](#快速运行) · [完整复现](#从零准备环境和数据)

## 主要结果与边界

| 评估方式 | 受试者范围 | 图块数 | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: |
| 三折受试者分组交叉验证的折外预测 | 10 个分组 | 84,815 | **99.39%** | **0.9816** |
| 常规训练模型在原保留测试受试者上的评估 | CRF049、CRF066 | 15,290 | **99.05%** | **0.9711** |

两行来自不同训练与评估流程，不能当成同一模型在同一数据集上的前后提升。交叉验证的 10 个分组由 11 个目录标识组成，`CRF101_2014` 与 `CRF101_2015` 保守地放在同组。原保留测试集已经用于多次事后分析，**不再是全新独立测试**；上述结果也不代表跨仪器泛化或临床性能。任务只对已裁切的四类细胞图块分类，未覆盖嗜碱性粒细胞，不能代替完整的五分类血常规。运行配置、每折结果和局限见[实验索引](docs/experiment_index.md)。

## 结果可视化

下图来自上表的**常规训练模型**：左侧为逐轮验证 macro-F1，右侧为训练／验证的未加权交叉熵。检查点按第 44 轮的最高验证 macro-F1（0.9893）选择；右图标出**同一轮**的训练损失 0.1054 和验证损失 0.1081，并非各自曲线的最低值。实际优化目标还包含类别权重和标签平滑，不能直接把它的数值与图中的 `val_loss` 比较；图中的 F1 **不是训练集 F1，也不是测试集曲线**。

<p align="center">
  <a href="docs/assets/training_curves.png"><img src="docs/assets/training_curves.png" alt="常规训练的验证 macro-F1 及训练和验证损失曲线" width="820"></a>
</p>

下图是同一模型在**原保留测试受试者** CRF049、CRF066 上的计数混淆矩阵，行是真实类别，列是预测类别。单核细胞召回率为 91.16%，仍是相对弱项；该测试集已经被多次分析，不能将图称为全新独立测试结果。

<p align="center">
  <a href="docs/assets/confusion_matrix_counts.png"><img src="docs/assets/confusion_matrix_counts.png" alt="原保留测试集的四类白细胞计数混淆矩阵" width="580"></a>
</p>

两张结果图分别来自运行 `resnet18_four_imagenet_flips_photometric_geometric_plateau_ls0.05_20261014_145319` 的 `csv/metrics.csv` 和 `test_evaluation/`；原始运行目录不随 Git 发布。训练指标也可在运行目录的 TensorBoard 日志中查看；评估时的混淆矩阵由[绘图模块](src/leukocyte_analysis/evaluation/plots.py)生成。详细分折结果与亮度局限见[实验索引](docs/experiment_index.md)。

## 项目实现

- 从 BBBC045 明场 montage 生成带来源索引的 55×55 图块、受试者划分与训练侧归一化统计量；原始 B/T 标签在四分类加载时合并。实现见[数据准备入口](src/leukocyte_analysis/prepare_data.py)。
- 将 ImageNet 预训练 ResNet-18 适配单通道小图，并在训练侧使用翻转、光度／几何增强、类别加权损失、标签平滑和学习率调度。实现见[模型](src/leukocyte_analysis/models/resnet.py)与[训练模块](src/leukocyte_analysis/training/module.py)。
- 提供按受试者分组的三折折外评估、原保留测试集混淆矩阵及可回溯原图的错分记录；亮度变化仅作事后敏感性诊断。入口见[交叉验证](src/leukocyte_analysis/cross_validate.py)与[测试评估](src/leukocyte_analysis/evaluate.py)。

## 项目流程

<p align="center">
  <img src="docs/assets/workflow.svg" alt="从 BBBC045 明场原图到图块、受试者分组、三折折外评估和原保留测试集评估的项目流程" width="860">
</p>

两条评估路径使用不同训练流程，不能将两组结果直接当作同一模型的前后提升。每折的归一化均值、标准差和类别权重仅由该折训练图块计算；常规训练也只用原训练集计算，不让验证或测试图块参与拟合。原始 B/T 标签保留在清单中，进入四分类模型时才合并。

下图是原始图块的缩略展示（按 B、T、嗜酸性粒、单核、中性粒细胞排列），不是模型预测结果。[点开查看完整尺寸](docs/assets/bbbc045_examples.png)。

<p align="center">
  <a href="docs/assets/bbbc045_examples.png"><img src="docs/assets/bbbc045_examples.png" alt="BBBC045 明场白细胞原始图块，五行分别为 B、T、嗜酸性粒、单核和中性粒细胞" width="520"></a>
</p>

## 模型结构

<p align="center">
  <img src="docs/assets/model_architecture.svg" alt="55 乘 55 单通道图块经三乘三输入卷积、ResNet-18 残差层、全局平均池化和四分类头输出" width="860">
</p>

残差主体使用 ImageNet-1K 预训练权重；单通道输入层和四分类头重新初始化。增强**只作用于训练图块**，验证／测试沿用同一训练侧归一化参数，不做随机扰动。实现见[模型构建](src/leukocyte_analysis/models/resnet.py)和[数据加载](src/leukocyte_analysis/training/loaders.py)。

## 从零准备环境和数据

数据来源：[Broad Bioimage Benchmark Collection 的 BBBC045v1 官方页](https://bbbc.broadinstitute.org/BBBC045)。按官网推荐，引用本数据集时注明 “We used image set BBBC045v1, available from the Broad Bioimage Benchmark Collection [Ljosa et al., Nature Methods, 2012].” 本项目只读取官方 `BBBC045_Stained_Montages.zip` 中的 `ch5_*.tif` 明场 montage；不需要另一个 CIF 压缩包。原始目录名提供 B、T、嗜酸性粒、单核和中性粒细胞标签，模型读取时才将 B/T 合并为淋巴细胞。

普通读者可先用 Python 3.11 建立虚拟环境。下面是 Windows PowerShell 的 **CPU 安装示例**；Linux/macOS 的虚拟环境激活命令不同，PyTorch 的 CPU/GPU 安装命令应按[官方安装选择器](https://pytorch.org/get-started/locally/)选择。`pyproject.toml` 声明其余运行依赖；已有 CUDA 13.2 环境的作者可继续使用 `environment.yml`，无需为 CPU 重建。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip check
```

在[官方页](https://bbbc.broadinstitute.org/BBBC045)下载 `BBBC045_Stained_Montages.zip`，将压缩包保存在项目外，然后解压到被 Git 忽略的目录：

```powershell
$zip = Read-Host "请粘贴下载的 BBBC045_Stained_Montages.zip 完整路径"
Expand-Archive -LiteralPath $zip -DestinationPath .\data\raw
Get-ChildItem .\data\raw -Directory
```

数据根目录必须**直接**包含 `CRF022/` 等受试者目录；例如 `data/raw/Stained_Montages/CRF022/T/batch1/ch5_*.tif`。若解压后多出一层目录，请把 `--data-root` 指向实际含 `CRF*` 子目录的那一层。原图也可以放在项目外，本机现有路径是 `../Stained_Montages/`；`data/raw/` 和生成的 `data/processed/` 均被 Git 忽略。

在项目根目录按顺序生成原始清单、固定受试者 train/val/test 划分、55×55 图块包和**仅从 train 计算**的归一化统计量：

```powershell
python -m leukocyte_analysis.prepare_data --data-root .\data\raw\Stained_Montages
# 若已将原图解压在项目外，改用：
# python -m leukocyte_analysis.prepare_data --data-root ..\Stained_Montages
```

命令复用 `data/` 下已有的清单、划分、打包、抽查和统计模块，默认输出到 `data/processed/`。目标文件只要已有一个就会拒绝启动，不覆盖旧实验数据；需要另一份结果时使用 `--output-dir` 指向**新的**目录，并在常规训练命令中指定对应的 `--split-csv`、`--tiles-npy` 和 `--stats-json`。图块旁的 `.meta.json` 须随 `.npy` 一起保留。若运行中断，已完成的输出不会自动删除；核对后可选择新的输出目录重做。数据准备会读取和打包全部 TIFF，耗时且需数 GB 磁盘空间；本仓库不附带原图、处理数据或预训练权重。完成后按下一节先跑轻量测试，再启动训练。

## 快速运行

以下命令均从项目根目录执行。**不下载 BBBC045 数据**也可先运行单元测试，检查代码接口；通过不等于复现上表的模型成绩。若尚未执行 `pip install -e .`，先设置源码路径；已安装者可跳过第一行。

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m unittest discover -s tests -v
```

准备好 `data/processed/` 图块与清单后，可用两批数据做训练入口冒烟测试（不保存完整模型）：

```powershell
python -m leukocyte_analysis.train --smoke-test --batch-size 8 --init random
```

确认数据与环境无误后再启动完整训练；以下命令**耗时且会写入 `runs/`**。默认值对齐最近完成的配置：常规训练最多 45 轮，三折交叉验证每折最多 30 轮；两者均用 ImageNet 初始化、学习率 `0.0001`、翻转／光度／几何增强、Plateau、标签平滑 `0.05`、早停耐心值 15 和最小改善量 0。

```powershell
python -m leukocyte_analysis.train
python -m leukocyte_analysis.cross_validate
```

轮数和学习率仍可指定，例如 `python -m leukocyte_analysis.train --epochs 60 --lr 0.00005`；`--epochs` 是上限，早停可能提前结束。显式参数、检查点评估和历史诊断命令见[运行与诊断指南](docs/reproduction.md)。

## 目录

```text
environment.yml                 Conda 环境
pyproject.toml                  Python 包与命令行入口
src/leukocyte_analysis/
  prepare_data.py               原图到训练输入的准备入口
  data/                         数据读取、裁切、打包与审计代码
  models/                       模型结构代码
  training/                     数据加载器、Lightning 模块和实验配置
  evaluation/                   独立测试推理、指标和图像报告
  train.py                      命令行入口
  cross_validate.py             按受试者交叉验证入口
  resume_cv.py                  指定一折从 last.ckpt 续训入口
  evaluate.py                   训练完成后的测试集评估入口
  record_errors.py              从预测文件或检查点导出可定位的错分记录
  brightness_stress.py           已训练交叉验证权重的亮度敏感性诊断入口
  audit_brightness.py            原始图块的受试者／类别／批次亮度分布入口
data/raw/                       可选的原图解压目录（Git 忽略）
data/processed/                 已处理图块、清单、划分和训练集统计
data/pretrained/                官方预训练权重缓存（初次下载后复用）
reports/                        审计结果及预览图
runs/                           每次完整训练生成的权重和指标（初始不存在）
docs/                           实验记录及 README 用的少量静态图
tests/                          不依赖完整数据集的自动化测试
```

`src/leukocyte_analysis/data/` 存放**处理数据的代码**；项目根目录的 `data/` 存放**数据文件**。本机现有的原始 TIFF 仍在项目外的 `../Stained_Montages/`，没有复制或修改；其他读者可自行选择原图位置。

历史亮度诊断脚本依赖未发布的旧权重和数据，已从仓库移除；当时的数值与解释仍保留在[亮度实验汇总](docs/bbbc045_brightness_findings.md)。

## 详细运行与诊断

数据文件说明、完整参数、交叉验证、亮度审计及测试评估命令见[运行与诊断指南](docs/reproduction.md)；每次实验的结果与边界见[实验索引](docs/experiment_index.md)。
