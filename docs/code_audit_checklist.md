# Python 代码逐文件审查清单

范围：`src/leukocyte_analysis/` 和 `tests/` 现有 60 个 Python 源文件。以下保留逐文件勾选、复用审查和注释复核记录；`[x]` 表示已检查，不表示必须修改。历史 `scripts/` 目录已按用户要求移除。

## `src/leukocyte_analysis/`（41）

该部分以 `rg --files src/leukocyte_analysis -g '*.py'` 为准，逐文件检查职责、重复实现、依赖方向及必要注释。用户已有改动与实验产物不在此清单中。

### 包根目录与命令入口（9）

- [x] `__init__.py` — 空包入口，无逻辑。
- [x] `audit_brightness.py` — 亮度审计命令入口，调用数据分析模块。
- [x] `brightness_stress.py` — 亮度压力测试命令入口。
- [x] `cross_validate.py` — 交叉验证参数与运行入口；与常规训练选项相近，但流程不同。
- [x] `evaluate.py` — 保留测试集评估入口。
- [x] `prepare_data.py` — 原图准备入口，仅编排现有数据模块并在开始前检查输出冲突。
- [x] `record_errors.py` — 错分导出入口，复用通用推理和记录函数。
- [x] `resume_cv.py` — 单折续训入口。
- [x] `train.py` — 常规训练参数与组装入口，不实现模型训练细节。

### 数据处理（12）

- [x] `data/__init__.py` — 空包入口。
- [x] `data/audit.py` — 清单、像素与清晰度都属于同一次非破坏性质量审计。
- [x] `data/bbbc045_config.py` — 唯一的数据格式、标签及受试者配置。
- [x] `data/cv_split.py` — scikit-learn 分折与该折清单写入，职责连贯。
- [x] `data/dataset.py` — 数据集读取、索引追溯和归一化，保持一类。
- [x] `data/grid.py` — 单一网格裁切函数。
- [x] `data/intensity_audit.py` — 亮度扫描、分组汇总和一张诊断图；绘图仅被此审计使用，暂不额外拆微型模块。
- [x] `data/manifest.py` — TIFF 扫描和清单生成。
- [x] `data/pack.py` — 同一图块包的创建与原图抽查。
- [x] `data/quality.py` — 单一清晰度代理指标。
- [x] `data/split.py` — 固定受试者划分和划分清单。
- [x] `data/stats.py` — 只从训练图块计算归一化统计量。

### 模型（2）

- [x] `models/__init__.py` — 模型目录包说明。
- [x] `models/resnet.py` — 使用 torchvision 构建并适配单通道小图，单一职责。

### 训练（11）

- [x] `training/__init__.py` — 空包入口。
- [x] `training/arguments.py` — 常规训练与交叉验证共用的超参数、数据路径和合法性检查；各自特有选项仍在命令入口。
- [x] `training/cross_validation.py` — 三折训练的编排；分折计划与报告已在独立模块，保留流程控制。
- [x] `training/cv_plan.py` — 建立、保存和读取同一交叉验证计划。
- [x] `training/cv_reports.py` — 折结果汇总与最终报告，复用评估指标函数。
- [x] `training/cv_resume.py` — 验证来源、续训和比较结果属于同一次续训事务；虽较长，但没有第二套训练算法。
- [x] `training/error_logging.py` — 验证错分记录回调，历史读取与频次统计服务同一输出。
- [x] `training/experiment.py` — 配置 Trainer、回调和训练产物路径。
- [x] `training/loaders.py` — 增强仅作用训练集，再创建数据集、类别权重和加载器；目前代码量有限，但增强构建可作为后续独立化候选。
- [x] `training/module.py` — LightningModule 的训练、验证和优化器钩子，属于框架约定的一类。
- [x] `training/progress.py` — 仅负责终端进度显示。

### 评估（7）

- [x] `evaluation/__init__.py` — 包用途说明，无逻辑。
- [x] `evaluation/brightness.py` — 两次交叉验证的受控亮度比较，复用通用推理、指标和绘图函数；计划校验与比较同属此诊断。
- [x] `evaluation/error_records.py` — 错分定位及 CSV 写入；后者供评估和训练报告共用。
- [x] `evaluation/inference.py` — 检查点校验、恢复和按清单顺序推理，供测试、错分记录与亮度诊断复用。
- [x] `evaluation/metrics.py` — 统一的 scikit-learn 分类指标入口。
- [x] `evaluation/plots.py` — 亮度、混淆矩阵和错分示例绘图；共同依赖 Matplotlib，未混入计算或训练逻辑。
- [x] `evaluation/runner.py` — 测试集评估编排与报告发布；推理已独立于此文件。

### 审查结论

初次审查时的 40 个文件已按清单顺序读完。只因文件超过 100 行，不作为拆分理由；独立职责混放或重复实现才作为修改依据。

- 已有的职责拆分：`training/cv_plan.py`、`training/cv_reports.py` 分担交叉验证计划与报告；`evaluation/inference.py` 从测试报告中抽出通用推理。它们仍是本清单审查的对象，不把“已拆分”当成未检查的理由。
- 本轮未发现需要再拆出独立文件的混合职责，也未发现第二套模型训练或分类指标实现。`training/loaders.py` 的增强构建是可进一步独立化的候选，但目前规模较小且只由一个加载器使用；贸然拆文件会增加调用层级而不减少行为复杂度。
- 首次审查给 `DonorFold` 补了说明受试者分组约束的类文档字符串；后续按同一清单做了完整注释复核，见下节。
- 该次 `src/` 审查没有改动训练、推理和评估行为；当时尚未审查测试、历史脚本、文档和数据产物。测试文件的后续审查记录见下文；历史脚本现已移除。

### 注释复核（2026-10-15）

初次审查按当时的清单逐个核对模块及类／函数定义。采用 [PEP 257](https://peps.python.org/pep-0257/) 的模块、类、函数文档字符串约定；复杂接口重点说明数据范围、形状、返回值、持久化副作用和验证／测试隔离。逐行解释赋值不属于目标。

- 补齐包入口、CLI `main`、数据集特殊方法、Lightning 回调／进度条覆写方法及亮度审计辅助函数的文档字符串。
- 扩充图块打包、训练统计、模型适配、加载器、训练编排、交叉验证、续训和测试评估等长接口的行为约束说明。
- 在亮度代理计算处说明外缘像素并非保证纯背景；核对文案不超过代码实际验证范围。
- 当时的 AST 复查结果：40/40 个模块、101/101 个类／函数定义有文档字符串；行为逻辑未因注释调整而改变。随后新增共用参数模块、移除了旧预览模块。

## `tests/`（19）

范围：以 `rg --files tests -g '*.py'` 为准，现有 19 个 Python 源文件。`__pycache__/` 内的 `.pyc` 是运行生成的缓存，不计入源文件清单。逐个检查职责、重复测试或实现、可复用性、必要注释及文件是否应合并/拆分。`[x]` 表示已检查，不代表必须修改。

### `tests/`（19）

- [x] `tests/_experiment_fixtures.py` — 三个训练实验公用的小数据加载器／临时目录，避免测试间重复准备。
- [x] `tests/test_augmentation.py` — 专测训练增强内容及验证集隔离。
- [x] `tests/test_brightness_stress.py` — 亮度压力测试的计划校验、分组指标和报告边界。
- [x] `tests/test_cross_validation.py` — 首折运行后接续其余折的完整编排场景。
- [x] `tests/test_cv_resume.py` — 单折从检查点续训且不更改原运行；较长的是合成三折运行的测试数据准备，不是多个被测职责。
- [x] `tests/test_cv_split.py` — 受试者分折、CRF101 分组、留出受试者及逐折归一化统计。
- [x] `tests/test_dataset.py` — 四分类映射、常规受试者划分和数据集读取；均属输入数据契约。
- [x] `tests/test_error_logging.py` — 错分记录回调的累计、历史续接和标签追溯。
- [x] `tests/test_evaluation.py` — 合成图块的通用推理及正式报告；辅助类只服务此场景。
- [x] `tests/test_intensity_audit.py` — 一套合成图块贯穿亮度报告及保留测试集排除开关。
- [x] `tests/test_learning_rate_scheduling.py` — Plateau、Cosine 和早停配置及调度步进。
- [x] `tests/test_metrics.py` — 共享分类指标的宏平均和缺类边界。
- [x] `tests/test_model.py` — ResNet 输入适配和分类器损失／预测接口；同属模型契约，规模不大。
- [x] `tests/test_progress.py` — 窄终端进度条显示，不检查训练逻辑。
- [x] `tests/test_quality.py` — 图块像素与清晰度筛查的非破坏性边界。
- [x] `tests/test_reporting.py` — 训练数据划分记录、检查点和 TensorBoard／CSV 产物。
- [x] `tests/test_training_resume.py` — Lightning 单次训练检查点续训，与三折的单折续训不同。
- [x] `tests/test_training.py` — 常规训练及交叉验证的 CLI 默认值与非法参数；已是从旧大型混合测试中拆出的配置测试。
- [x] `tests/test_validation_error_callback.py` — 回调挂载和一轮训练后产出错分文件的集成测试。

### 审查结论与变更

此前已逐个审查相关源文件；按用户要求先后移除六份历史／展示脚本，当前清单只列出保留的 19 个测试文件。README 的静态图和历史实验结论仍保留。

- 没有值得合并的整份测试文件：名称相近的 `test_cross_validation.py`／`test_cv_resume.py` 以及 `test_error_logging.py`／`test_validation_error_callback.py` 分别覆盖不同层次或操作。
- 没有必须拆开的长测试文件：`test_cv_resume.py` 和 `test_intensity_audit.py` 的长度来自单一完整场景所需的合成数据，拆到共享夹具会让场景更难核对。`test_training.py` 现在只管 CLI 配置，不再是以前混合模型、数据、增强和进度显示的大文件。
- 小重复保留：`test_error_logging.py` 和 `test_evaluation.py` 各有一个仅数行的 `PixelClassifier` 合成替身，放到共享夹具的收益低。

### 第二轮：已有模块与第三方库复用

依照上方清单逐个对照。这里的“保留”表示没有发现值得新增依赖或抽象层的重复，不表示代码永远不能改。

#### `tests/`

- `_experiment_fixtures.py`：训练测试已共用两个合成加载器和工作区内临时目录；保留。
- `test_augmentation.py`：已通过 `unittest.mock` 隔离数据加载，并检查现用 torchvision 增强；保留。
- `test_brightness_stress.py`：复用正式亮度模块和绘图函数；小型计划构造只服务本文件，保留。
- `test_cross_validation.py`：复用正式交叉验证入口；所需三折合成清单与其他测试的条件不同，保留局部构造。
- `test_cv_resume.py`：复用分折和报告函数；较长夹具准确模拟检查点、清单和三折结果，暂无第二处可复用同一夹具。
- `test_cv_split.py`：直接验证已用的 scikit-learn 分折封装与训练统计；不需要再引入分折库。
- `test_dataset.py`：已直接测试共享标签映射和 `LeukocyteDataset`；两个 CSV 准备片段相近，但分别覆盖标签映射和增强顺序，抽夹具收益小。
- `test_error_logging.py`：复用错分记录函数和回调；与 `test_evaluation.py` 有一个很短的合成像素分类器相同，暂不为此增加共享文件。
- `test_evaluation.py`：复用正式推理和报告；图块替身仅供本文件使用，保留。
- `test_intensity_audit.py`：复用正式亮度审计，一个合成数据集贯穿包括留出测试集在内的检查，保留。
- `test_learning_rate_scheduling.py`：已复用 `_experiment_fixtures.py`，检查 PyTorch 原生调度器和 Lightning 回调，保留。
- `test_metrics.py`：直接测试统一的 scikit-learn 指标封装，保留。
- `test_model.py`：直接测试 torchvision 主干适配和 LightningModule 接口；不重复实现模型，保留。
- `test_progress.py`：直接测试 Rich 进度条子类，保留。
- `test_quality.py`：使用已安装的 SciPy 模糊函数构造输入，测试像素／清晰度审核，保留。
- `test_reporting.py`：已复用共享夹具与 TensorBoard 的 `EventAccumulator` 读取实际事件文件，保留。
- `test_training_resume.py`：已复用共享夹具与 Lightning 的检查点恢复行为，保留。
- `test_training.py`：只检查 CLI 参数，不重复训练逻辑；使用标准库 `unittest.mock`，保留。
- `test_validation_error_callback.py`：已复用共享训练夹具，并验证真实一轮回调产物，保留。

第三方框架取舍：[pytest 的 `tmp_path`](https://docs.pytest.org/en/stable/how-to/tmp_path.html) 和[参数化](https://docs.pytest.org/en/stable/how-to/parametrize.html)确实可减少新测试的样板代码；但当前测试均为 `unittest.TestCase`，[pytest 官方说明](https://docs.pytest.org/en/stable/how-to/unittest.html)仅换运行器不能让这些方法直接接收普通 pytest 夹具或参数化。为减少少量 `TemporaryDirectory` 而迁移 52 项测试并新增依赖，目前得不偿失。现有 `unittest`、`tempfile`、`unittest.mock`、PyTorch、Lightning、torchvision、scikit-learn、NumPy、Pillow、tifffile、SciPy 和 TensorBoard 已覆盖所需功能。

### 注释复核（2026-10-15）

此前已核对测试与历史诊断脚本的文档字符串，重点写明所保护的行为和探索性结果的边界。随后新增一个命令行回归测试，并按用户要求删除三个一次性脚本；本段原有的 AST 数量不再代表当前文件集合。
