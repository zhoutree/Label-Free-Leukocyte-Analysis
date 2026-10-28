# Python 代码逐文件审查清单

范围：`src/leukocyte_analysis/` 和 `tests/` 现有 75 个 Python 源文件。以下保留逐文件勾选、复用审查和注释复核记录；`[x]` 表示已检查，不表示必须修改。历史 `scripts/` 目录已按用户要求移除。

维护约定：新增文件补入对应目录，修改文件重新检查并更新原条目；移动或删除文件同步路径、条目和文件总数。尚未检查的变更使用 `[ ]`，完成复查后才勾选；旧勾选不代表后续修改自动通过。具体变更和实际测试结果记在文末，不另建平行清单。本清单列源码与测试；`AGENTS.md`、README 和环境配置的配套修改另在变更记录说明，不计入 Python 文件数量。

## `src/leukocyte_analysis/`（50）

该部分以 `rg --files src/leukocyte_analysis -g '*.py'` 为准，逐文件检查职责、重复实现、依赖方向及必要注释。用户已有改动与实验产物不在此清单中。

### 包根目录与命令入口（12）

- [x] `__init__.py` — 空包入口，无逻辑。
- [x] `audit_brightness.py` — 亮度审计命令入口，调用数据分析模块。
- [x] `brightness_stress.py` — 亮度压力测试命令入口。
- [x] `cross_validate.py` — 交叉验证参数与运行入口；与常规训练选项相近，但流程不同。
- [x] `evaluate.py` — 保留测试集评估入口。
- [x] `paths.py` — 共用项目根目录及运行相对路径解析，由原 `training/run_paths.py` 迁入，不依赖训练配置。
- [x] `prepare_data.py` — 原图准备入口，仅编排现有数据模块并在开始前检查输出冲突。
- [x] `record_errors.py` — 只解析参数、调用独立错分导出业务及显示数量，不实现来源核对或模型推理。
- [x] `reporting.py` — 中立的报告暂存发布及 CSV 工具；数据、训练、评估共同复用，不依赖业务模块。
- [x] `resume_cv.py` — 单折续训入口。
- [x] `review_cells.py` — RMD 拟合、已有评分质量校准和无标签复核的 CLI；业务逻辑委托对应模块，参数与互斥规则保持不变。
- [x] `train.py` — 常规训练参数与组装入口，不实现模型训练细节。

### 数据处理（13）

- [x] `data/__init__.py` — 空包入口。
- [x] `data/audit.py` — 清单、像素与清晰度都属于同一次非破坏性质量审计。
- [x] `data/bbbc045_config.py` — 唯一的数据格式、标签及受试者配置。
- [x] `data/cv_split.py` — scikit-learn 分折与该折清单写入，职责连贯。
- [x] `data/dataset.py` — 数据集读取、索引追溯、归一化和输入指纹核对，保持一类。
- [x] `data/grid.py` — 单一网格裁切函数。
- [x] `data/intensity_audit.py` — 亮度扫描、分组汇总和一张诊断图；报告发布及 CSV 写入已复用公共模块，专用绘图保留在此。
- [x] `data/integrity.py` — 绑定新图块包的记录顺序与内容，并核对清单和统计文件指纹；旧包保留结构校验。
- [x] `data/manifest.py` — TIFF 扫描和清单生成。
- [x] `data/pack.py` — 同一图块包的创建、顺序元数据发布与原图抽查。
- [x] `data/quality.py` — 共享中央区域校验；保留历史拉普拉斯口径，补充 Sobel 梯度及对比度归一化质量评分。
- [x] `data/split.py` — 固定受试者划分、划分清单及参考校准来源检查；RMD 与质量校准共用受试者隔离校验。
- [x] `data/stats.py` — 只从训练图块计算归一化统计量。

### 模型（2）

- [x] `models/__init__.py` — 模型目录包说明。
- [x] `models/resnet.py` — 使用 torchvision 构建并适配单通道小图，单一职责。

### 训练（11）

- [x] `training/__init__.py` — 空包入口。
- [x] `training/arguments.py` — 常规训练与交叉验证共用的超参数、数据路径和合法性检查；各自特有选项仍在命令入口。
- [x] `training/cross_validation.py` — 三折训练与迁移续跑的编排；分折计划与报告已在独立模块，保留流程控制。
- [x] `training/cv_plan.py` — 建立、保存和读取同一交叉验证计划。
- [x] `training/cv_reports.py` — 折结果汇总与最终报告，复用指标和公共 CSV 写入；预测表使用仅创建模式。
- [x] `training/cv_resume.py` — 验证来源、续训和比较结果属于同一次续训事务；虽较长，但没有第二套训练算法。
- [x] `training/error_logging.py` — 验证错分记录回调，历史读取与频次统计服务同一输出；使用公共 CSV 工具，累计表允许更新。
- [x] `training/experiment.py` — 配置 Trainer、回调和训练产物路径。
- [x] `training/loaders.py` — 增强仅作用训练集，再创建数据集、类别权重和加载器；目前代码量有限，但增强构建可作为后续独立化候选。
- [x] `training/module.py` — LightningModule 的训练、验证和优化器钩子，属于框架约定的一类。
- [x] `training/progress.py` — 仅负责终端进度显示。

### 评估（12）

- [x] `evaluation/__init__.py` — 包用途说明，无逻辑。
- [x] `evaluation/brightness.py` — 两次交叉验证的受控亮度比较，复用通用推理、指标和绘图函数；计划校验与比较同属此诊断。
- [x] `evaluation/cell_review.py` — 仅组合质量与 RMD 特征偏离规则；不加载模型、不使用真实标签作决策。
- [x] `evaluation/error_export.py` — 从预测 CSV 或验证检查点导出错分的独立业务；核对完整预测顺序，API 也限制测试集重复推理。
- [x] `evaluation/error_records.py` — 只负责错分来源定位与记录字段；通用 CSV 写入已迁至根目录 `reporting.py`。
- [x] `evaluation/inference.py` — 检查点校验、恢复和按清单顺序推理，供测试、错分记录与亮度诊断复用。
- [x] `evaluation/metrics.py` — 统一的 scikit-learn 分类指标入口。
- [x] `evaluation/novelty.py` — 适配 PyTorch-OOD RMD 与原 ResNet 特征，保存有限张量状态，不重写距离算法。
- [x] `evaluation/plots.py` — 亮度、混淆矩阵和错分示例绘图；共同依赖 Matplotlib，未混入计算或训练逻辑。
- [x] `evaluation/quality_reference.py` — 质量下限计算、校准文件发布及恢复；先按完整来源清单核对验证集索引、受试者和类别，不加载模型。
- [x] `evaluation/review_runner.py` — RMD 参考拟合及无标签应用编排；质量参考读写已独立，共用根目录报告工具。
- [x] `evaluation/runner.py` — 测试集评估编排与报告发布；推理已独立于此文件。

### 审查结论

初次审查时的 40 个文件已按清单顺序读完。只因文件超过 100 行，不作为拆分理由；独立职责混放或重复实现才作为修改依据。

- 已有的职责拆分：`training/cv_plan.py`、`training/cv_reports.py` 分担交叉验证计划与报告；`evaluation/inference.py` 从测试报告中抽出通用推理。它们仍是本清单审查的对象，不把“已拆分”当成未检查的理由。
- 初次审查当时未发现需要再拆出独立文件的混合职责；这不代表后续新增代码已经合格，最新复核与整改见文末。未发现第二套模型训练或分类指标实现。`training/loaders.py` 的增强构建仍是可进一步独立化的候选，目前未为其新增文件。
- 首次审查给 `DonorFold` 补了说明受试者分组约束的类文档字符串；后续按同一清单做了完整注释复核，见下节。
- 该次 `src/` 审查没有改动训练、推理和评估行为；当时尚未审查测试、历史脚本、文档和数据产物。测试文件的后续审查记录见下文；历史脚本现已移除。

### 注释复核（2026-10-15）

初次审查按当时的清单逐个核对模块及类／函数定义。采用 [PEP 257](https://peps.python.org/pep-0257/) 的模块、类、函数文档字符串约定；复杂接口重点说明数据范围、形状、返回值、持久化副作用和验证／测试隔离。逐行解释赋值不属于目标。

- 补齐包入口、CLI `main`、数据集特殊方法、Lightning 回调／进度条覆写方法及亮度审计辅助函数的文档字符串。
- 扩充图块打包、训练统计、模型适配、加载器、训练编排、交叉验证、续训和测试评估等长接口的行为约束说明。
- 在亮度代理计算处说明外缘像素并非保证纯背景；核对文案不超过代码实际验证范围。
- 当时的 AST 复查结果：40/40 个模块、101/101 个类／函数定义有文档字符串；行为逻辑未因注释调整而改变。随后新增共用参数模块、移除了旧预览模块。

## `tests/`（25）

范围：以 `rg --files tests -g '*.py'` 为准，现有 25 个 Python 源文件。`__pycache__/` 内的 `.pyc` 是运行生成的缓存，不计入源文件清单。逐个检查职责、重复测试或实现、可复用性、必要注释及文件是否应合并/拆分。`[x]` 表示已检查，不代表必须修改。

### `tests/`（25）

- [x] `tests/_experiment_fixtures.py` — 三个训练实验公用的小数据加载器／临时目录，避免测试间重复准备。
- [x] `tests/test_augmentation.py` — 专测训练增强内容及验证集隔离。
- [x] `tests/test_brightness_stress.py` — 亮度压力测试的计划校验、分组指标和报告边界。
- [x] `tests/test_cross_validation.py` — 首折运行后接续其余折的完整编排场景。
- [x] `tests/test_cv_resume.py` — 单折从检查点续训且不更改原运行；较长的是合成三折运行的测试数据准备，不是多个被测职责。
- [x] `tests/test_cv_split.py` — 受试者分折、CRF101 分组、留出受试者及逐折归一化统计。
- [x] `tests/test_dataset.py` — 四分类映射、常规受试者划分、数据集读取和新图块包指纹；均属输入数据契约。
- [x] `tests/test_error_logging.py` — 错分记录回调的累计、历史续接和标签追溯。
- [x] `tests/test_error_export.py` — 错分导出 API 的防覆盖、顺序校验和来源互斥；测试集不重复推理及 CLI 委托。
- [x] `tests/test_evaluation.py` — 合成图块的通用推理及正式报告；辅助类只服务此场景。
- [x] `tests/test_intensity_audit.py` — 一套合成图块贯穿亮度报告及保留测试集排除开关。
- [x] `tests/test_learning_rate_scheduling.py` — Plateau、Cosine 和早停配置及调度步进。
- [x] `tests/test_metrics.py` — 共享分类指标的宏平均和缺类边界。
- [x] `tests/test_model.py` — ResNet 输入适配和分类器损失／预测接口；同属模型契约，规模不大。
- [x] `tests/test_progress.py` — 窄终端进度条显示，不检查训练逻辑。
- [x] `tests/test_quality.py` — 图块像素与清晰度筛查的非破坏性边界，补充归一化梯度、模糊方向与亮度变化测试。
- [x] `tests/test_quality_reference.py` — 独立质量校准的类内预算、完整验证来源检查、CRF101 隔离及旧参考兼容。
- [x] `tests/test_cell_review.py` — 质量／RMD 特征决策、无标签合成闭环、原测试隔离、检查点绑定与拒绝覆盖。
- [x] `tests/test_novelty.py` — 第三方 RMD 定义核对、分数方向、并列工作点、张量状态恢复和共享分类头。
- [x] `tests/test_reporting.py` — 训练数据划分记录、检查点和 TensorBoard／CSV 产物。
- [x] `tests/test_report_io.py` — 仅测试通用文件工具的列顺序、拒绝覆盖、累计更新与失败清理，不加载模型。
- [x] `tests/test_run_paths.py` — 中立路径模块的项目根位置、新旧路径及迁移后接续运行兼容性。
- [x] `tests/test_training_resume.py` — Lightning 单次训练检查点续训，与三折的单折续训不同。
- [x] `tests/test_training.py` — 常规训练及交叉验证的 CLI 默认值与非法参数；已是从旧大型混合测试中拆出的配置测试。
- [x] `tests/test_validation_error_callback.py` — 回调挂载和一轮训练后产出错分文件的集成测试。

### 审查结论与变更

此前已逐个审查相关源文件；按用户要求先后移除六份历史／展示脚本。当前清单列出 25 个保留的测试／夹具文件（24 个 `test_*.py` 加 `_experiment_fixtures.py`）。README 的静态图和历史实验结论仍保留。

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

第三方框架取舍：[pytest 的 `tmp_path`](https://docs.pytest.org/en/stable/how-to/tmp_path.html) 和[参数化](https://docs.pytest.org/en/stable/how-to/parametrize.html)确实可减少新测试的样板代码；但当前测试均为 `unittest.TestCase`，[pytest 官方说明](https://docs.pytest.org/en/stable/how-to/unittest.html)仅换运行器不能让这些方法直接接收普通 pytest 夹具或参数化。为减少少量 `TemporaryDirectory` 而迁移现有测试并新增依赖，目前得不偿失。现有 `unittest`、`tempfile`、`unittest.mock`、PyTorch、Lightning、torchvision、scikit-learn、NumPy、Pillow、tifffile、SciPy 和 TensorBoard 已覆盖所需功能。

### 注释复核（2026-10-15）

此前已核对测试与历史诊断脚本的文档字符串，重点写明所保护的行为和探索性结果的边界。随后新增一个命令行回归测试，并按用户要求删除三个一次性脚本；本段原有的 AST 数量不再代表当前文件集合。

### 路径与错误边界复查（2026-10-16）

- 重新按文件清单核对原有 42 个源码文件和 20 个测试文件，无遗漏；本轮新增数据完整性模块后为 43 个源码文件。完整轻量测试通过。新运行的交叉验证计划与检查点改存运行目录相对路径；旧绝对路径可读取，迁移旧运行时可显式覆盖输入路径。没有修改原始图像、已处理图块或旧运行产物。
- **已修复的校验缺口**：学习率和权重衰减拒绝 `nan`／`inf`；常规训练核对每个 donor 的固定划分，防止 CRF049／CRF066 等原测试受试者误入训练；错分审计对所有预测行核对真实标签，不再只核对错分行。
- **新数据输入一致性**：打包时新增图块顺序和内容指纹，统计文件记录清单与图块包指纹；加载和续训迁移时核对。旧数据及旧运行仍可读取，但若原本没有指纹，只能核对形状、行数及已保存折的清单记录，不能事后证明 `.npy` 内容完全相同。既有成绩没有发现受这些缺口影响。

### 再次逐项复核（2026-10-16）

- 当前清单与文件系统一致：43 个源码文件、20 个测试／夹具文件；完整轻量测试为 **64 项通过**，`git diff --check` 无格式错误。未运行完整训练或正式数据推理。
- [x] `evaluation/runner.py`：正式测试在推理前核对固定划分，并要求 `split=test` 恰好包含原保留受试者 CRF049／CRF066。
- [x] `data/audit.py`、`data/intensity_audit.py`：独立质量／亮度审计也核对新图块包的顺序和内容指纹，拒绝同长度错序清单。
- [x] `data/integrity.py`：缺少 `.npy.meta.json` 时明确警告按旧包兼容模式降级；新统计文件如已声明依赖元数据则直接拒绝。旧包仍无法事后证明原打包顺序。
- [x] `data/pack.py`：第二次重命名失败时回滚本次发布的元数据，避免通常的发布失败留下孤立文件；进程意外中断仍需人工检查输出目录。
- 上述修复面向新运行／自定义输入的边界风险；没有证据显示既有训练和测试成绩因此出错。未改动原始图像、已处理图块或既有运行产物。

### 特征与质量复核增量检查（2026-10-27）

- 按清单检查本次新增的命令入口、RMD 适配、纯决策规则、推理编排及对应测试；共享检查点恢复、数据集、质量模块、CSV 发布和来源摘要，没有新增第二套训练循环或手写 RMD 算法。
- 原拉普拉斯审计评分保持口径；新 Sobel 归一化评分只用于质量检查，不改变模型像素输入。质量阈值未指定时标记为未评估，不默认判废低纹理细胞。
- 训练侧拟合、验证侧校准；原保留测试受试者不得进入这两侧，CRF101 两年份沿用同组校验。参考绑定检查点摘要、类别顺序和训练归一化，换权重须重新拟合。
- 相关 **21 项合成／轻量测试通过**；覆盖第三方评分定义、状态恢复、高置信度特征待定、质量优先级、灰度线性变化、无标签闭环、失败不发布报告及拒绝覆盖。新增与修改函数均核对中文文档字符串，源码数量与清单一致。未启动真实图块参考拟合、完整训练或全量推理；未知类／真实失焦效果仍待样本验证。

### 仅保留 RMD 的简化（2026-10-27）

- 按用户要求移除分类置信度阈值规则、CSV 置信度复核入口、专用绘图和测试；分类置信度继续作为报告信息保存。待定判断仅使用 RMD，清晰度独立预警。
- 共享报告发布上下文保留在 `evaluation/review.py`，不新增重复发布实现；已有 RMD 张量参考格式不变，旧结果无需重新生成。文档及文件清单同步更新为 48 个源码、22 个测试／夹具文件。
- 只运行一轮相关轻量测试，**13 项通过**；覆盖已移除参数的拒绝、RMD 及质量决策、状态恢复与合成闭环。未重新训练或推理真实图块，历史报告保持原样。

### 保守质量阈值校准（2026-10-28）

- 复用 `evaluation/cell_review.py` 的纯规则与 `review_runner.py` 的报告编排，新增 `calibrate-quality` 子命令；不新增训练循环、依赖或文件层级。
- 每类验证评分的低尾预算默认 1%，取四类最保守的统一下限；严格低于阈值触发，推理不使用真实标签或预测类别选阈值。无质量真值，报告明确区分经验拦截比例和真正误判废率。
- `predict --quality-reference` 显式启用自动质量不合格标记，不删除图片或重新计算原模型成绩；与手动严重模糊阈值互斥，算法版本、来源摘要与限制留档。
- 相关轻量测试 **6 项通过**，覆盖各类预算、小样本／并列、无效评分、参考恢复、拒绝覆盖及合成应用；真实数据只读取已有评分 CSV，不启动模型推理。实测下限 `8.059967284809368`，44/16,221 张被标记，详见实验索引。

### 职责分离与共享工具整改（2026-10-28）

- [x] `evaluation/quality_reference.py`：从 `cell_review.py` 和 `review_runner.py` 迁出质量阈值计算、校准文件写入及读取；归为同一独立能力，不加载网络，不改算法、数值或 JSON 格式。
- [x] `reporting.py`：将原 `evaluation/review.py` 的报告发布与 `evaluation/error_records.py` 的通用 CSV 写入迁入中立模块；移除旧 `evaluation/review.py`，不保留两套实现。数据层不反向依赖评估层。
- [x] `evaluation/runner.py`、`evaluation/brightness.py`、`data/intensity_audit.py`、`evaluation/review_runner.py`、`evaluation/quality_reference.py`：统一复用报告暂存发布，失败不发布正式目录，已有报告拒绝覆盖。
- [x] `training/cv_reports.py`、`data/intensity_audit.py`：移除独立 CSV 写入实现。共享工具提供仅创建与原子更新两种语义；预测表拒绝覆盖，逐轮错分沿用既有防覆盖检查及原子写入，累计频次表允许更新。
- [x] `review_cells.py`、`record_errors.py`、`training/error_logging.py` 及相关测试：直接导入对应职责模块，清除旧工具路径；命令、参数、阈值、输出文件名称和列顺序不变。
- [x] `tests/test_report_io.py`：新增四项不加载模型的文件工具测试，覆盖中文／特殊字符和列顺序、空表头、创建拒绝覆盖、累计更新、失败清理。其他相关测试改为从新模块导入。
- 完整轻量测试 **80 项通过**；72 个 Python 文件语法解析通过，新模块与函数均有文档字符串。现有清单为 49 个源码、23 个测试／夹具文件；`git diff --check` 无格式错误。
- 本次未增加依赖，未采用 StandardScaler 或新的特征提取器，也未展开加载器和测试文件重分。原始清单流式生成与图块打包的多文件发布仍保留专用约束。不训练／推理正式数据，不修改既有权重、阈值参考或报告。

### 本轮新增与修改登记（2026-10-27—28）

以下对应本轮质量与 RMD 功能、共享工具整改的文件变更，逐文件职责见上方主清单；状态不替代后续修改后的重新检查。

| 状态 | 文件（源码路径相对 `src/leukocyte_analysis/`） | 本次内容 |
| --- | --- | --- |
| 新增／已检查 | `review_cells.py` | 拟合、质量校准与复核命令入口。 |
| 新增／已检查 | `evaluation/cell_review.py` | 纯决策规则，不再混入阈值校准。 |
| 新增／已检查 | `evaluation/novelty.py` | 第三方 RMD 适配和张量状态恢复。 |
| 新增／已检查 | `evaluation/quality_reference.py` | 独立质量阈值计算、保存与读取。 |
| 新增／已检查 | `evaluation/review_runner.py` | RMD 参考拟合与复核应用编排。 |
| 新增／已检查 | `reporting.py` | 数据、训练、评估共用报告发布和 CSV 工具。 |
| 修改／已复查 | `data/quality.py` | 保留拉普拉斯口径，补充归一化梯度评分。 |
| 修改／已复查 | `data/intensity_audit.py` | 移除独立 CSV 与报告发布实现，改用共享模块。 |
| 修改／已复查 | `evaluation/brightness.py` | 复用共享报告与 CSV 工具。 |
| 修改／已复查 | `evaluation/error_records.py` | 通用写入迁出，仅保留错分字段与定位。 |
| 修改／已复查 | `evaluation/runner.py` | 评估报告复用公共暂存发布。 |
| 修改／已复查 | `record_errors.py` | CSV 工具改从中立模块导入。 |
| 修改／已复查 | `training/cv_reports.py` | 预测 CSV 共用写入函数，仍拒绝覆盖。 |
| 修改／已复查 | `training/error_logging.py` | 共用 CSV 工具，保留逐轮防覆盖与累计更新语义。 |
| 新增／已检查 | `tests/test_cell_review.py` | 规则、命令、质量参考及合成闭环。 |
| 新增／已检查 | `tests/test_novelty.py` | RMD 评分、状态恢复与特征适配。 |
| 新增／已检查 | `tests/test_report_io.py` | 公共报告和 CSV 工具边界。 |
| 修改／已复查 | `tests/test_error_logging.py` | 更新公共写入工具导入，验证错分与续接语义。 |
| 修改／已复查 | `tests/test_quality.py` | 新质量评分与原有审计兼容测试。 |
| 迁出后移除 | `evaluation/review.py` | 报告发布迁入 `reporting.py`，不保留重复文件。 |

配套修改：README、`docs/reproduction.md`、`docs/experiment_index.md`、本清单记录功能、用法及限制；`pyproject.toml` 与 `environment.yml` 声明复核依赖及入口；本地 `AGENTS.md` 明确新增、修改、迁移和删除均须同步清单。上述配套文件不纳入 72 个 Python 文件计数。上一轮完整轻量测试为 **80 项通过**；本次仅维护约定和清单，不修改程序行为、不重复启动模型或测试。

### 校准来源与入口边界修复（2026-10-28，后续增量）

修复上轮未覆盖的质量评分来源检查；同时收敛错分导出入口和通用路径归属。前文 72 文件／80 测试为历史快照，当前文件清单以页首为准。

| 状态 | 文件（源码路径相对 `src/leukocyte_analysis/`） | 修改与复查结论 |
| --- | --- | --- |
| 新增／已检查 | `paths.py` | 迁入运行路径函数，集中项目根目录；只依赖 pathlib，不依赖训练或评估。 |
| 迁移后移除 | `training/run_paths.py` | 功能迁入根目录 `paths.py`，已更新全部调用，不留重复实现。 |
| 新增／已检查 | `evaluation/error_export.py` | 从 CLI 迁出错分导出业务，复用数据集、检查点恢复、预测、来源核对和 CSV 工具；直接调用也禁止测试集重新推理。 |
| 修改／已复查 | `record_errors.py` | 仅保留参数解析、业务调用与数量显示，参数及输出列不变。 |
| 修改／已复查 | `data/split.py` | 新增共用参考清单读取／隔离检查，RMD 与质量校准使用同一受试者和 CRF101 规则。 |
| 修改／已复查 | `evaluation/quality_reference.py` | 用完整来源清单验证评分覆盖、索引去重、受试者、标签与 val 归属；拒绝训练／测试图块或保留测试受试者参与；新产物额外记录清单摘要，旧质量参考仍可读取。 |
| 修改／已复查 | `evaluation/review_runner.py` | 复用数据层来源检查及中立路径模块，不重复实现隔离规则，不修改 RMD 算法或状态格式。 |
| 修改／已复查 | `review_cells.py` | 从中立模块取项目根，质量校准增加可选 `--split-csv`（默认原常规清单）并传给业务校验。 |
| 修改／已复查 | `evaluation/brightness.py` | 路径函数改从中立模块导入，比较流程不变。 |
| 修改／已复查 | `audit_brightness.py` | 项目根改为统一来源，默认报告路径不变。 |
| 修改／已复查 | `evaluate.py` | 项目根改为统一来源，评估逻辑不变。 |
| 修改／已复查 | `prepare_data.py` | 默认输出目录从统一项目根派生，不变更数据准备步骤。 |
| 修改／已复查 | `resume_cv.py` | 统一项目根来源，续训默认目录不变。 |
| 修改／已复查 | `train.py` | 不再从训练参数模块取项目根，训练参数与行为保持原样。 |
| 修改／已复查 | `cross_validate.py` | 统一项目根来源，交叉验证参数及行为不变。 |
| 修改／已复查 | `training/arguments.py` | 移除独立根目录计算，复用中立路径常量。 |
| 修改／已复查 | `training/cross_validation.py` | 路径函数更新导入，原计划兼容和分折规则不变。 |
| 修改／已复查 | `training/cv_plan.py` | 路径函数更新导入，计划字段与保存格式不变。 |
| 修改／已复查 | `training/cv_resume.py` | 路径函数更新导入，续训与原权重保护不变。 |
| 新增／已检查 | `tests/test_quality_reference.py` | 从复核测试迁入纯质量校准测试，补充训练／测试来源拒绝、索引重复／缺失、标签与 donor 错配、CRF101 隔离及旧参考兼容；不加载网络。 |
| 新增／已检查 | `tests/test_error_export.py` | 合成数据覆盖独立导出 API、现有预测复用、拒绝覆盖、顺序校验、禁止测试重复推理及薄 CLI 委托。 |
| 修改／已复查 | `tests/test_cell_review.py` | 质量参考测试迁出，闭环校准传入真实合成来源清单，共用来源检查的调用路径更新。 |
| 修改／已复查 | `tests/test_run_paths.py` | 更新路径导入，新增中立模块根目录位置断言，保留旧计划迁移兼容测试。 |

配套更新 README 目录说明及 `docs/reproduction.md` 的校准命令与输入契约；不新增依赖，不修改或覆盖 `data/processed/`、`runs/`、`reports/` 的已有产物，不重新计算现有阈值。不自动训练、推理正式图块或修改官方标签。

验证结果：首轮 87 项测试中，错分历史测试在 Windows 临时目录清理阶段遇到 `WinError 145`（目录非空），无业务断言失败；完整复跑 **87 项全部通过**。75 个 Python 文件语法解析通过，模块／类／函数文档字符串及主清单覆盖无缺项；当前为 50 个源码、25 个测试／夹具文件。旧质量参考读取、新校准来源拒绝和新旧运行路径兼容均包含在测试中。

### 复核流程展示图（2026-10-28，仅文档与静态图片）

- [x] 新增 `docs/assets/review_workflow.png`：沿用白底、蓝色分区、立体网络示意风格；展示独立质量评估、冻结 ResNet-18 的 512 维特征与 RMD、质量优先的三状态合并。不新增或重复实现程序功能。
- [x] 修改 `README.md`：在“图片质量与待定复核”嵌入可点击大图，说明启用质量参考的前提、质量不合格仍保留网络候选、阈值适用范围及非病变诊断边界。
- [x] 真实图片来源：现有图块包索引 `61318`（输入）、`7860`（低清晰度）、`4934`（高 RMD）、`62324`（规则通过）。用已有验证评分核对示例分别满足当前阈值条件；四图统一使用训练均值 ±4 标准差显示范围与最近邻插值，不合成、锐化或修改原图。来源信息与分数写入 PNG 的 Description 元数据。
- [x] 核对 `data/quality.py` 与 `evaluation/cell_review.py`：中央 45×45、平滑 σ=0.8、Sobel 能量／原灰度方差及质量优先规则与图一致。图中当前工作点为清晰度下限 8.060、RMD 上限 0.001496；不同真实样本仅作示例，不表示处理前后或病变标签。可选质量存疑分支仍以正文状态表为准。
- [x] 验证：PNG 3800×2000 可正常读取，已人工查看布局、文字及箭头，并检查 README 图片链接与补丁格式。本次不改 Python 或程序行为，不重跑单元测试、训练或推理；只读取四张已有原始图块与现有评分。未保留临时生成脚本；Python 文件数量仍为 75。
