---
title: "SAMpLE-A-SystemC-AMS-Machine-LEarning-based-Framework-for-Vi"
source: https://arxiv.org/pdf/2608.25910v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:26:11"
field: "电子系统级建模与虚拟原型"
keywords: ["SystemC-AMS", "虚拟原型", "Timed Dataflow", "在线学习", "ONNX", "机器学习集成"]
innovations: ["将ML模型作为原生TDF组件通过IMLModel统一接口集成到SystemC-AMS仿真", "双后端架构支持在线增量学习与离线ONNX部署在同一netlist下互换", "向SystemC-AMS标准提案新增ML原生primitive"]
benchmarks: ["UCI Appliances", "Tetuan City"]
---

# 论文速读：SAMpLE-A-SystemC-AMS-Machine-LEarning-based-Framework-for-Virtual-Prototyping

## 一句话总结
本文提出了 **SAMpLE**，一个开源的 SystemC-AMS 框架，将 ML 模型作为原生的 Timed Dataflow (TDF) 组件直接集成到嵌入式系统的虚拟原型仿真中，消除了现有 ad-hoc 集成方案导致的碎片化、不可复用和不可复现问题。

## 研究问题与动机
- **ML 模型在虚拟原型中的集成缺乏标准化机制**：SystemC/SystemC-AMS 语言本身未提供原生或标准化的 ML 模型集成接口，导致现有方法各自为政。
- **现有集成方案碎片化、工具链耦合严重**：主流做法包括（1）手动将模型重实现为 C++，工程量大且可移植性差；（2）通过 FMI/FMU 进行跨工具联合仿真，存在显著的同步开销；（3）围绕特定 ML 库开发应用专用接口，难以通用化。
- **异构 ML 模型无法在同一仿真环境公平比较**：不同模型通常在不同测试平台、数据集划分和评估指标下评测，缺乏统一对比基线。
- **在线学习（online learning）与离线部署（offline deployment）缺乏统一执行语义**：现有方法无法在同一 TDF 调度语义下支持"边仿真边学习"与"加载预训练模型"两种模式的互换对比。

## 核心贡献（创新点）
1. **ML 模型原生 TDF 集成方法论**：消除了对外部联合仿真框架、中间件层或手动模型翻译的依赖，使 ML 模型以"一等公民"身份运行于 SystemC-AMS TDF 调度内——这与 FMU 协作/跨进程通信方案有本质区别。
2. **统一的 IMLModel 抽象层**：通过 `required_window_size()` / `predict(window)` / `update(target)` / `supports_training()` 四个接口函数将"仿真语义"与"推理执行"解耦，使结构迥异的模型族（线性自适应、核方法、HMM 等）在相同 TDF 骨架下可无缝互换——现有工作均针对单一模型类型定制接口。
3. **双后端架构（ONLINE + OFFLINE）**：ONLINE 后端为轻量 C++ 原生实现，支持样本级增量学习；OFFLINE 后端基于 ONNX Runtime，支持 PyTorch/TensorFlow/scikit-learn 导出的预训练模型透明加载——二者共享同一 `IMLModel` 接口，这是现有 SystemC-AMS 工作所不具备的灵活性。
4. **开源可复现评估框架 + LRM 兼容生命周期管理**：提供 CSV 驱动的数据管线、确定性数据集划分（时序保持）、以及 R²/RMSE/MAE 自动评测报告，所有工件（配置、数据集、模型、预测、指标）随源码开放——对比现有方法缺乏系统级复现保障。
5. **向 IEEE 标准 primitives 的提案**：作者论证 IMLModel 合约（滑动窗口输入、固定尺寸输出、可选 ground-truth 更新、ready 标志）已足够整合异构模型族，建议将其纳入 SystemC-AMS 标准原生支持——这是现有框架均无的标准层贡献。

## 方法详解

**整体架构**：流程由两个输入驱动——`dataset.csv`（描述系统行为的时序数据）和 `config.json`（18 个键、五个 section 的统一配置源），经预处理后分流至两条训练路径。

**数据预处理流水线**（四阶段）：
1. **数据摄取**：从 CSV 提取 timestep 信息。
2. **数据清洗**：移除缺失/无效条目。
3. **特征工程**：推导时序与聚合特征，并对周期性属性做**循环编码**（cyclical encoding）：
   $$x_{\sin} = \sin\!\left(\frac{2\pi t}{T}\right),\quad x_{\cos} = \cos\!\left(\frac{2\pi t}{T}\right)$$
   其中 $t$ 为原始特征值、$T$ 为周期（如 $T=24$ 对应小时周期，$T=7$ 对应周周期），消除 23→0 时段的伪跳变。
4. **统计分析**：计算各特征的均值/标准差/最大/最小值，作为 ML 辅助输入及评测基准；最终按时序保持原则划分训练/验证/测试子集。

**在线训练路径（ONLINE backend）**：
- 在仿真启动时自动构建模型，无需独立训练阶段。
- 核心抽象 `IMLModel` 提供四个接口：
  - `required_window_size()`：首次有效输出前需积累的历史样本数。
  - `predict(window)`：将固定尺寸滑动窗口映射为预测值。
  - `update(target)`：根据最新真实标签执行单步增量参数更新。
  - `supports_training()`：运行时暴露是否支持框架内自适应。
- 已支持模型族：NLMS-ARX（归一化最小均方 ARX）、EW-RLS-ARX（指数加权递归最小二乘 ARX）、Hedge Ensemble（在线权重更新的集成预测器）、GHMM-Regime（高斯隐马尔可夫机制切换）、EKF-MLP（扩展卡尔曼滤波训练 MLP）、KRLS-ALD（近似线性依赖判据的稀疏核递归最小二乘）。

**离线训练路径（OFFLINE backend）**：
- 使用外部 Python 工具链（PyTorch/TensorFlow/scikit-learn）训练，导出为 ONNX 格式。
- ONNX 模型加载时执行两项校验：（1）静态输入形状检查（ONNX 输入节点数须匹配配置的特征向量宽度）；（2）零填充向量 dry-run 推理（确认图可调用且输出有限）。
- 配合序列化后的归一化/缩放参数，构成自包含部署工件。

**TDF 仿真语义**：
- ML 模型被表示为 TDF processing block，输入经 `features` 端口接收，预测从 `prediction` 端口发出，`ready` 信号在积累足够历史样本后拉高。
- 可选 `ground_truth` 和 `valid` 端口支持在线评测与异常检测。
- `processing()` 函数在每个激活步触发一次推理并更新预测端口；支持单步 ahead 预测（未来可扩展至多步）。
- 仿真结束时自动输出 R²、RMSE、MAE 三大回归指标。

## 实验与结果

**数据集**：
- **UCI Appliances**：19,735 条样本，10 分钟分辨率，噪声大、居住 occupancy 驱动，能耗 10–1080 Wh。
- **Tetuan City**：52,416 条样本，10 分钟分辨率，平滑周期性城市电力需求，13.9–52.2 kW。
- 两者均按**时序保持**原则划分，避免 temporal leakage。

**评估基线**：离线（Python）—— ARX Ridge、Random Forest、XGBoost；在线（C++ 原生）—— Hedge Ensemble、NLMS-ARX、Online-GP、EW-RLS-ARX、GHMM、GBT、EKF-MLP、KRLS-ALD。

**主要结果**：

| 维度 | 结论 | 关键数字 |
|------|------|----------|
| **Q1 保真度** | SystemC-AMS 离线推理与 Python 基线几乎一致 | 最大偏差 $\Delta R^2 = 0.003$（UCI 上 XGB） |
| **Q2 异构性** | 3 种离线 + 8 种在线模型在**同一 netlist** 下互换无代码修改 | 切换模型仅需改 `config.json` 一个字段 |
| **Q3 可移植性** | UCI → Tetuan 仅需换 CSV 与特征 schema，不改 C++/SystemC 代码 | — |
| **离线最佳（Tetuan）** | ARX Ridge $R^2 = 0.986$，所有离线模型聚拢 | RF 最慢 8.1s，ARX 仅 0.6s |
| **离线最差（UCI）** | $R^2 \in [0.13, 0.28]$，固定参数无法跟踪 regime shift | — |
| **在线最佳（UCI）** | **Hedge Ensemble $R^2 = 0.879$**，相对最优离线提升 >3 倍 | RMSE=31.5 Wh，运行时间仅 0.009s |
| **在线最佳（Tetuan）** | **NLMS-ARX $R^2 = 0.9961$**，超越离线 ARX（0.986） | 端到端 48ms，无需外部训练 |
| **在线速度跨度** | 9ms（线性滤波）→ 3.6s（Online-GP on Tetuan） | 三个数量级，均在 TDF 语义内测量 |

## 相关工作脉络
1. **FMU/FMI 联合仿真路线**（[7], [26]–[28]）：将 ML 模型包装为独立 FMU 由 master algorithm 编排。SAMpLE 定位差异——消除跨工具同步开销，直接在 TDF 内核内执行，无需中间件层。
2. **C++ 手动重实现路线**（[24]）：针对特定加速器模型手工翻译为 C++，移植性差。SAMpLE 通过 ONNX Runtime 实现框架无关部署，避免重实现。
3. **应用专用 TLM 接口路线**（[6], [29]）：围绕固定应用场景定制接口（如 NNSim 针对 DNN 加速器）。SAMpLE 通过 `IMLModel` 统一接口支持结构异质的模型族（线性→核方法→HMM→神经网络）。
4. **SystemC 建模 AI 加速器**（[24], [25]）：聚焦硬件加速器设计空间探索。SAMpLE 聚焦于 ML 模型作为**系统行为代理**融入虚拟平台仿真，而非加速器本身建模。
5. **Python 多保真联合仿真**（[5]）：外部 Python 运行时耦合方案。SAMpLE 消除 inter-process communication 开销，将推理完全嵌入 TDF 调度。
6. **ESL 虚拟原型与混合信号仿真**（[15]–[17]）：SystemC-AMS 通用框架扩展。SAMpLE 填补其中 ML 原生集成空白的细分位置。

## 局限性与未来方向
- **当前在线模型族有限**：仅覆盖 8 种算法（自适应线性、核方法、HMM 等），未包含深度序列模型（如 LSTM/Transformer），作者明确列为未来扩展。
- **仅支持单步 ahead 预测**：`processing()` 当前为 single-step-ahead，多步预测需抽象层扩展。
- **UCI 数据集上离线模型性能极差**（$R^2 \in [0.13, 0.28]$）：反映固定参数模型面对高度非平稳 occupancy 驱动场景的根本局限，需依赖在线自适应。
- **缺乏不确定性量化**：当前框架只输出点预测，无置信区间/方差估计，不利于安全关键场景决策。
- **标准提案尚未落地**：向 SystemC-AMS 委员会提交的 primitive 提案仍处于建议阶段，未被 IEEE 标准采纳。
- **超大规模平台验证缺失**：实验仅在两个公开时序数据集上验证，未在真实大型 SoC 虚拟平台（含复杂 TLM 网络）上测试。

## 研究启发与可借鉴点
1. **IMLModel 四接口模式可迁移至任何需要"在线学习 + 离线部署"互换的系统**：`required_window_size` / `predict` / `update` / `supports_training` 的解耦设计可作为通用模板，适配于边缘设备上的持续学习管线。
2. **循环编码公式（式 1）对任意周期性时序特征是即用技巧**：$T=24$（小时）/ $T=7$（周）/ $T=365$（年）可直接复用，避免周期性特征的边界伪跳变问题。
3. **双后端 + 统一接口的设计哲学可推广**：对于任何需同时支持"在线自适应"和"预训练加载"的仿真框架，可参考此 ONNX Runtime 桥接模式，减少重复开发。
4. **离线模型的 ONNX 校验两阶段法（静态形状检查 + zero-vector dry-run）值得借鉴**：可在模型加载层拦截错误，避免仿真中途崩溃，适用于所有 ONNX 部署场景。
5. **时序保持的数据集划分策略**：UCI/Tetuan 均按时间顺序切分训练/测试集，避免 future-to-past 信息泄漏——这对任何时序预测研究都是必要实践。

## 关键术语表
- **SystemC-AMS**：SystemC 的模拟/混合信号扩展标准（IEEE 1666.1），为 cyber-physical 系统提供多域虚拟原型仿真能力。
- **Timed Dataflow (TDF)**：SystemC-AMS 中的一种计算模型，系统将交互模块表示为按固定时钟周期消耗/产出样本的信号处理块。
- **IMLModel**：SAMpLE 定义的统一 ML 模型抽象接口，包含窗口大小查询、预测、增量更新、训练支持四个核心方法。
- **ONNX Runtime**：微软开发的跨平台 ONNX 模型推理引擎，支持 CPU/GPU 加速与图优化，本框架用于离线模型加载。
- **NLMS-ARX**：归一化最小均方（Normalized Least-Mean-Squares）增强的自回归外生（ARX）模型，支持样本级在线参数自适应。
- **Hedge Ensemble**：基于在线权重更新（multiplicative weights）的动态集成预测器，随仿真推进实时调整各子模型权重。
- **GHMM-Regime**：高斯隐马尔可夫模型（Gaussian Hidden Markov Model），用于捕获系统的潜在工作模式切换。
- **KRLS-ALD**：稀疏核递归最小二乘回归（Kernel Recursive Least Squares with Approximate Linear Dependency），通过线性依赖判据控制核方法计算复杂度。

## 可复现要素
- **数据集**：UCI Appliances（[36]）和 Tetuan City（[37]）均为 UCI Machine Learning Repository 公开数据。
- **代码开源**：仓库地址 https://github.com/andreialbu28/SAMpLE（[12]），含全部 artifacts（配置、数据集、模型、预测、指标报告）。
- **关键超参**：config.json 中指定模型类型、周期 $T$（默认 24/7）、滑动窗口大小、学习率等；具体数值见源码。
- **实验环境**：Intel Core i7-10700, 16 GB RAM, Ubuntu 22.04。
- **ONNX Runtime 版本**：1.20.0（[18]）。
- **测试集划分比例**：离线实验 15%（UCI: 2,954 / Tetuan: 7,855 样本）；在线实验 20%（UCI: 3,947 / Tetuan: 10,484 样本）。
