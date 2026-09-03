---
title: "TrustDABench-Benchmarking-Reliability-and-Robustness-of-LLMs"
source: https://arxiv.org/pdf/2608.24145v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:37:35"
field: "大模型评测基准"
keywords: ["结构化数据分析", "大语言模型", "可靠性", "鲁棒性", "基准评测", "扰动算子", "拒绝推理"]
innovations: ["从证据支撑路径统一形式化可靠性和鲁棒性两个正交诊断维度", "设计19个受控扰动算子（7可靠性+12鲁棒性）并建立Agent-LLM自动化构造管线", "首次在agentic数据分析场景下评测LLM拒答能力和表示不变推理能力"]
benchmarks: ["TrustDABench", "AIDABench-QA", "DABench"]
---

# 论文速读：TrustDABench: Benchmarking Reliability and Robustness of LLMs for Structured Data Analysis

## 一句话总结
本文提出 **TrustDABench**，一个面向结构化数据分析任务的评测基准，从"证据支撑路径"视角统一度量 LLM 的**可靠性**（无法回答时能否主动拒绝或澄清）和**鲁棒性**（在语义等价变换下能否保持正确答案）。共构建 2,340 个人工验证的扰动样本，评测了 8 个代表性 LLM。

## 研究问题与动机
- **正确答案 ≠ 可信分析**：LLM 在分析电子表格/CSV 时能生成看似合理的答案，但该答案可能依赖缺失列、歧义表头、冲突记录或被误解的表格表示，缺乏从问题到数据的**有效证据路径**。
- **现有基准的不足**：Table QA 基准主要测答案正确性；RADAR、ToRR 等未将"应触发拒绝的证据断裂案例"与"应保持答案不变的语义保留案例"分离评测。
- **两类能力不可互换**：可靠性要求模型识别证据边界并停止推理；鲁棒性要求模型在证据被改写/冗余干扰时仍能完成推理。两者揭示不同的失败机制。
- **实证缺口**：当前模型普遍存在"假设驱动"行为——即使证据不足也继续沿可执行路径计算，而非主动拒绝。

## 核心贡献（创新点）
1. **统一形式化**：从"证据支撑路径（evidence-supported path）"视角统一刻画可靠性与鲁棒性，区分**证据边界识别**与**有效路径恢复**两个独立能力维度。
2. **19 个受控扰动算子**：7 个可靠性算子（基于信息删除与证据冲突）+ 12 个鲁棒性算子（基于等价变换与冗余注入），按修改范围和复杂度分为 L0–L3 四级。
3. **Agent-LLM 自动化构造管线**：利用 LLM Selector → LLM Constructor → Rule-based/LLM Validator 三步流水线自动生成扰动样本，并通过 90% 专家共识阈值的人工审核保留约 86% 候选。
4. **系统性评测 8 个 LLM**：发现最佳可靠性（GPT-5.5，平均 MRS = 24.21%）与最佳鲁棒性（Claude-Sonnet-5，平均 ASR = 9.10%）**不重叠**，两者是不互通的诊断维度。

## 方法详解

### 任务形式化
将 Agent LLM 的结构化数据分析建模为 **Markov 状态序列**，每步 $s_k$ 需同时满足：
- $F_k \in \{0,1\}$：证据充分、一致、准确
- $R_k \in \{0,1\}$：模型能从当前证据正确过渡到下一状态

完整路径可达当且仅当 $\prod_{k=1}^{K} F_k R_k = 1$。

### 可靠性（Reliability）
对原始可回答任务施加扰动 $\delta \in \Delta_{\mathrm{rel}}$ 使 $A(Q, T') = 0$。单次实例判据：

$$\mathrm{Rel}_M(Q, T') = [\Gamma_M(s_{1:K}|s_0) = 0] \cap [M(Q, T') = \bot]$$

其中 $\bot$ 表示有依据地拒绝或请求澄清。度量指标：
- **MRS**（Mean Reliability Score）：$\frac{1}{M}\sum r_i$，$r_i \in \{1(\text{完全拒绝}), 0.5(\text{部分拒绝}), 0(\text{不拒绝})\}$
- FRR / PRR / NRR 分别报告三种响应模式比例

### 鲁棒性（Robustness）
对原始可回答任务施加 $\delta \in \Delta_{\mathrm{rob}}$ 保持 $A(Q, T') = 1$ 且答案不变。判据：

$$\mathrm{Rob}_M(Q, T') = [\Gamma_M(s_{1:K}|s_0) > 0] \cap [M(Q, T') = y^*]$$

度量指标：
- **ASR**（Average Success Rate of Perturbation）：原任务答对的情况下，扰动导致答错的比率，隔离基线能力
- **RAD**（Robustness Accuracy Drop）：按源问题归一化的准确率损失，防止扰动数量多的问题主导

### 19 个扰动算子
- **可靠性（7 个）**：`field_missing`（删除必要字段）、`data_missing`（替换为 NULL）、`file_missing`（删除多文件中的必要文件）、`deep_analysis_missing`（删除后期分析所需证据）、`structural_context_missing`（删除结构标记）、`evidence_conflict`（引入无法解决的冲突值）、`header_conflict`（不同字段赋予相同表头）。
- **鲁棒性（12 个，L0–L3）**：`row_order_shuffle`/`column_order_shuffle`/`header_synonym_substitution`（L0 基础不变性）；`equivalent_value_reencoding`/`unit_scale_conversion`（L1 值变换）；`csv_wide_long_reshape`/`csv_relational_decomposition`/`excel_hierarchical_header_relayout`/`excel_cross_sheet_relayout`（L2 结构重组）；`semantic_distractor_column`（L0 冗余）、`decoy_feature_pack_injection`/`non_observation_row_injection`（L3 强干扰）。

### 构造管线
Task Profiler → LLM Selector（选择适用算子并定位目标）→ ReAct-loop LLM Constructor（执行编辑生成 $T'$）→ Rule-based Validator + LLM Validator（独立验证回答性）→ 专家人工审核（≥90% 同意率保留）。

## 实验与结果

| 维度 | 数据集 | 样本数 | 算子数 |
|---|---|---|---|
| 可靠性 | AIDABench-QA | 562 | 7 |
| 可靠性 | DABench | 643 | 6 |
| 鲁棒性 | AIDABench-QA | 672 | 12 |
| 鲁棒性 | DABench | 463 | 6 |
| **合计** | — | **2,340** | **19** |

**关键结果**：
- **可靠性**：GPT-5.5 最优，平均 MRS = 24.21%，但仍对约 2/3 不可回答实例未能正确拒绝；所有模型 MRS 均未超过 25.19%。
- **鲁棒性**：Claude-Sonnet-5 最优，平均 ASR = 9.10%；Qwen3-30B-A3B 最差（ASR = 36.99%）。
- **冲突 vs 缺失**：冲突类算子（EC、HC）MRS 仅 0.46%，显著低于信息删除类（16.49%），是通用盲点。
- **最强扰动算子**：`non_observation_row_injection`（NRI）对所有模型都是最致命的鲁棒性扰动；`excel_cross_sheet_relayout`（CSR）具有强模型选择性。
- **可靠性失败阶段分布**：schema/证据绑定（FDM，27.1%）、schema 歧义检测（HC，19.6%）、值充分性检查（DM，18.4%）、证据一致性检查（EC，16.7%）。**无声不支持答案**占无拒绝案例的 74.8%。
- 模型排序跨数据集大致一致（Spearman $\rho = 0.81$），但无模型在绝对意义上表现可靠。

## 相关工作脉络
1. **Table QA 鲁棒性基准**（RobuT、FREB-TQA、ToRR）：仅测静态表格扰动下的正确性，不评估"何时应拒绝"，缺少可靠性维度。
2. **RADAR**：评估不完备表格下的数据感知推理，但未区分应拒绝案例与应保持答案的语义保留案例。
3. **Text-to-SQL 可靠性研究**（TrustSQL、CLARITY、TriageSQL 等）：聚焦 SQL 生成的不可行查询/歧义检测，未覆盖完整的 agentic 结构化数据分析工作流（文件检查→数据清洗→代码执行→结果合成）。
4. **结构化数据分析基准**（InfiAgent-DABench、DSBench、SpreadsheetBench）：评估真实任务完成能力，但未联合测试"拒绝不可回答"与"鲁棒于等价变换"两个条件。
5. **本文定位**：首次在有工具调用的 agentic 数据分析场景下，将可靠性（拒答能力）与鲁棒性（表示不变推理）作为正交诊断维度联合评测。

## 局限性与未来方向
- 基准从现有可回答样本出发，未覆盖从"天然不可回答"任务出发的可靠性评测。
- 部分算子（如 `structural_context_missing`，仅 18 例）样本量偏低，源于难以构造合法扰动。
- 构造管线本身依赖中间 LLM（claude-haiku 4.5 / Kimi-K3 / gpt-5.4），存在模型偏差风险。
- 专家人工审核效率较低，每实例需回顾完整构造轨迹。
- 未来方向：训练模型将证据边界检查内化为推理链中的持续停止条件，而非依赖提示词；增强对多样化等价表示的表征不变推理能力。

## 研究启发与可借鉴点
1. **正交维度分解**：将"是否应该回答"和"回答是否正确"分离为可靠性与鲁棒性两个诊断维度，避免了单一正确率指标的掩盖效应，该方法论可迁移到其他推理任务评测。
2. **算子分级框架（L0–L3）**：按修改范围和复杂度对扰动进行分类并量化难度梯度，为后续设计可控难度的评测提供了模板。
3. **Agent-LLM 自动化构造管线**：Selector-Constructor-Validator 三级流水线 + 人工专家最终仲裁的模式，在保证样本质量的同时大幅降低人工标注成本，可复用于其他基准构建。
4. **响应模式细粒度分析**（FRR/PRR/NRR）：不仅看是否拒绝，还区分"完全拒绝/部分承认但不拒绝/完全不拒绝"，为失败模式诊断提供更丰富信号。
5. **证据路径形式化**（Markov 状态序列 × $F_k \cdot R_k$ 乘积条件）为后续研究提供了一个可计算的"推理链可信度"建模框架。

## 关键术语表
**Evidence-supported path（证据支撑路径）**：从用户问题出发，经证据定位、工具调用、数据处理到答案生成的完整可追溯推理链，每一步均需有充分且一致的数据证据支撑。
**MRS（Mean Reliability Score）**：可靠性主指标，衡量模型在不可回答任务中给出有据拒绝/澄清请求的平均得分。
**ASR（Average Success Rate of Perturbation）**：鲁棒性主指标，衡量原任务答对的样本在语义保留扰动下答错的比率。
**RAD（Robustness Accuracy Drop）**：按源问题归一化的鲁棒性准确率损失，防止扰动数量不均导致的指标偏差。
**FRR / PRR / NRR**：完全拒绝率 / 部分拒绝率 / 不拒绝率，三分类响应模式统计。
**Agentic LLM**：具有工具调用能力（Python/Bash 代码沙箱）、可在多轮交互中执行数据分析的 LLM 智能体。
**Equivalence-preserving perturbation**：不改变问题语义和正确答案的表格表示变换，用于测试模型的表示不变推理能力。

## 可复现要素
- **数据集**：TrustDABench（基于 AIDABench-QA 和 DABench），2,340 个扰动实例；论文未明确声明开源链接，但 arXiv 附补充材料含算子定义、提示词和统计。
- **代码/权重**：论文未提供公开代码仓库链接，构造管线依赖 claude-haiku 4.5 / Kimi-K3 / gpt-5.4 等闭源模型。
- **关键超参**：任务级 prompt 和推理轮次跨模型共享；模型侧参数使用官方配置；专家审核阈值 ≥90% 同意率。
- **评测设置**：每模型在配备 Python + Bash 工具的代码沙箱中执行 agentic 推理。
