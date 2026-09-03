---
title: "TrustDABench-Benchmarking-Reliability-and-Robustness-of-LLMs"
source: https://arxiv.org/pdf/2608.24145v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:37:36"
field: "大语言模型评估与可信AI"
keywords: ["LLM可靠性", "表格推理鲁棒性", "结构化数据分析", "基准测试", "证据路径", "拒绝回答能力"]
innovations: ["提出统一证据路径视角下的可靠性与鲁棒性双维形式化定义", "设计19个受控扰动算子与Agentic-LLM自动构建流水线", "发现可靠性与鲁棒性排名不重合及冲突证据检测的跨模型盲区"]
benchmarks: ["TrustDABench", "AIDABench-QA", "DABench"]
---

# 论文速读：TrustDABench-Benchmarking-Reliability-and-Robustness-of-LLMs

## 一句话总结
论文提出了TrustDABench，一个用于评估大语言模型（LLM）在结构化数据分析中可靠性与鲁棒性的基准测试，通过19个受控扰动算子在AIDABench-QA和DABench上构建了2,340个人工验证样本，揭示了当前模型在证据边界识别和表示不变推理方面存在显著缺陷。

## 研究问题与动机
- **可信分析的证据路径依赖**：用户基于LLM进行电子表格和CSV数据分析时，期望答案不仅数值正确，还需有从问题到数据的可验证证据路径；现有工作仅关注答案正确性，忽视了证据完整性检查。
- **可靠性与鲁棒性的割裂评估**：现有基准（如RADAR、ToRR、RobuT）未区分"应拒绝回答的证据断裂场景"与"语义不变的表示变换场景"，导致模型诊断能力不完整。
- **模型在证据冲突上的系统性盲区**：实证观察显示，模型倾向于沿可执行路径继续计算而非验证证据充分性，对冲突证据（而非显式缺失）的拒答率极低（仅0.46% MRS）。
- **结构扰动下的表现脆弱性**：模型在改变观察边界或跨表关系的扰动（如NRI、CSR）下失败率高，暴露了对结构化表示的隐式依赖。

## 核心贡献（创新点）
1. **统一证据路径视角下的可靠性与鲁棒性形式化定义**：将两个维度归结为证据支持路径可达性与模型状态转换可靠性的联合约束，区别于以往仅衡量数值正确性的评测范式。
2. **19个可操作化的扰动算子体系**：7个可靠性算子（FDM、DM、FLM、DAM、SCM、EC、HC）与12个鲁棒性算子（L0-L3四个难度级别），通过属性对齐、任务相关性和最小干预原则设计，区别于表面编辑的对抗攻击。
3. **Agentic-LLM驱动的自动化构建与三层验证流水线**：采用选择器（Selector）→构造器（Constructor，ReAct循环）→验证器（Validator，规则+盲判）的级联框架，确保每个样本满足语义契约（答案可达性、证据充分性、答案等价性），人工验证保留率约86%。
4. **双维度的系统性模型诊断与失效模式分析**：发现可靠性与鲁棒性排名不重合（GPT-5.5最优可靠性24.21% MRS，Claude-Sonnet-5最优鲁棒性9.10% ASR），且冲突证据检测与观察边界变化是跨模型共性盲区。

## 方法详解
**形式化框架**：
- 定义答案可达性函数 $A(Q, T) \in \{0, 1\}$，当任务规格完整、语义锚定唯一、表证据充分且一致时为1。
- Agentic LLM的分析过程建模为Markov状态序列，完整路径可达条件为 $\Gamma_M(s_{1:K}|s_0) = \prod_{k=1}^K F_k R_k = 1$，其中 $F_k$ 为证据充分性（0/1），$R_k$ 为状态转换正确性（0/1）。

**可靠性算子（使 $F_k=0$）**：
- **field_missing (FDM)**：删除必要字段
- **data_missing (DM)**：将关键值替换为NULL
- **file_missing (FLM)**：多文件任务中删除必要文件
- **deep_analysis_missing (DAM)**：删除后期分析步骤所需的证据（Hard难度）
- **structural_context_missing (SCM)**：删除Excel多级表头/多Sheet结构标记（Hard难度）
- **evidence_conflict (EC)**：引入同一事实的不可解析冲突值
- **header_conflict (HC)**：将语义不同字段赋予相同表头

**鲁棒性算子（保持 $F_k=1$，改变表示）**：
- L0基础不变性：row_order_shuffle (ROS)、column_order_shuffle (COS)、header_synonym_substitution (HSS)、semantic_distractor_column (SDC)
- L1值变换：equivalent_value_reencoding (EVR)、unit_scale_conversion (USC)
- L2结构重组：csv_wide_long_reshape (WLR)、csv_relational_decomposition (RD)、excel_hierarchical_header_relayout (HHR)、excel_cross_sheet_relayout (CSR)
- L3语义干扰：decoy_feature_pack_injection (DFI)、non_observation_row_injection (NRI)

**验证契约**：
- 可靠性样本必须从 $A=1$ 变为 $A=0$，且无可靠恢复路径
- 鲁棒性样本必须保持 $A=1$，正确答案 $y^*$ 不变
- 三层验证：规则验证器（文件完整性、修改范围）→ LLM盲判验证器（独立重算答案）→ 人类专家验证（≥90%投票通过率）

## 实验与结果
**数据集**：基于AIDABench-QA（562可靠性+672鲁棒性）和DABench（643可靠性+463鲁棒性），共2,340个接受样本，覆盖19个算子。

**评估模型**：GPT-5.5、Claude-Sonnet-5、Qwen3.7-Max、Qwen3.6-27B、Qwen3-30B-A3B、DeepSeek-V4-Pro、GLM-5.2、Gemini-3.1-Pro。

**可靠性结果（MRS越高越好）**：
- GPT-5.5最优：AIDABench-QA 23.22%，DABench 25.19%，平均24.21%
- 所有模型均未超过25.19%，大部分NRR > 80%
- 冲突类算子（EC、HC）MRS仅0.46%，显著低于信息删除类（16.49%）
- 模型排名跨数据集Spearman ρ=0.81，但Qwen3-30B-A3B和Claude-Sonnet-5在DABench上下降明显

**鲁棒性结果（ASR越低越好）**：
- Claude-Sonnet-5最优：平均ASR 9.10%
- Qwen3-30B-A3B最差：平均ASR 36.99%
- NRI是普遍弱点（所有模型最高ASR），CSR是模型选择性弱点
- L3难度算子成功率是L0的2.82倍，Spearman ρ=0.59（p=0.044）

**跨维度差异**：可靠性最优模型（GPT-5.5）非鲁棒性最优（Claude-Sonnet-5），证明两维度为独立诊断维度。

## 相关工作脉络
- **RADAR (Gu et al. 2025)**：评估不完备表格上的数据感知推理，但未分离"应拒答"与"语义保持"两类场景
- **ToRR (Ashury-Tahan et al. 2026)**：表推理鲁棒性基准，仅测试prompt配置与结构扰动，缺少可靠性维度
- **RobuT (Zhao et al. 2023) 与 FREB-TQA (Zhou et al. 2024)**：表QA鲁棒性基准，聚焦静态扰动而非证据路径断裂
- **Text-to-SQL可靠性工作**（TrustSQL、CLARITY、TriageSQL等）：评估不可行查询检测，但未覆盖多文件、代码执行、中间结果综合的完整数据分析工作流
- **数据分析师基准**（InfiAgent-DABench、DSBench、SpreadsheetBench）：评估端到端任务完成，但未控制证据路径的可验证性

## 局限性与未来方向
- **算子分布不均**：未强制均匀分布因部分源任务无法构造合法扰动，可能低估某些操作类型的难度
- **单语言/单领域**：基准基于英文数据和商业/学术领域，跨语言与文化语境的泛化性待验证
- **静态扰动局限**：未评估动态交互场景（如多轮澄清对话、在线数据更新）下的可靠性演化
- **模型覆盖面有限**：仅评测8个主流模型，开源小模型与定制Agent框架的表现数据不足
- **未来方向**：将证据边界检查内化为模型 stopping condition，训练表示不变推理，开发动态扰动与交互式可靠性评估

## 研究启发与可借鉴点
1. **证据路径形式化可作为通用诊断框架**：将分析过程建模为状态转移序列并定义 $F_k$ 与 $R_k$ 分解，可迁移至代码生成、数学推理等链式任务的可信度评估
2. **三层验证流水线（规则→LLM盲判→人工）保障基准质量**：该架构可复用于其他需要高保真对抗样本的基准构建，特别是语义等价性验证环节
3. **难度分级（L0-L3）与操作类型正交分解**：将扰动按修改范围与复杂度分层，同时区分"等价变换"与"冗余注入"，为鲁棒性评测提供细粒度诊断
4. **响应模式三维分类（FRR/PRR/NRR）**：突破二元正确/错误，捕捉"部分识别但过度自信"的中态失败，适用于校准性评估
5. **跨维度不重合的发现方法论**：同时评估相互独立的能力维度可揭示模型架构的隐式假设，启发后续工作设计正交能力矩阵

## 关键术语表
- **Evidence-supported path（证据支持路径）**：从用户问题到答案的可验证推理链路，每步需有充分、唯一、一致的数据证据支撑
- **Mean Reliability Score (MRS)**：可靠性主指标，全拒答得1分、部分拒答得0.5分、无拒答得0分，均值反映模型在不可答任务上的证据 grounding 能力
- **Answer Success Rate (ASR)**：鲁棒性主指标，衡量语义保持扰动导致原正确任务失败的比例，排除基线能力干扰
- **Robustness Accuracy Drop (RAD)**：源问题归一化的准确率下降，防止扰动数量多的问题主导数据集级指标
- **Agentic-LLM generation framework**：基于ReAct循环的多LLM协作流水线，含Selector、Constructor（带Python/Bash工具调用）、Validator三层角色
- **Validity Contract（有效性契约）**：约束每个扰动样本必须满足的语义条件（答案可达性、证据充分性、答案等价性、无新歧义）
- **Failure-stage analysis（失效阶段分析）**：定位LLM在证据验证链中的具体断裂点（schema绑定、值充分性检查、一致性检查等）

## 可复现要素
- **数据集**：TrustDABench基于AIDABench-QA和DABench构建，原始数据地址：AIDABench-QA (arXiv:2603.15636)，DABench (arXiv:2401.05507)
- **代码/权重**：论文未提供官方代码仓库链接，但附录F提供了完整的Operator Registry Python代码与算子prompt定义
- **关键超参**：人工验证阈值≥90%专家投票通过率；构造器工具调用预算≤3次Python调用；验证器工具调用预算≤5次
- **评估配置**：Agentic reasoning with code sandbox (Python+Bash)，提示与推理轮数跨模型共享，模型专属参数遵循官方配置
