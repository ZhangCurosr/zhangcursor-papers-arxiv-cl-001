---
title: "WikiSkill-Compiling-Agent-Experience-into-Persistent-Knowled"
source: https://arxiv.org/pdf/2608.27454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:46:57"
field: "Agent 技能演化与知识积累"
keywords: ["Agent Skill Evolution", "Persistent Knowledge Base", "Wiki Layer", "Self-Improving Agent", "Cross-Model Transfer"]
innovations: ["提出 Raw/Wiki/Skill 三层架构，将执行经验持续编译为持久结构化知识库以支持跨迭代技能演化", "设计 Wiki 永不回滚而技能可回滚的双轨机制，确保失败教训累积复用", "系统性揭示技能发现（discovery）与技能执行（execution）是可分离的能力，并验证跨模型迁移的有效性"]
benchmarks: ["LiveMathematicianBench", "SealQA", "SpreadSheetBench", "OfficeQA", "ALFWorld"]
---

# 论文速读：WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution

## 一句话总结
WikiSkill 提出了一种将 Agent 执行经验持续编译为结构化持久知识库（Wiki）的框架，使技能演化能够建立在跨迭代积累且不断精炼的知识之上，从而在五个基准和五种模型上一致超越了现有技能演化方法和无技能基线。

## 研究问题与动机
- **核心问题**：现有 Agent 技能演化方法（如 EvoSkill、Trace2Skill、SkillOpt）在从执行轨迹中提取经验时，将学习到的洞察分散在优化历史中，缺乏独立的、可累积的知识表示层，导致跨迭代难以系统化复用。
- **动机1**：Agent 技能（Skill）作为轻量级、可复用的文件系统模块，能够封装领域专有知识和工作流而不更新模型参数，但有效技能的自动发现仍极具挑战——大多数技能依赖人工编写，需提前预判 Agent 所需的知识。
- **动机2**：Karpathy (2026) 提出将 LLM 经验编译为持久、复利的知识（LLM Wiki），本文受此启发，询问能否将 Agent 经验以同样方式编译为支持长期技能演化的持久知识。
- **动机3**：现有方法缺少对"什么已被学会"的独立、持续演化的知识表示，无法让后续技能更新建立在日益完善和整合的知识之上。

## 核心贡献（创新点）
1. **三层知识架构**：将 Agent 工作区划分为 Raw Layer（不可变执行轨迹）、Wiki Layer（持久结构化知识库）和 Skill Layer（可执行技能），实现了经验、知识与可执行技能的显式分离。*与已有工作相比，EvoSkill/Trace2Skill/SkillOpt 均未维护独立演化的知识表示层，本文的核心区别在于引入了跨迭代累积的 Wiki 层。*
2. **Wiki 感知的技能演化循环**：设计了 Wiki Maintainer（模式整合）→ Skill Proposer（ReAct 自主探索提出技能更新）→ Gating & Rollback（验证门控）的编排循环，其中 Wiki 永不回滚而技能可回滚。*与已有工作的本质区别在于：知识积累与技能演化解耦，失败的提议被记录但不阻止未来利用相同知识的改进。*
3. **系统性的跨模型技能迁移研究**：首次在五个基准和五个模型上系统研究了演化技能的跨模型/跨家族迁移，发现 Skill Discovery 与 Skill Execution 是可分离的能力。*与已有工作相比，现有方法仅关注自演化，本文揭示了其他模型演化的技能有时优于自身演化技能这一现象。*

## 方法详解
**三层知识架构**（§3.1）：
- **Raw Layer（raw/）**：存储每轮迭代中从训练样本收集的全部不可变执行轨迹 $\tau_i$，包含完整的多步交互（推理、工具调用、输出、最终答案）。Wiki Maintainer 和 Skill Proposer 可访问此层，但对推理 Agent 训练期间封闭。
- **Wiki Layer（wiki/）**：将原始轨迹编译为结构化的、跨迭代累积的知识。包含：patterns/ 目录（按失败模式或成功策略组织的 markdown 文件）、index.md（模式目录索引）、logs.md（演化日志）、skill-impact.md（技能影响追踪器，记录每次提议的 diff 和采纳结果）。Wiki 永不重置，持续累积。
- **Skill Layer（skills/）**：包含活跃的已演化技能集 $\mathcal{S}$，每个技能目录含 SKILL.md（完整技能内容）和 PURPOSE.md（追溯到激发动机的 Wiki 模式）。

**演化循环**（§3.2，Algorithm 1）：
- **Inference Agent**（§3.2.1）：第 $k$ 轮用当前技能集 $S_{k-1}$ 在训练集 $\mathcal{D}_{\text{train}}$ 上执行 rollout，生成轨迹 $\mathcal{T}_{\text{train},k}$。技能内容完整注入系统提示（full-injection），不使用检索机制。
- **Wiki Maintainer**（§3.2.2）：从轨迹中按分层采样（最多5条失败+3条成功，每条≤15,000字符）得到 $\mathcal{T}_{\text{sample},k}$，执行根因分析，提取成功策略，以增量 patch 方式更新 Wiki（创建/编辑 patterns、更新 index.md、追加 logs.md），产出中间 Wiki 状态 $W'_k$：
  $$W'_k \gets \mathcal{M}_{\text{WM}}(W_{k-1}, \mathcal{T}_{\text{sample},k})$$
- **Skill Proposer**（§3.2.3）：以 ReAct 风格自主运行，先获取 Wiki 索引和 skill-impact.md，再通过 `read_file` 按需读取特定 pattern 页和原始轨迹进行诊断，生成原子提议 $P_k$（创建新技能或对现有技能做 patch edit）：
  $$P_k \gets \mathcal{M}_{\text{P}}(W'_k, S_{k-1}, \mathcal{T}_{\text{train},k})$$
- **Gating and Rollback**（§3.2.4）：将 $P_k$ 应用于 $S_{k-1}$ 得候选 $S'_k$，在验证集上评估：
  $$S_k \gets \begin{cases} S'_k & \text{if } \mathcal{R}(\mathcal{T}_{\text{val},k}) > \mathcal{R}_{\text{best}} \\ S_{k-1} & \text{otherwise} \end{cases}$$
  若接受则更新 $\mathcal{R}_{\text{best}}$，否则回滚技能。**但 Wiki 永不回滚**，每次验证后通过 `Update(W'_k, P_k, R(T_{val,k}), a_k)` 记录采纳结果到 skill-impact.md。当 $\mathcal{R}_{\text{best}} = 1.0$ 时提前终止。

## 实验与结果
**数据集**（5个基准）：
- LiveMathematicianBench（数学推理，单选，35 train / 18 val / 124 test）
- SealQA（网页搜索问答，16 train / 10 val / 85 test）
- SpreadSheetBench（电子表格操作，80 train / 40 val / 280 test）
- OfficeQA（长文档QA，50 train / 24 val / 172 test）
- ALFWorld（交互式具身任务，39 train / 18 val / 134 test）

**模型**：Qwen-3.5-4B、Qwen-3.5-9B、Qwen-3.6-27B、Gemma-4-31B、Gemini-3.5-Flash（各3次独立运行取平均）。

**主要结果**（Table 1）：
- WikiSkill 在所有五个模型上均取得最高平均性能。
- 相对于最强竞品方法，WikiSkill 平均提升：Qwen-4B +3.3、Qwen-9B +5.1、Qwen-27B +10.0、Gemma-31B +5.8、Gemini-Flash +12.0 分。
- 相对于无技能基线：Qwen-4B 26.2%→38.5%（+12.3）、Qwen-9B 29.9%→47.4%（+17.5）、Qwen-27B 39.4%→63.3%（+23.9），增益随模型规模递增。
- **最强结果**：WikiSkill + Gemini-3.5-Flash 在 SpreadSheet 上达 76.6%（vs 基线 50.5%，+26.1），LiveMath 72.6%（vs 33.0%，+39.6）；Qwen-3.6-27B 在 SpreadSheet 上达 81.7%（vs 40.8%，+40.9）。
- **小模型超越大模型**：Qwen-3.5-9B + WikiSkill (47.4%) 超过 Qwen-3.6-27B 无技能 (39.4%)。

**跨模型迁移**（Table 2）：
- WikiSkill-27B 演化的 SpreadSheet 技能将 Qwen-9B 从 24.3% 提升至 50.5%（远超自演化的 33.6%）。
- WikiSkill-9B 演化的 ALFWorld 技能使 Qwen-9B 达 69.2%，超过其自演化的 63.4%。
- **负迁移案例**：WikiSkill-4B 的 SpreadSheet 技能使 Gemini-Flash 从 50.5% 降至 18.1%，原因是小模型技能编码了低层 workaround 且引入冗余工具调用耗尽了大模型的交互预算。

**消融**（Table 3，Gemini-3.5-Flash）：
- 移除 Wiki（仅用 Trace2Skill 风格）：平均分从 63.7% 降至 48.7%（-15.0%），证明持久知识积累至关重要。
- 训练期间给 Inference Agent 开放 Wiki 访问：平均分从 63.7% 降至 60.9%（LiveMath 72.6%→64.8%），说明推理时访问 Wiki 会削弱技能本身的学习信号。

## 相关工作脉络
1. **EvoSkill**（Alzubi et al., 2026）：将技能演化建模为候选程序搜索，维护有界 frontier，仅向 Proposer 提供失败轨迹和扁平反馈历史。*区别：EvoSkill 无持久知识层，不累积结构化经验。*
2. **Trace2Skill**（Ni et al., 2026）：三阶段并行轨迹分析+分层合并，每轨迹一次独立 LLM 调用，复杂度 $O(N_{\text{train}})$。*区别：Trace2Skill 无跨迭代知识持久化，且 API 调用量随训练集规模线性增长。*
3. **SkillOpt**（Yang et al., 2026）：六阶段 ReflACT 流水线（Rollout-Reflect-Aggregate-Select-Update-Evaluate），更新单一单体技能文档。*区别：SkillOpt 无独立知识库，经验未结构化积累。*
4. **GEPA**（Agrawal et al., 2026）：通用自动 prompt 优化器。*本文聚焦专用技能演化框架，因专用管道普遍优于通用 prompt 优化（Yang et al., 2026）。*
5. **Skill retrieval 相关**（SkillRetriever、SkillRouter、Skill1 等）：关注从技能库中检索相关技能。*本文与检索正交——聚焦技能质量本身，假设技能已全量注入提示。*
6. **Karpathy (2026) LLM Wiki**：提出将 LLM 经验编译为持久、复利知识的理念。*本文是其在 Agent 技能演化场景的首个系统性实现。*

## 局限性与未来方向
1. **技能检索未评估**：本文采用 full-injection 设置（所有技能直接注入提示），未评估技能检索/触发机制，而随着技能数量增长检索变得重要。
2. **严格门控排除中性提议**：验证门控要求每次提议必须提升验证分数，排除了可能短期持平但长期有益的"中性提议"；探索更灵活的接受准则是未来方向。
3. **Wiki 缺乏自动剪枝机制**：Wiki 层持续累积 pattern 页面和日志，但当前无自动化 pruning 机制，长期演化后可能需剪枝。
4. **未覆盖超长时域任务**：基准不包括跨越数百环境动作或多小时的任务；开发单次长 rollout 内的在线技能适应方法是有价值的未来方向。

## 研究启发与可借鉴点
1. **"知识-技能分离"的三层架构设计**：Raw/Wiki/Skill 三层的显式分离思路可迁移到任何需要跨迭代学习的 Agent 系统中——尤其是反思式 Agent、self-improving Agent pipeline，可作为通用的知识积累基础设施。
2. **Wiki 不回滚 + 技能可回滚的双轨机制**：这一设计保证了失败教训不会丢失（记录在 Wiki 中供未来参考），同时允许技能灵活试错，值得在 RL-based skill discovery 或 prompt evolution 场景中借鉴。
3. **Skill Proposer 的 ReAct 自主探索模式**：Proposer 不预设采样轨迹，而是通过工具调用按需阅读 Wiki 和原始轨迹进行诊断，这种"按需信息检索+诊断"模式可降低固定采样的信息损失，适用于复杂根因分析场景。
4. **小样本分层采样策略**（最多5失败+3成功）：高效利用有限上下文预算获取诊断信号，对资源受限的 Agent 实验有直接参考价值。
5. **跨模型技能迁移的系统性评估范式**：本文的"同源技能在不同目标模型上的迁移矩阵"实验设计，可作为评估技能可移植性的标准范式的参考。

## 关键术语表
**WikiSkill**：一种将 Agent 执行经验持续编译为持久结构化知识库（Wiki）以支持跨迭代技能演化的框架。
**Skill Layer**：包含可执行技能（SKILL.md + PURPOSE.md）的工作区层，直接注入推理 Agent 的 system prompt。
**Wiki Layer**：持久化的知识积累层，包含 pattern 页面、演化日志和技能影响追踪，跨迭代不重置。
**Raw Layer**：不可变的原始执行轨迹存储层，记录 Agent 的完整多步交互历史。
**Gating and Rollback**：基于验证集性能的提议筛选机制——技能接受或回滚，但 Wiki 永久保留。
**Skill Proposer**：基于 ReAct 范式的 LLM Agent，负责探索 Wiki 和轨迹后生成原子技能修改提议。
**Wiki Maintainer**：分析执行轨迹、执行根因分析并增量更新 Wiki 的 LLM Agent。
**Full-injection**：将所有技能内容直接注入 Agent 系统提示的设置，避免技能检索/触发成为混淆变量。

## 可复现要素
- **数据集**：LiveMathematicianBench、SealQA、SpreadSheetBench、OfficeQA、ALFWorld——论文引用原工作，数据格式和切分与先前工作对齐（附录 B 提供了详细统计）。
- **代码/权重**：论文未明确声明代码开源状态。使用的模型包括 Qwen-3.5-4B/9B-Instruct、Qwen-3.6-27B、Gemma-4-31B-It（open-weight）和 Gemini-3.5-Flash（closed）。
- **关键超参**：每轮采样轨迹上限 8 条（最多5失败+3成功），每条轨迹截断至 15,000 字符；批量大小 $B = N_{\text{train}}$（全批训练）；ReAct 推理轮数 $T_{\text{ReAct}} \approx 10-20$；独立运行 3 次取平均；配对 bootstrap 显著性检验 $p < 0.05$（1,000 次迭代）。
- **部署**：open-weight 模型使用 vLLM 框架部署。
