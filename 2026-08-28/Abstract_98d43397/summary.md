---
title: "Abstract"
source: https://arxiv.org/pdf/2608.26563v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-04 06:47:23"
field: "Agentic LLM 预训练与工具调用"
keywords: ["Skill Pre-Training", "Agentic Language Models", "Tool Use", "Pre-training Data", "Reference Insert", "ClawHub", "Mid-training"]
innovations: ["提出 SPT，将可复用技能包作为 pre-training/mid-training 语料以替代高成本的 agent 轨迹数据", "设计 Reference Insert 多文件组装策略，显式保留技能包跨文件依赖结构", "构建去污染的多文件 SkillCorpus（38k 包/218k 文件），验证技能先验训练的有效性"]
benchmarks: ["API-Bank", "MetaTool", "APTBench", "ToolEyes", "OLMES", "BFCL", "OSWorld-MCP"]
---

# 论文速读：Abstract

## 一句话总结
本文提出 **Skill Pre-Training (SPT)**，将公开可复用的技能包（skill packages）作为 pre-training/mid-training 语料，在 post-training 之前为 Agentic LLM 注入工具语义与工作流先验，以替代高成本、低覆盖的 agent 轨迹数据。

## 研究问题与动机
- **现有方法依赖后训练轨迹**：当前 Agentic LLM 主要在 post-training 阶段通过工具调用轨迹与 agent trajectory 进行监督，数据生成需依赖任务环境、工具执行与结果验证，成本极高。
- **轨迹数据的结构性缺陷**：一条轨迹仅记录单次执行路径，无法泛化为可复用工作流；且合成数据规模与工具覆盖度难以满足 pre-training 级别的需求。
- **Skill 数据的未被充分利用**：公开 skill 包编码了可复用的工具语义、依赖关系与工作流，但目前仅被用作推理时的上下文输入，尚未被系统性地引入训练阶段。
- **数据供给快速增长**：npm 注册表中公开可用 skill 从 2026 年 2 月的 **170,226** 增至 6 月的 **1,640,440**（增长 **9.6×**），证明 skill 作为可扩展预训练数据源的可行性。

## 核心贡献（创新点）
- **提出 SPT 训练范式**：在 post-training 前引入基于 skill 包的 mid-training，采用标准因果语言建模目标，无需新增训练损失或任务环境。与现有轨迹 SFT/RL 方法相比，将能力获取前置至预训练阶段，显著降低对合成交互数据的依赖。
- **构建 SkillCorpus 清洗管线**：从 ClawHub 构建多文件技能数据集，建立涵盖发布者可信度审计、基准污染检测（API-Bank/MetaTool/APTBench/ToolEyes/OLMES）、二进制度过滤的流程，剔除约 **2%** 低质量包与 **~0.3%** 含 benchmark 特定 schema 的包。与以往直接使用原始仓库不同，本文强调训练前的 decontamination 与工程规范化。
- **设计 Reference Insert 多文件组装策略**：通过解析 `SKILL.md` 中的无歧义引用，将支持文件就地插入并在首次提及处标记 `<referenced_file_sep>`，剩余文件追加为 `<unreferenced_file_sep>`，最终打包为 **4,096-token** 训练块。与 DeepSeek-Coder Packing、Random File Order 等基线相比，显式保留跨文件依赖结构。
- **验证 Skill × 通用数据混合方案**：提出线性混合分布 $p_\alpha(x) = \alpha \cdot p_{\text{skill}}(x) + (1-\alpha) \cdot p_{\text{gen}}(x)$，在固定 token 预算下系统探索纯 skill、纯通用及混合条件，为后续研究提供 ablation 参考。

## 方法详解
- **训练目标**：沿用标准 causal language modeling，不修改模型架构或 loss 函数，仅在 mid-training 阶段替换语料分布。
- **SkillCorpus 构建流程**：
  1. 数据源：ClawHub 2026-05-01 快照。
  2. 包级筛选：仅保留身份验证、独立审计的完整发布包。
  3. 污染检测：调用 DeepSeek-V4-Flash API 对 5 个主流 benchmark 进行 cross-check，剔除衍生内容。
  4. 文本清洗：移除空/短内容、encoded blob、lock 文件、二进制及敏感字段，规范化路径；最终保留 **38,040 个包，218,277 个文件**（平均 **5.74** 文件/包）。
- **Reference Insert 组装逻辑**：
  - 顺序扫描 `SKILL.md`，识别对支持文件的明确引用路径。
  - 遇到引用时插入 `<referenced_file_sep>` 标记并立即嵌入对应文件内容，每个文件最多插入一次。
  - 扫描结束后追加所有未引用文件（带 `<unreferenced_file_sep>`）。
  - 包边界以分隔符闭合，切分为 **4,096-token** 序列；最终产出 **84,905 个训练块，约 347.8M tokens**。
- **混合策略**：固定总 token 预算 $B$，通过调节 $\alpha$ 控制 skill 语料占比；$\alpha=1$ 为纯 SPT，$\alpha=0$ 为通用 mid-training 对照组，$0<\alpha<1$ 为混合训练。

## 实验与结果
> 注：所提供的分段笔记中未包含实验结果的具体数值与图表，以下仅整理实验设置与评估框架，定量结论需参见原文后续部分。

- **Backbone 模型**（均仅经历单阶段通用 pre-training，未经 annealing/post-training）：
  - MeCo-1.6B-DCLM-160B
  - Instella-3B-Stage1
  - OLMo-3-1025-7B
- **Mid-training 对比条件**（等 token 数）：
  - Dolmino（通用数据基线）
  - AgentBank（agent trajectory 基线）
  - SkillCorpus（本文 skill 数据条件）
  - 直接 SFT（跳过 mid-training）
- **Post-training recipes**：Tulu 3（通用指令微调）、xLAM-FC（函数调用 SFT）
- **硬件与实现**：4× NVIDIA A100 80GB GPU，BF16 + TF32，gradient checkpointing
- **消融对比的组装策略**：DeepSeek-Coder Packing、Random File Order、Metadata Packing、Skill FIM

## 相关工作脉络
- **轨迹驱动工具学习（ToolLLM、Gorilla、ToolAlpaca、AgentBANK）**：依赖模拟或真实交互轨迹进行 SFT/RL；本文定位差异在于用静态、可复用的 skill 包替代动态轨迹，将能力注入提前至 pre-training/mid-training 阶段。
- **连续预训练增强框架（MetaData Conditioning、MaskSearch、Hephaestus）**：聚焦于通过元数据注入或掩码搜索提升基础能力；本文补充了“以结构化技能包为语料”的预训练路径，强调工作流与依赖关系的显式保留。
- **工具调用评测基准（API-Bank、MetaTool、APTBench、ToolEyes、OLMES、BFCL、OSWorld-MCP）**：主要用于 post-training 阶段的能力验证；本文将其纳入 pre-training 前的 decontamination 管线，防止测试集泄漏。
- **代码/工具大模型（DeepSeek-Coder、Qwen2.5-Coder、StarCoder 2）**：侧重代码生成能力；本文与之交叉但聚焦于“技能封装形式”而非纯源码，更贴近 agentic 场景中的多文件工作流组织。
- **RL/偏好优化方法（ToolRL、DPO、RLHF）**：通过 reward 信号或人类反馈调整行为分布；本文坚持无额外损失的标准 CLM 范式，证明仅靠语料结构优化即可实现能力前置。

## 局限性与未来方向
- **数据生态依赖**：SkillCorpus 主要来源于 npm/ClawHub 软件生态，对非代码类工具（如硬件控制、跨域 API、企业私有工具）覆盖有限。
- **跨文件依赖的隐式损失**：Reference Insert 虽保留显式引用顺序，但复杂的运行时依赖、环境变量与动态加载逻辑仍可能在 tokenization 过程中被扁平化。
- **混合比例的经验性**：$\alpha$ 的最优值依赖具体 backbone 与任务域，缺乏理论界的 scaling law 分析；当前仅展示 0/1/中间值的对比。
- **评测覆盖度**：实验聚焦于通用 agentic 基准，对长程多轮 agent、实时环境交互、跨平台 MCP 协议等新兴场景的验证尚待补充。

## 研究启发与可借鉴点
- **将开放仓库直接转化为训练语料**：证明 npm/ClawHub 等生态的 skill 包可作为低成本、高覆盖的 pre-training 数据源，适用于其他能力导向的模型构建（如数学工具、数据分析 agent）。
- **Reference-aware 序列组装**：Reference Insert 策略可直接迁移至多文件文档、代码库、技术手册的预训练中，帮助模型保留依赖拓扑结构。
- **Decontamination 前置化**：在 mid-training 阶段即使用多 benchmark 进行污染检测，为后续 SFT/RL 阶段提供更干净的基底，值得纳入标准数据清洗流程。
- **能力获取阶段前移**：将工具使用能力从 post-training 移至 mid-training，可显著减少后续 SFT 阶段的轨迹需求量，为资源受限团队提供更具性价比的 agentic 训练路径。

## 关键术语表
- **Skill Pre-Training (SPT)**：在 post-training 之前，使用技能包语料对模型进行因果语言建模的 mid-training 训练范式。
- **SkillCorpus**：从 ClawHub 筛选、审计、去污染后构建的多文件技能包数据集，包含 38,040 个包与 218,277 个文件。
- **Reference Insert**：多文件技能组装策略，依据 `SKILL.md` 中的无歧义引用将支持文件就地插入，并通过 `<referenced_file_sep>` / `<unreferenced_file_sep>` 标记维持跨文件语义结构。
- **Agent Trajectory**：记录模型与环境单次完整交互过程的序列数据（含工具调用、执行结果与状态转换），常用于 post-training 监督。
- **Decontamination**：利用 LLM 对候选语料进行多 benchmark 交叉检测，剔除与评测集重叠或包含基准特定 schema 的数据。
- **Mixing Formula**：$p_\alpha(x) = \alpha \cdot p_{\text{skill}}(x) + (1-\alpha) \cdot p_{\text{gen}}(x)$，控制 skill 语料与通用语料在 mid-training 中的分布权重。
- **ClawHub**：公开多文件 skill 包托管平台，本文选取其 2026-05-01 快照作为原始数据源。
- **APTBench / OSWorld-MCP**：分别用于评估基础 LLM 预训练期 agentic 潜力与计算机使用 agent 中 MCP 协议工具调用的评测基准。

## 可复现要素
- **数据集**：ClawHub 公开快照（2026-05-01）；经筛选后构建 SkillCorpus（38,040 包 / 218,277 文件）。**论文未明确声明是否开源**，建议查阅配套代码仓库确认。
- **代码/权重**：论文未提及
- **关键超参**：训练块长度 **4,096 tokens**；混合比例 $\alpha \in \{0, 1, \text{intermediate}\}$；硬件 **4× NVIDIA A100 80GB**；精度 **BF16 + TF32**；启用 **gradient checkpointing**；decontamination 调用 **DeepSeek-V4-Flash** API。
