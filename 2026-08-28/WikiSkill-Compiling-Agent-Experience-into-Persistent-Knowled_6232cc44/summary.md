---
title: "WikiSkill-Compiling-Agent-Experience-into-Persistent-Knowled"
source: https://arxiv.org/pdf/2608.27454v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:46:50"
---

# 论文速读：WikiSkill: Compiling Agent Experience into Persistent-Knowled

## 一句话总结
提出 WikiSkill 框架，通过引入持久化知识库（Wiki）将智能体的执行经验系统编译并跨迭代复用，实现技能与知识的协同演化；在五大基准与多种模型上持续优于现有技能演化方法及无技能基线，并揭示技能演化与模型规模的互补性及跨模型迁移规律。

## 研究问题与动机
- 现有自动技能演化方法（EvoSkill、Trace2Skill、SkillOpt）将优化过程中的洞察分散存储于历史记录或单次轨迹中，缺乏独立的、持续演进的知识表征层。
- 智能体在迭代中积累的成败经验难以被后续轮次系统性复用，导致技能演化停留在碎片化反馈层面，重复提出无效修改。
- 技能发现（经验编译）与技能执行（推理时调用）的能力在实际流程中被混淆，需探索解耦机制以提升泛化与跨模型迁移效果。
- 缺乏可审计的演化历史追踪，难以支撑需要多轮累积修正的复杂程序性知识构建。

## 核心贡献（创新点）
- 提出 Raw-Wiki-Skill 三层架构，将不可变执行轨迹、持久化知识层与可执行技能解耦，使后续技能更新能基于累积知识而非碎片历史进行。
- 设计 Wiki Maintainer 与 ReAct 驱动的 Skill Proposer 协同闭环，前者负责模式归纳与日志维护，后者基于 Wiki 审计追踪自主检索轨迹并生成增量补丁。
- 系统验证持久化 Wiki 对技能演化的关键作用，消融表明开启 Wiki 使平均性能提升 15.0%，且 Wiki 内容跨迭代永不回滚。
- 揭示技能演化与模型规模的互补关系及跨模型迁移规律：更大模型从技能演化中获益递增，小模型携带演化技能可超越未携带技能的大模型，且他模演化技能常优于自演化技能。

## 方法详解
- **三层架构**：Raw Layer（`raw/`）存储完整执行轨迹；Wiki Layer（`wiki/`）维护结构化模式页、演化日志（`logs.md`）与技能影响追踪器（`skill-impact.md`）；Skill Layer（`skills/`）包含活跃技能（`SKILL.md` + `PURPOSE.md`）。
- **迭代循环**：第 $k$ 轮中，Inference Agent 使用当前技能集 $S_{k-1}$ 在训练集 rollout（禁止访问 Wiki）；Wiki Maintainer 采样轨迹 $ \mathcal{T}_{\text{sample},k} $ 进行根因分析与成功策略提取，产出 $ W_k' \leftarrow \mathcal{M}_{\mathrm{WM}}(W_{k-1}, \mathcal{T}_{\text{sample},k}) $；Skill Proposer 以 ReAct 模式读取 Wiki 索引与影响日志，按需查阅具体轨迹，生成单技能原子提案 $ P_k \leftarrow \mathcal{M}_{\mathrm{P}}(W_k', S_{k-1}, \mathcal{T}_{\text{train},k}) $。
- **Gating 与回滚**：应用提案得 $ S_k' $，在验证集评估得 $ \mathcal{R}(\mathcal{T}_{\text{val},k}) $；若严格优于历史最佳 $ \mathcal{R}_{\text{best}} $ 则接受并更新阈值，否则回滚技能至 $ S_{k-1} $。无论是否接受，Wiki $ W_k $ 均永久保留，并通过 $ W_k \leftarrow \mathrm{Update}(W_k', P_k, \mathcal{R}(\mathcal{T}_{\text{val},k}), a_k) $ 追加审计记录。达到满分时提前终止。

## 实验与结果
- **数据集与模型**：LiveMathematicianBench、SealQA、SpreadSheetBench、OfficeQA、ALFWorld；Qwen-3.5-4B/9B、Qwen-3.6-27B、Gemma-4-31B、Gemini-3.5-Flash。
- **基线**：Trace2Skill、EvoSkill、SkillOpt 及 No-skill。
- **主要结果**：WikiSkill 在五模型五基准上平均性能最高，较最强基线分别提升 3.3、5.1、10.0、5.8、12.0 分。Qwen 系列随规模增大增益显著（4B/9B/27B 平均提升 +12.3/+17.5/+23.9）。Qwen-3.5-9B+WikiSkill (47.4%) 超越 Qwen-3.6-27B 无技能 (39.4%)。
- **跨模型迁移**：Qwen-3.6-27B 演化技能使 Qwen-3.5-9B 在 SpreadSheet 达到 50.5%（自演化仅 33.6%），证明程序性知识可跨架构迁移且有时优于自演化。
- **消融结论**：移除 Wiki 累积使 Gemini-3.5-Flash 平均性能从 63.7% 降至 48.7%；允许 Inference Agent 访问 Wiki 会干扰技能学习，平均降至 60.9%。

## 相关工作脉络
- **EvoSkill / Trace2Skill / SkillOpt**：同属经验驱动的技能演化框架，但依赖扁平历史或单次轨迹分析，缺乏独立知识层；WikiSkill 通过 Wiki 实现跨轮次知识复用与客观审计。
- **通用 Prompt 优化器（如 GEPA）**：针对整体提示搜索优化，WikiSkill 聚焦文件系统级可复用程序性技能的自动化发现，二者正交。
- **Skill Retrieval 工作（SkillRet、SkillRouter 等）**：关注多技能场景下的检索与触发，WikiSkill 刻意剥离检索环节（全量注入），专注于提升单技能质量本身。
- **Agent Harness 优化（Meta-harness、Autoharness）**：优化 prompt、工具、工作流等系统级组件，WikiSkill 保持 harness 固定，仅演化技能模块。
- **Karpathy (2026) LLM Wiki 理念**：启发本文核心假设——将经验编译为持久、复利式知识可支持长期技能演化，本文将其形式化为可验证的三层架构与演化循环。

## 局限性与未来方向
- 技能以全量注入方式提供，未评估大规模技能库下的检索与触发机制。
- 验证门控采用严格单调提升准则，可能错过短期内持平但利于长期演化的中性提案。
- Wiki 缺乏自动剪枝机制，长期演化可能导致知识库膨胀与检索噪音。
- 基准未覆盖超长 horizon 任务（数百步交互或多小时环境），在线单轮内的技能自适应尚待探索。

## 研究启发与可借鉴点
- **知识-技能分离架构**：将执行轨迹、持久知识与可执行规程分层存储的设计可直接迁移至其他 agent 自我改进系统，避免经验污染与历史覆盖。
- **ReAct 驱动的自主提议者**：Skill Proposer 按需检索轨迹而非全量喂入，结合影响日志避免重复试错，适用于需要长程归因的技能/提示优化任务。
- **跨模型迁移解耦分析**：通过对比自演化与他演化技能表现，清晰区分“经验发现能力”与“规程执行能力”，为小模型补偿与大模型增效提供标准实验范式。
- **审计日志与不可回滚 Wiki**：`skill-impact.md` 记录每次提案的 diff 与验证结果，形成客观进化史，可借鉴于任何基于迭代的自动优化 pipeline。
- **严格验证门控与早期终止**：$\mathcal{R}_{\text{best}}$ 阈值机制与满分早停策略兼顾效率与稳定性，适合资源受限的 agent 演化实验。

## 关键术语表
- **WikiLayer**：持久化知识层，以 Markdown 文件形式存储跨迭代积累的模式、日志与审计记录，内容永不回滚。
- **Skill Proposer**：基于 ReAct 范式的 LLM 智能体，负责阅读 Wiki 与原始轨迹，生成单技能的创建或增量补丁提案。
- **Wiki Maintainer**：负责分析采样轨迹、进行根因诊断并将成功/失败模式整理写入 Wiki 的知识策展智能体。
- **Gating and Rollback**：基于验证集性能阈值的决策机制，仅当新技能集严格超越历史最佳时接受，否则回滚技能但保留 Wiki。
- **Patch-based Editing**：对现有技能文档仅进行追加、替换或插入片段的操作，避免全量重写导致的上下文丢失或结构破坏。
- **Cross-model Skill Transfer**：在不同架构/大小的模型间复用已演化技能，验证程序性知识的模型无关性与可迁移性。
- **Raw Layer**：不可变的原始执行轨迹存储目录，记录多轮交互中的观测、动作与工具调用结果，供分析智能体查阅。

## 可复现要素
- **数据集**：LiveMathematicianBench、SealQA、SpreadSheetBench、OfficeQA、ALFWorld（均为公开基准）。
- **代码/权重**：论文未明确声明开源仓库链接；模型使用 Qwen/Gemma/Gemini 官方发布版本，推理部署依赖 vLLM。
- **关键超参**：
