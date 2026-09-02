---
title: "Break-It-Down-Pass-It-On-Cross-Task-Skill-Transfer-in-LLM-Ag"
source: https://arxiv.org/pdf/2608.20274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 23:59:52"
field: "LLM Agent Memory and Skill Transfer"
keywords: ["LLM Agents", "Skill Transfer", "Cross-Task Generalization", "Skill Memory", "Subtask Decomposition", "Skill Utility Score"]
innovations: ["系统对比任务级vs子任务级、文本vs代码四个组合的跨任务技能迁移", "提出无需执行的Skill Utility Score（specificity×abstractness）预测技能价值", "发现任务级技能平均损害Agent性能而子任务级技能平均提升性能"]
benchmarks: ["AppWorld", "OfficeBench", "KramaBench"]
---

# 论文速读：Break-It-Down-Pass-It-On-Cross-Task-Skill-Transfer-in-LLM-Ag

## 一句话总结
本文系统研究了 LLM Agent 从历史任务中提取的技能如何在跨任务场景中可靠迁移，对比了任务级 vs 子任务级技能诱导、文本 vs 代码技能格式两个关键维度，发现**子任务级诱导 + 文本格式**组合效果最佳；同时提出**技能效用评分（Skill Utility Score）**，仅需技能和任务描述即可在执行前预测技能价值。

## 研究问题与动机
- **核心问题**：Agent 从已完成任务中诱导的技能，在什么条件下能够可靠地跨任务迁移？现有方法常因技能过度绑定源任务而导致迁移失败甚至损害新任务表现。
- **现有方法不足**：多数工作采用任务级技能诱导（对整个轨迹提取一个技能），导致技能高度 specialization，泛化到新任务时成为无关上下文，干扰模型推理（Shi et al., 2023; Yoran et al., 2024）。
- **子任务级工作的证据局限**：虽有 SSO、MUSE 等工作尝试子任务级技能诱导，但仅在 1–2 个领域、少数模型上验证，且仅覆盖文本格式，缺乏统一控制变量下的系统对比。
- **评估指标缺失**：现有工作多用端到端分数掩盖单个技能的贡献，缺少在任务执行前对技能库质量的预评估工具。

## 核心贡献（创新点）
1. **首个统一对比任务级 vs 子任务级、文本 vs 代码四个组合的实验分析**，覆盖 3 个长程基准、11 个模型，隔离了技能诱导级别和格式两个轴的影响。
2. **提出技能效用评分（Skill Utility Score）= specificity × abstractness**，仅需技能和任务描述（无需执行），可在任务运行前诊断技能库质量，与任务成功率稳定相关。
3. **发现任务级技能平均损害 Agent 性能**（Text 降 1.2 分，Code 降 4.1 分），而子任务级技能平均提升性能（Text 升 1.9 分，Code 升 0.5 分），揭示了技能诱导级别是决定迁移成败的关键。
4. **揭示文本技能优于代码技能**：在两个诱导级别上，Text 格式均表现更好，且其 self-retrieval 率更低但迁移效果更好，说明 abstraction 质量比检索频率更重要。
5. **因果验证技能效用**：将技能库按效用中位数分成高/低两半，高效用半库在相同任务上 consistently 取得更高成功率（如 KramaBench 上 Subtask 库 48.4% vs 47.0%），证明评分具有预测力。

## 方法详解
- **任务级 Agent**：平铺 ReAct 循环，将所有 action-observation 追加到单一上下文中，完成任务后从整条轨迹 $\tau$ 提取一个技能。
- **子任务级 Agent**：由 Planner → Executor（ReAct 子任务求解）→ Summarizer 三角色循环组成，将任务分解为子任务 $\tau = (\tau_1, \dots, \tau_K)$，每个子轨迹 $\tau_k$ 独立提取一个技能。
- **文本技能**：以工作流笔记形式存储，包含 procedure（步骤序列）和环境特定注意事项（API 参数名陷阱、边界情况等），用于检索的描述即标题。
- **代码技能**：以 Python 函数形式存储，实例特定值作为参数，环境注意事项写为注释；检索后将函数加载到 namespace 供 Agent 直接调用。
- **技能诱导**：两级共用同一诱导 prompt，通过"一条轨迹对应一个技能"的规则防止任务级诱导被拆分为子任务式技能，确保对比隔离。
- **技能检索**：使用 all-MiniLM-L6-v2 嵌入技能和查询，取 top-5 匹配（阈值 cosine ≥ 0.30）注入上下文；Code 技能还需加载函数。
- **技能效用评分公式**：
  - Specificity：技能与其最近任务余弦相似度高于随机任务对的概率，惩罚与所有任务都远的技能。
  - Abstractness：相似度分布的 perplexity 归一化，衡量技能相关性均匀分散在多任务上的程度。
  - Utility(s) = specificity(s) × abstractness(s)，乘积捕捉两者平衡。

## 实验与结果
- **数据集**：AppWorld（417 任务，多应用工具调用）、OfficeBench（300 任务，办公文档流程）、KramaBench（92 任务，数据科学管线），均采用官方确定性评分器。
- **模型**：11 个模型，含 3 个 MoE（Qwen3-235B-A22B、GPT-OSS-120B、Nemotron-Super-120B）、6 个 Dense（Qwen3 4B/8B/14B/32B、Gemma-3 4B/12B/27B）、1 个商业模型（Gemini-3.1-Pro）。
- **最强结果**：在 AppWorld 上，Subtask+Text 对 Gemini-3.1-Pro 达 **72.4%**（基线 68.3%，+4.1 分）；Subtask+Code 达 **77.5%**（+9.2 分）。跨模型平均：Subtask+Text 较无记忆基线平均提升 **+1.9 分**，Subtask+Code 提升 **+0.5 分**；而 Task+Text 平均下降 **-1.2 分**，Task+Code 下降 **-4.1 分**。
- **效率分析**：在同等 dependency/latency 预算下，子任务级 Agent 从中等预算开始超越任务级，且在高预算下饱和更高。
- **难度分层**：在 easy/medium/hard 三个难度层级中，子任务级技能始终优于任务级，Text 始终优于 Code。
- **自我检索率验证**（App. E.6）：Text 技能 self-retrieval 率（75.6% Task / 79.6% Subtask）低于 Code（88.1% / 86.1%），但 Text 迁移效果更好，排除检索质量差异的解释。

## 相关工作脉络
- **任务级技能工作**：Voyager（任务+代码+游戏）、TroVE（任务+代码+数学/表格）、AWM（任务+文本+网站）、ExpeL（任务+文本+QA）、CLIN（任务+文本+科学模拟）——本文指出这些方法普遍存在技能过度绑定源任务的问题，且缺乏跨格式的公平比较。
- **子任务级技能工作**：SSO（子任务+文本+科学模拟/游戏）、MUSE（子任务+文本+生产力）、Shen et al.（2026）（子任务+文本+SWE）——本文扩展了其在多领域、多模型、多格式下的验证，并引入效用评分。
- **RL 训练记忆方法**：SkillRL（Xia et al., 2026）、MemPO（Li et al., 2026）、Memory-R1（Yan et al., 2026）——通过 RL 优化记忆操作或策略，与本文基于诱导+检索的轻量方案形成对比。
- **不诱导直接检索 episode 的工作**：Synapse（Zheng et al., 2024）、Buffer of Thoughts（Yang et al., 2024）——本文聚焦诱导过程对迁移的影响，而非检索策略本身。
- **定位差异**：本文为首个在统一 prompt/设置下隔离诱导级别和格式两个轴的工作，并提出了无需执行的预评估指标。

## 局限性与未来方向
- **评估范围受限**：仅在 3 个基准（AppWorld/OfficeBench/KramaBench）上验证，computer use（OSWorld）、agentic coding（SWE-bench Pro）、web search（BrowseComp）等场景的迁移行为尚待研究。
- **技能记忆静态**：固定诱导/检索/去重规则，未支持 Agent 随时间修订已存技能（如 MemGPT、Memp 的动态更新机制），后者需沙盒文件系统支持。
- **缺乏细粒度过程分析**：仅用最终状态评分，无 step-level ground truth，中间决策质量需依赖 judge model，留作未来工作。
- **安全/恶意技能风险**： adversaries 可能注入恶意技能通过复用机制操控 Agent，本文假设技能全部由 Agent 自身从沙盒任务诱导产生，信任问题留作后续研究。

## 研究启发与可借鉴点
- **子任务分解是技能迁移的关键**：将任务拆解为原子子任务并逐个子任务诱导技能，可显著提升跨任务复用率；可迁移到本团队的 long-horizon agent 研究中，替代全轨迹 skill 提取。
- **文本格式在跨任务场景更具优势**：代码技能虽 self-retrieval 率更高，但抽象度和通用性不如文本笔记；建议在新系统评估中优先尝试文本格式，尤其在意迁移泛化时。
- **技能效用评分可作为离线诊断工具**：仅需 skills + task descriptions（无需执行），即可预测技能库质量；可集成到本团队的 skill memory pipeline 中作为 pre-deployment 筛选器。
- **实验设计借鉴：控制变量交叉**：通过固定 induction prompt、仅改变诱导级别和格式，干净隔离两个轴的影响；这种设计模式适用于其他 memory/skill 系统的消融实验。
- **创新机会**：可将 Skill Utility Score 与 RL-based memory 优化（如 SkillRL、Memory-R1）结合，作为 reward shaping 信号引导 Agent 学习更具 utility 的技能表示。

## 关键术语表
**Skill Induction Level**：技能从多长的轨迹中提取，任务级指整个任务轨迹，子任务级指单个子轨迹。
**Skill Format**：技能的存储形式，文本（workflow note）或代码（Python function）。
**Specificity**：技能与其最相似任务的接近程度，衡量技能与真实任务的相关性。
**Abstractness**：技能相关性在任务集合上的均匀分散程度，用 softmax 相似度分布的 perplexity 归一化衡量。
**Skill Utility Score**：specificity × abstractness 的乘积，用于评估单个技能的综合可用性。
**Cross-Task Skill Transfer**：在任务流中，早期任务诱导的技能被后期任务检索复用的现象。
**Dependency**：MEM1 计算代价指标，近似任务对增长型上下文的注意力开销。
**Self-retrieval Rate**：技能能被其源任务/子任务自身检索到的比例，用于验证检索质量。

## 可复现要素
- **数据集**：AppWorld（https://github.com/StonyBrookNLP/appworld）、OfficeBench（https://github.com/zlwang-cs/OfficeBench）、KramaBench（https://github.com/mitdbg/Kramabench），均已开源。
- **代码/权重**：论文声明将发布完整代码库和所有条件下的诱导技能库（MIT License）；骨干模型（Qwen3、GPT-OSS、Nemotron、Gemma-3、Gemini-3.1-Pro）均已公开或可通过 API 访问。
- **关键超参**：Embedding 模型 all-MiniLM-L6-v2；检索阈值 cosine ≥ 0.30，top-5；温度 τ = 0.1（abstractness）；任务级 50 步 ReAct 上限；子任务级 15 个子任务上限、共享 50 步 executor 预算；生成长截断至 8192 tokens。
