---
title: "WHAT-IS-MISSING-FROM-AI-POST-TRAINING-AI-AN-EMPIRICAL-ANALYS"
source: https://arxiv.org/pdf/2608.19072v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:56:53"
field: "LLM Post-Training & Agent R&D"
keywords: ["LLM Post-Training", "AI Agent", "Strategy vs Execution", "PostTrainBench", "AI for AI", "Empirical Analysis", "Agent Capabilities"]
innovations: ["提出执行层-策略层两级能力框架并系统量化策略锁定现象", "通过递进式三重干预（经验/指导/计算）证明缺失的是策略重评机制而非资源", "发现策略可塑性仅在首次训练前短窗口内存在，训练启动后锁定不可逆"]
benchmarks: ["PostTrainBench", "GSM8K", "HumanEval", "AIME 2025"]
---

# 论文速读：WHAT IS MISSING FROM AI POST-TRAINING AI: AN EMPIRICAL ANALYSIS

## 一句话总结
本文系统分析了 PostTrainBench 上 1,338 条公开 agent 轨迹，发现 frontier agent 的训练策略在开始时就已锁定，后续几乎全部预算均消耗在选定策略内的局部调优；通过经验驱动、人类指导和增加推理计算的递进干预实验，证明当前 agent 缺失的不是经验、指导或计算量，而是在执行过程中自发重新评估并修正高层训练策略的机制。

## 研究问题与动机
1. **执行层与策略层的混淆**：当前 AI-for-AI 讨论将"在选定策略内迭代"（execution-level）和"根据证据修正高层判断"（strategy-level）两种能力混为一谈，忽视了策略层能力才是自动化 AI R&D 的真正瓶颈。
2. **策略锁定现象的实证基础薄弱**：虽已有工作展示 frontier agent 可端到端 post-train，但缺乏对大规模轨迹的系统分析来回答"agent 是否会随实验证据改变训练策略"这一关键问题。
3. **缺失成分的三假设亟待验证**：策略锁定可能源于经验不足、缺乏指导或缺少推理计算，三者需逐一对比验证。
4. **策略窗口的边界不明**：若策略可在训练前被引导，那这一"可塑性窗口"的确切边界和关闭后的不可逆性尚未被量化。

## 核心贡献（创新点）
1. **提出执行层–策略层两级能力框架**：将 post-training 拆解为"执行既定 pipeline"与"根据证据修订策略"两类动作，为 agent 行为分析提供了可操作的分类体系。
2. **大规模轨迹实证揭示策略锁定规律**：对 1,338 条轨迹的标注显示，agent 在训练前极短时间内确定策略，之后 97.9% 的相邻实验对均未发生策略变更，且锁定反映的是 agent 先验而非任务特性。
3. **递进式三重干预实验**：依次施加经验脚手架（实验日志+技能库+评估器 agent）、人类策略指导、额外推理计算，量化各因素对执行层与策略层的影响边界。
4. **发现策略可塑性的时间窗口**：证明策略仅在首次训练前的短窗内可被经验与指导重塑，一旦训练启动，同一机制均失效；缺失的核心是"在执行中自发重启策略重评"的机制，而非资源。

## 方法详解
**两级能力框架**：定义训练策略 $s = (k, d, g)$，其中 $k$ 为训练范式（Full SFT / PEFT / RL / DPO / Distillation），$d$ 为数据源类型，$g$ 为阶段结构；策略变更定义为 $s_{i,t} \neq s_{i,t-1}$，其余动作（调超参、改 reward、修 bug、选 checkpoint）归为执行层动作。

**轨迹分析方法**：统计相邻训练实验对的策略持久率 $R_{\text{persist}}$ 与变更率 $R_{\text{change}}$；用初始策略收敛度 $C_G = \max_k \frac{1}{|G|}\sum_i \mathbf{1}[k_{i,1}=k]$ 衡量 agent 群体的初始策略一致性。

**经验驱动框架**（Section 4.1）：由三组件构成——（1）实验日志（append-only，记录 plan/observation/lesson/eval_result）；（2）技能库（从 verl/TRL/OpenRLHF 等 908 份文档蒸馏出约 60 个 SKILL.md 文件）；（3）独立评估器 agent（ inspect pipeline → 调用评估脚本 → 返回诊断与建议）。

**人类指导干预**（Section 4.2）：训练前由人类 reviewer 对 agent 的初始方案进行迭代审查（固定格式 JSON 决策），批准后进入完全自主运行；此干预仅作用于 planning stage，不介入后续执行。

**推理计算扩展**（Section 4.3）：经验驱动框架较基线多消耗 2–8× 推理 token，用于对比"更多计算"是否能带来策略层面的改进。

## 实验与结果
**轨迹分析（Section 3）**：
- 1,338 条轨迹、5,111 次有效训练实验，覆盖 7 个 benchmark（AIME 2025、ArenaHardWriting、BFCL、GPQA Main、GSM8K、HealthBench、HumanEval）、4 个 base model（Gemma-3-4B-PT、Qwen3-1.7B-Base、Qwen3-4B-Base、SmolLM3-3B-Base）、20 种 agent 配置。
- 策略锁定：仅 74/3,557（2.1%）相邻对发生策略变更；Claude Code 80.7% 以 Full SFT 开局，Codex CLI 89.6% 以 PEFT 开局，同一 agent 跨任务高度一致，不同 agent 同任务显著分化。

**控制实验（Section 4，Qwen3-1.7B-Base，4× NVIDIA A800，10h）**：

| 设置 | GSM8K | HumanEval | AIME 2025 |
|---|---|---|---|
| Base model | 10.84% | 5.48% | 0.00% |
| 官方 instruct | 88.70% | 66.46% | 33.33% |
| Opus 4.6 (Claude Code) | 64.70% | 22.00% | 3.33% |
| GLM 5.2 (Claude Code) | 49.51% | 44.51% | 3.33% |
| GPT-5.2 (Codex CLI) | 43.44% | 13.41% | 0.00% |
| **经验驱动 (Opus 4.6)** | **77.30%** (+12.6) | **62.80%** (+40.8) | **5.56%** |
| – 去掉实验日志 | 74.50% | 50.20% | 4.44% |
| – 去掉技能库 | 73.10% | 54.50% | 3.33% |
| – 去掉评估器 agent | 68.20% | 42.60% | 3.33% |

- **经验干预**：全面改善执行表现，但评估器 agent 提出 8 次策略级建议（如 HumanEval 上从 SFT 切换到带代码执行的 RL），agent 全部采纳执行级建议、零采纳策略级建议。
- **人类指导**（AIME 2025）：成功将初始 SFT 方案重定向为 GRPO，但 agent 达到早期峰值后回落到超参微调循环，未能持续修正策略。
- **推理计算**：经验驱动框架消耗 7.9× token 在 AIME 2025 上仅多解 1 题（在评估方差内）；计算扩展对简单任务有效，对最难任务触及天花板。

**最强结果**：经验驱动 + Opus 4.6 在 HumanEval 达 62.80%（接近官方 instruct 的 66.46%），GSM8K 达 77.30%；但策略层能力在所有干预下均未被激活。

## 相关工作脉络
1. **PostTrainBench（Rank et al., 2026）**：本文轨迹数据来源，首次形式化 LLM post-training agent benchmark；本文进一步区分其内部执行层与策略层，指出此前工作将二者混同。
2. **AI Scientist / The AI Scientist-v2（Lu et al., 2024; Yamada et al., 2025）**：关注端到端科学发现工作流自动化；本文与其定位差异在于聚焦 post-training 这一更具体的 AI R&D 子任务，并首次对策略锁定进行量化测量。
3. **经验驱动 agent（Reflexion、ExpEL 等）**：通过反思、技能库、评估反馈提升 agent 表现；本文借用类似组件（实验日志、技能库、评估器 agent），但发现这些执行层增强无法触发策略层变更。
4. **Agent 行为诊断（AgentRx、MLE-Bench、MLE-Dojo）**：关注 agent 在 ML 工程中的失败模式；本文与之互补，从策略维度而非实现维度解释 agent 的" competent but stuck"现象。
5. **RL post-training 系统（Nemotron RL、OpenRLHF、slime、verl）**：提供技能库来源；本文反向利用这些系统的文档知识来诊断 agent 行为边界。

## 局限性与未来方向
1. **观察性数据的内生偏差**：轨迹数据来自不同 agent/模型/接口的非随机组合，无法严格分离各因素因果效应；附录 E 的跨 agent 分析可部分弥补但样本有限。
2. **人类指导仅作用于 planning stage**：未测试持续的人机协同策略修正；训练开始后策略衰退的具体机制仍需更细粒度实验验证。
3. **AIME 2025 基线得分为零的评估方差问题**：pass@8 评估下微小提升难以统计显著， hardest task 上的结论相对谨慎。
4. **未来方向**：① 设计能在执行中自发触发策略重评的训练信号或交互协议；② 研究策略锁定与 agent 先验（system prompt、interface scaffold）的定量关系；③ 探索"探索–利用"平衡在 post-training 中的显式建模。

## 研究启发与可借鉴点
1. **两级能力框架的可迁移性**：执行层/策略层的二分法可直接用于分析其他 agent R&D 场景（如算法设计、实验规划），作为行为诊断的分析工具。
2. **经验驱动框架的组件设计**：实验日志（append-only 结构化记录）+ 技能库（文档蒸馏为可检索条目）+ 独立评估器 agent（高信息密度诊断）的三层架构，可复用于任何长程 agent pipeline。
3. **递进干预的实验设计范式**：从"缺资源"到"缺引导"再到"缺计算"的 escalate 顺序，清晰分离了不同假设，是 agent 能力剖析的优秀实验范式。
4. **策略锁定对训练信号设计的启示**：当前 RL/RLHF 训练目标隐含优化执行效率，未来需在 reward design 中显式奖励"证据触发的策略重评"，或在 system prompt 中设立策略重评的决策节点。
5. **可结合本团队方向的机会**：若团队关注 agent 的 long-horizon reasoning 或 self-improvement，可将"策略可塑性窗口"作为 agent 架构设计的约束条件，或在 meta-learning 层面研究如何让 agent 学会"何时放弃当前策略"。

## 关键术语表
**Execution-Level Capability（执行层能力）**：在既定训练策略内完成数据构建、超参调优、debug、checkpoint 选择等操作的可靠执行能力。

**Strategy-Level Capability（策略层能力）**：根据累积的实验证据，决定切换训练范式、增减阶段或重定向剩余预算的高层判断与修正能力。

**Strategy Lock-in（策略锁定）**：agent 在首次训练前极短时间内确定训练策略，此后在整个预算内不再进行策略级变更的现象。

**Experience-Driven Framework（经验驱动框架）**：由实验日志、技能库和独立评估器 agent 三者组成的脚手架，用于提升 agent 的执行质量与信息密度。

**Experiment Journal（实验日志）**：append-only 的结构化记录，按 plan/observation/lesson/eval_result 四类条目持续累积跨迭代的实验知识。

**Evaluator Agent（评估器 agent）**：独立于主 agent 的辅助 agent，负责 inspect pipeline、运行评估、返回诊断与下一步建议，提高主 agent 上下文的信息密度。

**Strategy Change（策略变更）**：相邻训练实验之间训练范式（k）、数据源类型（d）或阶段结构（g）中任一维度的改变；超参调整、reward shaping、格式化等归为执行层变更。

**Pass@k Accuracy**：对每道题采样 k 次生成，只要至少一次通过即计为正确；AIME 2025 因题目少（30 题）采用 pass@8 以降低方差。

## 可复现要素
- **数据集**：PostTrainBench 公开轨迹（1,338 条）；论文使用数据已随 benchmark 公开。
- **代码/权重**：PostTrainBench 与经验驱动框架实现见论文附录与补充材料；base model 为 Qwen3-1.7B-Base（开源）。
- **关键超参**：10-hour compute budget；4× NVIDIA A800 80GB GPU（控制实验）/ 1× H100 80GB GPU（轨迹收集）；pass@1（GSM8K/HumanEval）与 pass@8（AIME 2025）评估协议；经验驱动框架三组件（实验日志、60 个 SKILL.md、评估器 agent）配置见附录 C。
