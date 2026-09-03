---
title: "Recursive-Experiential-Working-Memory-Evolution-for-Long-Hor"
source: https://arxiv.org/pdf/2608.24876v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:53:48"
field: "LLM Agent 记忆与自改进"
keywords: ["long-horizon agent", "recursive self-improvement", "experiential memory", "working memory", "skill invocation", "memory evolution", "failure localization"]
innovations: ["EM–WM 耦合的验证式技能调用与状态更新闭环", "基于结构化轨迹的组件级失败归因与定点修补", "有界递归记忆演化及其跨模型可迁移性"]
benchmarks: ["tau2-Retail", "tau2-Airline", "SkillFlow", "Terminal-Bench 2.1"]
---

# 论文速读：Recursive-Experiential-Working-Memory-Evolution-for-Long-Horizon-Agent-Harnesses

## 一句话总结
论文提出 **Recuris**，一种递归式“经验记忆–工作记忆”（Experiential–Working Memory, EM–WM）耦合架构，用于长程 Agent 编排层。通过将任务状态追踪（WM）与可复用技能检索（EM）闭环耦合，并将执行轨迹转化为结构化诊断证据，Recuris 实现了对外部记忆控制层的递归、组件级、验证门控修补，从而在多个长程基准上将从 3B 到前沿模型的 Task Success 大幅提升。

## 研究问题与动机
- **核心问题**：长程任务中，Agent 的历史上下文持续膨胀，导致任务状态模糊、技能调用与当前需求错位，难以实现稳定的递归自改进（RSI）。
- **现有方法不足**：
  1. 既有经验记忆方法多从初始指令或完整交互历史检索技能，随着执行推进，噪声与过期信息稀释检索可靠性。
  2. 工作记忆的状态更新多由模型自述或固定规则驱动，缺乏基于环境观测的“验证式”提交，易出现未执行动作被误判为完成。
  3. 现有递归自改进方案多针对模型权重、工具集或工作流图进行重写，评估信号往往只是最终成败，无法将失败定位到具体记忆组件，导致修补粗粒度、易破坏已有能力。
  4. 长程任务收益常随交互长度增长而衰减，缺乏一种能持续对齐“当前未完成目标”与“可用经验”的显式控制层。

## 核心贡献（创新点）
1. **EM–WM 耦合的验证式长程编排机制**：提出以紧凑工作状态 $w_t$ 作为当前任务进展的接口，驱动从经验记忆 $\mathcal{E}_k$ 中的状态接地（state-grounded）技能调用，并引入 Checker 集合 $C_k$ 对 proposed 状态变更进行观测验证后才提交，形成 “状态→技能选择→执行反馈→更新状态” 的闭环。
2. **结构化轨迹用于组件级失败定位**：将每轮执行记录为包含 $(w_t, \mathcal{E}_t, a_t, o_t, \tilde w_{t+1}, c_t, w_{t+1})$ 的结构化 trace $\Gamma_k$，使 Meta-Agent 能够依据轨迹证据将失败归因到 $\mathcal{E}$、$\mathcal{W}$、$\rho$ 或 $C$ 中某一组件，而非仅凭最终 outcome 做整体判断。
3. **有界的跨任务递归记忆演化框架**：在固定基础 LLM 与外部改进流程的前提下，通过诊断 $D_k$ → 组件特化修补 $\Delta m_z$ → 保留非 implicated 组件不变 → 门控检验是否在不退化保留集的前提下修复目标失败，实现 memory-control layer 的有界递归演化。
4. **大规模多模型多基准验证与可迁移性证明**：在四个长程基准、十个模型上评测，35/37 对 model–benchmark 显著提升；前沿模型经同一份由小模型演化出的 Skill Memory 加持后仍获大幅增益（如 GPT-5.6 Sol +17.8、Claude Opus 5 +15.6 on $\tau^2$-Retail），证明记忆层可作为可复用的可训练表面。

## 方法详解
- **Skill Memory 架构**：$\mathcal{M}_k = (\mathcal{E}_k, \mathcal{W}_k, \rho_k, C_k)$，分别对应经验技能库、工作记忆规范、调用策略和检查器集合。每次候选 patch 只修改诊断指出的组件，其余原样拷贝。
- **状态接地调用**：在预定义执行事件 $e_t$（如草拟状态变更 tool call 或回合边界）触发 $\rho_k(x, w_t, e_t, \mathcal{E}_k)$，将匹配技能注入上下文，而非让模型自行从历史中挑选。
- **证据接地状态更新**：$U_{\mathcal{W}_k}$ 根据动作与观测提出 $\tilde w_{t+1}$；$C_k$ 对目标 $g$ 逐项求值 $c_{t,g}$；固定内核 $K$ 仅在观测支持时才把 pending 改为 done 或 blocked，否则保留原状并记录 false-pending 错误。
- **跨任务递归循环**：$\mathcal{M}_k \to \Gamma_k \xrightarrow{\mathcal{A}_{fixed}} D_k \xrightarrow{\mathcal{P}_{fixed}, \oplus} \mathcal{M}_k^+ \xrightarrow{\mathcal{G}_{fixed}} \mathcal{M}_{k+1}$。门控要求：在源失败任务上修复且在保留开发集（含已解 anchor）上不退化才准入。
- **测试时单任务适配模式**：限制 Meta-Agent 只能看到任务指令、失败轨迹与验证器返回的 0/1 位，不暴露期望输出或测试本身，用于 Terminal-Bench 等无跨任务结构的孤立场景。

## 实验与结果
- **基准与模型**：$\tau^2$-Bench（Retail/Airline）、SkillFlow、Terminal-Bench 2.1；覆盖 Qwen3、Granite、gpt-oss 系列与 GPT-5.6 Sol/Claude Opus 5/Gemini 3.7 Flash 等，共十款。
- **主要结果**：35/37 个 model–benchmark 对提升。$\tau^2$-Retail 上 Recuris 将 GPT-5.6 Sol 从 58.3% 提至 76.1%（+17.8），Claude Opus 5 从 72.4% 提至 87.9%（+15.6，达到当时评测中的最高值）；SkillFlow 上 Qwen3.6-27B/35B 分别 +16.6/+13.5。
- **长程收益**：优势随交互长度增加而扩大，最长任务区间达 +32.2 点；六类常见长程失败模式降低最高达 80%。
- **失效定位**：基于结构化 trace $\Gamma$ 的组件定位 macro 准确率达 64.8%，显著高于单 outcome（13.0%）与原始轨迹（37.0%）。
- **消融结论**：纯 EM 增益不显著，纯 WM 即可带来约 +23.9；模型自主调用全量技能库反而不如无技能基线且消耗更多 token，说明“何时调用”比“调用多少”更关键。

## 相关工作脉络
1. ** experiential memory / agent skills**（Voyager、AWM、ExpeL、Dynamic Cheatsheet、SkillOpt、SkillComposer 等）：重在存什么，少关注何时由谁决定是否将技能带入上下文；本文强调以工作记忆为条件的显式调用控制。
2. **LLM working memory**（StateAct、ReflAct、Magentic-One、StateFlow、StructAgent）：多由模型自写状态或依赖手写规则，缺乏观测验证与技能联动；本文用 checker 门控提交，并将状态用于驱动检索。
3. **递归自改进 LLM Agent**（Gödel machine 传统、Promptbreeder、MetaSkill-Evolve、Memento、EvolveMem、Memskill 等）：常以模型自我评估或单一指标提升决定更新，难以归因到组件；本文通过结构化 trace 实现 component-scoped patching。
4. **动态/上下文感知检索**（AutoGuide、SGDR、SkillsInjector）：检索信号多为环境当前外观的相似度匹配；本文以已验证的任务状态与执行事件为条件，对齐当前未完成目标。
5. **记忆门控与稳定性**（PACE、SSGM 等）：关注更新是否被保留；本文在此基础上进一步将“为什么失败”显式化到组件级别，避免误修补。

## 局限性与未来方向
- **单次演化起点依赖**：Skill Memory 在当前实验中均从部署模型的中档实现起步，若底层 agent 初始行为极差，可能影响首轮证据质量与后续收敛速度。
- **跨任务结构前提**：Terminal-Bench 2.1 等无共享工具/策略的基准上，跨任务演化未能 admitting 任何 patch，需退化为单任务适配，且实测增益与重试预算混叠、统计效力不足。
- **门控保守性**：dev 集合较小时置信区间宽，部分在 held-out 上实际有效的 patch 被 gate 拒绝，存在错过改进空间的可能。
- **泛化边界不清**：同一份 memory 对不同模型增益差异较大，尤其接近 saturate 模型（如 Claude Opus 5 on Airline）增益有限，说明“纪律型”知识对天花板模型的价值受剩余 headroom 制约。
- **潜在方向**：自动 pruning 冗余技能、自适应扩大 dev 集或引入统计功效更大的门控、将组件定位扩展到流程编排层、探索多轮单任务持续适应的分离评估协议。

## 研究启发与可借鉴点
- **用结构化 trace 替代 outcome-only 信号做归因**：把 $w_t$、$\mathcal{E}_t$、$a_t$、$o_t$ 与 proposed/committed 状态对齐记录，可显著提升组件级故障定位精度，值得推广到其他自改进或调试系统。
- **调用控制优于内容堆叠**：相同技能库，状态接地的事件触发调用比全量注入让模型自行筛选效果更好且更省 token，提示后续工作应优先设计“何时/为何检索”的显式策略。
- **验证门控与保留集设计**：以 anchor 任务保证不退化为硬约束，可在不冻结整个 harness 的前提下允许记忆层稳步增长，便于工程落地与审计。
- **单一来源记忆的跨模型复用**：在小模型上完成一次演化后直接迁移到更大模型，可作为低成本、低过拟合风险的“纪律模板”分发方式。

## 关键术语表
- **Experiential–Working Memory Coupling（EM–WM 耦合）**：用当前已验证任务状态驱动从经验记忆中选取技能，并通过执行观测回写状态，实现检索与执行的闭环对齐。
- **State-grounded skill invocation（状态接地技能调用）**：在指定执行事件处，依据当前工作状态而非完整历史检索并注入技能。
- **Evidence-grounded state update（证据接地状态更新）**：工作记忆 proposal 须经 checker 对照真实观测后才被内核提交，防止误标完成。
- **Structured execution trace $\Gamma$**：记录每步状态、被检索技能、动作、观测、提议状态与 checker 决定的诊断级轨迹。
- **Component-specific patching（组件特化修补）**：仅对被诊断归属的 $\mathcal{E}/\mathcal{W}/\rho/C$ 之一进行改动，其余原样保留。
- **Validation-gated admission（验证门控准入）**：候选更新必须在源任务上修复且在保留开发集上不退化才可被接受。
- **Bounded recursive evolution（有界递归演化）**：只递归更新外部记忆控制层，基础 LLM 与外部改进流程保持固定。
- **Test-time adaptation mode（测试时单任务适配）**：限制 Meta-Agent 仅见任务指令、失败轨迹与最终 0/1 分数，用于无跨任务结构的孤立场景。

## 可复现要素
- **数据集**：$\tau^2$-Bench、SkillFlow、Terminal-Bench 2.1 均为公开基准；论文提供 split 说明（evolve/dev/test）并在附录 C 给出大小。
- **代码/权重**：项目主页 https://github.com/Gen-Verse/Recuris；论文强调未更新任何模型权重，记忆内容以 Skill Memory 包形式分发。
- **关键超参**：每任务独立尝试次数 $k=4$（除特别说明）；最大步数 200；连续工具错误上限 10；temperature=0；bootstrap 重复 10,000 次计算 95% CI。
