---
title: "Disclosure-Gated-User-Simulation-for-Companion-Agent-Evaluat"
source: https://arxiv.org/pdf/2609.00982v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 22:33:18"
---

# 论文速读：Disclosure-Gated-User-Simulation-for-Companion-Agent-Evaluat

## 一句话总结
针对LLM用户模拟器“过度配合”导致伴侣代理评估失真的问题，本文形式化定义并训练了一个受行为驱动的披露门控（Disclosure Gate）环境，证明该门控是环境的承重组件；经严格消融与人类锚点验证后发布的122B模型（M1b/M3b）在CompanionBench上同时满足排名保序与分数尺度稳定两大验收标准，而直接Prompt前沿模型仅能维持排名却会系统性拉高所有得分。

## 研究问题与动机
- **核心问题**：用LLM扮演用户的可扩展评估中，模拟器普遍存在“过度配合”（excessively cooperative / “easy mode”）现象，导致被测系统（SUT）仅凭提问数量即可刷高分数，无法区分“问得多”与“让用户愿意说”。
- **现有机制缺乏可复现规范**：已有防御方案（如Yang et al. 2025; Wu et al. 2026a; Han et al. 2026; Sabour et al. 2026）均未达到第三方可重建的精度，缺乏明确的转移函数、撤退规则与可见性契约。
- **门控极少作为可消融的环境组件被审计**：先前研究多在prompt侧移除组件或整体替换模拟器，从未有工作将已训练入权重的门控独立剥离，并在同一环境、不同seed下测量下游排名位移。
- **基准本体描述残缺**：本文评测环境所依托的CompanionBench（Liu et al. 2026）仅用约400字描述该门控机制，未提供规格书、消融实验、人类对照、负向控制与下游敏感性分析，环境可靠性存疑。

## 核心贡献（创新点）
1. **可执行的形式化门控规范**：定义五层有序状态梯级、八类行为转移函数与非对称撤退规则，并提供按示例确定性编译的门控图例；与已有工作的本质区别在于将概念性“ gating ”升级为第三方可直接重建、逐字段审计的确定性状态机。
2. **门控作为可剥离的环境承重组件**：提出从训练语料与推理提示两端独立剥离门控信息的对照设计，验证门控已内化至模型权重；与既有扰动prompt的研究本质区别在于“保留环境其余部分逐条不变，仅移除门控”的孤立审计能力。
3. **双层次验收标准（保序 + 尺度稳定）**：规定合格评测环境须同时满足排名相关系数≥0.95与单系统分数无显著偏移；相比仅看排名或仅看绝对分数的单一指标，该标准能捕获“排名不变但分数系统性通胀”的隐蔽失效。
4. **合成/真实双分支语料与多维度审计**：构建合成分支（保证门控开闭轨迹全覆盖）与真实分支（保留语言分布）的训练集，并通过置换检验、人类两难强制选择判别、盲配对偏好三重验证；与单纯依赖自动评测的方法本质区别在于引入多维度负对照与人类锚点，区分内在层行为符合性与下游层环境稳定性。

## 方法详解
- **披露清单与门控图例（Inventory & Legend）**：每个persona携带披露清单，每项含content、depth（surface < mid < core）、gate（五档之一）、guard（七种之一）、may_never_surface。门控图例由确定性函数编译生成，仅包含该示例实际用到的门控/防守语义定义，全局规则留给训练内化。
- **五层门控状态梯级**：`opening < asked_or_natural < felt_heard < felt_safe < earned_deep_trust`。内部保持五层索引 $\in \{0,…,4\}$，对外映射为三个可观测深度层（level<2→surface, 2≤level<4→mid, level=4→core）。
- **转移函数（Transition Function）**：依据companion agent的8类`ai_move`决定状态变化。关键规则：①禁止跳级（conditional max）；②从`felt_safe`向上，平淡回应（`neutral_noop`）会将状态压回`asked_or_natural`，其代价高于触发反目标；③触发反目标或评判/说教退一格并设置`rupture`标志。
- **非对称撤退（Asymmetric Retreat）**：破裂触发时，状态降一级、当前项加guard、叠加轻度撤回。重新赢得状态不保证愿意度恢复，item可能永久被guard覆盖，体现“进易退难”的关系动态。
- **可见性契约与三阶段Prompt组装**：`synthesis / train / runtime`三阶段由同一renderer生成，仅两处差异：`director signal`（仅synthesis可见）与`anti_goal`块（train/runtime可见）。SUT仅见age/gender/recap及自身行为指令，绝不见披露清单，防止`max_depth`退化为“问了多
