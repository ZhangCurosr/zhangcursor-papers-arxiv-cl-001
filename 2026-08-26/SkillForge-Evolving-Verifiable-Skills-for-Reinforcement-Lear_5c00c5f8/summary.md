---
title: "SkillForge-Evolving-Verifiable-Skills-for-Reinforcement-Lear"
source: https://arxiv.org/pdf/2608.24747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:56:23"
field: "智能体强化学习与技能演化"
keywords: ["LLM Agent", "Reinforcement Learning", "Skill Learning", "Continuous Evolution", "Skill Verification", "GRPO"]
innovations: ["显式技能调用机制使RL可联合优化动作与技能决策", "基于EMA成功率和调用次数的技能低表现分数验证闭环", "多路径技能诱导（提取/精炼/对比分析）驱动技能库质量增长"]
benchmarks: ["ALFWorld", "WebShop", "AppWorld"]
---

# 论文速读：SkillForge: Evolving Verifiable Skills for Reinforcement Learning Agents

## 一句话总结
SKILLFORGE 提出了一种持续技能演化框架，通过显式技能调用、基于证据的技能验证和多路径技能诱导，使 LLM 智能体在 RL 训练中能够持续积累、验证和精炼可复用技能，显著优于 SKILLRL 等追加式技能库方法。

## 研究问题与动机
- **RL 智能体缺乏跨 episode 知识积累**：现有 RL 训练的 LLM 智能体是偶发性的，每个 episode 从零开始，无法复用过去的成功或失败经验。
- **现有技能基方法（如 SKILLRL）将技能库视为追加存储**：技能只增不减，缺乏有效性验证和质量控制机制，错误技能会长期残留污染知识库。
- **技能使用情况不可观测、无法归因**：SKILLRL 在批量注入技能到 prompt 后，无法确定智能体是否真正依赖某个技能，也无法将 episode 结果归因到特定技能。
- **技能质量缺乏闭环控制**：没有机制识别和修订过时或误导性的技能，技能库会随着训练逐步劣化。

## 核心贡献（创新点）
- **提出显式技能调用策略**：将技能调用设计为轨迹中的离散事件（`<skill_call>` 标签），使 RL 能联合优化环境动作和技能调用决策，技能使用情况变得可观测、可归因。
- **引入基于证据的技能验证机制**：维护每个技能的使用统计（EMA 成功率、调用次数），构建低表现分数（underperformance score），优先对证据充分的退化技能进行 LLM 反思修订。
- **设计多路径技能诱导机制**：从成功轨迹（提取）、失败轨迹（修正）、成功-失败对比分析三条路径合成新技能，确保技能库在增长的同时提升质量。
- **无需 SFT 冷启动的端到端 RL 训练**：与 SKILLRL 不同，SKILLFORGE 不需要额外的指令微调阶段，直接从 RL 训练开始演化技能库。

## 方法详解

**技能表示与检索**：每个技能表示为结构化元组 $(title, intent, principle, applicability, category, status)$。初始技能库 $B_0$ 通过教师 LLM 从初始 rollout 的成功/失败轨迹中提炼而成，分为通用技能 $B_g$ 和任务类型特定技能 $B_k$。每个 episode 开始时，使用嵌入检索（cosine similarity）从 $B_g \cup B_k$ 中召回 Top-K 技能，仅将标题和一行意图摘要以紧凑目录形式注入 prompt，完整技能内容仅在显式调用时返回。

**显式技能调用**：智能体在每个 step 输出环境动作 $a_t^{env}$ 和可选技能调用 $c_t \in \{\emptyset\} \cup S_{ret}$。调用时智能体发出 `<skill_call>NAME</skill_call>` 标签，框架在下一步观测中将完整技能内容（intent、principle、applicability）返回，使每次调用成为轨迹中的可追踪离散事件。

**策略优化**：采用 GRPO 对策略 $\pi_\theta$ 进行优化，目标函数为：
$$J(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G} \min\left(\rho_i \hat{A}_i, \operatorname{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$
其中 $\hat{A}_i = (R_i - \bar{R})/\sigma_R$ 为组相对优势。由于 `<skill_call>` 标签嵌入在生成 token 序列中，该目标同时优化动作和技能调用决策。

**多路径技能诱导**：每 $I$ 步，教师 LLM $M_T$ 根据轨迹缓冲区的内容选择模式：仅成功轨迹时做 extraction，仅失败轨迹时做 refinement，两者都有时做 contrastive analysis，综合生成新技能 $S_{new} = M_T(\mathcal{T}^+, \mathcal{T}^-, B, m)$，经词法+语义去重后加入技能库。

**基于证据的技能验证**：每次技能调用后更新其 EMA 成功率 $\hat{p}_s \leftarrow \alpha \cdot r + (1-\alpha) \cdot \hat{p}_s$，同时累计调用次数 $n_s$。低表现分数定义为 $\operatorname{conf}(s) = (1 - \hat{p}_s) \cdot (1 - 0.5^{n_s/h})$，高分数技能优先被送入 LLM 反思，决定是否 keep 或 revise（重写 principle 和 applicability）。

## 实验与结果

**数据集与基线**：在 ALFWorld、WebShop、AppWorld 三个 benchmark 上评估，对比基线包括 ReAct、Reflexion、Mem0、SimpleMem、RLOO、GRPO、Memory-Augmented RL 方法及最强技能基 SKILLRL。

**主要结果（Qwen2.5-7B）**：ALFWorld 整体成功率 93.6%（SKILLRL: 89.9%），WebShop 得分 89.8 / 成功率 83.0%（SKILLRL: 85.2 / 72.7%），AppWorld TGC 23.8% / SGC 14.3%（SKILLRL: 19.0% / 5.36%），较 SKILLRL 平均提升 6.3%。在 AppWorld 上 SGC 近乎翻三倍。

**Scaling 结果**：Qwen3-4B 达 ALFWorld 87.9%、WebShop 84.0%，接近或超过 Qwen2.5-7B + SKILLRL；Qwen3-30B-A3B 达 ALFWorld 94.3%、AppWorld TGC 59.5%。

**消融（Qwen3-4B）**：移除显式调用 ALFWorld 从 87.9→77.9；移除技能库→79.3；移除多路径诱导→82.1；移除效果追踪→83.6；移除 LLM 反思→82.1。多路径诱导和效果追踪是最关键组件。

**训练效率**：SKILLFORGE 相比 SKILLRL 在三个基准上端到端训练时间更短（AppWorld 16.5h vs 18.7h，ALFWorld 5.9h vs 6.7h），技能相关开销占总时间低于 10%。

## 相关工作脉络
- **SKILLRL**：最直接前作，采用追加式技能库+trajectory-level 递归扩展，但缺乏 per-skill 验证与显式调用；本文通过显式调用和证据验证实现 skill-level credit assignment，且无需 SFT 初始化。
- **ReAct / Reflexion**：经典 in-context 智能体方法，依赖即时推理或自我反思，不具备跨 episode 知识积累能力。
- **Mem0 / SimpleMem**：外部记忆增强方法，存储压缩后的经验摘要，但信息噪音大、难以提取可复用的结构化技能。
- **GRPO / RLOO**：RL 训练基线；本文在其基础上引入可演化的技能库，实现 Action + Skill 联合优化。
- **Agent0 / SkillClaw**：同样探索 agent 自我进化，但 SkillClaw 关注集体协作演化，Agent0 侧重工具集成推理；本文聚焦技能的质量验证闭环。

## 局限性与未来方向
- **依赖外部教师 LLM**：技能诱导和修订质量受教师模型能力制约，自教师设置虽有效但性能低于强教师。
- **技能库持续增长带来的检索开销**：虽然去重机制控制冗余，但长期训练中技能数量仍会增加，检索延迟需关注。
- **显式调用引入额外 token**：在大规模部署场景下可能增加 prompt 长度和推理成本。
- **未来方向**：轻量化自教师方案、动态技能库压缩/淘汰策略、探索技能跨环境泛化机制。

## 研究启发与可借鉴点
- **显式调用将技能使用转化为可训练信号**：通过离散 token 机制让 RL 直接优化"何时调用哪个技能"，这一设计可迁移至任何需要外部知识调用的智能体系统。
- **EMA 成功率 + 低表现分数的组合验证思路简洁有效**：无需额外标注即可在线评估技能质量，可推广至其他 memory/skill 系统的自动化质量控制。
- **多路径诱导中的对比分析路径**：从成功-失败配对中提取决策差异，比单纯提取成功经验更具判别性，可作为通用技能提取范式。
- **技能跨模型规模可迁移性**：小模型演化出的技能可直接用于大模型且无需再训练，提示了技能作为独立知识资产的潜力。
- **无需 SFT 的端到端训练设置**：降低了技能增强 RL 的启动门槛，对资源受限团队具有参考价值。

## 关键术语表
- **Skill Calling（显式技能调用）**：智能体通过 `<skill_call>NAME</skill_call>` 标签主动请求技能内容，使每次调用成为轨迹中的离散可追踪事件。
- **Skill Bank（技能库）**：存储结构化可复用技能的外部知识集合，分为通用技能和任务类型特定技能两级层次。
- **Underperformance Score（低表现分数）**：综合技能 EMA 成功率和调用次数的指标，用于识别需要优先验证和修订的技能。
- **Multi-Pathway Skill Induction（多路径技能诱导）**：从成功提取、失败精炼、成功-失败对比分析三条路径合成新技能的机制。
- **GRPO（Group Relative Policy Optimization）**：基于组内相对优势的 RL 优化算法，本文用于联合优化动作和技能调用策略。
- **EMA Success Rate（指数移动平均成功率）**：对技能每次调用结果进行加权滚动平均，用于平滑估计技能的有效性。

## 可复现要素
- **数据集**：ALFWorld、WebShop、AppWorld（均为公开 benchmark）。
- **代码**：论文未提及代码开源情况。
- **权重**：使用 Qwen2.5-7B-Instruct、Qwen3-4B-Instruct、Qwen3-30B-A3B-Instruct（开源权重可用）；教师模型为 Qwen3-Max（闭源）。
- **关键超参**：学习率 1e-6、group size G=8、KL coefficient 1e-3、rollout temperature 0.9、EMA 平滑因子 α=0.1、使用半衰期 h=20、更新间隔 I=5、检索 Top-6+Top-6、最大 response length 4096。
