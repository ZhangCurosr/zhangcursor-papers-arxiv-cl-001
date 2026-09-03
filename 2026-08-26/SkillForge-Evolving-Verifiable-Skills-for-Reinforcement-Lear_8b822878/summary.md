---
title: "SkillForge-Evolving-Verifiable-Skills-for-Reinforcement-Lear"
source: https://arxiv.org/pdf/2608.24747v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:56:22"
field: "LLM Agent 强化学习与技能演化"
keywords: ["LLM Agent", "Reinforcement Learning", "Skill Learning", "Self-Evolving", "GRPO", "Memory-Augmented Agent"]
innovations: ["显式技能调用机制实现 skill-level credit assignment", "基于 EMA 成功率和使用次数的低效评分驱动的 LLM reflexion 验证", "多路径技能归纳（extraction/refinement/contrastive）协同技能库演化"]
benchmarks: ["ALFWorld", "WebShop", "AppWorld"]
---

# 论文速读：SkillForge: Evolving Verifiable Skills for Reinforcement Learning Agents

## 一句话总结
SKILLFORGE 提出一种连续技能演化框架，通过显式技能调用和基于证据的验证机制，使 LLM agent 在 RL 训练中能够持续提炼、评估和更新技能库，从而克服现有方法将技能视为静态追加仓库的缺陷。在 ALFWorld、WebShop 和 AppWorld 三个基准上，相比最强 baseline SKILLRL 平均提升 6.3%。

## 研究问题与动机
1. **RL 训练 agent 缺乏跨 episode 知识积累**：现有 RL-trained agents 多为 episodic，每次 episode 从零开始，无法复用历史成功经验，导致重复探索和低效学习。
2. **现有技能方法 treating skill bank as append-only**：SKILLRL 等方案从轨迹中提取技能并追加到技能库，但从不验证已存储技能是否仍有效，导致错误/过时技能长期污染知识库。
3. **技能使用情况不可观测**：技能以批量方式注入 prompt，无法追踪 agent 是否真正依赖某个技能，也无法进行细粒度的技能级信用分配（skill-level credit assignment）。
4. **技能有效性难以归因与质量控制**：缺乏基于使用证据的效果跟踪机制，无法判断成功轨迹是由特定技能还是偶然因素导致。

## 核心贡献（创新点）
1. **显式技能调用策略**：将技能使用设计为 trajectory 中的离散可观测事件，使 RL 能够联合优化环境动作和技能调用决策，与 SKILLRL 的 prompt 批量注入形成本质区别。
2. **基于证据的技能验证机制**：通过 EMA 成功率和使用次数计算技能"低效评分"（underperformance score），结合 LLM reflexion 对低质量技能进行 keep 或 revise 决策，实现技能库的持续质量保障。
3. **多路径技能归纳（Multi-pathway Skill Induction）**：从成功轨迹提取策略、从失败轨迹提炼纠正方案、通过对比分析识别决定性行为差异，三种路径协同驱动技能库增长，区别于 SkillRL 仅依赖递归扩展的设计。
4. **免 SFT 初始化的纯 RL 训练范式**：无需额外 SFT 预热阶段，直接从 instruction-tuned 模型出发，在 RL 训练过程中同步演化技能库，简化训练流程。

## 方法详解
**1. 技能表示与检索（Skill Representation and Retrieval）**
- 每个技能 $s$ 为结构化元组：$(title, intent, principle, applicability, category, status)$，其中 title 为可调用标识符，intent 为用途描述，principle 为核心决策策略，applicability 为触发条件，category 分为 general 或 task-type $k$。
- 技能库初始化：在目标环境 rollout 初始 instruction-tuned 模型，通过 teacher LLM $M_T$ 蒸馏轨迹——成功轨迹提炼为战略模式，失败轨迹合成纠正教训。
- 检索：用 embedding 计算任务描述与技能 intent 的余弦相似度，取 top-K 技能格式化为紧凑目录（仅含 title + 一行 intent summary）注入系统 prompt。

**2. 显式技能调用与策略优化（Skill Calling and Policy Optimization）**
- 每个 step 动作定义为 $a_t = (a_t^{\text{env}}, c_t)$，其中 $c_t \in \{\emptyset\} \cup S_{\text{ret}}$，agent 通过 emit `<skill_call>NAME</skill_call>` tag 显式调用技能。
- 调用后框架返回完整技能内容（intent, principle, applicability）作为下一次 observation 的一部分。
- 策略优化使用 GRPO：$J(\theta) = \mathbb{E}[\frac{1}{G}\sum_{i=1}^G \min(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})]$，其中 $\hat{A}_i = (R_i - \bar{R})/\sigma_R$ 为组内相对优势。
- 轨迹被 LLM 摘要为结构化 abstraction，按 reward 分为成功集 $\mathcal{T}^+$ 和失败集 $\mathcal{T}^-$，存入 per-task-type buffer。

**3. 技能演化（Skill Evolution）**
- **多路径归纳**：每 $I$ 步，teacher LLM $M_T$ 根据缓冲区状态选择模式合成新技能：$S_{\text{new}} = M_T(\mathcal{T}^+, \mathcal{T}^-, \mathcal{B}, m)$，其中 $m$ 为 contrastive（两者均非空）、extraction（仅成功）或 refinement（仅失败）。新技能经 lexical matching + semantic similarity 去重后加入银行。
- **基于证据的验证**：每次技能 $s$ 被调用且 episode 结果为 $r \in \{0, 1\}$ 时，更新 EMA 成功率：$\hat{p}_s \leftarrow \alpha \cdot r + (1-\alpha) \cdot \hat{p}_s$，同时记录总使用次数 $n_s$。计算低效评分：$\text{conf}(s) = (1 - \hat{p}_s) \cdot (1 - 0.5^{n_s/h})$，其中 $\alpha=0.1$，$h=20$。高 conf(s) 的技能优先触发 LLM reflexion（keep 或 revise principle 和 applicability）。

## 实验与结果
**数据集与评估指标**：
- ALFWorld（文字交互家居任务）：每子任务成功率(%) + 总体平均成功率
- WebShop（电商搜索购买）：平均 score + 成功率(%)
- AppWorld（应用模拟交互）：Task Goal Completion (TGC) + Scenario Goal Completion (SGC)

**主要结果（Qwen2.5-7B backbone）**：
| 基准 | 方法 | 指标 |
|------|------|------|
| ALFWorld All | SKILLRL | 89.9% |
| ALFWorld All | SKILLFORGE | **93.6%** (+3.7) |
| WebShop Score | SKILLRL | 85.2 |
| WebShop Score | SKILLFORGE | **89.8** (+4.6) |
| WebShop Success | SKILLRL | 72.7% |
| WebShop Success | SKILLFORGE | **83.0%** (+10.3) |
| AppWorld TGC | SKILLRL | 19.0% |
| AppWorld TGC | SKILLFORGE | **23.8%** (+4.8) |
| AppWorld SGC | SKILLRL | 5.36% |
| AppWorld SGC | SKILLFORGE | **14.3%** (×2.67) |

**Scaling 结果**：Qwen3-4B 已达 87.9（ALFWorld）、84.0%（WebShop success）；Qwen3-30B-A3B 达到 94.3（ALFWorld）、59.5（AppWorld TGC）。

**消融实验（Qwen3-4B）**：
- 移除显式调用：ALFWorld 87.9 → 77.9（-10.0）
- 移除技能库：87.9 → 79.3（-8.6）
- 移除多路径归纳：87.9 → 82.1（-5.8）
- 移除去重：AppWorld TGC 44.6 → 38.7（-5.9）
- 移除效果跟踪：87.9 → 83.6（-4.3）
- 移除 LLM reflexion：87.9 → 82.1（-5.8）

**训练效率**：SKILLFORGE 端对端训练时间优于 SKILLRL（ALFWorld 5.9h vs 6.7h，AppWorld 16.5h vs 18.7h），技能相关开销 <10%。

## 相关工作脉络
1. **SKILLRL (Xia et al., 2026)**：最接近的前作，同样维护技能库并与 policy 共同演化，但将技能作为 prompt 上下文批量注入，无显式调用机制，无 per-skill 验证，且需 SFT 预热。SKILLFORGE 通过显式调用实现 skill-level credit assignment 和持续验证。
2. **ReAct / Reflexion**：经典 agent 框架，依赖 in-context reasoning/reflection，但本质仍为 episodic，无法跨 episode 积累结构化知识。
3. **Mem0 / SimpleMem**：外部记忆增强方法，存储或压缩历史轨迹，但保留 noisy 信息，难以提取可复用的决策原则。
4. **GRPO (Shao et al., 2024)**：群体相对优势策略优化算法，本文作为 policy update 的基础优化器，与技能机制正交结合。
5. **Agent0 (Liu et al., 2025)**：零数据 self-evolving agent，探索工具集成的推理能力，但与技能演化框架的设计目标不同。
6. **SkillClaw (Ma et al., 2026)**：集体演化的 agent skill 框架，与 SKILLFORGE 同期工作，侧重不同（集体 vs 单体 + 显式调用验证）。

## 局限性与未来方向
1. **依赖外部 Teacher LLM**：技能归纳和验证质量受 teacher 模型能力制约（虽有 self-teacher 实验证明框架本身有效，但 Qwen3-Max 仍带来额外增益）。
2. **技能库持续增长带来检索开销**：新技能不断 induction 入库，长期训练中检索成本可能上升（尽管论文显示 growth 是 controlled 的）。
3. **显式调用引入额外 token**：`<skill_call>` tag 增加 prompt 长度和推理成本，在大尺度部署中需权衡。
4. **未探索跨环境/跨任务的 skill transfer**：当前工作局限于单环境内演化，技能跨环境迁移能力待研究。
5. **Teacher 敏感性与成本权衡**：附录 A.7 显示 teacher 质量影响性能，如何降低对强 teacher 的依赖是未来方向。

## 研究启发与可借鉴点
1. **显式调用设计实现 skill-level credit assignment**：将技能使用建模为 trajectory 中的离散可观测事件，使 RL 能够直接优化调用决策，这一设计可迁移到任何需要外部知识调用的 agent 系统。
2. **基于 EMA + 半衰期的低效评分机制**：用 $\text{conf}(s) = (1-\hat{p}_s) \cdot (1-0.5^{n_s/h})$ 量化技能老化程度，兼顾成功率和累计使用量，可用于其他 memory/skill 系统的质量控制。
3. **多路径归纳策略（extraction/refinement/contrastive）**：区分"从成功提取"、"从失败纠正"、"对比分析"三种归纳模式，可按场景组合使用，对知识提取模块设计有通用参考价值。
4. **免 SFT 的纯 RL 训练范式的可行性**：证明无需预热 SFT 即可从 instruction-tuned 模型起步同步演化技能，可降低训练 pipeline 复杂度，适合资源受限场景。
5. **技能库 evolution visualization**：通过 t-SNE 投影和 bank size 曲线直观展示技能演化过程，这种可视化方法可用于其他 self-evolving 系统的诊断分析。

## 关键术语表
**Skill Calling**：Agent 通过 emit 结构化 tag（`<skill_call>NAME</skill_call>`）在交互过程中显式调用技能，使调用行为成为 trajectory 中的可追踪离散事件。
**Underperformance Score (conf(s))**：综合技能 EMA 成功率和累计使用次数的量化指标，用于识别需要 reflexion 审查的低质量技能。
**Multi-Pathway Skill Induction**：从成功轨迹（extraction）、失败轨迹（refinement）和成败对比（contrastive）三条路径分别合成新技能的知识归纳机制。
**GRPO (Group Relative Policy Optimization)**：群体相对优势策略优化算法，通过组内归一化优势估计策略梯度，本文作为 policy update 的基础优化器。
**Reflexion**：由 teacher LLM 执行的技能审查过程，根据使用证据和使用上下文决定对 flagged 技能执行 keep 或 revise（重写 principle 和 applicability）。
**EMA (Exponential Moving Average)**：用于平滑更新技能成功率的指数移动平均方法，使近期表现获得更高权重。
**Skill Transferability**：指在不同模型规模间迁移已演化技能库的能力，论文证明较小模型的后期技能可匹敌甚至超越大模型的自演化技能。
**TGC / SGC**：AppWorld 评估指标，Task Goal Completion（单任务完成率）和 Scenario Goal Completion（同一场景所有变体均完成率）。

## 可复现要素
- **数据集**：ALFWorld、WebShop、AppWorld（均为公开 benchmark）
- **代码/权重**：论文未提及代码开源声明
- **关键超参**：
  - Learning rate: 1e-6
  - Group size (G): 8
  - KL coefficient: 1e-3
  - Clip ratio: [0.20, 0.28]
  - Rollout temperature: 0.9
  - Max response length: 4096
  - Skill bank update interval (I): 5 steps
  - EMA smoothing factor (α): 0.1
  - Usage half-life (h): 20
  - Retrieval: top-6 general + top-6 task-specific skills per episode
- **训练框架**：VeRL
- **Teacher LLM**：Qwen3-Max（也可使用 policy model 自身作为 self-teacher）
- **Embedding model**：Qwen3-Embedding-0.6B
- **硬件配置**：Qwen2.5-7B/Qwen3-4B 使用 8× NVIDIA H20；Qwen3-30B-A3B 使用 16× H20
