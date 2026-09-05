---
title: "S-sup-3-sup-Gym-Can-LLMs-Turn-Self-Testing-and-Self-Judging"
source: https://arxiv.org/pdf/2608.31100v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:29:24"
field: "LLM Agent 自我改进与自适应评估"
keywords: ["self-improving agents", "self-testing", "self-judging", "in-context learning", "parameter training", "interactive benchmarks", "experience-driven learning"]
innovations: ["提出S³Gym统一框架，将自我改进形式化为探索-评判-整合-评估五阶段迭代并分离宽松探索与严格hold-out评估", "在相同协议下系统比较History ICL、Summary Memory和Parameter Training三条经验整合路径", "显式评估Self-Judging可靠性及其与后续改进效果的耦合关系，发现局部评分准确不足以预测改进"]
benchmarks: ["S³Gym"]
---

# 论文速读：S³Gym: Can LLMs Turn Self-Testing and Self-Judging into Self-Improvement?

## 一句话总结
本文提出了 S³Gym，一个评估大语言模型通过自我测试、自我评判和自我改进闭环实现行为提升的交互式基准，系统比较了历史 ICL、摘要记忆和参数训练三种经验整合路径在七款文本游戏中的表现。

## 研究问题与动机
- 现有 Agent 基准（如 AgentBench、KORGym）将模型视为固定策略进行静态评估，无法回答"模型能否利用自身交互经验改进未来行为"这一问题。
- 自我改进不是经验的自动产物：积累轨迹 ≠ 学习； agent 必须主动生成诊断证据、解释成败原因，并将判断转化为可执行策略。
- 不同经验整合路径（上下文级 vs. 参数级）在不同任务结构下的有效性差异尚不明确，缺乏统一协议进行对比。
- 现有工作多聚焦单一改进机制，难以剥离"经验质量""评判可靠性"和"整合机制"三者对最终改进效果的各自贡献。

## 核心贡献（创新点）
1. **统一形式化体验驱动的自我改进流程**：将自我改进建模为 Explore→Judge→Consolidate→Update→Evaluate 五阶段迭代，并分离宽松探索与严格 hold-out 评估，使经验转化过程可测量。
2. **首次在同一协议下系统比较三种经验整合路径**（History ICL / Summary Memory / Parameter Training），揭示路径选择高度依赖任务结构而非模型能力本身。
3. **显式评估 Self-Judging 可靠性及其与改进效果的耦合关系**：通过对比模型自评分与 verifier 计算的真实奖励，发现局部评分准确并不能可靠预测后续改进。
4. **提供七款文本游戏实例化的统一基准**，覆盖规则归纳、约束满足、空间规划、资源分配、多智能体策略等多种交互推理范式。

## 方法详解
- **核心循环**：$R_t^{(p)} = \text{Explore} \rightarrow \text{Judge} \rightarrow \text{Consolidate} \rightarrow \text{Update}_p \rightarrow \text{Evaluate}$，其中 $p \in \{\text{History, Memory, Training}\}$ 表示改进路径。
- **交互协议**：每步 agent 联合输出动作与自我评分 $(a_{x,i}, s_{x,i}) = \pi_\theta(O_{x,i}, H_{x,i}; Z_t^{(p)})$，环境返回下一状态、可见反馈 $F_{x,i}$ 和 verifier 奖励 $r_{x,i}$；关键设计是探索阶段 **agent 不接收** verifier 真实奖励，仅依赖自评分进行经验整理。
- **三种整合路径**：
  - **History ICL**：将带分数的轨迹直接序列化追加到后续 episode 上下文（$C_{t+1} = \text{Serialize}(C_t, \mathcal{T}^{\text{exp}}_t, S^{\text{exp}}_t)$）。
  - **Summary Memory**：将轨迹压缩为可复用策略三元组 $M_{t+1} = (R^{\text{retain}}_t, R^{\text{avoid}}_t, D^{\text{next}}_t)$。
  - **Parameter Training**：高评分动作作为正样本进行 SFT（$\mathcal{D}^{\text{SFT}}_t = \text{BuildSFT}(\cdot)$，$\theta_{t+1} = \text{SFT}(\theta_t, \mathcal{D}^{\text{SFT}}_t)$）。
- **评估协议**：每 3 个探索 episode 后进行一次严格模式 hold-out 评估（3 个 episode），共 10 个检查点（episodes 0–30），探索与评估使用不相交随机种子。
- **三个核心指标**：Avg（全程平均分）、Max（最高分）、AUC⁺（高于初始基线的正面积）。

## 实验与结果
- **数据集/游戏**：7 款文本游戏（Chess、Minesweeper、Nullify、Tetris、Snake、Plants-vs-Zombies、Trust Evolution），每款有探索配置（更宽松）与评估配置（更严格）。
- **评测模型**：GPT-4o/4.1、o3-mini、Gemini-2.5-Flash/Pro、GPT-5.5、Gemini-3.5-Flash 共 7 款。
- **关键结果**：
  - **历史 ICL 最优**：GPT-5.5 在 PvZ 上 AUC⁺ 达 548.499（最强单结果），Gemini-3.5-Flash 在 Trust Evolution 上 AUC⁺ 达 200.000。
  - **摘要记忆选择性优势**：Gemini-2.5-Flash Minesweeper AUC⁺ 从 0.000 提升至 7.794；PvZ 从 24.402 提升至 238.501；但 GPT-5.5 PvZ 从 548.499 降至 33.219，说明摘要不是无条件优于原始历史。
  - **参数训练结果分化**（Qwen3-8B，20 轮检查点）：Trust Evolution 提升显著（AUC⁺=163.5，18/19 检查点高于基线）；PvZ 出现严重负迁移（从 23 降至 6）；Minesweeper/Nullify/Tetris 无改善。
  - **Self-Judging 部分可靠**：Chess 事件一致性仅 0.496（近随机），Trust 为 0.665；整体自评准确率与后续改进相关性接近零（$\rho(A,g)=-0.010$）。
  - **结论**：自我改进既非自动也非均匀；最强路径依赖任务结构——摘要适合可压缩为策略规则的领域，历史 ICL 适合需精确状态信息的反应式控制任务。

## 相关工作脉络
1. **AgentBench / AgentBoard**：评估 Agent 多轮交互能力但视模型为固定策略；S³Gym 进一步关注经验驱动的动态行为改进。
2. **Reflexion / Voyager**：前者用口头强化学习将反思文本注入上下文，后者维护可重用技能库；S³Gym 在统一协议下横向比较了类似机制与原始历史的优劣。
3. **STaR / ReST / Re-ReST**：通过自我生成的推理轨迹进行参数训练；S³Gym 将此类训练路径纳入与上下文级方法的同一比较框架，并揭示负迁移风险。
4. **ContinualSkillBench / SEAGym / PAST-Bench**：分别评估持续技能学习、持久 harness 演化、跨会话经验保留；S³Gym 的独特定位是显式耦合 Self-Testing + Self-Judging，并在可执行环境验证下进行 hold-out 评估。
5. **Social Gym / AI4AI Bench**：前者关注多智能体博弈推理，后者聚焦递归自改进算法设计；S³Gym 专注于单 Agent 经验→改进闭环的诊断分析。

## 局限性与未来方向
- **参数训练稳定性不足**：当前实现下训练易导致负迁移（如 PvZ 场景），需要更好的轨迹过滤和经验质量评估机制。
- **Self-Judging 校准误差显著**：尤其在 Chess 和 Trust Evolution 中接近随机，限制了经验整合的上限。
- **仅测试了 7 款游戏**：任务类型仍有限，且所有游戏均为文本接口，未覆盖视觉或多模态环境。
- **探索配置与评估配置的分布差距**可能使某些任务上的"无改善"归因于探索不足而非模型本身缺陷。
- 未来方向包括改进 judgment 校准、发展更鲁棒的轨迹筛选与元学习机制、扩展至更长 horizon 和更复杂的交互环境。

## 研究启发与可借鉴点
1. **探索-评估严格分离的设计**：宽松探索 + 严格 hold-out 评估的协议可有效剥离"探索质量"与"泛化改进"的混杂效应，值得在多领域 Agent 评测中借鉴。
2. **自评与真实奖励的显式解耦**：不向 agent 暴露 verifier 奖励而只依赖自评分，能干净地诊断 Self-Judging 瓶颈；这一设计可用于研究外部反馈缺失场景下的鲁棒学习。
3. **多维度指标组合（Avg/Max/AUC⁺）**：兼顾整体性能、偶发突破和持续改进趋势，比单一指标更能刻画自我改进的动态过程。
4. **任务结构决定路径选择的发现**：对工程实践的直接启示——不存在通用最优的经验整合方式，应根据任务的可压缩性（是否适合抽象为规则）选择 History ICL 或 Summary Memory。

## 关键术语表
**Self-Testing**：agent 在交互环境中主动探索策略并生成诊断性证据的能力。
**Self-Judging**：agent 对自身的动作、结果及其可复用性进行评估和打分的能力。
**Self-Improvement**：agent 将经过测试和评判的经验转化为未来更好决策的能力。
**History ICL**：将带分数的交互轨迹直接序列化追加到后续 episode 上下文中的经验整合方式。
**Summary Memory**：将长轨迹压缩为策略规则、失败模式和方向建议的中间层记忆机制。
**AUC⁺**：衡量模型在多次检查点中持续高于初始基线的改进正面积，反映稳定改进程度。
**Negative Transfer**：参数训练后模型在特定任务上性能反而下降的现象，源于错误经验的内部化。
**Forward Deployed Engineer (FDE)**：嵌入部署环境、将通用模型转化为可用系统的工程师角色，其工作突显了现有基准与真实部署之间的鸿沟。

## 可复现要素
- **数据集**：七款文本游戏（Chess、Minesweeper、Nullify、Tetris、Snake、Plants-vs-Zombies、Trust Evolution），项目页面 https://self-developing-agents.github.io/
- **代码/权重**：论文未明确说明代码是否开源；提供了项目页面链接，项目名为 self-developing-agents
- **关键超参**：历史 ICL 输入上限 100,000 tokens；每次响应输出上限 14,000 tokens；每 episode 最多 64 步；30 个探索 episode；每 3 个 episode 评估一次（共 10 个检查点）；每个检查点 3 个严格模式评估 episode
