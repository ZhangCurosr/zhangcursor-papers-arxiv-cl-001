---
title: "Write-Execute-Refine-From-Skill-Followers-to-Skill-Optimizer"
source: https://arxiv.org/pdf/2608.17587v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:04:54"
field: "Agent 技能学习与执行反馈优化"
keywords: ["skill optimization", "agent reinforcement learning", "tool-use", "refinement state construction", "group-relative optimization", "phase-wise self-bootstrapping"]
innovations: ["将 refinement-state construction 作为独立训练问题，提出 Skill Optimizer 独立于 executor 的框架", "phase-wise self-bootstrapping：用 mixed-outcome 配对重组跨 phase 训练状态", "4B 专用 optimizer 在 BFCL v4 上超过 GPT-5.5 等通用大模型"]
benchmarks: ["BFCL v4 Multi-Turn", "τ²-bench"]
---

# 论文速读：Write, Execute, Refine: From Skill Followers to Skill Optimizers via Reinforcement Learning from Execution Feedback

## 一句话总结
论文提出了 WER（Write-Execute-Refine）多阶段框架，通过训练一个独立的 Skill Optimizer，使其学会从执行轨迹和程序化验证反馈中迭代改进 agent 所用的自然语言 skill；在 BFCL v4 和 τ²-bench 上，WER 显著优于无技能基线及通用大模型。

## 研究问题与动机
1. **Skill 写作能力的显著缺口**：Expert 编写的 skill 能将 SkillsBench 均分从 33.9% 提升至 50.5%，但 agent 自己写的 skill 反而比不用 skill 低 8–11 个百分点，说明"遵循程序指导"与"从执行证据改进指导"是两种不同能力。
2. **Inference-time loop 无法训练 writer**：现有反思/修订循环（如 Reflexion、SkillRevise）能修复当前 artifact，但每次任务仍依赖基础模型重新诊断执行日志，无法让 skill writer 本身变得更会写 skill。
3. **Plausible-but-wrong 纠正的代价**：仅注入 10% 看似合理但错误的经验就会使 τ²-bench Pass@1 从 82.5 降至 77.2，且自验证几乎无法恢复，凸显需要有更强执行 grounding 的训练机制。
4. **如何组织中间技能的执行经验为训练状态**：需要设计 rollout 机制、credit assignment 与跨阶段经验构造，将执行后果转化为可复用的 refinement state。

## 核心贡献（创新点）
1. **将 refinement-state construction 定义为迭代 skill 优化的核心训练问题**，并形式化为独立于 executor 的 Skill Optimizer 策略；与已有工作相比，后者多聚焦单轮 in-context 修订或 prompt artifact 优化，未训练 skill writer 本身。
2. **提出 phase-wise self-bootstrapping**：将同任务/同 skill 下的成功与失败轨迹配对，重组为下一阶段的 refinement state，使 optimizer 从自身前一轮输出的执行后果中学习；区别于 SkillMaster/SkillOS 的 episode 后 mutation 或仓库级管理决策。
3. **组内相对优化（Group-Relative Optimization）**：在固定 refinement state 下对 K 个候选 skill 进行 n 次 rollout，用 verifier 得分计算组内相对优势并更新 πθ，无需 KL 项；相比 Skill-R1 的 intra- 与 inter-generation 双级优势，本文强调同 phase 内的局部对比。
4. **4B 参数 optimizer 在 BFCL v4 上超过所有评测的通用大模型**（GPT-5.5 等），证明 specialized refinement training 的收益独立于通用 in-context reasoning 能力。

## 方法详解
- **Skill 定义**：短 markdown 文档，含 name、description、numbered workflow、notes 四部分；执行时被 prepend 到下游 agent system prompt，不修改 agent 参数。
- **Refinement state** $x = (q, \mathcal{C}, h, s, e)$：$q$ 为用户查询，$\mathcal{C}$ 为可见工具定义与环境描述，$h$ 为交互历史，$s$ 为当前 skill，$e$ 为上轮执行证据。
- **Skill Optimizer**：$\pi_\theta$ 仅输出文档 $s' \sim \pi_\theta(\cdot|x)$，不直接发出工具动作；cold start 时 $s$ 由 GPT-5.1 从零生成。
- **执行通道**：冻结 executor $\pi_A$（GPT-4o）在环境 $\mathcal{E}$ 中对每个候选 $s'$ 进行 $n$ 次 rollout，得到完整轨迹 $\tau = (a_1,o_1,...,a_T,o_T,y)$；轨迹以原文形式传入 optimizer，不压缩。
- **奖励设计** $R(s') = \frac{1}{3}(R_{fmt} + R_{task} + R_{len})$：$R_{fmt}$ 检查输出可解析（reasoning 与 skill body 分块）并随训练衰减；$R_{task}$ 为 verifier 聚合得分（BFCL 对比最终对象状态，τ²-bench 对比终态数据库与必需操作）；$R_{len}$ 约束 reasoning 长度。
- **Group-Relative Advantage**：$\hat{A}_k = R_k - \mu(\mathbf{R})$，采用 clipped GRPO 损失 $\mathcal{I}(\theta) = \mathbb{E}_x[\frac{1}{K}\sum_k \min(\rho_k \hat{A}_k, \text{clip}(\rho_k,1-\epsilon,1+\epsilon)\hat{A}_k)]$，无 KL 项。
- **Phase-wise Self-Bootstrapping**：每 phase 结束后，从经验缓冲区 $B^{(p)}$ 中**仅保留 mixed-outcome 记录**（$n=2$ 次 rollout 得分为 1 的记录），即同一 skill 在同一任务下一次成功一次失败；将其重组为下一 phase 的 refinement state：成功轨迹放 success block，失败轨迹放 failure block，skill 与工具上下文保留原样，直接作为 optimizer 的输入。

## 实验与结果
- **数据集与划分**：BFCL v4 multi-turn-base（200 实例，50 train / 150 test，四领域均衡）；τ²-bench base tasks（同样 1:3 划分）；训练集 pooling 两个 benchmark。
- **评估指标**：Pass@1（单次尝试成功率），每个评测重复 3 次取平均。
- **主要结果（Pass@1, %）**：

| Method | BFCL v4 AVG | τ²-bench AVG |
|---|---|---|
| No Skill | 68.83 | 46.87 |
| GPT-5.1 Seed Skill | 67.28 | 47.70 |
| Qwen3-4B (untrained optimizer) | 67.28 | 40.43 |
| Skill-R1 | 71.25 | 41.47 |
| Trace2Skill | 72.14 | 43.54 |
| **WER (Ours)** | **76.63** | **50.72** |

- WER 相对 No Skill：BFCL v4 +7.80 pts，τ²-bench +3.85 pts。
- 相对同 backbone 无 optimizer 训练：BFCL v4 +9.35 pts，τ²-bench +10.29 pts。
- 相对 GPT-5.1 seed：BFCL v4 +9.35 pts，τ²-bench +3.02 pts。
- **消融**：Phase 1→2→3 平均 Pass@1 单调提升：69.35% → 71.29% → 76.63%。
- **推理深度**：Depth 0/1/2/3 对应 67.28% / 70.67% / 76.63% / 75.33%，两轮修订后饱和。
- **与通用模型对比**：WER 4B 优化器 76.63% > GPT-5.5 74.75% > DeepSeek-V4-Flash 72.61% > Gemini 3.5 Flash 71.96% > Claude Sonnet 4.6 69.91%。
- **定性案例**：三轮修订依次修复 file-creation、numerical-aggregation 等不同失败模式。

## 相关工作脉络
1. **Inference-time refinement**（EvoSkill、SkillOpt、SkillRevise、Execute-Distill-Verify）：在线修订 skill artifact，但 writer 模型不变；WER 训练 optimizer 本身。
2. **Agentic RL 中的 skill**（SAGE、Skill1、SkillRL、ReSkill、Skill0/Skill0.5）：多更新 task agent 或与 skill 同步进化；WER 保持 executor 冻结，只训练独立 optimizer。
3. **Learned skill management**（SkillMaster、SkillOS、Skill-R1）：分别侧重 episode 后 mutation、仓库管理、跨代改进；WER 在同 phase 组内比较 + 跨 phase 配对构造。
4. **Auto prompt optimization**（OPRO、PromptAgent、DSPy、TextGrad）：优化文本 artifact 但不训练 writer；WER 保留外部文本接口同时训练 optimizer。
5. **Reflection / trajectory distillation**（Reflexion、Contextual Experience Replay、Trace2Skill）：基于轨迹回放或局部 lesson 提取；WER 以 verifier 确定信号 + 混合轨迹配对提供更强 credit assignment。

## 局限性与未来方向
- 仅在 BFCL v4 与 τ²-bench 两个具备程序化 verifier 的 benchmark 上验证，未证明向 open-ended 评估、未见工具接口或不同 executor 的泛化。
- 完整保留成功/失败轨迹导致 refinement state 体积随交互长度线性增长，未测试更长 horizon、多模态轨迹或大规模 skill repository 下的 context 可扩展性。
- 未来将扩展至更复杂 setting 并研究如何在保留诊断证据的前提下压缩经验。

## 研究启发与可借鉴点
1. **Refinement-state construction 作为独立训练问题**：将"如何把执行经验组织成训练状态"置于核心，而非仅优化单轮 prompt；该思路可迁移至 prompt 优化、tool-use policy 训练等领域。
2. **混合结果配对（Matched Outcomes）机制**：仅在 skill 与任务固定时对比成功/失败分支，提供强因果局部信号；可用于任何需要归因"哪个指令差异导致了行为差异"的 agent 训练 pipeline。
3. **Executor 冻结 + 程序化 verifier 解耦**：将 executor 能力与 optimizer 训练分离，避免 credit assignment 被模型主观评分污染；设计可复现 reward 的系统时可直接复用。
4. **Phase-wise self-bootstrapping 的 curriculum 效应**：通过自举让训练分布逐步迁移至"skill 近乎充分"的 hard cases；这种 progressive difficulty 构造法适用于多数 iterative policy improvement 场景。
5. **4B 专用 optimizer 胜过通用大模型**：证明 refined execution-conditioned policy 存在独立于 in-context reasoning 的收益，值得在更多垂直 task 上复现和扩展。

## 关键术语表
**Skill Optimizer**：独立于 executor 的策略 πθ，输入 refinement state 并输出自然语言 skill 文档，不直接与环境交互。
**Refinement State**：复合训练状态 $x=(q,\mathcal{C},h,s,e)$，包含任务、工具定义、交互历史、当前 skill 与执行证据。
**Phase-Wise Self-Bootstrapping**：每训练 phase 将前 phase 的 mixed-outcome 经验重组为下一 phase 的 refinement state，形成 revision tree。
**Group-Relative Optimization**：在同一 refinement state 下对 K 个候选 skill 比较相对优势，以组内偏差驱动 GRPO 更新。
**Matched Outcomes**：同一 skill 在相同任务下一次成功一次失败的两次 rollout 构成的配对，提供可归因的对照信号。
**Programmatic Verifier**：基于环境终态与参考解的程序化对比器，输出确定性强奖励，避免模型自评引入 bias。
**Refinement Depth**：推理时从 seed skill 出发连续应用 optimizer 的轮数；论文显示深度 2 后收益饱和。
**Pass@1**：单次尝试（无重采样）任务完成率，本文主要评估指标。

## 可复现要素
- **数据集**：BFCL v4（200 多轮 base 任务）、τ²-bench（Airline/Retail/Telecom 三领域）；训练集各取 1/4，测试集 3/4。论文未说明是否完全公开，但 github 仓库提供训练脚本。
- **代码/权重**：代码已开源 https://github.com/littlepkk/WER4skill-optimizer-training；模型权重论文未声明开源链接。
- **关键超参**：verl 框架，8×Ascend 910B NPU；GRPO batch_size=6，每 prompt 4 rollouts；lr=1e-6 cosine schedule；prompt max 19,000 tokens，response max 4,096 tokens；temperature=0.95，top-k=50；K=4 候选，n=2 rollouts；P=3 phases。
- **训练配置细节**：Appendix A.3 提供完整设置；附录 B 含 prompt 示例（Figure 6/7）。
