---
title: "Co-Evolving-Actor-Conditioned-Critics-for-Non-Verifiable-Gen"
source: https://arxiv.org/pdf/2608.30397v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:03:52"
---

# 论文速读：Co-Evolving-Actor-Conditioned-Critics-for-Non-Verifiable-Gen

## 一句话总结
论文针对缺乏确定性验证器的开放式生成任务，提出将 critique 视为“actor-conditioned”的修订指导，设计了评估完整反馈-修订链路的 TAISCORE 奖励，并通过 GRPO 与 DPO 交替训练实现 critic 与 actor 的协同进化，使 8B 训练型 critic 在 WritingBench 等基准上超越零样本 120B 冻结 critic。

## 研究问题与动机
- **核心问题**：非可验证生成任务（创意写作、深度研究等）中，如何训练能真正提供“可执行修订指导”的 critic？现有方法仅凭 critique 孤立质量或最终修订质量打分，无法判断 actor 是否真正遵循了反馈并改进了目标维度。
- **actor 依赖性实证**：受控零样本分析表明，相同 critique 对强 actor 带来显著提升，对弱 actor 几乎无效；反之，更大的 critic（如 32B）产出的 critique 孤立评分更高，但在固定 8B refiner 时 adherence 反而下降。
- **错误归因风险**：单纯奖励最终 revision 质量会将“critique 无关的偶然改进”计入 credit；单纯奖励 critique 质量会忽略 actor 执行能力，产生“看起来好但无法落地”的空泛反馈。
- **动态对齐需求**：actor 能力提升后，原先有效的 critique 可能变得冗余，原先过难的 critique 可能变得可行，critic 必须随 actor 演化保持对齐，否则监督信号会错位。

## 核心贡献（创新点）
- **提出 actor-conditioned critique 视角**：论证 critique 有用性并非内生于文本本身，而是由 critique 与目标 actor 的交互（执行可行性 × 目标改进）共同决定，打破了“critique 质量独立于执行者”的隐含假设。
- **设计 TAISCORE（Targeted Actionable Improvement Score）**：首次将 reward 建模建立在 $(x, y_0, c, y_1)$ 完整四元组上，通过四维度诊断分分配 credit，避免孤立打分导致的错误归因。
- **构建 critic-actor 协同进化训练循环**：以 TAISCORE 为 GRPO 奖励更新 critic，以 critique-guided refinement 构造 DPO 偏好对更新 actor，交替迭代使 critic 始终适配 actor 当前能力。
- **实证“针对性训练胜过参数规模”**：8B TAISCORE 训练 critic 全面超越冻结 120B gpt-oss-120B critic；且 critic 必须与 target actor 规模匹配才能获得最大下游收益。

## 方法详解
- **TAISCORE 构造**：给定指令 $x$、初始回答 $y_0$、critique $c$、修订 $y_1$，judge 优先生成四个诊断分 $(q_{\mathrm{qual}}, q_{\mathrm{adh}}, q_{\mathrm{gain}}, q_{\mathrm{faith}})$，分别对应 critique 有效性、actor 遵循度、目标维度增益、忠实度；随后在同一推理链中输出最终标量 $T(\tau) \in [1,10]$ 作为 critic 训练 reward。诊断分提供显式推理支架，确保最终分数不混入无关改进或空泛好评。
- **Critic 更新（GRPO）**：每个 prompt 采样 $N{=}=4$ 条 critique，同一 actor 对每条 critique 做修订，得到 rollout 组 $\{\tau_i\}$。计算 group-relative advantage $A_i = (r_i - \mathrm{mean}(\{r_j\}))/\mathrm{std}(\{r_j\})$，用 GRPO 策略梯度 $\mathcal{J}_t(\kappa) = \mathbb{E}\left[\frac{1}{N}\sum_i A_i \log \kappa(c_i|x,y_0)\right]$ 更新 critic $\kappa$，迫使 critic 倾向产出对当前 actor 更有用指导的反馈。
- **Actor 更新（DPO）**：用更新后的 critic $\kappa_{t+1}$ 生成 critique-guided revision，由 blind pairwise judge（不看见 critique）比较 $y_1$ 与 $y_0$ 质量，选出 $y_1 \succ y_0$ 构造成 $M$ 对偏好数据，用 DPO 损失 $\mathcal{L}_{\mathrm{DPO}}(\pi) = -\mathbb{E}[\log\sigma(\beta\Delta_\pi)]$ 直接更新 actor $\pi$（$\beta=0.1$）。
- **协同进化（Co-evolution）**：从 $(\pi_0, \kappa_0)$ 出发，每轮 $t$ 先固定 actor 用 GRPO 更新 critic得 $\kappa_{t+1}$，再用新 critic 生成修订对 actor 做 DPO 更新得 $\pi_{t+1}$。实验共 3 轮，critic 与 actor 能力曲线同步上升。

## 实验与结果
- **数据集/基准**：创意写作（DeepWriting-20K 训练，WritingBench、HelloBench 的 OEQA/HTG 子集评估）；深度研究（OpenScholar 6K 查询训练，DeepResearch-Gym 评估，指标 KPR/KPC/Report Quality）。
- **模型与超参**：默认 actor/critic 均为 Qwen3-8B；judge 与零样本 baseline 使用 gpt-oss-120B；GRPO 每 prompt 采样 $N{=}4$ critique，学习率 $1{\times}10^{-6}$，KL 系数 0.02，mini-batch 8，temperature 0.8/top-p 0.95；DPO $\beta{=}0.1$，max length 4096。
- **主结果**：
  - 8B TAISCORE critic 在 WritingBench 达 75.96，超越冻结 gpt-oss-120B critic 的 75.41（+0.55），且全面优于 Outcome-gain（75.63）与 Critique-quality（75.18）。
  - Co-evolution 进一步提升至 76.72（相对 base 72.33 提升 +4.39），HelloBench OEQA 达 39.84，DeepResearch-Gym KPR 达 76.14、Quality 达 83.15，均为各基准最优。
  - 消融显示：直接修订（未 DPO）阶段 TAISCORE critic 已带来最大瞬时增益（72.33→75.11，+2.78）；匹配 target actor 的 critic（Qwen3-8B）增益 3.63，显著高于面向更小 actor（Llama-3.2-3B、Qwen3-4B）训练的 critic。
- **人类评估**：50 个 WritingBench prompt 盲评，TAISCORE 及 co-evolution 均获显著多数偏好，自动指标与人工判断一致。Cross-model judge 验证显示 gpt-oss-120B 与 Claude Opus 4.8 的 top-1 重合率达 89.5%，pairwise 一致率 77.9%。

## 相关工作脉络
- **Self-Refine / Reflexion / Critique-guided refinement**（Madaan et al., 2023; Shinn et al., 2023; Wadhwa et al., 2024 等）：仅评估 critique 孤立合理性或最终 revision
