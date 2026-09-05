---
title: "TTPO-TEST-TIME-POLICY-OPTIMIZATION"
source: https://arxiv.org/pdf/2608.27448v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 01:48:55"
field: "LLM 推理能力优化"
keywords: ["test-time training", "on-policy self-distillation", "reinforcement learning", "large language models", "mathematical reasoning", "pseudo-label", "token-level selection"]
innovations: ["非对称目标：正样本蒸馏+负样本RL惩罚，容忍高频伪标签错误", "Token级双分支选择：蒸馏分支降权已收敛位置，RL分支掩码局部正确token", "无标签TTT性能超越有标签OPSD基线，并实现自演化循环"]
benchmarks: ["AIME 2025", "AIME 2026", "HMMT 2025", "HMMT 2026", "BRUMO 2025"]
---

# 论文速读：TTPO-TEST-TIME-POLICY-OPTIMIZATION

## 一句话总结
TTPO 提出了一种**非对称目标函数**，在无标签测试时训练（TTT）场景中，对与多数投票伪标签**一致**的 rollout 施加 OPSD 蒸馏损失，对**不一致**的 rollout 施加 GRPO 惩罚损失；结合 token 级加权与掩码，使模型在完全无监督条件下匹配甚至超越带标签的 OPSD。

## 研究问题与动机
- 现有 RLVR 和 OPSD 等后训练方法依赖 ground-truth 标签，无法应用于 TTT（模型在推理时于测试样本上自训练，无标签可获）。
- 用多数投票伪标签替代 ground-truth 是自然思路，但竞赛级问题中伪标签约 **~85%** 是错误的，直接蒸馏会随教师分布将错误传播至每个 token。
- **非对称性观察**：即使伪标签错误，约 **~79%** 的不一致（negative）rollout 本身也是错误的；因此"惩罚不一致"这一信号几乎不依赖伪标签内容，而"蒸馏向伪标签"则高度脆弱。
- TTRL 等纯 RL 方法仅提供序列级二值奖励，缺乏 token 级密集监督；而纯 OPSD 对所有 rollout 蒸馏会放大伪标签错误的影响。

## 核心贡献（创新点）
1. **非对称目标设计**：OPSD 仅作用于 positive（一致）rollout，GRPO 仅作用于 negative（不一致）rollout，本质区别于全量蒸馏或纯 RL 的单一信号处理方式，前者利用错误伪标签下的结构性鲁棒性，后者利用负样本无需伪标签答案即可判定。
2. **Token 级加权（蒸馏分支）**：基于学生熵与教师-学生 KL 发散度的 Soft-OR 组合，对已收敛位置降权，避免梯度稀释——区别于 TIP 等仅用单一信号的 token 选择策略。
3. **Token 级掩码（RL 分支）**：针对"负样本中局部正确 token 被误罚"问题，以低概率低熵为锚点掩码掉混淆位置的梯度，区别于 STAPO 仅在正样本上做掩码的设计。
4. **无标签场景下超越有标签基线**：在 OpenThoughts 上 TTPO（仅用伪标签）平均精度超过有标签 OPSD（40.1 vs 39.7, 1.7B），打破"需要 ground-truth 才能有效训练"的假设。
5. **自演化循环**：随着模型能力提升，多数投票质量改善，训练信号质量随之提升，形成良性正反馈；这是 ground-truth 路由无法实现的特性。

## 方法详解
**整体框架**：对每个问题 $x$，采样 $K=64$ 条 trajectory，提取答案后进行多数投票得到伪标签 $\hat{a}$，将 rollout 划分为正样本集 $\mathcal{P}$（答案一致）和负样本集 $\mathcal{N}$（答案不一致）。

**Teacher-Student 构造**：沿用 OPSD 范式，teacher 以伪标签 $\hat{a}$ 作为特权信息，student 无特权信息：
- $q_t^{(\hat{a})} = \pi_\theta(\cdot \mid [x;\hat{a}]_{\text{teacher}}, y_{<t})$（无梯度）
- $p_t = \pi_\theta(\cdot \mid x_{\text{student}}, y_{<t})$（有梯度）

**蒸馏分支（$\mathcal{P}$）损失**：
$$\mathcal{L}_{\text{OPSD}}(k) = \frac{1}{T_k}\sum_{t=1}^{T_k} w(t) \cdot \text{KL}(q_t^{(\hat{a})} \| p_t)$$
- **Token 加权**：$w(t) = \hat{H}(t) + \hat{\Delta}(t) - \hat{H}(t)\cdot\hat{\Delta}(t)$，其中 $\hat{H}(t)$ 为归一化学生熵，$\hat{\Delta}(t)$ 为归一化 KL 发散度；当两者均低（学生已收敛且与教师一致）时权重趋零。

**RL 分支（$\mathcal{N}$）损失**：
$$\mathcal{L}_{\text{GRPO}}(k) = -\frac{A_k}{T_k}\sum_{t=1}^{T_k} m(t) \cdot \log\pi_\theta(y_k^{(t)} \mid x, y_k^{(<t)})$$
- 优势 $A_k$ 基于 group-relative 计算，负样本 $A_k < 0$ 构成惩罚。
- **Token 掩码**：$s(t) = -\log p_t(y_k^{(t)}) \cdot (1-\hat{H}(t))$，取 score 前 50% 为 $m(t)=1$；用未归一化的 $-\log p$ 确保局部正确的高概率 token 自然被排除，只惩罚"低概率且高置信"的异常 token。

**总损失**：$\mathcal{L}_{\text{TTPO}} = \frac{1}{|\mathcal{B}|}\left(\sum_{k\in\mathcal{P}}\mathcal{L}_{\text{OPSD}}(k) + \lambda\sum_{k\in\mathcal{N}}\mathcal{L}_{\text{GRPO}}(k)\right)$，$\lambda=0.1$。

## 实验与结果
- **数据集**：AIME 2025/2026、HMMT 2025/2026、BRUMO 2025（五竞赛级数学基准）；训练数据为 OpenThoughts。
- **模型**：Qwen3-1.7B / 4B / 8B，LoRA（r=64, α=128）。
- **基线**：OPSD†（有标签）、GRPO†（有标签）、TTRL（无标签）、OPSD-TTT（无标签）。
- **OpenThoughts 训练（Table 1）**：TTPO 超越有标签 OPSD，1.7B（40.1 vs 39.7）、4B（58.6 vs 58.4）、8B（62.6 vs 61.7）。
- **纯 TTT（Table 2）**：1.7B 从 38.0% 提升至 **45.2%**（+7.2），超越 OPSD-TTT（41.9）和 TTRL（40.2）；4B 达到 61.1%，超越 8B 基线（60.7%）。
- **无思考模式评估（Appendix D.2）**：TTPO 相比基线提升 **+25.2%~+36.4%**，约为 OPSD 提升的数倍。
- **跨任务泛化（Figure 4）**：在任一基准上训练均能提升其他两个基准的表现，证明获得了可迁移的推理能力而非过拟合。
- **最强结果**：Qwen3-8B TTT Avg@12 达 **65.3%**；Qwen3-4B TTT（61.1%）已超越 Qwen3-8B 基线（60.7%）。

## 相关工作脉络
- **TTRL**（Zuo et al., 2026）：首次将 TTT 引入 LLM 推理，用多数投票生成伪奖励+GRPO。TTPO 在 TTRL 基础上引入 token 级密集蒸馏信号，且通过非对称设计容忍伪标签噪声。
- **OPSD**（Zhao et al., 2026）：利用同模型以特权信息（如 ground-truth）构建教师进行 on-policy 蒸馏。TTPO 将其扩展至无标签场景，用伪标签替代 ground-truth，并通过非对称目标解决伪标签错误问题。
- **TIP**（Xu et al., 2026c）：基于学生熵和师生发散度选择重要 token 进行蒸馏。TTPO 借鉴此思路但设计了双信号 Soft-OR 组合，并首次将其同时应用于蒸馏和 RL 两个分支。
- **STAPO**（Liu et al., 2026a）：在 RL 中对正样本的低概率低熵 token 做掩码。TTPO 将此思想延伸至负样本，且结合 group-relative 优势而非绝对 reward。
- **NLNL**（Kim et al., 2019）：负学习用于含噪标签场景。TTPO 的非对称设计本质上延续了"陈述样本不是什么比陈述是什么更可靠"的思想。
- **Hi-TTRL / SCRL**（Xu et al., 2026b; Yan et al., 2026）：改进 TTRL 对共识质量的敏感性。TTPO 不依赖共识质量过滤，而是通过非对称目标天然兼容低共识场景。

## 局限性与未来方向
- **依赖多数投票质量**：当采样数 $K$ 过小或问题极难导致无正确 rollout 时，投票信号退化，两分支均受噪声影响；自适应调整正负比例或低共识时回退到纯 RL 是可探索方向。
- **领域受限**：实验仅限可验证最终答案的数学推理；扩展到代码生成（需执行验证）或开放推理（需学习式 reward model）尚待研究。
- **固定非对称目标**：训练全程使用固定 $\lambda$ 和固定正负比例；随着模型能力提升伪标签准确率上升，最优 balance 可能变化，动态 curriculum 值得探索。

## 研究启发与可借鉴点
1. **非对称信号分配的通用范式**：在 noisy label/伪标签场景下，"正样本用密集监督、负样本用鲁棒惩罚"的非对称设计可迁移至其他需要 self-supervision 的领域（如代码生成、agent 任务）。
2. **Token 级双信号加权（熵+KL 发散度 Soft-OR）**：蒸馏分支的 token 选择策略可有效复用，适用于任何 on-policy 蒸馏场景以聚焦"真正需要学习"的位置。
3. **"最短路径优先"的 rollout 选择策略**：$K_{\text{train}}$ 选取最短 completion 效果最佳（因只更新前 1024 token），这一直觉可在任意 rollout-based 训练方法中借鉴。
4. **自演化循环的工程价值**：TTPO 展示了无标签 TTT 可在训练过程中持续改善监督信号质量，对部署时的在线适应（online adaptation）有直接启发。
5. **虚假惩罚问题的显式建模**：负样本 RL 中"局部正确 token 被误罚"的问题通过 token masking 解决，这一诊断-对策对任何 critic-free RLVR 方法均有参考价值。

## 关键术语表
- **Test-Time Policy Optimization (TTPO)**：一种无标签测试时训练方法，通过非对称目标将对齐/不对齐 rollout 分别进行蒸馏和 RL 惩罚。
- **On-Policy Self-Distillation (OPSD)**：以同一模型在特权信息（如 ground-truth）条件下的输出为教师，对模型自身 rollout 做 forward KL 蒸馏。
- **Group Relative Policy Optimization (GRPO)**：在组内相对计算 advantage 的 RL 方法，无需 critic model，仅依赖轨迹级奖励信号。
- **Majority-Vote Pseudo-label**：对多条 rollout 的最终答案聚类后取最大簇作为伪标签，用于无监督场景下的信号生成。
- **Thinking Mode / Non-Thinking Mode**：Qwen3 等模型的两种推理模式，thinking mode 启用链式推理，non-thinking 直接使用最终答案输出。
- **Privileged Information**：训练时可提供给教师模型但不提供给学生的额外上下文（如答案、完整轨迹），用于构造强教师分布。
- **Forward KL Distillation**：最小化教师分布到学生分布的 KL 散度 $\text{KL}(q\|p)$，鼓励学生在每个 token 上逼近教师。
- **Self-Evolving Cycle**：TTPO 中模型能力提升→多数投票质量改善→训练信号质量提升的良性正反馈机制。

## 可复现要素
- **数据集**：AIME 2025/2026、HMMT 2025/2026、BRUMO 2025、OpenThoughts（论文未明确声明是否全部公开，建议查 arxiv 源）。
- **代码**：已开源，https://github.com/ZJU-REAL/TTPO。
- **权重**：论文未提及发布预训练/微调权重。
- **关键超参**：LoRA r=64, α=128；学习率 5e-6；K=64 rollouts/problem，K_train=8；λ=0.1；Max new tokens=16,000（采样）/1,024（梯度更新）；Temperature=1.1；Top-p=0.95；训练步数 100 步；Batch size=32。
