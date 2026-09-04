---
title: "A-Model-with-No-Head-and-Many-Thoughts"
source: https://arxiv.org/pdf/2608.31069v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:47:36"
field: "大语言模型推理与强化学习"
keywords: ["soft thinking", "chain-of-thought", "reinforcement learning", "latent representation", "language model reasoning", "Gumbel-Softmax", "pass@k"]
innovations: ["用轻量级latent projector替换完整LM head进行连续推理，降低每步计算量约5倍", "在压缩K维latent空间中应用Gumbel-Softmax采样实现policy gradient可训练", "证明压缩latent空间虽降低单样本精度但提升多rollout多样性，显著改善pass@32性能"]
benchmarks: ["AIME2024", "AIME2025", "AMC23", "MATH-500", "GSM8K", "GPQA Diamond", "HumanEval"]
---

# 论文速读：A Model with No Head and Many Thoughts

## 一句话总结
论文提出 Soft Latent Thinking (SLT) 方法，在推理阶段用轻量级 latent projector 替换完整的词汇表 LM head，使语言模型能在连续 embedding 空间中进行自回归 rollout，将推理步骤从离散 token 生成转为连续状态转移，同时降低每步计算成本并提升 pass@k 性能。

## 研究问题与动机
- 现有 soft thinking 方法虽然将推理步骤置于连续 embedding 空间，但每一步仍需通过完整的 V 维词汇表 head 计算概率分布，计算成本高且成为长推理链的瓶颈。
- 现有软思考方法将潜在推理状态绑定在离散 token 语义上（状态必须是 token embedding 的加权和），限制了推理空间的表达能力。
- 连续推理与 GRPO/RL 结合时缺乏有效的探索机制，导致策略梯度难以有效更新。

## 核心贡献（创新点）
- **提出 SLT 框架**：在推理阶段用轻量级 encoder-decoder projector 替代完整 LM head，将推理维度从 V≈150k 压缩至 K≈12k–24k，降低每步计算量约 5 倍。与 SofT-GRPO 的本质区别在于解耦了推理状态生成与离散 token 语义空间。
- **设计可训练的连续推理机制**：基于 Gumbel-Softmax 重参数化，为 latent 空间中的推理动作定义可微分的似然函数，使 policy gradient 可在连续推理轨迹上直接应用。与 SofT-GRPO 的区别在于无需经过完整词汇表计算 soft token。
- **证明多样性-精度权衡的积极效应**：单个推理样本精度略低于全词汇表 soft thinking，但多 rollout 之间的多样性提升显著改善高 k 下的 pass@k 表现（如 pass@32 提升明显）。
- **提出模块化部署策略**：推理时可根据任务类型动态启用/禁用 projector，在非目标域（如代码）上禁用 projector 后性能仍可匹敌最强 baseline。

## 方法详解
- **项目器架构**：编码器 $W_{enc} \in \mathbb{R}^{K \times d}$ 将隐藏状态 $h_t$ 压缩为 K 维 logits $z_t = W_{enc} h_t$；解码器 $W_{dec} \in \mathbb{R}^{K \times d}$ 将 mixture weights 映射回 embedding 空间 $s_t = W_{dec}^\top y_t$。
- **采样机制**：对 K 个 latent 类别应用 Gumbel-Softmax 采样（温度 $\tau_g$），得到 mixture weights $y_t \in \mathbb{R}^K$，再经解码器合成 soft embedding $s_t$。
- **初始化策略**：从目标域中最频繁的 K 个 token（如数学推理数据中 top 12k）的 LM head 行和 embedding table 行初始化 $W_{enc}$ 和 $W_{dec}$，训练过程中独立演化。
- **训练目标**：沿用 SofT-GRPO 的 policy gradient 框架，但对 latent 推理步使用 Gumbel 变量的似然估计（替换原公式中的 V 为 K），对最终答案 token 使用标准 categorical log-probability ratio。
- **推理终止条件**：当 soft embedding 与 `<|endthink|>` 或 `\boxed` token embedding 的余弦相似度超过阈值时停止推理，切换回标准自回归解码生成最终答案。
- **训练配置**：冻结 backbone，仅训练 LoRA（rank 64）+ projector，使用 DeepScaleR 数据集和 outcome-based reward。

## 实验与结果
- **数据集**：五个数学推理基准——AIME2024、AIME2025、AMC23、MATH-500、GSM8K；外加 GPQA Diamond（科学）和 HumanEval（代码）做域外评估。
- **模型**：DeepSeek-R1-Distill-Qwen-1.5B 和 LLaMA-3.2-3B-Instruct。
- **最强结果**：在 DeepSeek-R1-Distill-Qwen-1.5B 上平均 pass@32 达 **86.22**，超越 SofT-GRPO（85.18）和 base + GRPO（83.23）；在 LLaMA-3.2-3B 上平均 pass@32 达 **60.70**，超越 SofT-GRPO（57.28）和 base + GRPO（55.32）。
- **Token 效率**：SLT 生成的推理 token 总数少于 SofT-GRPO（如 DeepSeek 模型上平均 6073 vs 6517），且每步 FLOPs 降低约 5×。
- **域外表现**：GPQA 上性能相当；HumanEval 上启用 projector 时下降（因数学初始化），但禁用 projector 后配合 LoRA 可达到与 SofT-GRPO 相当的 92.7%（vs 94.5%）。
- **更大模型验证**：Qwen3.5-9B 初步实验中 pass@32 达 93.3%，与 base 持平但 token 数大幅减少（9288 vs 23292）。

## 相关工作脉络
- **CoT / Soft Thinking（Zhang et al., 2025）**：基础软思考方法，通过词汇表 softmax 生成 soft token 作为 weighted mixture，计算成本高且易陷入 greedy 陷阱。本文在此基础上解耦推理与 token 空间。
- **Stochastic Soft Thinking（Wu et al., 2025）**：引入 Gumbel-Softmax 增加多样性，但仍依赖完整词汇表投影。本文将其推广到压缩 latent 空间。
- **SofT-GRPO（Zheng et al., 2025）**：将 GRPO 适配到 soft thinking，用 Gumbel reparameterization 定义 policy gradient。本文保留其 RL 框架但替换投影模块，进一步降低计算量。
- **Coconut（Hao et al., 2024）**：使用 recurrent hidden-state loop 进行连续推理。本文方法不同在于显式压缩投影到 K 维 latent basis。
- **Diffusion-of-Thought（Ye et al., 2024）**：通过连续去噪进行推理。本文方法是 autoregressive 风格而非 diffusion 风格。

## 局限性与未来方向
- **域泛化受限**：projector 初始化依赖目标域高频 token（如数学），在代码等非目标域上直接启用会导致性能下降，需额外设计 domain-agnostic 初始化或多域 projector。
- **低 k 场景优势不明显**：pass@1 略低于 full vocabulary soft thinking，说明压缩 latent 空间牺牲了单样本精度。
- **超参敏感性**：训练温度 $\tau_g$ 因模型而异（LLaMA 用 0.1，DeepSeek 用 0.5），需针对不同 backbone 调优。
- **未来方向**：探索 PCA/SVD 全局初始化（初步实验效果不佳）、多域共享 projector、更优的 domain-agnostic 初始化策略。

## 研究启发与可借鉴点
- **推理-输出解耦设计**：将推理过程的 embedding 空间与最终 token 空间分离，通过独立 projector 实现，避免推理状态被离散 token 语义束缚，可迁移至其他需要连续推理的场景。
- **压缩 latent basis + Gumbel-Softmax 组合**：用小规模 latent basis 替代全词汇表投影，同时保留随机探索能力，是一种兼顾效率与多样性的实用技巧。
- **模块化部署策略**：推理时按需启用/禁用 projector 的设计思路，允许模型在不同任务间灵活切换，减少对泛化能力的损害。
- **训练与推理温度分离**：训练时用较低温度（0.1）保证稳定性，推理时用较高温度（0.5）鼓励探索，这种设定值得参考。
- **轻量化适配方案**：仅训练 LoRA + projector 而不更新 backbone，在保持域外能力的同时实现推理加速，适合资源受限场景。

## 关键术语表
- **Soft Latent Thinking (SLT)**：在推理阶段用轻量级 latent projector 替换完整词汇表 LM head 的方法，使推理在连续 embedding 空间中进行。
- **SofT-GRPO**：将 GRPO 强化学习算法适配到 soft thinking 的方法，通过 Gumbel reparameterization 实现 policy gradient 更新。
- **Gumbel-Softmax**：用于分类变量重参数化的技术，使离散采样过程可微分，常用于引入可控随机性。
- **pass@k**：在 k 次独立采样中至少有一次正确的概率，衡量模型在多次尝试下的覆盖率。
- **Latent Projector**：由 encoder（$W_{enc}$）和 decoder（$W_{dec}$）组成的模块，将 hidden state 压缩并映射回 embedding 空间。
- **Chain-of-Thought (CoT)**：通过生成中间推理步骤来提升模型复杂问题求解能力的方法。
- **Soft Token**：由词汇表概率分布加权混合 token embedding 得到的连续向量，作为推理状态传递。
- **Outcome-based Reward**：仅根据最终答案正确性给出的奖励信号，不依赖中间步骤反馈。

## 可复现要素
- **数据集**：DeepScaleR（训练），AIME2024/2025、AMC23、MATH-500、GSM8K（评估）；数学子集来自 OpenThoughts-114k。论文未明确说明数据开源状态。
- **代码/权重**：论文未提及开源。
- **关键超参**：$K = 8d$（Qwen: 12288, LLaMA: 24576）；LoRA rank=64；训练温度 $\tau_g=0.1$（LLaMA）/ $0.5$（DeepSeek）；推理温度 $\tau_g=0.5$。
