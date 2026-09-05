---
title: "HSRM-Hidden-State-Reward-Models-for-Test-Time-Verification"
source: https://arxiv.org/pdf/2608.30841v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:05:25"
field: "大语言模型推理验证"
keywords: ["test-time verification", "hidden-state reward model", "best-of-N reasoning", "mathematical reasoning", "implicit correctness signal"]
innovations: ["提出HSRM，首次利用生成器步骤边界隐状态进行轻量级候选验证", "设计tie-safe ranking loss适配多正确解的Best-of-N场景", "2M参数隐状态验证器在15/16设置中匹配或超越55M文本验证器"]
benchmarks: ["GSM8K", "MATH-500", "AIME", "OlympiadBench"]
---

# 论文速读：HSRM-Hidden-State-Reward-Models-for-Test-Time-Verification

## 一句话总结
论文提出 HSRM（Hidden-State Reward Model），一种轻量级隐状态奖励模型，通过直接读取冻结生成器在推理步骤边界处的内部表示来对候选解进行排序验证，仅需约 2M 参数即可在数学推理任务上匹配或超越 55M 参数的纯文本验证器。

## 研究问题与动机
- 现有测试时推理流程依赖文本验证器重新读取每个生成解，使验证成为推理成本的主要组成部分。
- 已有研究表明 LLM 内部表示中编码了正确性相关信号，但尚未被有效利用于验证任务。
- 验证质量高度依赖验证器的能力与成本，当前大参数 PRM 成本过高，小参数文本验证器性能不足。
- 生成过程中已计算丰富内部表示，若可直接读取而非重新编码，可显著降低验证开销。

## 核心贡献（创新点）
- 提出 HSRM，首次利用生成器冻结状态在推理步骤边界提取隐状态作为验证输入，避免文本重编码。
- 设计 tie-safe ranking loss，仅要求正确候选得分高于错误候选，不强制正确候选间的全序关系。
- 在 4 个数学推理基准上，2M 参数的 HSRM 在 15/16 设置中匹配或超过 55M 文本验证器 EORM。
- 揭示正确性信号分布于生成器上层多个层中，顶层 4 层拼接可进一步提升排序性能。
- 验证跨家族泛化（Qwen/Llama）与跨数据集 zero-shot 迁移能力，证明隐状态验证信号的普适性。

## 方法详解
- **步骤边界隐状态提取**：使用步终止分隔符（如 `\n\n`, `\n`, `:\n` 等）将生成解分段，提取每步末尾 token 的最后一层隐状态 $h_{t_s}^L \in \mathbb{R}^{d_{\text{gen}}}$，形成序列 $H^L(y) \in \mathbb{R}^{S \times d_{\text{gen}}}$。
- **HSRM 架构**：输入投影 $z_s^{(0)} = W_{\text{in}} h_{t_s}^\ell + b_{\text{in}}$ → 2 层 Transformer 编码器（$d_{\text{model}}=256$，4 heads）→ 均值池化 + LayerNorm → 线性读出头输出标量分 $f_\phi(x,y) = w^\top z + b$。
- **训练目标**：tie-safe Bradley-Terry ranking loss $\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{P}||\mathcal{N}|} \sum_{i \in \mathcal{P}} \sum_{j \in \mathcal{N}} \log(1 + e^{-(s_i - s_j)})$，忽略全对或全错问题。
- **训练流程**：一次性运行生成器采样 N=64 候选并缓存隐状态，后续验证器训练无需额外生成器调用。
- **推理**：对 N=8 候选批量通过 HSRM 评分，取最高分候选返回（Best-of-N）。

## 实验与结果
- **生成器**：Qwen3 1.7B/4B/8B/14B（non-thinking），另测 Llama-3.2-1B/3B、Llama-3.1-8B。
- **基准**：GSM8K、MATH-500、AIME、OlympiadBench，每问题训练集采样 64 候选，评测时 Best-of-8。
- **主要结果**（GSM8K，Qwen3-14B）：HSRM 85.8% vs EORM 55M 82.2%，接近 Oracle 95.8%；在 15/16 设置中优于或匹配 EORM。
- **跨家族**：Llama 系列 6 组设置中 HSRM 均优于匹配 EORM（GSM8K 提升 3.1–8.1pp，MATH-500 提升 3.6–4.5pp）。
- **效率**：HSRM 验证 FLOPs 比 7B PRM 低约 5 个数量级，参数量仅 2–3.4M。
- **Ablation**：Top-4 层拼接 + step boundary 输入达最优 86.86% Acc / 0.724 AUROC；thinking 模式下 answer-only 提取优于 full-trace（AUROC 0.736→0.884）。
- **零样本迁移**：在 GSM8K/MATH-500 上训练后直接评测 OlympiadBench，HSRM 持续优于 EORM 且接近 7B PRM。

## 相关工作脉络
- **EORM (Jiang et al., 2025)**：55M 参数文本能量验证器，基于 Bradley-Terry 排序损失；HSRM 用隐状态替代文本重编码，参数量降低 27 倍且性能相当或更优。
- **Qwen2.5-Math-PRM-7B (Zhang et al., 2025)**：7B 领域专用 PRM，作为强外部基线；HSRM 以 2M 参数接近其 GSM8K 性能，验证成本极低。
- **SWIFT/ELHSR (Guo et al., 2025)**：对 token 级隐状态施加轻量奖励头；HSRM 在步骤边界聚合而非 token 级，避免冗余计算。
- **ReProbe (Ni et al., 2026)**：在 token 级隐状态上训练 Transformer probe 进行步骤级验证；HSRM 聚焦步骤边界表示，更简洁高效。
- **正确答案信号研究** (Kadavath et al., 2022; Azaria & Mitchell, 2023; Burns et al., 2024)：证明隐状态含正确性信号；本文将其直接用于学习式验证而非仅探测。
- **过程奖励模型 (PRM)** (Uesato et al., 2022; Lightman et al., 2023; Wang et al., 2024)：监督中间推理步骤；HSRM 仅用最终答案标签训练，无需过程标注。

## 局限性与未来方向
- 仅评估数学推理任务，未验证于其他领域（如代码生成、科学推理）。
- Thinking-mode 实验仅在 MATH-500 上对两种规模模型进行，需更广泛评估显式 deliberation 下的隐状态利用。
- 训练数据规模可扩展性有限，大 K 值带来更高标注成本。
- 未探索隐状态提取层数、步边界定义、 pooling 策略的更多变体。
- 跨域 zero-shot 迁移在 OlympiadBench 上仍与 Oracle 有较大差距，反映生成质量瓶颈而非验证瓶颈。

## 研究启发与可借鉴点
- **隐状态验证范式**：可直接迁移至代码生成、多步规划等需测试时验证的场景，复用生成器内部表示避免额外推理。
- **步骤边界提取策略**：利用语义分隔符（如 `\n\n`、特定 token）聚合序列表示，适用于任意链式推理输出结构。
- **Tie-safe ranking loss**：适用于存在多个正确解的 best-of-N 场景，避免过度排序压力，可复用于其他候选选择任务。
- **训练数据一次缓存**：生成器仅运行一次，后续验证器训练完全离线，大幅降低实际部署成本。
- **跨家族泛化验证**：在 Qwen 上训练后在 Llama 上测试，证明隐状态信号不依赖特定架构，增强方法可信度。

## 关键术语表
**HSRM**：Hidden-State Reward Model，一种利用生成器内部隐状态而非生成文本进行候选验证的轻量级奖励模型。
**Best-of-N (BoN)**：采样 N 个候选解并由验证器选出最优一个的测试时推理策略。
**Tie-safe Ranking Loss**：仅要求正确候选得分高于错误候选的成对排序损失，不对正确候选间强制全序。
**Step-Boundary Hidden State**：在推理步骤分隔符处提取的生成器隐状态，用于压缩序列表示并捕捉步骤级语义。
**EORM**：Energy-based Outcome Reward Model，55M 参数文本能量验证器，基于 Bradley-Terry 损失训练。
**PRM (Process Reward Model)**：过程奖励模型，对中间推理步骤给予逐步监督信号的大型验证模型。
**Within-Problem AUROC**：在同一问题内候选池中对正确/错误候选的排序区分度指标。
**Thinking Mode**：Qwen3 的显式内部推理模式，先生成 deliberation 轨迹再输出最终答案。

## 可复现要素
- **数据集**：GSM8K、MATH-500、AIME、OlympiadBench（公开）
- **代码/权重**：论文未提及开源；训练与评测细节完整
- **关键超参**：AdamW，lr=1e-4，batch_size=8（problem-level），steps=1000，dropout=0.1，5 seeds
- **硬件**：L40S、H100 GPU
- **Generator 配置**：Qwen3 1.7B–14B，float16，temperature=0.7，top-p=0.9，non-thinking 模式
