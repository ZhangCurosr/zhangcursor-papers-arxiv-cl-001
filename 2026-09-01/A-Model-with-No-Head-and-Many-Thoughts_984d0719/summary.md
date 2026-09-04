---
title: "A-Model-with-No-Head-and-Many-Thoughts"
source: https://arxiv.org/pdf/2608.31069v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:47:40"
field: "大语言模型推理与强化学习"
keywords: ["Soft Latent Thinking", "软思维", "GRPO", "Gumbel-Softmax", "链式推理", "低秩适应"]
innovations: ["用K维潜投影器替代V维LM头进行推理，降低每步计算5倍并提升pass@k多样性", "提出基于高频token的投影器初始化策略并证明专用子空间优于全局SVD初始化", "设计与SofT-GRPO兼容的可训练框架，仅更新投影器+LoRA即可实现连续空间推理优化"]
benchmarks: ["AIME2024", "AIME2025", "AMC23", "MATH-500", "GSM8K", "GPQA Diamond", "HumanEval"]
---

# 论文速读：A Model with No Head and Many Thoughts

## 一句话总结
本文提出 Soft Latent Thinking (SLT)，在推理阶段用轻量级潜投影器（encoder-decoder，潜基大小 K≈12k–24k）替代完整的 V≈150k 词汇表 LM 头，使模型能在连续嵌入空间进行自回归推理；在 DeepSeek-R1-Distill-Qwen-1.5B 和 LLaMA-3.2-3B-Instruct 上，SLT 均取得最高的平均 pass@32，并显著降低每步计算成本与推理 token 数。

## 研究问题与动机
1. **计算瓶颈**：现有软思维方法每步仍需通过完整词汇表头形成 V 维分布（V≈150k），成为长推理链的算力瓶颈。
2. **语义受限**：软 token 被强制约束为 token 嵌入的线性组合，导致潜推理状态无法突破离散 token 语义边界。
3. **探索多样性不足**：SofT-GRPO 等基于 Gumbel-Softmax 的方法虽引入随机性，但每一步仍依赖完整词汇分布，样本多样性提升有限。
4. **训练效率**：希望仅用轻量级 LoRA + 投影器适应即可训练，避免全参重训练。

## 核心贡献（创新点）
1. **提出 SLT 投影器架构**：用 K≪V 的潜基替代 LM 头进行推理，推理时每步 FLOPs 从 2dV 降至 4dK（约 5× 加速）。与 SofT-GRPO 的本质区别在于：SLT 彻底解耦了推理嵌入与词汇表投影，推理状态不再受 token 语义张成空间限制。
2. **可训练的潜投影器初始化策略**：从数学领域高频 token 的 LM 头和嵌入表中拷贝 K 行作为 W_enc 和 W_dec 的初始值。与直接使用 PCA/SVD 等全局初始化不同，本文证明了专用领域子空间更适合推理轨迹。
3. **与 SofT-GRPO 兼容的 RL 训练框架**：在 rollout 时存储 Gumbel 变量 g_t，更新时只重算 p_t 下的 Gumbel 似然（Eq.3），保持梯度可信。相比全参数 GRPO，仅需更新投影器+LoRA，训练成本大幅降低。
4. **揭示压缩词汇的多样性–准确性权衡**：消融表明 K 增大（12k→24k）提升 pass@16/@32 但略降 pass@1，证明压缩投影器通过引入可控随机性提升多样本覆盖。
5. **可插拔的模块化部署**：推理时可启用投影器（域内数学任务）或禁用投影器回退到完整软思维（域外代码/科学任务），LoRA 权重保持不变。

## 方法详解
**投影器结构（Encoder-Decoder）：**
- Encoder：$z_t = W_{\text{enc}} h_t$，其中 $W_{\text{enc}} \in \mathbb{R}^{K \times d}$，将 d 维隐藏状态压缩为 K 维 logit。
- Sampling：对 $z_t$ 做 softmax 得 $p_{t,i}$，注入 Gumbel 噪声 $\epsilon_{t,i}$ 得 $g_{t,i} = \log p_{t,i} + \epsilon_{t,i}$，再经 Gumbel-Softmax（温度 $\tau_g$）得权重 $y_t \in \mathbb{R}^K$。
- Decoder：$s_t = W_{\text{dec}}^\top y_t$，$W_{\text{dec}} \in \mathbb{R}^{K \times d}$，输出 d 维软嵌入作为下一步输入。

**初始化（Section 4.1）：**
- 选取目标域（数学推理）中最频繁的 K 个 token 索引 $\mathcal{K}$。
- $W_{\text{enc}}[i,:] = W_{\text{head}}[k_i,:]$，$W_{\text{dec}}[i,:] = E[k_i,:]$，即从预训练 LM 头和嵌入表复制对应行。
- K=8d（Qwen: 12288，LLaMA: 24576）效果最优。

**训练（Section 4.2）：**
- 基于 SofT-GRPO 目标，冻结 backbone，仅训练 LoRA（rank=64，作用于所有 attention 和 MLP 模块）+ 投影器。
- Rollout 时存储 $(c_t, g_t, \epsilon_t, s_t)$；更新时固定 $g_t$，重算 $p_t^\theta$，计算 Gumbel 似然 $\log p_\theta(g_t|c_t)$（Eq.3，V→K）。
- 最终答案 token 使用标准 categorical log-prob ratio，与离散 GRPO 一致。

**推理停止条件（Section 4.3）：**
- 有 `<|im_end|>` token：$\cos(s_t, e_{</think>}) > \delta$。
- 无显式边界 token：$\cos(s_t, e_{\boxed{}}) > \gamma$。
- 阈值敏感度低：中间推理阶段余弦相似度通常 <0.2，答案生成前急剧上升。

## 实验与结果
**数据集与模型：**
- 数学推理：AIME2024、AIME2025、AMC23、MATH-500、GSM8K
- 域外：GPQA Diamond（科学）、HumanEval（代码）
- 模型：DeepSeek-R1-Distill-Qwen-1.5B、LLaMA-3.2-3B-Instruct、Qwen3.5-9B（初步验证）

**主要结果（Table 1）：**

| 模型 | 方法 | 平均 @1 | 平均 @16 | 平均 @32 |
|------|------|--------|---------|---------|
| DeepSeek-1.5B | No-Finetune + GRPO | 61.28 | 80.16 | 82.39 |
| DeepSeek-1.5B | + SofT-GRPO | 61.39 | 83.54 | 85.18 |
| **DeepSeek-1.5B** | **+ Ours (SLT)** | **57.32** | **82.66** | **86.22** |
| LLaMA-3B | + SofT-GRPO | 32.60 | 53.00 | 57.28 |
| **LLaMA-3B** | **+ Ours (SLT)** | **32.83** | **54.96** | **60.70** |

- SLT 在两个模型上均取得最高平均 pass@32：DeepSeek 1.5B 为 86.22（+1.04 vs SofT-GRPO），LLaMA 3B 为 60.70（+3.42 vs SofT-GRPO）。
- Qwen3.5-9B 初步验证（Table 3）：SLT 用 9288 avg tokens 达到 pass@32=93.3，与基座模型 23292 tokens 持平。

**Token 效率（Table 2）：** SLT 比 SofT-GRPO 少用约 7–15% 的推理 token（DeepSeek 模型平均 #Token: 6073 vs 6517）。

**计算效率（Section 5.5）：** 每步 FLOPs 降低约 5×（2dV → 4dK）；原型 serving 吞吐量提升 1.03×–1.26×（表4）。

## 相关工作脉络
1. **Chain-of-Thought (CoT)** (Wei et al., 2022; Kojima et al., 2022)：通过逐步中间推理改善复杂任务性能——本文继承此思想但将中间步骤从离散 token 改为连续嵌入。
2. **Soft Thinking** (Zhang et al., 2025)：首次将中间推理置于连续嵌入空间，通过加权 token 嵌入混合实现平滑状态转移——本文在此基础上用投影器替代完整词汇投影，解耦推理与输出语义。
3. **Stochastic Soft Thinking** (Wu et al., 2025)：发现 vanilla soft thinking 易陷入 greedy pitfall，引入 Gumbel-Softmax 增加多样性——本文沿用 Gumbel 重参数化但将其作用域从 V 维降至 K 维。
4. **SofT-GRPO** (Zheng et al., 2025)：将 GRPO 适配到软思维 setting，通过 Gumbel 似然定义策略梯度——本文在此基础上进一步用投影器替换 LM 头，降低计算并提升多样性。
5. **Coconut** (Hao et al., 2024) 与 **Diffusion-of-Thought** (Ye et al., 2024)：分别在循环 hidden-state 和去噪框架下实现连续推理——本文定位差异：SLT 是"无头"投影方案，直接嵌入到现有 autoregressive 架构中，无需修改 backbone 结构。

## 局限性与未来方向
1. **域外迁移受限**：投影器以数学高频 token 初始化，在 HumanEval（代码）上启用投影器会导致性能下降（pass@32 从 94.5 降至 87.2），但禁用投影器后可恢复。
2. **全局初始化策略不佳**：PCA/SVD 初始化实验表现差，说明推理轨迹占据的是专门子空间而非高方差全局子空间，如何自动发现该子空间仍是开放问题。
3. **超参依赖基座模型**：训练温度 $\tau_g$ 在 DeepSeek(0.5) 和 LLaMA(0.1) 上最优不同，缺乏统一调参指南。
4. **未来方向**：多领域投影器、域无关初始化、更大模型规模系统性验证。

## 研究启发与可借鉴点
1. **投影器作为"推理适配器"**：将推理过程与词汇表投影解耦的思路可迁移到其他需要长推理链的场景（如程序合成、多步规划），仅需替换投影器即可适配新领域。
2. **Gumbel-Softmax 在压缩空间的多样性价值**：K 维压缩 + Gumbel 采样的组合在 pass@k 场景下表现突出，可作为多样本推理的标准组件。
3. **LoRA + 投影器联合训练的轻量化范式**：冻结 backbone 仅训练低秩适配 + 小投影器，兼顾域外能力保留与域内性能提升，适合工业部署。
4. **余弦相似度停止判据**：用 $\cos(s_t, e_{\text{boundary}})$ 判断推理终止，避免了额外分类头，设计简洁且实验验证有效。
5. **与团队方向结合机会**：若团队关注多智能体推理或 pass@k 优化的场景，SLT 的投影器可直接作为推理阶段的替代组件集成。

## 关键术语表
**Soft Latent Thinking (SLT)**：用轻量级 encoder-decoder 投影器替代 LM 头进行推理的方法，使中间状态在连续嵌入空间中演化而不经过完整词汇分布。

**Gumbel-Softmax**：对 categorical 分布的连续可微近似，通过注入 Gumbel 噪声实现可导的随机采样，常用于重参数化技巧。

**SofT-GRPO**：将 GRPO 强化学习算法适配到软思维 setting 的方法，通过 Gumbel 似然定义策略梯度，解决软 token 不可微的问题。

**pass@k**：在 k 次独立采样中至少有一次正确的概率，衡量多样本推理的覆盖能力。

**LoRA (Low-Rank Adaptation)**：通过低秩分解微调 LLM 参数的高效适配器方法，本项目中用于适配 backbone 处理软嵌入。

**潜基 (Latent Basis)**：大小为 K 的向量集合（由 $W_{\text{dec}}$ 的列构成），作为推理嵌入空间的基底，替代完整 V 维词汇嵌入。

## 可复现要素
- **数据集**：DeepScaleR（训练）；AIME2024/2025、AMC23、MATH-500、GSM8K、GPQA Diamond、HumanEval（评测）——数学训练数据来自 OpenThoughts-114k 数学子集
- **代码/权重**：论文未明确声明开源
- **关键超参**：K=8d（Qwen: 12288, LLaMA: 24576）；LoRA rank=64；训练温度 $\tau_g$（DeepSeek: 0.5, LLaMA: 0.1）；推理温度 $\tau_g=0.5$
