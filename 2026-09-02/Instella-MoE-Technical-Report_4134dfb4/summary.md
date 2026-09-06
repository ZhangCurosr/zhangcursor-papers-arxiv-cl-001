---
title: "Instella-MoE-Technical-Report"
source: https://arxiv.org/pdf/2609.00791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:19:55"
field: "开源 MoE 语言模型"
keywords: ["Mixture-of-Experts", "Fully Open LLM", "Gated MLA", "FarSkip-Collective", "RL Post-Training", "MOPD", "AMD GPU"]
innovations: ["Gated MLA：在 MLA 输出端引入 input-conditioned sigmoid gate 增强注意力表达能力", "FarSkip-Collective：通过部分 activation 替换实现 MoE 通信与计算完全重叠，提升 12.7% 训练吞吐", "IF-specialized RL + MOPD 两阶段后训练：先专项强化指令遵循再蒸馏回原模型防止灾难性遗忘"]
benchmarks: ["MMLU", "GSM8K", "HumanEval+", "IFEval", "HELMET", "RULER", "AIME24/25", "BBH", "GPQA", "AGIEval"]
---

# 论文速读：Instella-MoE-Technical-Report

## 一句话总结
本文发布了 **Instella-MoE-16B-A3B**，一个完全开源的 MoE 语言模型（16B 总参数、2.8B 激活参数），在 AMD Instinct MI300X/MI325X 上从零训练，结合 Gated MLA 和 FarSkip-Collective 两项架构/系统创新，在预训练基准上平均得分 76.7，在 post-training 基准上 Think checkpoint 平均得分 73.2，均为同参量级完全开源模型中的最强结果。

## 研究问题与动机
1. **闭源模型主导进展**：GPT、Claude、Gemini 等前沿模型权重与训练数据均不公开，阻碍科学复现与公平访问。
2. **开放权重模型缺乏完整透明度**：Qwen3、Llama 4、GLM-5.3 等虽开源权重，但训练数据配比、预处理管线和完整训练配方未公开，研究者无法审计数据污染或复现全流程。
3. **完全开源 MoE 模型稀缺**：目前仅有 OLMoE（6.9B 总参、1.3B 激活参）在小规模上验证了可行性，Marco-MoE 依赖 dense checkpoint upcycling 而非从零训练，缺乏从预训练到 RL 的完整开源流水线。
4. **MoE 通信开销大**：Expert-parallel all-to-all 通信导致 GPU 空闲，限制了 MoE 模型的训练和推理效率，需要系统级优化。

## 核心贡献（创新点）
1. **Instella-MoE-16B-A3B 完全开源 MoE 模型**：16B 总参数/2.8B 激活参数，从零训练并开源全部阶段 checkpoint、训练配置、数据配比和代码；与 OLMoE 等的本质区别在于从预训练到 RL 的全流程透明，且参量级更大。
2. **Gated MLA（门控多头潜在注意力）**：在 MLA 的 attention output 与 output projection 之间插入 input-conditioned sigmoid gate，增强模型表达能力；与 DeepSeek-V2/V3 MLA 的本质区别在于引入了可学习的输出通道门控机制，消融显示 MMLU 和代码生成提升显著。
3. **FarSkip-Collective 连接模式**：通过允许部分/过时的 activation（\(\hat{h}_k\)）作为后续层输入，将 Dispatch/Combine all-to-all 通信与 attention 和 shared-expert 计算重叠；与标准 MoE 连接的本质区别在于消除了通信依赖屏障，预训练吞吐量提升 12.7% 而不显著损失精度。
4. **多阶段训练流水线（含 MoE 适配的 DPO 与双阶段 RL）**：提出在 DPO 阶段禁用 load-balancing bias 更新以稳定 expert routing（图 6 显示 top-1 expert 差异从 54% 降至 5%）；提出 IF-specialized RL + Multi-Teacher On-Policy Distillation (MOPD) 两阶段策略，将指令遵循能力整合进模型而不破坏数学/编码能力；与单纯 SFT+DPO 的本质区别在于 RL 仅针对 IF 专项强化，再通过 MOPD 回灌以避免灾难性遗忘。

## 方法详解

### 架构设计
- **Decoder-only MoE Transformer**，27 层（1 个 dense FFN + 26 个 MoE 层），总参数 16B，每 token 激活 2.8B。
- MoE 层采用 DeepSeek-V3 风格的 fine-grained shared-plus-routed 设计：每 token dispatch 到 top-K=6 of N=64  routed experts（FFN intermediate size 1,408）+ 2 个 fused shared experts（intermediate size 2,816），router scaling factor=2.5。
- 注意力采用 **Gated MLA**：
  - KV 状态压缩为低秩 latent representation（KV latent rank=512，16 heads，value dim=128）。
  - 在 scaled dot-product attention 输出 \(\hat{o}_t \in \mathbb{R}^{Hd_v}\) 与 output projection \(\mathbf{W}_o\) 之间插入 gate：
    \[
    y_t = \mathbf{W}_o \left[ \operatorname{Sigmoid}(\mathbf{W}_g \boldsymbol{x}_t) \odot \hat{\boldsymbol{o}}_t \right]
    \]
    其中 \(\boldsymbol{x}_t\) 为 RMSNorm 归一化后的层输入。
- **FarSkip-Collective**：修改 transformer 层间连接，用部分 activation \(\hat{h}_k\) 替代完整 \(h_k\) 作为下一层输入，使 Dispatch 与 attention 共享 expert 计算可并行重叠。具体地：
  - \(\mathrm{moE-in}_k = \hat{h}_k(\mathrm{moE}) = h_{k-1}\)（Dispatch 可与 attention 重叠）
  - \(\mathrm{atm-in}_k = \hat{h}_k(\mathrm{attn}) = h_{k-2} + \mathrm{attn-out}_{k-1} + \mathrm{shared-exp-out}_{k-1}\)（routed expert Combine 可与 shared expert 计算和 attention 部分计算重叠）
- **Multi-Token Prediction (MTP)**：预训练和 mid-training 阶段附加一层 MTP head（权重 0.3/0.1），长上下文扩展及之后禁用。

### 负载均衡
- 主机制：bias-based loss-free balancing（per-expert bias 加入 router score 但不加入 mixture weights），bias 更新率 \(1 \times 10^{-3}\)。
- 辅机制：sequence-level auxiliary load-balancing loss，系数 \(1 \times 10^{-4}\)。
- **DPO 阶段**：冻结 bias 更新，auxiliary loss 系数设为 0，防止 expert routing drift。

### 多阶段训练流水线
1. **Pre-training**：7.1T tokens，seq len=4K，global BS=4096，AdamW（\(\beta_1=0.9, \beta_2=0.95\)），LR peak \(4\times10^{-4}\)，WSD schedule，MTP 激活，FarSkip 启用。
2. **Mid-training**：~100B tokens，训练三个 variant（v1/v2/v3 仅 STEM/reasoning 子集不同），等权 model souping 合并，峰值 LR \(2\times10^{-4}\)。
3. **Long-Context Extension**：两阶段将 context 从 4K 扩展至 64K：
   - Stage 1：~194B tokens 在 Dolma 3 Longmino 100B 上训练，YaRN RoPE scaling（\(\theta=8\times10^6\)），doc masking。
   - Stage 2：~20B tokens 在 37.32B STEM mixture 上训练，恢复 Stage 1 退化的数学/编码能力。
4. **SFT**：两阶段，Phase 1 用 Dolci-Think-SFT-7B + 数学/编程/科学 slice（seq len=32K，2 epochs）；Phase 2 使用 **feedback-driven data curation** 从 512K 候选池中精准选取数据。
5. **DPO**：使用 Dolci-Think-DPO-7B 对比偏好数据（delta learning 思想：weak vs strong trace 的质量差），\(\beta=5.0\)，禁用 load balancing。
6. **RL Stage 1（IF-specialized RL）**：GRPO + DAPO + R3，1,400 steps，verifiable reward（约束满足分数），partial rollouts，async on-policy。
7. **RL Stage 2（MOPD）**：domain-routed multi-teacher on-policy distillation，IF prompt → IF-RL teacher，其余 prompt → frozen DPO teacher（self-anchor），token-level reverse-KL 目标。

### RL 关键技术细节
- **Rollout Routing Replay (R3)**：记录 rollout 时的 expert 分配，在训练前向传播中 replay，确保 \(\log\pi^{\mathrm{train}}\) 与采样路径一致。
- **Truncated Importance Sampling (TIS)**：修正 residual numerical mismatch，IF-RL 截断范围 [0.5, 1.5]，MOPD 为 [0.5, 2.0]。
- GRPO 改进：zero-gradient filtering、active sampling、token-level loss、无 KL penalty、clip-higher（\(\varepsilon_{\mathrm{low}}=0.2, \varepsilon_{\mathrm{high}}=0.272\)）、无 std normalization。

## 实验与结果

### 基线模型评估（Base）
- **Instella-MoE-16B-A3B-Base** 在 13 个预训练基准上平均得分 **76.7**，为 fully open 模型最强：
  - 超越 OLMo-3-7B（70.1）、SmolLM3-3B（70.5）、OLMoE-1B-7B（61.9）。
  - 与 open-weight 模型 Moonlight-16B-A3B（76.2）相比更强，仅落后于 Qwen3.5-4B-Base（79.5，激活参数更多）。
  - WinoGrande 86.5（全场最高），HumanEval+ 65.7，GSM8K 81.5。

### 长上下文评估
- HELMET avg（8K–64K）：41.5；RULER avg：79.4。
- 与 OLMo-3-7B（RULER 80.2）相当，优于 SmolLM3-3B（78.6）。
- Stage 1 checkpoint 更长上下文更强（HELMET 43.7，RULER 83.9），Stage 2 以小幅长上下文退化为代价换取短上下文 STEM 恢复。

### Post-trained 评估（Think Checkpoint）
- 平均得分 **73.2**，为 fully open post-trained 模型最强：
  - 超越 OLMo-3-7B-Think（71.97）、Gemma-4-E4B-Think（70.47）、Qwen3.5-4B（69.73）。
  - IFEval 83.70，AIME25 73.40，GPQA 61.20，LCB 54.30。

### Ablation
- **架构消融**（200B tokens）：Gated MLA 使平均分从 49.86 提升至 50.33（+0.47），FarSkip 保持 50.38 同时带来 12.7% 训练吞吐提升。
- **RL 消融**：IF-RL 单独使 IFEval 从 77.1 升至 84.1 但 AIME24/25 和 GPQA 退化；MOPD 恢复大部分 IF 增益同时保持 DPO 水平的数学/编码/MMLU 表现，ablation 平均分 75.7 为最高。

### 效率
- 预训练吞吐：FarSkip-Collective 较标准 MoE 基线提升 **12.7%**。
- 推理 TTFT 吞吐：expert-parallel serving with SGLang 提升 **39.2%**。

## 相关工作脉络
1. **DeepSeek-V2/V3**：Instella-MoE 采用其 fine-grained shared-plus-routed MoE 设计（top-K routing、router scaling）和 MLA 注意力作为基础，但在此基础上增加 Gated MLA 和 FarSkip-Collective，且完全开源全部训练细节。
2. **OLMoE（Muennighof et al., 2025）**：首个完全开源 MoE 模型，但仅 6.9B 总参/1.3B 激活参；Instella-MoE 在参量级和性能上大幅超越，且提供完整训练流水线。
3. **OLMo-3 / SmolLM3**：完全开源 dense 模型；Instella-MoE 证明在同等活跃参数下 MoE 架构可达到甚至超越 dense 模型性能。
4. **Marco-MoE（Jiang et al., 2026）**：完全开源 MoE 但依赖 dense checkpoint upcycling 而非从零训练；Instella-MoE 是从零训练的全流程开源。
5. **Qwen3.5 / Gemma-4 / Moonlight-16B**：open-weight MoE/dense 基线；Instella-MoE 在相同或更小活跃参数下达到可比或更优性能，且完全开源。
6. **GRPO/DAPO/Dr.GRPO**：Instella-MoE 的 IF-RL 阶段集成了这些 RL 改进技术，并结合 R3 和 TIS 解决 MoE 特有的 train/inference routing 不一致问题。
7. **MOPD（Ma et al., 2026）**：本文在其基础上应用于 MoE 模型的 IF-specialized RL 后知识整合，利用 domain-routed teacher 实现 self-anchoring 防止灾难性遗忘。

## 局限性与未来方向
1. **长上下文与短上下文能力的权衡**：Stage 2 STEM recovery 以 HELMET/RULER 小幅下降为代价换取 GSM8K 和 HumanEval+ 的恢复，表明 64K 长上下文能力仍有提升空间。
2. **FarSkip-Collective 的 activation 近似**：使用部分/过时 activation 理论上可能累积误差，虽然 200B-token 消融显示影响极小，但在更长训练或更大模型上是否保持需进一步验证。
3. **仅支持英文/开源语料**：训练数据均为公开开源语料，多语言能力未在本文评估中体现。
4. **RL 阶段的计算成本**：IF-RL + MOPD 两阶段 RL 需要大量 rollout 计算资源（每步 512 responses × 16K tokens），在更大模型上扩展成本较高。
5. **模型规模限制**：16B 总参在 MoE 模型中属于中小规模，能否在更大尺度（如 67B/141B）上复现相同效率增益尚未验证。

## 研究启发与可借鉴点
1. **Feedback-driven 数据筛选可用于其他模型训练**：SFT 阶段用 judge model 分析错误 → reflection model 生成检索 query → embedding 检索修复数据的 pipeline（图 5），可有效替代均匀采样，提升 IFEval (+4.8)、AIME25 (+3.6)、LCB (+3.2)。该方法可迁移至任何 SFT 数据配比优化场景。
2. **DPO 阶段禁用 load balancing 稳定 expert routing**：MoE 模型在低 LR 偏好优化中容易出现 routing drift，关闭 bias 更新和 auxiliary loss 可保持 top-1 expert 差异仅 5%（vs 54%），这对所有 MoE 偏好对齐工作具有直接参考价值。
3. **MOPD 的 self-anchoring 设计**：以 DPO model 作为非 IF domain 的"教师"（实为自身），使非目标域的 advantage 接近零，从而在强化 IF 能力同时保护已有能力。该思想可推广至多能力并行增强的 post-training 场景。
4. **R3 + TIS 解决 MoE train/inference routing 不一致**：在 RL 中记录 rollout 时的 expert assignment 并在训练 replay，结合 token-level truncated importance sampling 修正数值差异，这对所有基于 MoE 的 RL post-training 工作具有通用借鉴意义。
5. **FarSkip-Collective 的通信-计算重叠思路可扩展**：通过容忍少量 activation 近似来换取大量通信重叠，在attention-heavy 和 MoE 混合架构中有推广潜力，特别是在 H100/MI300X 等高带宽硬件上。

## 关键术语表
- **Gated MLA（Gated Multi-head Latent Attention）**：在 MLA 的 attention output 和 output projection 之间插入 input-conditioned sigmoid gate，对每个 head 的 value channel 进行逐元素调制，增强表达力。
- **FarSkip-Collective**：一种 MoE 层间连接模式，通过用部分/过时 activation 作为下一层输入，消除 Dispatch/Combine all-to-all 通信的计算阻塞，实现通信与计算的完全重叠。
- **MOPD（Multi-Teacher On-Policy Distillation）**：在 RL post-training 中，用 domain-routed 的多位 frozen teacher（包括 student 自身作为 self-anchor）对当前 student on-policy rollout 进行 token-level reverse-KL 蒸馏，整合专项能力而不破坏其他能力。
- **R3（Rollout Routing Replay）**：在 MoE RL 训练中记录 rollout 阶段的 expert 分配决策并在训练前向传播中 replay，确保训练 log-probability 与采样路径一致。
- **TIS（Truncated Importance Sampling）**：对 on-policy/off-policy 分布差异进行 token-level 重要性采样修正，并将 ratio 截断到有界范围（如 [0.5, 2.0]），避免梯度方差过大。
- **Delta Learning（偏好优化中的 delta 思想）**：DPO 训练信号来自 chosen 和 rejected response 之间的质量差距而非绝对质量，因此即使 chosen trace 来自强模型而 rejected 来自弱模型也能提供有效学习信号。
- **Model Souping**：将多个独立训练的 checkpoint 以等权（或其他权重）线性平均，常在 mid-training 多 variant 合并中使用以提升泛化性能。
- **IF-RLVR（Instruction-Following Reinforcement Learning with Verifiable Rewards）**：使用确定性 verifier（约束检查函数）而非 reward model 为 instruction-following 响应打分，支持 partial credit 并按约束满足比例评分。

## 可复现要素
- **数据集**：全部为公开开源数据（Nemotron-CC-v2、Dolma 3 Dolmino 100B、MegaMath、FineMath、RefineCode、TxT360、Dolci-Think 系列、Instella-GSM8K-synthetic 等），license 多为 ODC-BY-1.0 或 dependent，见论文 Table 4。
- **代码**：开源，GitHub 仓库 `github.com/AMD-AGI/Instella-MoE`，包含 FarSkip-Collective 重叠实现代码。
- **权重**：开源全部阶段 checkpoint（Table 2），涵盖 Pretrain → Midtrain → Base → SFT → DPO → Think 六个阶段。
- **训练配置**：所有阶段超参在 Table 3 详细列出（tokens/seq len/global BS/peak LR/LR schedule/key settings）。
- **数据配比**：各阶段训练数据明细在 Table 4 和 Appendix C（Table 22-24）中给出。
- **硬件**：AMD Instinct MI300X 和 MI325X GPUs，Primus + Miles 框架。
