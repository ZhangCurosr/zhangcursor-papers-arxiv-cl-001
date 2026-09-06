---
title: "Instella-MoE-Technical-Report"
source: https://arxiv.org/pdf/2609.00791v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:19:54"
field: "开源高效大语言模型训练"
keywords: ["Mixture-of-Experts", "MoE", "Gated MLA", "FarSkip-Collective", "fully open LLM", "instruction tuning", "reinforcement learning", "MOPD"]
innovations: ["Gated MLA 门控多头潜注意力提升表达能力", "FarSkip-Collective 通信-计算重叠提升训练/推理吞吐", "反馈驱动数据选择与域路由多教师 on-policy 蒸馏保障后训练稳定"]
benchmarks: ["MMLU", "GSM8K", "HumanEval+", "HELMET", "RULER", "IFEval", "AIME24/25", "BBH", "GPQA", "AGIEval"]
---

# 论文速读：Instella-MoE-Technical-Report

## 一句话总结
本文提出了 Instella-MoE-16B-A3B，一个基于 MoE 架构、**完全开源**（含权重、代码、配方）的 16B 总参数 / 2.8B 激活参数语言模型，采用 Gated MLA 与 FarSkip-Collective 两项创新实现高效训练与推理，在同等激活参数量下取得了最强的完全开源模型表现（Base 平均 76.7，Think 平均 73.2）。

## 研究问题与动机
1. **专有模型主导与科研透明性缺失**：当前 SOTA 模型多为专有发布，权重与训练数据不公开，阻碍了科学理解与可复现性。
2. **“开放权重”≠“完全开源”**：Qwen、Llama、DeepSeek 等虽开放权重，但通常未公开预训练语料、预处理流水线与完整训练配方，研究者无法审计数据污染或复现全流程。
3. **MoE 效率优势尚未在完全开源模型中充分验证**：MoE 能以少量激活参数获得大模型容量，但目前性能具备竞争力的完全开源 MoE 模型仍较少（仅 OLMoE 等早期探索）。
4. **AMD 硬件生态下的开源 MoE 支持不足**：缺乏在 AMD Instinct GPU 上端到端训练的开源 MoE 模型与工具链，限制了硬件多样性研究。

## 核心贡献（创新点）
1. **Instella-MoE-16B-A3B 完全开源 MoE 模型**：16B 总参数 / 2.8B 激活参数，在相同激活规模下超越 SmolLM3-3B、OLMo-3-7B 等完全开源模型，并与 Moonlight-16B-A3B、Qwen3.5-4B 等开放权重模型持平或领先。
2. **Gated MLA（门控多头潜注意力）**：在 MLA 压缩 KV 的基础上加入输入条件门控，以极小开销提升模型表达能力；相较标准 MLA，在 200B token 消融中平均提升 +0.47 分，尤其在代码与 MMLU 上增益明显。
3. **FarSkip-Collective 连接模式**：修改 MoE 层内数据流依赖，使 Dispatch/Combine 通信与注意力、共享专家计算重叠，预训练吞吐提升 12.7%，推理 TTFT 吞吐提升 39.2%。
4. **端到端多阶段训练流水线与全量开源**：涵盖 Pretrain → Mid-train → Long-context → SFT（含反馈驱动数据选择）→ DPO（含路由稳定化）→ IF-RL → MOPD 的完整流程，并开源所有阶段权重、配置与代码，建立可复现研究基准。
5. **MoE 后训练稳定性优化**：发现 DPO 阶段启用负载均衡会导致路由漂移，提出在 DPO 时冻结偏置更新并关闭辅助损失，使 top-1 专家偏移率从 54% 降至 5%，专家集 Jaccard 重叠从 0.42 升至 0.94。

## 方法详解
- **架构概览**：Decoder-only Transformer，27 层（1 dense + 26 MoE），hidden size 2048，16 个注意力头（KV latent rank 512），64 个路由专家（每 token top-6，FFN 1408）+ 2 个共享专家（FFN 2816）。使用 RMSNorm + SwiGLU + RoPE。
- **Gated MLA**：对 MLA 输出的 per-head 通道施加 element-wise sigmoid 门控，公式为 $y_t = W_o[\text{Sigmoid}(W_g x_t) \odot \hat{o}_t]$。门控由归一化后的输入 $x_t$ 产生，在不增加 KV-cache 前提下提升表达力。
- **FarSkip-Collective**：标准 MoE 中 Dispatch 依赖前一层 Combine 完成，Combine 依赖专家计算完成；FarSkip 改为使用“可用激活”（可能缺少部分 routed expert 输出）作为下一层输入，使得通信可与本层注意力、共享专家计算并行，从而消除计算等待气泡。
- **预训练**：7.1T tokens，混合包括 Nemotron-CC-v2、Nemotron-CC-Math、MegaMath、RefineCode 等，峰值 LR $4\times10^{-4}$，WSD 调度，激活 MTP（权重 0.3）。
- **Mid-training**：3 个基于 Dolma 3 Dolmino 的变体（v1/v2/v3），仅在 STEM/推理子集比例上不同，分别训练后等权 model souping，MTP 权重降至 0.1。
- **长上下文扩展**：两步从 4K 延伸至 64K。Stage 1 使用 Dolma 3 Longmino 100B（~194B tokens）+ YaRN（$\theta=8\times10^6$）+ document masking；Stage 2 使用 37.32B token STEM 混合恢复数学/代码能力，最终得到 Base 模型。
- **SFT（反馈驱动数据选择）**：Phase 1 使用 Dolci-Think-SFT-7B 及数学/代码/科学切片；Phase 2 针对 512K 样本，通过 judge 模型诊断错误、reflection 模型生成加权检索查询、embedding 检索选取针对性样本，相比均匀采样平均提升 +1.5 分。
- **DPO**：使用 Dolci-Think-DPO-7B 偏好对（delta learning 设定，强/弱轨迹对比）；关键修改为关闭负载均衡偏置更新与辅助损失，以稳定路由分布。
- **强化学习（RL）**：
  - **IF-specialized RL**：基于 GRPO + DAPO/Dr.GRPO 改进（零梯度过滤、active sampling、token-level loss、无 KL、clip-higher 等），使用 IF-RLVR 验证式奖励（多约束部分得分），并启用 R3（Rollout Routing Replay）与 TIS 解决 MoE 训练/推理路由不一致问题。
  - **MOPD（Multi-Teacher On-Policy Distillation）**：以 DPO 模型为锚，IF 专家仅对 IF 提示打分，其他提示由 DPO 模型自锚打分；学生通过 token-level reverse KL 在自身 on-policy  rollout 上更新，融合 IF 能力同时保持数学/代码能力。

## 实验与结果
- **数据集与基线**：基础模型评测使用 ARC-E/C、BoolQ、SciQ、PIQA、HSwag、WG、OBQA、MMLU、GSM8K、HumanEval+、MBPP+、MATH；长上下文使用 HELMET、RULER。基线包括 OLMo-3-7B、SmolLM3-3B、OLMoE-1B-7B、Moonlight-16B-A3B、Qwen3.5-4B、Gemma-4-E4B 等。
- **Base 模型**：Instella-MoE-16B-A3B-Base 在标准预训练基准上平均 **76.7**，为完全开源模型最高；超越 OLMo-3-7B（70.1）+6.6、SmolLM3-3B（70.5）+6.2、OLMoE-1B-7B（61.9）+14.8；接近开放权重 Qwen3.5-4B-Base（79.5），优于 Moonlight-16B-A3B（76.2）。长上下文 HELMET 41.5 / RULER 79.4。
- **Post-trained 模型**：Think 检查点在指令遵循/推理/数学/代码/聊天基准上平均 **73.2**，超越 OLMo-3-7B-Think（72.0）、Gemma-4-E4B（70.5）、Qwen3.5-4B（69.7），为对比中最强。IFEval 提升最显著（反馈驱动 SFT 带来 +4.8 分）。
- **消融**：Gated MLA 带来 +0.47 平均分；FarSkip 在精度持平（50.38 vs 50.33）同时带来通信重叠收益；IF-RL 单独导致数学/推理退化，MOPD 可恢复 IF 增益并保持 DPO 水平（RL 消融平均 75.7 最高）。
- **效率**：FarSkip-Collective 预训练吞吐 +12.7%，SGLang expert-parallel 推理 TTFT 吞吐 +39.2%。

## 相关工作脉络
1. **OLMoE / OLMo-3 / SmolLM3**：完全开源序列，但 OLMoE 参数规模小（1.3B active）、SmolLM3 为 dense；Instella-MoE 首次在 2.8B active MoE 上实现同等或更强性能且全量开源。
2. **DeepSeek-V3 / V4**：提出 MLA、MoE 高效训练与 MTP；Instella-MoE 继承 MLA 思想并引入 Gated 门控，同时在 FarSkip 通信重叠方面独立创新。
3. **Moonlight-16B-A3B / Qwen3.5 / Gemma-4**：开放权重 MoE/dense 模型；Instella-MoE 在相当激活参数规模下达到可比性能，并开源完整训练流。
4. **GRPO / DAPO / Dr.GRPO**：强化学习算法改进；本文在其基础上针对 MoE 引入 R3 + TIS 以对齐训练/推理路由，并设计 delta-learning DPO 与域路由 MOPD。
5. **Multi-Token Prediction (MTP)**：常见于 DeepSeek-V3；本文在预训练/中期训练阶段使用 MTP（权重 0.3/0.1），长上下文及之后禁用。
6. **Context extension (YaRN / document masking)**：OLMo 系列已应用；本文采用两阶段扩展策略，并在 Stage 2 以 STEM 优先混合恢复 short-context 能力。

## 局限性与未来方向
- **仅英文与文本模态**：当前模型为单语言文本模型，未涉及多语言或视觉/多模态扩展。
- **推理长度与长上下文权衡**：Stage 2 STEM 恢复以略微降低 HELMET/RULER 得分为代价换取短上下文数学/代码能力。
- **仅 AMD GPU 验证**：训练与系统优化（FarSkip、Primus/Miles）主要针对 AMD Instinct 硬件，跨平台泛化未经系统评估。
- **MoE 路由漂移仅在 DPO 阶段缓解**：后训练其他阶段（如 RL）仍可能存在路由分布变化，需进一步验证长期稳定性。
- **未来方向**：可探索多语言扩展、更大上下文（>64K）、跨硬件移植、多模态架构集成，以及更细粒度的领域路由与 continual post-training 策略。

## 研究启发与可借鉴点
1. **反馈驱动数据选择流水线**：judge 模型结构化错误分析 → reflection 模型生成检索策略 → embedding 近邻检索，可作为通用 SFT 数据精筛范式，迁移至其他模型规模与任务。
2. **MoE 后训练的“路由稳定化”意识**：DPO/RL 阶段关闭辅助负载均衡与偏置更新可有效防止 expert routing drift；后续 MoE 后训练工作可直接沿用此经验。
3. **Multi-Teacher On-Policy Distillation 解耦能力**：通过域路由将不同 teacher 分配给不同能力维度，可在不破坏原有能力的前提下添加专项能力（如 IF、安全、代码），适合模块化能力增强场景。
4. **FarSkip-Collective 的通信-计算重叠思想**：对 any all-to-all heavy 架构（如专家并行、序列并行）具有参考价值，可降低训练/推理气泡。
5. **Delta-learning 偏好数据构造**：使用强/弱轨迹对比而非单纯模仿优质轨迹，可提升 DPO 数据效率；适合资源受限下构建高质量偏好集。

## 关键术语表
- **MoE（Mixture-of-Experts）**：将每层拆分为多个专家网络，每个 token 仅路由至少数专家计算，从而以较少激活参数获得较大模型容量。
- **Gated MLA**：在多头潜注意力（MLA）的输出投影前加入由输入决定的 element-wise sigmoid 门控，用以自适应调制各注意力头输出通道。
- **FarSkip-Collective**：一种 MoE 层连接改造，允许 Dispatch/Combine all-to-all 通信与同层注意力、共享专家计算重叠执行，减少通信等待。
- **MTP（Multi-Token Prediction）**：除预测下一 token 外，额外预测后续 token，提供更强训练信号；本文仅在预训练/中期训练阶段启用。
- **DPO（Direct Preference Optimization）**：直接基于偏好对优化策略，无需显式奖励模型；本文结合 delta learning 构造强弱轨迹对比对。
- **GRPO**：Group Relative Policy Optimization，以组内相对优势进行策略梯度更新；本文辅以 zero-gradient filtering、active sampling、clip-higher 等改进。
- **MOPD**：Multi-Teacher On-Policy Distillation，通过域路由将不同 teacher 的 log-prob 信号蒸馏到 student 的 on-policy rollout 上，实现多能力融合。
- **R3（Rollout Routing Replay）**：记录推理阶段的专家分配并在训练前向中重放，保证 train/inference 的专家路径一致，配合 TIS 校正数值偏差。

## 可复现要素
- **数据集**：预训练/中期/长上下文/SFT/DPO/RL 各阶段使用的语料均源自公开开源数据集（Nemotron-CC、Dolma 3 Dolmino、MegaMath、RefineCode、TxT360 等），许可证多为 ODC-BY 或依赖型；具体配比见论文 Table 4 与 Appendix C。
- **代码**：训练代码已开源至 `github.com/AMD-AGI/Instella-MoE`（论文声明），使用 Primus（Megatron-LM 后端）与 Miles 框架。
- **权重**：从 Pretrain → Midtrain → Base → SFT → DPO → Think 各阶段权重均开源。
- **关键超参**：
  - 预训练：7.1T tokens、seq len 4096、global BS 4096、peak LR $4\times10^{-4}$、WSD schedule、MTP weight 0.3。
  - Mid-training：~100B tokens、3 变体等权 souping、peak LR $2\times10^{-4}$。
  - Long-context：Stage 1 ~194B tokens / Stage 2 ~20B tokens、seq len 65536、YaRN $\theta=8\times10^6$、document masking。
  - SFT：32K seq len、2 epochs、phase-2 512K 反馈驱动样本。
  - DPO：$\beta=5.0$、0.75 epoch、关闭负载均衡偏置与辅助损失。
  - RL：IF-RL 1400 steps、LR $1\times10^{-6}$、clip [0.2, 0.272]、TIS clip [0.5, 1.5]；MOPD LR $1\times10^{-6}$、30-step warmup、TIS clip [0.5, 2.0]。
- **硬件**：AMD Instinct MI300X / MI325X，EP=8，bfloat16 混合精度。
