---
title: "A-X-K2-Technical-Report"
source: https://arxiv.org/pdf/2608.30181v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:48:13"
field: "大规模语言模型系统与方法"
keywords: ["Mixture-of-Experts", "Sparse Gated Attention", "Long Context", "Think-Fusion", "Low-Precision Deployment", "Reinforcement Learning", "Korean LLM"]
innovations: ["提出稀疏门控注意力(SGA)配合稀疏warmup策略，在1.6%注意力预算下保持128K/256K上下文质量", "提出Think-Fusion配对SFT recipe实现单一模型内可切换的思维/非思维模式", "端到端blockwise FP8训练-部署一致性与数据并行异步rollout解决MoE低精度RL稳定性问题"]
benchmarks: ["RULER", "LongBench v2", "AIME26", "Apex", "KMMLU-Pro", "CLIcK", "GDPval", "τ²-Bench", "KS-Eval", "Manufacturing Benchmark", "LiveCodeBench"]
---

# 论文速读：A-X-K2-Technical-Report

## 一句话总结
SK Telecom 发布了 A.X K2，一个总参数 688B、激活参数 33B 的 MoE 语言模型，通过在更高质量的 8.5T token 数据上从头训练，结合稀疏门控注意力（SGA）、门控归一化（GN）和 Think-Fusion SFT 等技术，在数学、韩语和智能体任务上达到与领先开源模型竞争力，同时原生支持 128K/256K 长上下文并实现高效的低精度部署。

## 研究问题与动机
1. **大模型规模化与部署效率的矛盾**：前沿 LLM 在推理成本、长上下文服务和可控推理计算方面面临严峻挑战，单纯扩大规模无法解决部署问题。
2. **长上下文推理的算力瓶颈**：深度推理产生超长思维链会显著增加延迟和服务成本，而简单查询则浪费算力，需提供推理深度的用户可控切换机制。
3. **低精度部署的数值稳定性**：大规模 MoE 模型在 NVFP4 等窄块缩放格式下，隐藏状态异常值会导致精度崩溃，需要有效的抑制手段。
4. **韩国本土 AI 主权需求**：作为韩国国家"主权 AI 基础模型"项目的一部分，需构建深刻理解韩语和文化的领先规模模型，减少对国外闭源系统的依赖。

## 核心贡献（创新点）
1. **Token 高效的预训练**：在比 A.X K1 更少的 8.5T token（约 8.2T 预训练）上实现了全面超越，部分基准提升超过 30pp，体现了通过改进多阶段数据处理管道实现的显著 token 效率增益。*与 A.X K1 相比，这是同框架下数据质量优先策略的直接验证。*

2. **稀疏门控注意力（SGA）与原生 128K 训练**：将轻量级索引器与门控注意力结合，每个查询仅关注 2,048 个位置（128K 上下文的 1.6%），并通过"稀疏 warmup"——直接从稀疏 top-k 选择优化索引器而非先拟合密集注意力分布——大幅降低适应成本，使 RULER 在 256K 上下文下达到 94.6 分。*与 DeepSeek-V3.2 和 GLM-5 的先密集后稀疏 warmup 方案本质不同，节省了昂贵的高质量适应阶段。*

3. **门控归一化（GN）稳定大规模训练并赋能低精度服务**：在 RMSNorm 后引入输入依赖的门控操作，抑制隐藏状态异常值放大，使 4-bit NVFP4 服务精度与 FP8 相差不到 1 个点（平均仅下降 0.76 分，保留 99.0% FP8 精度）。*与 A.X K1 使用的双归一化方案相比，GN 以单一门控层替代了更复杂的 stacked normalization 架构。*

4. **Think-Fusion SFT + 多阶段 RL**：通过在配对思维/非思维响应数据上联合训练，使用显式控制 token 而非表面分布线索实现模式切换；多阶段 RL 在共享奖励框架下联合优化指令遵循、人类偏好对齐、智能体工具使用和安全。*与 EXAONE 4.0 等的双轨训练策略不同，本方法通过数据工程而非架构改造解决模式混淆问题。*

5. **端到端 Blockwise FP8 训练-部署一致性**：针对 RL 中 trainer（Transformer Engine）与 rollout 引擎（vLLM）数值不一致导致的奖励崩溃问题，设计了端到端 blockwise FP8 方案，并通过数据并行异步 rollout 流水线（TP1/DP8）解决 MoE 大规模 RL 的吞吐量瓶颈。*这是首次系统性解决 MoE 模型低精度 RL 中 trainer-rollout 精度对齐的工程难题。*

## 方法详解

**架构设计**：A.X K2 为 MoE 架构，688B 总参数、33B 激活参数，61 层（首层为密集 FFN，其余 60 层为 MoE），每层 256 个路由专家中激活 8 个（分成 8 组，每组 top-4），采用 MLA（Multi-head Latent Attention）、QK 归一化和 head-specific output gate。

**SGA（Sparse Gated Attention）**：由两部分组成：(1) 轻量级索引器基于 DeepSeek-AI (2025b) 的设计，对每个查询选择 top-k=2048 个关键 token；(2) head-specific output gate $G = \sigma(W_g q_{latent})$ 调制 attention 输出。索引器通过 KL 散度损失对齐其选择分数与注意力分布。SGA 使注意力计算从二次方缩放变为近乎线性：128K 上下文每个查询仅读取 2,048 个位置（1.6%），256K 时降至 0.8%。

**Sparse Warmup 策略**：Stage 3C 引入 SGA 时，冻结非索引器参数，直接从稀疏 top-k 选择优化索引器（而非先对密集注意力分布做 warmup），大幅降低适应成本，且质量损失可忽略（LongBench v1: 62.80→62.99）。

**GN（Gated Norm）**：在 RMSNorm 后立即应用门控操作，平滑异常值分布。实验表明 GN 足以替代 A.X K1 中的双归一化方案（图 3 消融显示叠加 post-MLP normalization 无额外收益），同时将训练吞吐量降低约 5%。

**长上下文扩展**：使用 ABF（Adjusted Base Frequency）方法，RoPE base 从默认 $10^4$ 逐步提升至 $10^6$，原生训练至 128K；推理时通过 YaRN scaling factor 扩展至 256K（NIAH 得分为 100）。

**Checkpoint Merging（WSM）**：对最后 6 个 checkpoint（间隔约 1.6B token）在权重空间做 SWA，使用 $1-\sqrt{\cdot}$ 加权方案，最近 checkpoint 权重更大。

**Pre-training 三阶段课程**：
- Stage 1（6.4T token，seq len 4096）：通用知识，WSD 的 warmup+stable 阶段
- Stage 2（1.4T token，seq len 4096）：高质量推理，学习率余弦衰减
- Stage 3（0.36T token）：32K→128K 渐进扩展 + SGA 适应

**思考/非思考模式控制（Think-Fusion）**：构建配对 SFT 数据集（思维:非思维 ≈ 13:1 token 比），使用控制 token `<think>...</think>` 区分模式。Indexer SFT 阶段同时训练索引器和模型参数（KL-loss=0.1），之后固定索引器进行主 SFT。

**多阶段 RL**：使用 CISPO 优化器（裁剪重要性采样权重而非 token 级更新）+ GDPO（分组奖励解耦归一化），无 KL 惩罚，16 rollouts/prompt，temperature=1.0，lr=$2\times10^{-6}$。奖励体系涵盖指令遵循（规则验证）、人类偏好（LLM-as-judge）、智能体工具使用（分层 judge：有效性门控+参考比较）和安全（9 维安全评估）。

**Blockwise FP8 训练-部署一致性**：对 Blackwell TE 打 patch 强制 FP32 block scale，使 trainer 与 vLLM rollout 引擎使用一致的 blockwise FP8 格式，避免 MXFP8 与 blockwise FP8 混用导致的奖励崩溃（图 6）。

**Data-Parallel Asynchronous Rollout**：rollout 引擎使用 TP1/DP8（而非训练器的 TP8/DP1），将注意力并行从张量并行转移到数据并行，消除逐层 all-reduce，保持 EP8 不变以实现策略权重 clean resharding。

## 实验与结果

**评测基线**：Qwen3.5-397B-A17B、Nemotron 3 Ultra (550B)、DeepSeek-V4 Flash (284B)、GLM-5.1 (754B)、Kimi-K2.6 (1T)、MiniMax-M2.7 (230B)，以及自身前代 A.X K1 (519B)。

**数学**：A.X K2 在 AIME26 达 97.1（最佳）、Apex 达 45.8（最佳，较 A.X K1 的 1.0 提升巨大）、Apex-shortlist 88.6（与 Nemotron 持平）、IMO 2025 获 35/42（金牌线）、KMO26 二试全对 8/8。

**韩语**：KMMLU-Pro 80.5（最佳）、CLIcK 91.6（最佳）、KoBALT 73.0（竞争力）。

**代码**：LiveCodeBench v6 (Feb-May) 84.0，SciCode 41.0，Terminal Bench v2.1 36.0（落后于 DeepSeek-V4 Pro 的 61.8）。

**科学&知识**：AA-Omniscience 85.6，Humanity's Last Exam 27.8（较 A.X K1 的 8.3 大幅提升），GPQA Diamond 39.6。

**通用**：IFBench 75.9，AA-LCR 66.0（较 A.X K1 的 26.0 提升 40pp）。

**智能体**：$\tau^2$-Bench Telecom 98.0（最佳），GDPval Elo 1031（较 A.X K1 的 500 大幅提升），$\tau^3$-Bench Banking 13.0（接近 Qwen3.5 的 13.4）。

**长上下文**：RULER 在 256K 上下文下总体 94.6；NIAH 在 128K/256K/512K 均得满分 100；LongBench v2（416项过滤集）63.9；SGA 引入后 LongBench v1 从 62.80 升至 62.99（质量无损）。

**KS-Eval（韩语专属）**：常识 57.45（第一）、指令遵循 71.67（第一）、长上下文 66.85（+27.11pp vs K1）、STEM 75.47。

**Manufacturing Benchmark（韩国制造业）**：Overall F1 69.0%（第一），拒答 F1 61.1%（领先 6.9pp）。

**Red-Teaming（韩语对抗）**：非思考模式 ASR 12.1%（最低，最安全）；思考模式 ASR 33.9%。

**低精度鲁棒性**：NVFP4（W4A4 experts-only）平均仅下降 0.76 分（保留 99.0% FP8 精度），NIAH 完美保持。

**推理效率**：SGA  alone 在 64K 输入时超越 A.X K1；加 EAGLE3 投机解码后在 32K 即超越；NVFP4 PTQ 在全长度范围超越，120K 输入时吞吐量较 A.X K1 提升 +123.7%（20.3K total tok/s）。

## 相关工作脉络
1. **DeepSeek-V3.2（DeepSeek-AI, 2025b）**：SGA 的稀疏注意力机制源自此工作，但本文提出"稀疏 warmup"替代其先密集后稀疏的两阶段索引器训练，大幅降低了适应成本。
2. **GLM-5（GLM-5 Team, 2026）**：同样采用先密集后稀疏的索引器 warmup 策略；本文与之相反，证明从稀疏 top-k 直接优化更高效且质量无损。
3. **Gated Attention（Qiu et al., 2025b）**：head-specific output gate 的理论基础，用于抑制 attention sink 并改善 loss 收敛，本文将其与稀疏注意力整合为 SGA。
4. **Gated Norm（Qiu et al., 2026）**：本文采用的 GN 技术来源，用于抑制隐藏状态异常值，支撑 NVFP4 低精度部署；本文验证其替代双归一化方案的可行性。
5. **EAGLE3（Li et al., 2025）**：投机解码方法，本文将其与 SGA+NVFP4 组合形成分层优化策略，实现 32K+ 上下文下的吞吐优势。
6. **CISPO + GDPO（MiniMax, 2025; Liu et al., 2026）**：多奖励 RL 的优化器与归一化方案，本文将其应用于统一的多能力 RL 训练框架，避免了单能力 RL 导致的过优化退化。

## 局限性与未来方向
1. **并行度受限**：Expert parallelism 限于节点内 8-GPU NVLink 域（EP=8），未能探索更高维度的多节点 expert 并行，限制了更细粒度的专家划分。
2. **计算预算约束下的性能差距**：在固定 GPU 预算和时间约束下，尽管参数层面已优化，仍略逊于同等规模但训练 Compute 更大的模型。
3. **单模态限制**：当前仅为纯文本模型，缺乏原生多模态理解能力。
4. **长上下文推理模式下的安全风险**：思考模式下的 ASR 升至 33.9%，说明扩展推理会增加对抗提示的脆弱性，需在后续工作中进一步改进安全对齐。
5. **文档断言抵抗较弱**：在 Manufacturing Benchmark 的 document-assertion 条件下仅得 41.4%，远低于 user-assertion 的 67.8%，工业部署中高度依赖参考文档的场景仍需改进。

## 研究启发与可借鉴点
1. **稀疏 warmup 替代密集 warmup**：对于 sparse attention 适配阶段，直接从稀疏 top-k 选择优化索引器而非先拟合密集分布，可大幅节省高质量适应阶段的 compute 成本——这对任何引入 sparse attention 的大模型训练都有借鉴价值。
2. **Think-Fusion 配对数据设计**：通过构建 prompt-思维/非思维响应对，让模型从控制 token 学习模式切换而非从样本长度等表面特征推断，有效解决了多模式训练的混淆问题——可迁移到任何需要推理深度控制的模型。
3. **Trainer-rollout 精度一致性对低精度 RL 的关键性**：Blockwise FP8 端到端一致性是 MoE 模型低精度 RL 稳定的必要条件，这一发现对任何使用低精度部署的大模型 RL 训练具有直接指导意义。
4. **GN 替代双归一化的架构简化**：单一 post-normalization 门控操作可取代复杂的 stacked normalization 设计，在保持训练稳定性的同时降低架构复杂度——为后续 MoE 模型设计提供了更简洁的稳定化方案。
5. **KS-Eval 作为 eval-dev 闭环**：将内部评测（KS-Eval、Manufacturing Benchmark、Red-Teaming）直接融入模型迭代循环，以评估结果驱动数据构造和训练改进——为韩国语/特定领域模型的持续迭代提供了可复用的方法论。

## 关键术语表
**Sparse Gated Attention (SGA)**：将轻量级索引器（top-k token 选择）与 head-specific output gate 结合的注意力机制，使每个查询在 128K 上下文中仅读取 2,048 个位置，注意力计算从二次方缩放近似变为线性。

**Gated Norm (GN)**：在 RMSNorm 后立即引入的可学习输入依赖门控操作，抑制隐藏状态异常值放大，改善低精度（FP8/NVFP4）部署的数值稳定性。

**Think-Fusion**：通过配对思维/非思维 SFT 数据训练单一统一模型，使模型能在推理时根据显式控制 token 在深思模式和简洁模式之间切换的技术。

**Blockwise FP8**：每小块（如 128×128 权重块）共享单一 scaling factor 的 FP8 格式，与 microscaling MXFP8 不同，可与主流 FP8 推理内核（如 DeepGEMM）兼容。

**NVFP4**：专家权重 4-bit（E2M1）、激活 4-bit 的低精度量化格式，在 GN 和 SGA output gate 的异常值抑制下可保留 99% 的 FP8 精度。

**WSD（Warmup-Stable-Decay）**：预训练学习率调度策略，包含 warmup 阶段、恒定最大学习率的稳定阶段和余弦衰减阶段。

**ABF（Adjusted Base Frequency）**：通过逐步提升 RoPE base frequency（从 $10^4$ 到 $10^6$）而非插值来原生扩展上下文长度的方法。

**CISPO**：一种 RL 优化算法，裁剪重要性采样权重（而非 token 级更新），使低概率但行为关键的 token 继续贡献梯度。

## 可复现要素
- **数据集**：预训练使用约 8.5T token，包括 Nemotron-CC v2.1、FineWeb2、内部韩语爬取数据、合成数据和 SFT 格式数据；具体组成见附录 B.1-B.2。*论文未提及公开*
- **代码/权重**：模型已在 HuggingFace 开源（https://huggingface.co/skt/A.X-K2）；*代码未提及*
- **关键超参**：总参数 688B，激活 33B，61 层，64 注意力头，hidden size 7168，专家 256/激活 8，vocab 163,840；学习率 peak $2.2\times10^{-4}$，batch size 16,384 seq，AdamW($\beta_1=0.9, \beta_2=0.95, \epsilon=10^{-8}$)；RL lr $2\times10^{-6}$，batch size 80 prompts，16 rollouts/prompt
- **硬件**：512 NVIDIA B200 GPU，约 70 天预训练
- **上下文长度**：原生 128K，YaRN 扩展至 256K
