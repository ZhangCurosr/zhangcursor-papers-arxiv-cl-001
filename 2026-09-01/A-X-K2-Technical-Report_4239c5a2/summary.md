---
title: "A-X-K2-Technical-Report"
source: https://arxiv.org/pdf/2608.30181v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:48:05"
field: "大规模语言模型训练与推理优化"
keywords: ["Mixture-of-Experts", "Sparse Gated Attention", "Gated Norm", "Long-context LLM", "Think-Fusion", "Reinforcement Learning", "Low-precision Serving", "Agentic AI"]
innovations: ["SGA 稀疏门控注意力结合 sparse warmup 实现 128K/256K 长上下文高效推理", "GN 门控归一化统一训练稳定与 NVFP4 低精度部署", "Think-Fusion 配对数据实现统一模型的 thinking/non-thinking 可控切换"]
benchmarks: ["RULER", "LongBench v2", "AIME26", "Apex", "KMMLU-Pro", "CLIcK", "τ²-Bench Telecom", "GDPval", "KS-Eval", "Manufacturing Benchmark"]
---

# 论文速读：A-X-K2-Technical-Report

## 一句话总结
SK Telecom 发布的 A.X K2 是一个 688B 参数 MoE 语言模型，仅用 8.5T tokens（少于前代 A.X K1）完成预训练，通过稀疏门控注意力（SGA）和门控归一化（GN）实现高效 128K/256K 长上下文推理，并借助 Think-Fusion SFT 在多阶段 RL 后统一支持 thinking/non-thinking 双模式推理，在数学、韩语和理解基准上媲美或超越主流开源基线。

## 研究问题与动机
- **Agentic 能力与部署效率的矛盾**：前沿 LLM 已具备多步推理、工具调用等 agentic 能力，但推理链膨胀导致延迟和成本剧增，实际部署需要可控推理开销。
- **Token 效率瓶颈**：参数量增大与训练 token 预算之间存在权衡，希望以更少数据达成更高性能。
- **长上下文服务质量与计算成本的平衡**：原生 128K 训练后，稀疏注意力引入会显著降低 KV-cache 负载，但需避免质量衰减。
- **统一模型的推理模式控制**：同一模型需在复杂推理与快速响应之间灵活切换，避免部署多模型。

## 核心贡献（创新点）
1. **Token-Efficient Pre-training**：688B 参数 MoE 模型在 8.5T tokens 下完成训练，部分基准较 A.X K1 提升超 30pp，证明数据质量与处理 pipeline 可显著改善 token efficiency。
2. **Gated Transformer Blocks（SGA + GN）**：将 Sparse Gated Attention（轻 indexer + head-specific output gate）与 Gated Norm 结合，使每个 query 仅读取 2,048 个位置（128K 上下文的 1.6%），同时保持 LongBench 质量不变，并支持 NVFP4 4-bit 部署且仅损失 0.76 分。
3. **Sparse warmup adaptation**：索引器从一开始就在自身 sparse top-k selection 上优化，而非先 fit dense attention distribution，大幅降低 SGA 适配成本。
4. **Think-Fusion SFT + Multi-Stage RL**：配对 thinking/non-thinking 数据训练，使模式切换由显式控制 token 而非分布特征驱动；后续多阶段 RL 在共享 reward framework 下联合优化指令遵循、偏好对齐、agentic 工具使用与安全。
5. **端到端工程验证**：512×NVIDIA B200 GPU、约 70 天稳定完成 688B MoE 全链路训练，并在 TP1/DP8 异步 rollout 与 blockwise FP8 trainer-rollout 一致性上给出系统级方案。

## 方法详解
- **架构**：688B 总参数 / 33B 激活参数；61 层（首层 dense FFN，其余 60 层 MoE）；每层 64 个 attention head；256 个 routed experts（每 token 激活 8 个，分 8 组，group top-k=4）；MLA + QK-normalization + 全局辅助损失 + 可学习 expert bias。
- **SGA 设计**：轻量 indexer 对 KV 候选打分并选出 top-k=2048；head-specific output gate $G=\sigma(W_g q_{latent})$ 注入非线性并抑制 attention sink；indexer 训练目标为 KL divergence 对齐 attention distribution。
- **GN（Gated Norm）**：在 RMSNorm 后立即施加输入依赖的门控，平滑 hidden-state outliers；消除 A.X K1 的双归一化结构，单点 GN 已足以稳定训练并改善 FP8/NVFP4 块尺度利用率。
- **Stage 3 长上下文适配**：ABF 方式将 RoPE base 从 $10^4$ 提升至 $10^6$，原生训练至 32K→128K；SGA 阶段冻结主干，仅 warmup indexer（sparse warmup）后再放开全参数微调；推理时通过 YaRN scaling factor 扩展至 256K，NIAH 保持 100。
- **Think-Fusion SFT**：构造 paired 数据（每 prompt 配 thinking 与 non-thinking 两组响应），thinking:non-thinking token 比约 13:1，防止长样本主导训练；indexer SFT 阶段同时训练 indexer，之后冻结。
- **多阶段 RL**：CISPO 优化（clip importance-sampling weight 而非 token-level update）+ GDPO 多 reward 归一化；全局 batch=80 prompts，temperature=1.0，lr=$2\times10^{-6}$；四类 reward：rule-based 指令遵循、LLM-as-judge 人类偏好、verifiable + gated judge 的工具调用、九维度 safety judge。
- **Blockwise FP8 一致性**：Blackwell TE 默认 MXFP8，通过 patch 强制 blockwise FP8（per-block scale=FP32）与 vLLM rollout 对齐，避免 trainer-rollout 精度不匹配导致的 reward collapse。
- **CheckPoint Merging**：WSM 风格 SWA，取最后 6 个 checkpoint（间隔约 1.6B tokens）在权重空间平均，采用 $1-\sqrt{\cdot}$ 权重偏向最新 checkpoint。

## 实验与结果
- **数学**：AIME26 97.1%（SOTA 对比中最佳）；Apex 45.8%（A.X K1 仅 1.0%）；Apex-shortlist 88.6%；KMO26 92.5%；IMO 2025 获 35/42（达金牌线），KMO26 第二轮 8/8 全对。
- **韩语**：KMMLU-Pro 80.5%（领先）、KoBALT 73.0%、CLIcK 91.6%；KS-Eval 各子项全面超越 A.X K1（Instruction Following +27.98pp、Long Context +27.11pp）。
- **代码**：LiveCodeBench 84.0%；Terminal Bench v2.1 36.0%（弱于 DeepSeek/GLM，留优化空间）。
- **科学/通用**：Humanity's Last Exam 27.8%（vs K1 8.3%）；AA-LCR 66.0%（vs K1 26.0%）；GPQA Diamond 39.6%（略低于 GLM-5.1 42.3%）。
- **Agentic**：τ²-Bench Telecom 98.0%（最佳）；GDPval Elo 1031（vs K1 500）；τ³-Bench Banking 13.0%（接近 Qwen3.5 13.4%）。
- **长上下文**：RULER 全长度平均 94.6；128K 保留 95.1%、256K 保留 88.8% 的 4K 分数；NIAH 在 128K/256K/512K 均 100；LongBench v2（416-item filtered）63.9%，与 Qwen3.5 64.2% 接近。
- **SGA 质量中性**：LongBench v1 由 dense 62.80 → sparse 62.99（+0.19），证明 1.6% 注意力预算下质量未损。
- **低精度**：NVFP4 (W4A4 experts-only) 11 项平均 78.19 vs FP8 78.95，保留 99.0%；NIAH 在 NVFP4 下仍完美。
- **推理吞吐（单 B200，concurrency=32，OSL=1K）**：FP8 在 64K 输入处超越 A.X K1；加 EAGLE3 投机解码后 crossover 提前至 32K；NVFP4 PTQ 在全部长度（1K–120K）均领先 A.X K1，120K 处 +123.7%（20.3K tok/s）。

## 相关工作脉络
- **DeepSeek-V3.2/V4**：稀疏注意力与 indexer 设计参考；本文差异在于 sparse warmup 直接优化 sparse top-k 而非先 fit dense，且将 SGA 与 GN 联合用于低精度部署。
- **GLM-5**：Stage 3 复用 Stage 3B 数据做 SGA adaptation 的方案；本文在此基础上加入 ABF 原生扩展与 YaRN 静态/动态 scaling 选择。
- **Kimi-K2/Kimi-K2.6**：YaRN 与 scaling-law 指导的 MoE 设计参考；本文强调 native 128K 训练而非插值外推。
- **Qwen3.5-397B-A17B**：主流开源 MoE baseline；本文在数学、韩语与部分 agentic 指标上匹敌或超越，展现更低参数（33B active）竞争力。
- **Gemma / GLM-4 早期设计**：双归一化（attention/MLP 前后各一）稳定性方案；本文以单点 GN 取代，简化结构并实现同等或更好收敛。
- **MiniMax-M1/M2.7**：RL 侧 CISPO 优化器与小 epsilon AdamW 配置；本文沿用相似 recipe 并扩展至多 reward 联合框架。

## 局限性与未来方向
- **模态单一**：当前仅支持文本，缺少原生多模态理解能力。
- **Expert Parallelism 受限**：为规避跨节点慢速互联，将 EP 限制在单节点 NVLink（EP=8），未探索更高维度多节点 expert sharding。
- **算力预算约束**：在固定 70 天/512 B200 预算下做出工程取舍，与更大训练预算的同规模模型相比仍有性能缺口。
- **Thinking 模式下的安全 Trade-off**：thinking mode 的 ASR 从 12.1% 升至 33.9%，extended reasoning 对 adversarial prompt 的脆弱性需进一步缓解。
- **文档断言抗性不足**：Manufacturing Benchmark 中面对"文档提供错误答案"场景，ASR 抗性显著弱于"用户断言"场景，工业部署中需重点改进。

## 研究启发与可借鉴点
- **Sparse warmup 直接优化 sparse selection**：避免先 fit dense 再切换到 sparse 的两阶段高成本流程，可直接将 indexer loss 作用在其自身的 top-k mask 上，节省 adaptation 时间且质量无损。
- **GN 一举解决训练稳定与低精度部署**：用单一门控归一化替代双重归一化，既平滑 outliers 又缩小 NVFP4 块尺度动态范围，是 MoE 大规模部署的有效正则化手段。
- **Think-Fusion 配对数据胜过 token ratio 调参**：通过显式 paired 样本让模型从 control token 学习模式，而非依赖 length/token-ratio 等分布偏差，模式切换鲁棒性更强。
- **Trainer-rollout blockwise FP8 一致性是关键**：RL 训练中若 trainer 与 rollout 的 FP8 scaling 不一致会导致 reward collapse，需 patch TE 强制 blockwise 配方。
- **TP1/DP8 异步 rollout 适配大 MoE**：将 attention 并行从 TP 改为 DP 可消除 per-layer all-reduce，显著提升 MoE 生成吞吐，适合 RL rollout 阶段。

## 关键术语表
- **MoE（Mixture-of-Experts）**：通过 router 将 token 动态分配给多个专家网络，仅激活部分专家以降低计算量、扩大总参数规模。
- **SGA（Sparse Gated Attention）**：结合轻量 indexer 的 top-k 稀疏选择与 head-specific output gate，以恒定计算预算服务超长上下文。
- **GN（Gated Norm）**：在 RMSNorm 后附加输入依赖门控，抑制 hidden-state outliers，提升训练稳定性与低精度量化表现。
- **MLA（Multi-head Latent Attention）**：将 Q/K/V 投影压缩至低维 latent 表示，减少 KV cache 体积并降低长上下文显存占用。
- **Think-Fusion**：在 SFT 阶段构造 paired thinking/non-thinking 响应对，使模型通过控制 token 而非分布特征切换推理模式。
- **ABF（Adjusted Base Frequency）**：通过在预训练不同阶段调整 RoPE base frequency 实现原生长上下文扩展，避免插值外推误差。
- **YaRN**：在已有 RoPE 基础上通过缩放因子扩展上下文长度，支持 static/dynamic 两种模式。
- **CISPO**：对 importance-sampling weight 进行 clip 的 RL 优化器，保留低概率但行为关键 token 的梯度贡献。

## 可复现要素
- **代码与权重**：https://huggingface.co/skt/A.X-K2（已开源）
- **数据集**：预训练与 SFT 数据为内部构建（含 Nemotron-CC v2.1、FineWeb2、内部韩语 crawl 及合成数据），未公开
- **训练硬件**：512×NVIDIA B200 GPU，约 70 天
- **预训练 Token**：总计约 8.5T（pretrain 8.2T + posttrain 0.3T）
- **优化器**：AdamW（β1=0.9, β2=0.95, ε=1e-8），peak lr=2.2e-4，global batch=16384 seq
- **精度配置**：训练 MXFP8 E4M3，发布 checkpoint 为 blockwise FP8 E4M3；推理可选 NVFP4 W4A4
- **并行策略**：PP=8（interleaved virtual-pipeline degree=2），EP=8，DP 占余下，无 TP；长上下文阶段 CP=2（32K）/ CP=8（128K）
- **激活重计算**：full activation recomputation，micro-batch=4（短序列）/1（长序列）
