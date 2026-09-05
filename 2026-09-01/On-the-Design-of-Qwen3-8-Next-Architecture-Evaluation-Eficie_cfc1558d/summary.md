---
title: "On-the-Design-of-Qwen3-8-Next-Architecture-Evaluation-Eficie"
source: https://arxiv.org/pdf/2608.30320v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:27:06"
field: "大语言模型架构设计"
keywords: ["sparse MoE", "Gated DeltaNet", "Gated Residual", "sparse attention", "Muon optimizer", "training stability", "scaling laws", "n-gram embedding"]
innovations: ["GDN-Attention混合架构：每四层中三层GDN+一层全注意力，在9个基准中赢8个", "Gated Residual：将残差流扩宽4分支并用elementwise sigmoid gate读取，替代HC的H_res并提升训练稳定性", "Qwen Sparse Attention：micro-block粒度compressed indexer使索引开销随序列长度亚二次增长，1M上下文prefill提速7.6x"]
benchmarks: ["MMLU", "MMLU-Pro", "SuperGPQA", "MATH", "GSM8K", "BBH", "MMMLU", "EvalPlus", "MultiPL-E", "RULER", "MRCR", "GPQA", "MGSM", "INCLUDE", "SWEBench-Pretrain"]
---

# 论文速读：On the Design of Qwen3.8-Flash-Next Architecture: Evaluation, Efficiency, and Training Stability

## 一句话总结
本文提出了 Qwen3.8-Flash-Next（125B 总参数量、6B 激活）的完整架构设计，通过 GDN-注意力混合、Gated Residual (GR)、Qwen Sparse Attention (QSA) 及 n-gram embedding 四大组件，在仅用前任旗舰 397B-A17B 模型约 1/3 激活参数、1/3 token 和约 1/9 FLOPs 的条件下，在 14 个基准上持平或超越前代旗舰。

## 研究问题与动机
- **性能-成本失衡**：前任旗舰 Qwen3.7-Plus-Base（397B 总参、17B 激活）训练成本极高，如何在大幅削减计算预算的同时保持旗舰级能力是核心问题。
- **架构变更的三重耦合**：任何架构改动同时影响下游能力、训练/推理成本、超参数最优性与训练稳定性，三者需联合优化而非孤立评估。
- **长上下文推理效率瓶颈**：full attention 的计算和 KV cache 随序列长度二次/线性增长，限制了百万token级别的实用部署。
- **大规模训练稳定性**：模型规模达万亿参数、训练 token 达数十万亿时，loss spike 和梯度异常频繁出现，需要架构层面（而非仅靠 clipping）的稳定性保障。
- **Loss 与下游精度不一致**：预训练 loss 最优不等于下游 benchmark 最优，需要多轴联合评估才能发现真正的架构收益。

## 核心贡献（创新点）
1. **GDN-Attention 混合架构**：每四层中三层使用 Gated DeltaNet (GDN) 实现线性复杂度的前缀压缩，一层使用全局 attention 保留直接 token 级检索；与全 attention Transformer 相比在 9 个基准中赢 8 个，相比 SWA 混合赢 7 个。
2. **Gated Residual (GR)**：将残差流扩宽为 4 条分支并通过 elementwise sigmoid gate 读取，既增加容量又提供训练稳定所需的 rescaling；与 Hyper-Connections (HC/mHC) 的本质区别在于将表达能力集中于 read（elementwise gate）而非 write 或 branch mixing（H_res），同时省去了一个全量残差读取，降低内存开销。
3. **Qwen Sparse Attention (QSA)**：采用 compressed lightweight indexer 在 micro-block 粒度上评估上下文重要性并选择 top-k 块，将索引复杂度从 O(n²) 降至 O(n²/r)；与传统 sparse attention 的本质区别在于 indexer 本身也随序列长度压缩，且与 GDN 混合架构天然兼容（层内压缩比跨层共享更可靠）。
4. **N-gram embedding 作为加速器外容量扩展**：将 51B 参数的 n-gram embedding 表置于 host memory 并通过 prefetching 与计算重叠，以近乎零额外 per-token FLOPs 和延迟的方式扩展模型容量；与 MoE 专家的本质区别在于确定性寻址 + 稀疏访问使得 off-accelerator 存储完全可行。
5. **稳定性-效率-能力联合设计方法论**：通过 stress test（固定高学习率放大不稳定）和 scaling law 重拟合（Muon + GR 使最优 batch size 和学习率同时上移、batch warmup 不再必要），证明架构设计与优化器共同构成一个耦合系统，单独优化任一维度都会遗漏关键收益或引入隐性缺陷。

## 方法详解

### Token Mixing：GDN Hybrid Architecture
每四个 sublayer 块中包含 3 个 GDN 层 + 1 个 full-attention 层。GDN 使用 gated delta recurrence 将前缀压缩为固定大小的 recurrent state：

- **衰减门 α_t**：控制现有 state 的生存周期（全局缩放）；**写入门 β_t**：控制 delta 更新的强度。
- 核心递推公式：
  - S̃_{t-1} = α_t · S_{t-1}
  - e_t = v_t - S̃_{t-1}^⊤ k_t（残差误差估计）
  - S_t = S̃_{t-1} + β_t · k_t · e_t^⊤（只写入残差，避免无界累积）
  - y_t = S_t^⊤ q_t
- q/k 经过 short causal convolution + L2 normalization，v 经过 short causal convolution + SiLU。
- 输出门采用 sigmoid（而非原 GDN 的 SiLU），配合 zero-centered RMSNorm，实验显示更稳定。
- 全 attention 层保留 RoPE 位置编码（去掉位置编码在预训练 loss 上无差异，但 post-training 后 endless generation 风险显著升高）。

**QSA（持续预训练阶段替换 full attention 层）**：
- Indexer 采用 MQA 结构（4 query heads + 1 shared key head），key 序列以 r=4 压缩为 micro-block，经 AvgPool 后施加 partial RoPE（仅 64/128 维度），通过 block-causal scoring（ReLU 激活的 q-k 相似度求和）为每个 query 选出 top-K_B 个最重要块。
- 训练分两阶段：Stage 1（Dense Distillation，1000 steps，lr=1e-3，约 2B tokens）先用 teacher forcing 蒸馏 full-attention 的 softmax 分布到 indexer（max pooling 对齐 block 级）；Stage 2（8000 steps，lr=2.5e-5，约 200B tokens）联合训练 backbone 和 indexer。
- 在 1M context 下，QSA 相比 dense attention 在 prefill 提速 7.6×，decode 提速 4.9×（kernel 级）。

### Gated Residual (GR)
- 残差流扩宽为 n_r=4 条独立分支 R ∈ ℝ^{n_r×d}。
- **Read（elementwise gate）**：先对每条分支独立做 RMSNorm，再通过低秩瓶颈（rank r=d/8）从全部 4 条分支预测 elementwise sigmoid gate G ∈ ℝ^{n_r×d}，加权平均后得到块输入 x = (1/n_r) Σ_i G_i ⊙ R̂_i。
- **Write（per-branch scalar）**：从全部分支预测 per-branch scalar s ∈ ℝ^{n_r}（值域 [0,2] via 2·σ(·)），将块输出 y 按 s_i 写入各分支：R'_i = R_i + s_i · y。
- 与 HC/mHC 的本质区别：去掉了 H_res（n_r×n_r 的 branch mixing operator），将全部表达能力投入 read 的 elementwise gate；同时 GR 的 read 自身承担了 pre-norm 的功能，不再需要单独的 LayerNorm。
- 分析表明：GR 的一条分支专门用于跨层保留早期 GDN 输出（典型 skip ≈ 10.9 层），其余三条保持局部连接（median skip 1.2–3.5 层）；softmax attention 层是整合这些 long-range 信息的关键 hub。

### N-gram Embedding Layer
- 放置于 Layer 2（经消融验证浅层效果最佳，且可与 host prefetch 重叠）。
- 使用 multi-head hashing 进行 embedding lookup，通过 contextual gating mechanism 注入残差流。
- 固定 300 tokens per active parameter (TPP)；在固定总参数预算下，10× 词表规模（25% 参数占比）取得最低 loss；允许额外参数时，loss 随词表单调下降，但部分下游精度饱和甚至回退（loss-accuracy 不一致的典型例证）。中文基准（C-Eval、CMMLU）随词表扩大持续提升。
- Embedding 表存储在加速器外（host memory），通过异步 prefetching 与 Layer 1 计算重叠，per-token 额外计算和延迟可忽略。

### Optimizer：Muon + 工程适配
- Muon 应用于二维线性映射权重（attention q/k/v/out、GDN proj、MoE fc1/fc2、n-gram key proj）；input/output embedding、MoE router、GR 的低秩投影仍用 AdamW（router 维度相互独立无线性结构可正交化；GR 投影形状过于细长）。
- Newton–Schulz 迭代 8 步（采用 Polar Express 系数调度），数值稳定性常数 1e-14。
- **Fused parameter splitting**：Megatron 中将 qkv/swiglu fc1/GDN input 存为单一 fused 矩阵，正交化会跨子块混合奇异方向，因此需按 head 粒度拆分后独立做 NS 迭代再合并。
- **Canzona 框架**（Wang et al., 2026a）：解耦逻辑优化器分配与物理参数布局，α-balanced 静态分区器使 DP rank 间 NS FLOPs 均衡，异步 Micro-Group pipeline 通过 fused All-to-All 重建 Muon 所需矩阵。
- 分裂后单层约百个子矩阵导致 step 被大量小 kernel 主导，整体 step 用 CUDA graph 捕获消除 launch overhead。

### Hyperparameter Scaling Law
- 新 scaling law 预测：batch size 从 Qwen3.5 的 12.6M 提升至 25.2M（4T-token 预算下 loss 降低 7.2×10⁻³），学习率同步上移。
- **Batch-size warmup 被证明不必要**：逐步 ramp 从 6.3M 至 25.2M 不仅未改善 loss，且多消耗 18.8% optimizer step（墙钟时间增加）；原因是在相同 lr 下小 batch 引入更高梯度噪声，早期劣势在后期无法补偿。
- 新 scaling law 的预测 optimum 处于一个平坦的 bowl 底部（±√2 倍 lr 和 +25% batch 范围内性能无显著差异）。

### Training Stability
- Stress test 设计：在 28-layer 25B-A3B MoE 上以 2× 和 4× 最优 lr 恒定训练（跳过 lr decay），复现大规模训练中的不稳定模式。
- 结果：4× lr 下 AdamW 每 10k step 产生 183 次 loss spike 且频繁触发 clip（213/19932 step 越过阈值）；Muon+GR 零 spike、零 clip 穿越。
- GR 中 GatedNorm 提供显式 rescaling：使 3× lr 下 spike 率从 32.0/10k 降至 3.2/10k，clip 穿越从 256 降至 20。
- 得益于稳定性提升，全规模训练无需 qk-clip 或 SwiGLU-clip 等显式 activation control。

## 实验与结果

### 主要对比实验（Table 11）
| 模型 | 总参数 | 激活参数 | 关键优势 |
|------|--------|----------|----------|
| Qwen3.8-Flash-Next-Base | 125B | 6B | 本文目标模型 |
| Qwen3.8-27B-Base | 27B | 27B | 同代稠密基线 |
| Qwen3.7-Plus-Base | 397B | 17B | 前任旗舰 |

**与前任旗舰（397B-A17B）对比**：在 14 个 pre-training benchmark 中 **8 个领先**，其余 6 个最多落后 2.6 分；激活参数仅为前任的 ~1/3，训练 token 约 1/3，训练 FLOPs 约 1/9。

**具体关键数字**：
- MMLU: 90.36（优于 397B 的 90.43，差距 0.07）
- MMLU-Pro: 73.23 vs 70.90（+2.33）
- SuperGPQA: 51.36 vs 48.42（+2.94）
- BBH: 90.87 vs 89.41（+1.46）
- MATH: 72.78 vs 74.38（-1.60）
- GSM8K: 93.29 vs 92.95（+0.34）
- EvalPlus: 78.76 vs 78.06（+0.70）
- SWEBench-Pretrain: 50.99 vs 49.24（+1.75）
- MMMLU: 84.86 vs 84.53（+0.33）

**QSA 评估**（Table 2）：在 8 个短上下文基准中 7 个持平或超越 full attention，平均 76.8 vs 75.9（+0.9）。

**长上下文检索**（Table 3）：
- RULER @ 512K–1M: 93.00 vs 90.08（+2.92）
- MRCR @ 512K: 40.53 vs 30.66（+9.87）；@ 1M: 26.44 vs 20.71（+5.73）
- QSA 在更长上下文下反而表现更好。

**GDN 混合 vs 全 attention vs SWA 混合**（Table 1，28-layer 25B-A3B）：
- GDN hybrid 平均 53.81%，超过全 attention（49.87%）+3.94 个百分点，超过 SWA hybrid（51.15%）+2.66 个百分点。
- 在 7/9 基准上取得单列最高分。

**GR 消融**（Table 5，25B-A3B，560B tokens）：
- Pre-norm: avg 50.91 → mHC static: 52.49（+1.58）→ mHC dynamic: 54.47（+1.98）→ GR: 54.66（+0.19）
- Loss 从 1.617 降至 1.590，其中 static→dynamic 仅降 0.002 但 benchmark 提升 1.98 分，体现 loss-benchmark 不一致。

**FlashQLA 加速**：相比 Triton FLA baseline 实现 2–3× forward / ~2× backward 加速（NVIDIA GPU）。

## 相关工作脉络
1. **Gated Delta Networks (GDN)** (Yang et al., 2024)：线性注意力变体，通过 gated delta rule 实现内容感知的 recurrent state 更新；本文将其嵌入 Transformer 混合架构并配合 FlashQLA 高效内核。
2. **Hyper-Connections (HC/mHC)** (Zhu et al., 2024; Xie et al., 2025)：将残差流扩宽并通过 learnable operators 做 read/write/mix；本文 GR 与之同源，但将表达能力集中到 elementwise read gate 并去除 H_res，降低内存开销和稳定性风险。
3. **Sparse Attention (DSA/QSA lineage)** (Liu et al., 2025a)：用 lightweight indexer 生成 token-level 稀疏 mask；本文 QSA 的进步在于 micro-block 粒度的 compressed indexer，使索引开销本身随序列长度亚二次增长。
4. **N-gram Embedding / Lookup Tables** (Google DeepMind, 2025; Cheng et al., 2026; Liu et al., 2026)：将 n-gram 作为确定性 key 查表增强容量；本文系统研究了 placement、vocabulary scale 及 loss-accuracy 脱钩现象，确认其作为 off-accelerator 容量扩展的有效性。
5. **Muon Optimizer** (Jordan et al., 2024)：对二维权重做 Newton-Schulz 正交化的矩阵优化器；本文揭示了 fused parameter splitting 和 Canzona 分布式实现的必要性，并扩展了适用/不适用 Muon 的参数类型清单。
6. **Attention Residual (AttnRes)** (Kimi Team, 2026)：用 softmax attention 在跨层输出上做 read；本文 GR 在同等 28-layer 设置下达到相当 loss（1.762 vs 1.762 Full AttnRes），但以更低内存开销（无注意力计算）实现。

## 局限性与未来方向
- **Loss 与下游精度的不一致难以统一建模**：n-gram 词表扩大单调降低 loss 但部分 benchmark 饱和；去掉 attention 层的位置编码在预训练 loss 上无差异但影响 post-training 生成质量——说明现有预训练评估无法可靠预测最终能力排序。
- **N-gram 嵌入的容量扩展存在边际递减**：10× 词表在固定参数预算下最优，继续扩大（50×、100×、200×）虽持续降 loss 但部分下游任务波动或饱和。
- **稀疏写入（sparse write）在 post-training 后退化**：仅读 top-2 分支在 pre-training loss 上几乎无影响，但后续训练质量下降，暴露预训练指标对后期行为的预测盲区。
- **长上下文检索仍依赖特定 sparse pattern**：QSA 在 512K+ 显著提升 RULER/MRCR，但在极短上下文下优势不明显，且 indexer 质量完全依赖 distillation 阶段的教学信号。
- **未来方向**：作者明确指出最紧的瓶颈是评估吞吐量——需要一个更便宜的 mid-scale probe 来可靠预测 post-training 排序，这将使架构搜索空间大幅扩大。

## 研究启发与可借鉴点
1. **多轴联合评估方法论**：任何架构改动必须同时从 loss+benchmark、训练/prefill/decode 成本、超参数稳定性和训练稳定性三个维度评估；单一维度的"无害改动"可能在另一维度引入隐性缺陷（如 sparse write、no-RoPE）。建议在本团队研究中复现此三轴评估框架。
2. **Gated Norm 作为通用稳定性组件**：GR 中的 GatedNorm（elementwise sigmoid gate + low-rank bottleneck）在 stress test 中将 spike 率降低 10 倍；此组件可迁移至任何深层 Transformer 架构的训练稳定性优化。
3. **Fused parameter splitting for Muon**：将 Megatron 中的 fused qkv/swiglu 矩阵按 head/语义子块拆分后再做 NS 正交化，既避免跨子块奇异方向混淆，又为 per-submodule optimizer 选择提供粒度；这一工程技巧可直接应用于其他使用 Muon 的大模型训练。
4. **Batch-size warmup 并非普遍有益**：在 Muon + 大 batch 稳定场景下，warmup 不仅无收益反而增加 18.8% optimizer step；挑战了"大 batch warmup 是标准实践"的默认假设，建议根据优化器特性重新审视 warmup 策略。
5. **N-gram embedding 的 placement 敏感性与 host prefetch 策略**：单层置于 Layer 2 即足够，且可与第一层计算重叠——为低资源场景下的容量扩展提供了低成本方案；可探索将此思路与 LoRA/adapter 等方法结合。

## 关键术语表
- **Gated DeltaNet (GDN)**：一种线性注意力变体，通过数据依赖的衰减门和 delta 写入门将前缀压缩为固定大小 recurrent state，实现 O(n) 复杂度的 token mixing。
- **Gated Residual (GR)**：将残差流扩宽为 4 条独立分支，通过 elementwise sigmoid gate 从全部分支读取并做 per-branch scalar 写入，兼具容量扩展与训练稳定功能。
- **Qwen Sparse Attention (QSA)**：在 micro-block 粒度上用 compressed lightweight indexer 评估上下文重要性并选择 top-k 块进行 sparse attention，索引开销随序列长度亚二次增长。
- **N-gram Embedding**：以短 n-gram 为 key 查表获取 embedding 向量，通过 contextual gating 注入残差流；确定性寻址使其可存于 host memory 并通过 prefetching 零延迟访问。
- **Muon Optimizer**：对二维权重矩阵用 Newton-Schulz 迭代做 Nesterov 动量正交化的优化器，更新方向经形状无关缩放，适合大规模线性层训练。
- **FlashQLA**：基于 TileLang 的融合线性注意力内核库，为 GDN 等 recurrent attention 提供 2–3× forward / ~2× backward 加速。
- **Canzona**：解耦逻辑优化器分配与物理参数布局的分布式框架，通过 α-balanced 静态分区和异步 Micro-Group pipeline 支持 Muon 等矩阵优化器的高效多卡训练。
- **Stress Test**：通过固定高学习率（2×/4× 最优值）并在中小规模模型上训练，放大大规模训练中才会出现的 loss spike 和梯度异常，以低成本验证架构稳定性。

## 可复现要素
- **数据集**：评估使用 MMLU、MMLU-Pro、SuperGPQA、MATH、GSM8K、BBH、MMMLU、EvalPlus、MultiPL-E、RULER、MRCR、GPQA、MGSM、INCLUDE、SWEBench-Pretrain（论文未提及训练数据细节，仅说明 QSA CPT 使用 256K context、共约 200B tokens）
- **代码开源**：FlashQLA 内核已开源（https://github.com/QwenLM/FlashQLA）；Canzona 框架有技术报告（Wang et al., 2026a）
- **模型权重**：Qwen3.8-Flash-Next 系列模型通过 qwen.ai 发布（论文引用 Qwen Team, 2026）
- **关键超参数**：
  - GDN: α_t (exp-of-exp 参数化), β_t (sigmoid), short causal conv, L2 norm on q/k
  - QSA: r=4（压缩比）, K=2048（token budget）, K_B=512（block budget）, 4 query heads + 1 shared key head, partial RoPE (64/128 dims)
  - GR: n_r=4 branches, bottleneck rank r=d/8, sigmoid gate, write scalar range [0,2]
  - Muon: NS iteration=8 steps, Polar Express coefficient schedule, γ(A,B)=0.2√max(A,B), μ=0.95
  - Batch size (predicted optimum): 25.2M (4T-token budget), 8.4M (419B-token budget)
  - Learning rate: 1.76×10⁻³ (419B run)
  - Gradient clipping threshold: 0.5
  - N-gram: 300 TPP, Layer 2 placement
