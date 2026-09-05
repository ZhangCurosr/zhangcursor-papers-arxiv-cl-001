---
title: "On-the-Design-of-Qwen3-8-Next-Architecture-Evaluation-Eficie"
source: https://arxiv.org/pdf/2608.30320v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:26:57"
field: "大语言模型架构设计"
keywords: ["Sparse Mixture-of-Experts", "Gated DeltaNet", "Sparse Attention", "Gated Residual", "Muon Optimizer", "N-gram Embedding", "Training Stability", "Scaling Law"]
innovations: ["提出GDN与全局注意力的层級混合架构，在每4层中3层GDN+1层全局注意力，兼顾效率与长程信息保留", "设计Gated Residual（GR），将残差流加宽至4分支并用elementwise sigmoid门控读取，在提升训练稳定性的同时降低内存带宽开销", "提出Qwen Sparse Attention（QSA），在micro-block粒度上用压缩轻量级indexer实现O(n²/r)索引开销，1M上下文prefill加速7.6×"]
benchmarks: ["MMLU", "MMLU-Pro", "SuperGPQA", "MATH", "GSM8K", "BBH", "MMMLU", "EvalPlus", "MultiPL-E", "RULER", "MRCR", "GPQA", "SWEBench-Pretrain", "MGSM", "INCLUDE"]
---

# 论文速读：On-the-Design-of-Qwen3.8-Next-Architecture-Evaluation-Eficiency

## 一句话总结
本文介绍了 **Qwen3.8-Flash-Next**（125B 总参数、每 token 激活 6B 参数 + 51B n-gram 嵌入表）的架构设计与系统评估，通过 GDN 混合层、Qwen Sparse Attention、Gated Residual 和 n-gram 嵌入四大组件，在约 **1/9 训练 FLOPs** 和 **1/3 激活参数** 下达到了与上代旗舰 Qwen3.8-397B-A17B 相当甚至更强的能力。

---

## 研究问题与动机
1. **核心问题**：如何在大幅削减训练成本（参数、token、FLOPs）的同时，保持甚至超越上一代 397B 旗舰模型的能力。
2. **现有方法的不足**：
   - 全局注意力计算复杂度为 $O(n^2)$，KV cache 随序列长度线性增长，长上下文场景不可扩展。
   - 滑动窗口注意力（SWA）无法直接访问远距离 token，信息跨层传播效率低。
   - 标准残差连接中早期写入的特征与后续写入的特征存在竞争，信号逐层衰减。
   - 传统超参数配方（学习率、batch size）针对上一代架构与 AdamW 设计，在新架构下并非最优。
3. **设计目标耦合性**：架构改动同时影响 downstream 能力、训练/推理开销、以及最优超参数与训练稳定性，必须**联合优化 loss、benchmark、效率、稳定性四个维度**。
4. **Loss 与 Benchmark 的不一致性**：loss 最低的配置不一定对应最佳下游性能（如 n-gram 词汇扩大时 loss 单调下降但 benchmark 饱和），需多维度联合评估。

---

## 核心贡献（创新点）
1. **GDN 混合层架构（GDN Hybrid）**：每 4 层中 3 层使用门控 DeltaNet（GDN）压缩前缀为固定大小状态，1 层使用全局注意力保留直接 token 级检索；相比 SWA 混合在 9 个 benchmark 中占优 7 个，相比全注意力占优 8 个。
2. **Gated Residual（GR）**：将残差流加宽至 4 条分支，通过元素级 sigmoid 门控读取（elementwise gate）而非逐分支标量读取，同时省略了 branch-mixing 算子 $H_{\mathrm{res}}$；牺牲少量内存带宽换取更高表达能力与显著的训练稳定性提升。
3. **Qwen Sparse Attention（QSA）**：采用压缩轻量级 indexer（micro-block 粒度 + 部分 RoPE + block-causal scoring），将索引成本从 $O(n^2)$ 降至 $O(n^2/r)$；在 1M 上下文下 prefill 加速 **7.6×**、decode 加速 **4.9×**，长上下文 RULER 从 90.08 提升至 **93.00**。
4. **N-gram 嵌入外置扩展**：51B 参数的 n-gram 嵌入表驻留 host 内存并通过预取机制透明接入，以几乎零额外 per-token FLOPs 扩展模型容量；loss 随词汇规模单调下降，中文 benchmark（C-Eval、CMMLU）持续改善。
5. **Muon 优化器 + 新 Scaling Law 的稳定训练配方**：结合 GDN + GR 的新架构将最优学习率和 batch size 同时上移，**消除了 batch-size warmup 的必要性**，且在 4× 最优学习率压力测试下零 loss spike，实现了无需显式 clipping（qk-clip/SwiGLU-clip）的稳定训练。

---

## 方法详解

### 2.1 Token Mixing：GDN 混合架构
**动机**：全局自注意力有 $O(n^2)$ 计算瓶颈和线性增长的 KV cache；滑动窗口注意力缺乏全局信息访问。GDN 以线性代价将前缀压缩为固定大小状态，通过门控 delta 规则实现内容驱动的写入/擦除。

**Gated Delta Recurrence（公式 1–5）**：
- 状态更新：$\widetilde{S}_{t-1} = \alpha_t S_{t-1}$，残差误差 $e_t = v_t - \widetilde{S}_{t-1}^\top k_t$，状态写入 $S_t = \widetilde{S}_{t-1} + \beta_t k_t e_t^\top$
- 输出：$y_t = S_t^\top q_t$
- $\alpha_t$ 控制状态全局寿命，$\beta_t$ 控制 delta 写入强度；重复/相似 key 只更新残差而非累积，区别于纯加性线性注意力。

**GDN 参数化（公式 6–11）**：
- $q_t, k_t$ 经过 ShortConv → SiLU → L2Norm，提供局部归纳偏置并稳定 rank-one delta 转移
- $\beta_t = \sigma(W_\beta x_t)$，$\alpha_t$ 采用 double-exp 参数化保证 $(0,1)$ 范围
- 输出门采用 **sigmoid**（非原文 SiLU），配合 zero-centered RMSNorm

**混合策略**：每 4 层中 1 层使用全局注意力（保留 RoPE），3 层使用 GDN；全注意力层在继续预训练阶段替换为 QSA。

### 2.1.2 Qwen Sparse Attention（QSA）
**动机**：DSA 等稀疏注意力方法在 $n$ 增大时 $O(n^2)$ indexer 开销不可忽视。

**压缩轻量级 Indexer（公式 12–16）**：
- MQA 结构（4 query heads + 1 shared key head）
- Key 按 micro-block（大小 $r=4$）平均池化压缩，压缩后再施加部分 RoPE（64/128 维）
- Block-causal scoring：$I_{ib} = \sum_h \mathrm{ReLU}(\langle q_i^h, \bar{k}_b \rangle)$，仅对已完整观察的 block 计分
- Top-$K_B$ 选择后展开为 token 级 mask，$K=2048, r=4 \Rightarrow$ 选 512 个 block

**两阶段训练**：
- Stage 1（Dense Distillation）：用全序列 attention 分布对 indexer 做 $L_1$ 归一化后 MaxPool 对齐 block 级别，KL 散度蒸馏 1000 步（~2B tokens）
- Stage 2（Sparse Training）：联合训练 backbone 和 indexer 8000 步（~200B tokens），KL loss 仅在所有 top-$K_B$ block 内计算

**高效实现**：融合 kernel 同时计算 sparse attention 输出和 KL loss；多步 MTP 复用 top-k 索引。

### 2.2 Gated Residual（GR）
**动机**：Pre-norm 逐层衰减信号；Alternating Updates / Hyper-Connections 通过加宽残差流提升容量。

**设计消融要点**：
- Sigmoid 门优于 tanh（loss + 稳定性双优）
- 数据依赖的门控比静态门增益更大（benchmark 提升 1.98 分 vs 静态 1.58 分），但 loss 仅差 0.002——**loss 与 benchmark 分化典型案例**
- 读取粒度比写入粒度更重要：$H_{\mathrm{mix}}$ 细化到 per-channel 有效，$H_{\mathrm{combine}}$ 细化收益可忽略
- $H_{\mathrm{res}}$（branch mixing 算子）几乎无贡献，移除后节省一次全残差流读取

**GR 具体设计（公式 29–34）**：
- 每条分支独立 Group RMSNorm：$\widehat{R}_i = \mathrm{RMSNorm}(R_i; \gamma_i)$
- 元素级门控读取（融合 GatedNorm）：$G = \mathrm{unvec}\,\sigma(W_u \mathrm{SiLU}(\frac{1}{n_r} W_d \mathrm{vec}(\widehat{R})))$，输入 $x = \frac{1}{n_r}\sum_i G_i \odot \widehat{R}_i$
- 写入门控：$s = 2\sigma(\frac{1}{n_r} W_w \mathrm{vec}(\widehat{R}))$，$R_i' = R_i + s_i y$
- $n_r = 4$ 条分支，attention 和 MLP 子层各一个 GR 模块

**Branch 功能分解**：一条分支专门保存早期（Layer 0）输出形成 **长距离路径**（平均跳过 10.9 层），其余三条保持局部（1.2–3.5 层）；softmax 注意力层是长距离信息汇聚的关键 hub。

**推理优化**：FP8 存储残差状态（体积减半），read/write kernel 各自融合为单次遍历。

### 2.3 N-gram Embedding
- 多层放置位置消融：单层 Layer 2 效果最好（可与第一层计算重叠 prefetching）
- 固定参数预算下 10× 词汇量最优（loss 最低），但下游 benchmark 饱和
- 额外参数扩展下：loss 单调下降，中文 benchmark 持续改善，证明 n-gram 与 MoE expert 扮演不同容量角色

### 3.1 Muon 优化器
- 对 2D 线性投影权重应用 Muon（NS 迭代 8 步，Polar Express 系数调度）
- Input/output embedding、MoE router、GR 低秩投影继续使用 AdamW
- **Splitting fused parameters**：将 fused qkv/fc1/GDN 投影拆分为独立子矩阵分别正交化，避免混合不相关子块的奇异方向
- 实现挑战：NS 迭代需全矩阵操作，通过 **Canzona** 框架解耦逻辑分配与物理布局（α-balanced 静态分区 + 异步 Micro-Group All-to-All）；CUDA graph 捕获整个 step 消除小 kernel 开销

### 3.2 超参数 Scaling Law
- 新配方预测更大 batch size 和更高学习率
- Batch size 从 12.6M → 25.2M 带来 $7.2 \times 10^{-3}$ loss 提升
- **Batch-size warmup 不必要**：与直接从目标 batch size 开始相比性能无提升，且多花费 18.8% 优化步数
- 学习率优化 landscape 非常平坦（±$\sqrt{2}$ 范围内 loss 差异在噪声水平）

### 3.3 训练稳定性压力测试
- 设计：恒定 LR = 2×/4× 最优值（跳过 decay schedule）以复现大规模训练的峰值 LR 压力
- 结果：4× LR 下 AdamW 基线每 10k 步 183 次 spike，Muon+GR 零 spike；pre-clip gradient norm 仅为 AdamW 的一半以下
- **Gate 的作用机制**：高 LR 下网络通过增大 activation outlier 隐式完成 rescaling，显式 gate 直接提供 rescaling，消除 outlier 增长

---

## 实验与结果

**评估基准（14 项）**：MMLU、MMLU-Pro、MMLU-Redux、BBH、SuperGPQA、GPQA、GSM8K、MATH、EvalPlus、MultiPL-E、SWEBench-Pretrain、MGSM、MMMLU、INCLUDE。

**主结果（Table 11）**：
| 模型 | 总参数 | 激活参数 | 对比基线表现 |
|---|---|---|---|
| Qwen3.8-Flash-Next-Base | 125B | 6B | 14 项中 8 项超过 Qwen3.7-Plus-Base（397B/17B 激活） |
| Qwen3.8-27B-Base | 27B | 27B | Flash-Next 全面超越 |

**关键数字**：
- MMLU: 90.36（vs Qwen3.7-Plus 90.43，接近持平）
- MATH: 72.78（vs Qwen3.8-27B 60.54，大幅提升）
- EvalPlus: 78.76（vs Qwen3.8-27B 76.05）
- SWEBench-Pretrain: 50.99（vs Qwen3.7-Plus 49.24）

**QSA 效率（Table 3, 6）**：
- RULER 1M 上下文：93.00（vs 全注意力 90.08）
- MRCR 512K：40.53（vs 30.66）；1M：26.44（vs 20.71）
- 1M 上下文 kernel 级加速：prefill **7.6×**，decode **4.9×**

**GDN vs SWA vs Full Attention（Table 1）**：
- GDN hybrid 平均 53.81%，优于 SWA hybrid 51.15% 和 Full attention 49.87%

**训练成本**：激活参数为前代的 **1/3**，训练 token 为 **1/3**，训练 FLOPs 约为 **1/9**。

---

## 相关工作脉络
1. **Gated DeltaNet (GDN, Yang et al., 2024)**：线性注意力变体，维护固定大小 state 实现内容驱动写入；本文在此基础上引入 layer-wise 混合和工程优化（FlashQLA kernel）。
2. **Hyper-Connections (HC, Zhu et al., 2024) / mHC (Xie et al., 2025)**：通过三算子（read/write/mix）扩展残差流；本文 GR 与之同源但将表达能力集中在 elementwise read gate 并省略 $H_{\mathrm{res}}$。
3. **Qwen Sparse Attention 对标 DSA (Liu et al., 2025a)**：DSA 使用 token 级 sparse mask；QSA 改进为 micro-block 级索引，索引成本随序列长度增长而降低。
4. **N-gram Embedding (Roy et al., 2022; Google DeepMind, 2025; Cheng et al., 2026)**：利用 n-gram 做 lookup-based memory；本文系统研究了 placement、vocab size 与 MoE 的参数分配关系。
5. **Muon Optimizer (Jordan et al., 2024)**：对 2D 权重做 Newton-Schulz 正交化；本文将其引入 LLM 训练并解决分布式实现挑战（Canzona 框架）。
6. **Attention Residual (Kimi Team, 2026)**：使用 softmax attention 跨层读取；本文 GR 在同等 loss 下以更低的内存 traffic 实现相似效果。

---

## 局限性与未来方向
1. **Sparse write 在 post-training 退化**：仅读取最高门控的两条分支在预训练 loss 上几乎无损失，但后续训练/后训练质量下降，说明预训练指标不足以判断此类改动。
2. **N-gram 下游饱和**：vocabulary 扩大时 loss 单调下降但下游 accuracy 饱和，参数的有效利用率有待进一步挖掘。
3. **Evaluation throughput 成为瓶颈**：作者指出"更便宜的中等规模 probe 若能可靠预测后训练排序"将大幅提升设计空间搜索效率，这是当前验证流程的主要短板。
4. **未探索的方向**：larger $n_r$（如 xHC 所做）因内存开销未深入；token normalization / 非均匀分配等 n-gram 效率优化策略未见显著提升。

---

## 研究启发与可借鉴点
1. **四维联合评估范式**：将 loss + benchmark + efficiency + stability 作为统一设计目标，并在每个组件上做消融，避免单一指标的误导（如 sparse write 在 pretrain 看似无害但 posttrain 退化）。
2. **Gated Residual 的可迁移设计**：elementwise sigmoid gate + 加宽残差流 + FP8 存储的组合可同时提升能力和稳定性，值得在其他架构（RWKV、Mamba 等）中验证。
3. **QSA 的 micro-block 索引思想**：将索引粒度从 token 级降为 block 级使索引成本随序列长度亚二次增长，对长上下文 LLM 和 agent 系统有直接应用价值。
4. **Muon + 大 batch 的稳定训练配方**：证明了 Muon 可以安全地配合更大的 batch size 和更高的学习率，且 batch-size warmup 在 Muon 下非必需，为训练效率优化提供了新参考。
5. **压力测试作为大规模训练的代理**：通过固定高 LR（2×/4×）在小规模模型上复现大规模训练的不稳定性，是一种高效的设计验证策略。

---

## 关键术语表
**GDN (Gated DeltaNet)**：一种线性注意力变体，通过 decay gate 和 delta write gate 将前缀信息压缩为固定大小的 state，以线性复杂度实现内容驱动的 long-range 依赖建模。

**QSA (Qwen Sparse Attention)**：Qwen 提出的稀疏注意力机制，在 micro-block 粒度上用压缩轻量级 indexer 评估上下文重要性，将索引开销从 $O(n^2)$ 降至 $O(n^2/r)$，适用于超长上下文推理。

**GR (Gated Residual)**：将残差流加宽至多条分支并用元素级 sigmoid 门控读取的设计，在添加容量的同时提供训练稳定性所需的 rescaling 能力。

**Muon Optimizer**：对 2D 权重矩阵应用 Newton-Schulz 迭代进行正交化 momentum 的优化器，据称在大模型训练中比 AdamW 具有更优的训练效率和稳定性。

**N-gram Embedding**：用短 n-gram 序列作为键查表获取嵌入向量，并将其通过与上下文相关的门控机制注入残差流，以外挂方式扩展模型容量。

**Scaling Law (超参数)**：描述最优学习率、batch size 等超参数随模型规模变化的经验规律；本文针对新架构重新拟合了 scaling law。

**Stress Test**：通过在高 LR（2×/4× 最优值）下保持恒定学习率的方式，在中等规模模型上放大训练不稳定性，用于快速验证架构改动的大规模可扩展性。

**FlashQLA**：基于 TileLang 的融合线性注意力 kernel 库，相比 FLA Triton kernel 实现 2–3× forward 和 ~2× backward 加速。

---

## 可复现要素
- **数据集**：标准预训练语料（具体组成论文未详细说明）；评测基准包括 MMLU、MMLU-Pro、SuperGPQA、MATH、GSM8K、BBH、MMMLU、EvalPlus、MultiPL-E、RULER、MRCR 等
- **代码**：FlashQLA 已开源（https://github.com/QwenLM/FlashQLA）；Canzona 框架见 arXiv:2602.06079
- **权重**：Qwen3.8-Flash-Next 模型权重公开（via Qwen 官方渠道）
- **关键超参**：$n_r = 4$ 条残差分支；QSA 压缩比 $r=4$、token budget $K=2048$、indexer query heads=4；NS 迭代 8 步；$\mu=0.95$（Nesterov momentum）；gradient clip threshold=0.5；Newton-Schulz 数值稳定性常数 $10^{-14}$；GDN ShortConv 局部感受野；0-centered RMSNorm

---
