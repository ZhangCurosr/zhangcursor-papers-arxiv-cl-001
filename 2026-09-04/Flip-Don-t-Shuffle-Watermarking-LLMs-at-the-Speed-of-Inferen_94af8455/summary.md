---
title: "Flip-Don-t-Shuffle-Watermarking-LLMs-at-the-Speed-of-Inferen"
source: https://arxiv.org/pdf/2609.03844v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:18:39"
field: "大语言模型可追溯性与内容认证"
keywords: ["LLM watermarking", "statistical watermark", "stateless Bernoulli", "CBRNG", "fused kernel", "self-salt", "Jenkins hash", "vLLM"]
innovations: ["以逐token伯努利试验替代KGW全局置换，实现O(1)单内核融合水印", "首次在生产规模支持全词表self-salt，延迟较KGW低6000×", "识别哈希函数设计为影响水印零假设校准与文本多样性的独立因素"]
benchmarks: ["C4 validation", "Qwen3-8B generation", "ROC-AUC", "perplexity degradation", "vLLM serving latency"]
---

# 论文速读：Flip, Don't Shuffle: Watermarking LLMs at the Speed of Inference

## 一句话总结
本文提出 Stateless Bernoulli Watermarking (SBW)，一种将统计水印的绿色列表构造从全局词汇表置换/锦标赛改为独立逐token伯努利试验的方法，使水印过程降低至 O(1) 单次比较，支持单内核融合执行与零中间分配，同时保持与 KGW 相同的检测统计保证。

## 研究问题与动机
- 现有统计水印方法（KGW、SynthID）在推理阶段存在显著的"水印税"：KGW 需要 vocab 级别的 `randperm`（O(V log V)），SynthID 需要 m=30 次顺序重加权；随着词汇表超过 100K 且并发规模扩大，这些方法会产生不可接受的延迟与内存开销。
- KGW 的 `randperm` 包含不可融合的排序操作，必须物化 `(B, V)` 索引张量并分别执行 mask 与偏置，导致 GPU 内存与 KV-cache 竞争。
- SynthID 虽可用 `torch.compile` 融合，但每步仍做 O(k·m) 计算（k=40, m=30），并在 top-k 近似下无法对全词表施加水印。
- 现有 self-salt 方案（KGW）仅对 top-40 候选施加候选相关种子，剩余词表无偏置，在高熵上下文中削弱水印信号；SynthID 则完全不支持 self-salt。
- 缺乏对哈希函数设计这一维度对水印质量影响的系统性研究——KGW 使用预生成置换表，其较小输出范围会引入绿列表分配的自相关性。

## 核心贡献（创新点）
- **伯努利绿列表替代全局置换**：用 `CBRNG(v, r_t) < γ` 的逐 token 独立试验取代 KGW 的 `randperm(V)`，实现 per-token O(1) 复杂度和单内核融合，与 KGW 保留相同的 z-score 零分布。
- **首次在生产规模实现全词表 self-salt**：SBW 将每候选成本降至 O(1)，使对整个 151K 词表的候选相关种子（self-salt）变得可行，比 KGW self-salt（top-40）快 6000×、比 SynthID 快 2×，同时零额外内存分配。
- **识别哈希函数为水印质量的新轴**：提出用 GPU 原生 Jenkins hash 替换 KGW 的置换表 hash，使零假设校准改善 1.8×、负均值偏差缩小 2.3×，并在高 δ 下维持更高的文本多样性。
- **提供 vLLM 生产级实现**：给出集成到 `vLLM` logits processor 的开源实现（`pip install sbw`），端到端生成开销在所有 batch size 下均低于 1%。
- **理论证明与经验验证统计等价**：严格证明 Bernoulli 绿列表的每 token 绿标指示变量 Y_t 满足 E[Y_t]=γ 与 Var(Y_t)=γ(1−γ)，z-score 在 H₀ 下保持 N(0,1)；跨 8 组 (γ, δ) 配置与两种 seeding 方案的 ROC-AUC 差异均 < 0.01。

## 方法详解
- **CBRNG 驱动的绿色列表构造**：对每个生成步 t，由前序上下文经 PRF 导出种子 r_t；每个 token v ∈ V 独立地通过 `CBRNG(v, r_t) < γ` 决定是否进入绿列表 G_t，期望大小 E[|G_t|] = γ|V|（式 1）。
- **Logit 偏置与 KGW 相同**：若 v ∈ G_t 则 l'_{t,v} = l_{t,v} + δ，否则 l'_{t,v} = l_{t,v}（式 2），采样过程不变。
- **Philox 4x32-10 CBRNG**：采用具备强统计性质与原生 GPU 支持的计数器型 RNG，给定种子与位置可在 O(1) 直接计算该位置随机值，无需依次生成前置值，使绿列表成员判定成为无依赖的并行点运算。
- **Jenkins 整数哈希**：对 selfhash/minhash/skipgram 等基于 token ID 的 seeding 方案，使用 Bob Jenkins integer hash（1997）替代 KGW 的置换表；Jenkins hash 可编译为纯 ALU 指令、避免散列内存访问造成 GPU cache thrashing，且输出范围更大，改善绿列表分配的均匀性。
- **单内核融合与零中间分配**：由于每个 token 的绿/蓝判定相互独立，水印可表达为单个 pointwise tensor 表达式，经 `torch.compile(mode="max-autotune")` 编译为一条 fused CUDA kernel，原地修改 logits，无需额外张量。
- **检测时 O(1)**：对任意 token 只需一次 CBRNG 评估即可判定是否绿，不必重建完整 G_t；z-score 仍使用标准 KGW 公式 z = (W − Tγ) / √(Tγ(1−γ))，在 H₀ 下服从 N(0,1)（式 5）。
- **理论等价性证明要点**（Proposition 1 & Appendix H）：利用全期望定律得 E[Y_t]=E[Γ_t]=γ；利用全方差定律分解 Var(Y_t)=E[Γ_t(1−Γ_t)]+Var(Γ_t)，代入 E[Γ_t²]=Var(Γ_t)+γ² 后 Var(Γ_t) 项恰好消去，得到 Var(Y_t)=γ(1−γ)，与固定大小绿列表完全一致。
- **集中界限**：Hoeffding 不等式表明 |Γ_t−γ|>1% 的概率 < 2×10^{−13}（|V|=151,936, γ=0.5），绿色列表比例波动可忽略。

## 实验与结果
- **数据集与模型**：生成模型 Qwen3-8B（vocab 151,936），困惑度评估用 Qwen3.5-27B；提示取自 C4 validation split（500 条，截断至 30 tokens）；GPU 为 NVIDIA RTX 3090，temperature=1.0（纯 sampling，无 top-p）。
- **参数网格**：γ ∈ {0.25, 0.50}，δ ∈ {1, 2, 5, 10}；两种 seeding：context-only（simple_1）与 self-salt（selfhash，上下文宽 4 + 候选 token）。
- **Z-score 分布等价**：非 self-salt 场景下，SBW-1 与 simple_1 的两样本 KS 检验 p=0.14（γ=0.25）与 p=0.002（γ=0.50），最大 CDF 差异 <5pp；watermarked 状态下 8 组中有 7 组 t-test 不拒绝均值相等。self-salt 场景下 SBW-ss 因 Jenkins hash 零假设校准更优（KS 相对 selfhash 改善 1.74×/1.60×）。
- **检测准确率**：所有配置的 ROC-AUC 差异 < 0.01；高 δ 下两者均达到 ≥ 0.999 的近乎完美区分。
- **文本质量（PPL 退化）**：非 self-salt 时 KGW 与 SBW-1 的 ΔPPL 几乎一致；self-salt 在高 δ≥5 时 SBW-ss 的 ΔPPL 略高，但伴随更高的文本多样性（δ=10 时 diversity 0.62 vs 0.40，baseline 0.70）。
- **端到端延迟**：SBW 在全部 batch size（16–256）下引入 <1% 额外开销（9–76 ms，相比 3.4–14.5 s 总生成时间）；SynthID（top-k=40）为 0.6–4.4%；B=64 时 SBW 21 ms vs SynthID 136 ms（6.5×）。KGW selfhash (top-40) 在 B=1 时已高出 57%，B=256 时 +7.0s（221%）。
- **隔离 kernel 延迟**：SBW 比 SynthID 低 2–3×，比 KGW 低 6000×+；SBW 峰值额外显存为 0 MB，SynthID 在 B=512 时达 3.8 MB。
- **vLLM 生产部署**：≥32 并发请求下开销约 1%（B=64: 1.1%，B=128: 1.3%）。
- **鲁棒性（19 种攻击）**：在 character/word/structural/semantic 攻击下 SBW 与 KGW 在 11/19 配置分布等价；其余 8 配置中 SBW 保留更多水印信号（更高 z），归因于 Jenkins hash 降低相邻 token 绿列表分配的自相关。

## 相关工作脉络
- **KGW (Kirchenbauer et al., 2023)**：开创性统计水印，基于 PRF 种子与固定大小绿列表；SBW 在检测统计上等价，但以伯努利试验替代置换，实现 O(1) 单内核并支持全词表 self-salt。
- **SynthID-Text (Dathathri et al., 2024, Google)**：工业级部署的多层 tournament 水印（m=30 次重加权）；SBW 消除顺序重加权，延迟降低 2×，且首次在 prod 规模支持 self-salt。
- **Kuditipudi et al., 2024 (RDW)**：无损分布水印；SBW 保留 δ 偏置但实践中 PPL 影响极小（δ≤2），优势在于 kernel 融合与 zero-allocation。
- **Christ et al., 2024; Wu et al., 2024**：不可检测/鲁棒分布保留水印；SBW 属"可检测统计水印"阵营，侧重推理效率而非理论不可检测性。
- **Lai et al., 2025; Zhang & Koushanfar, 2024**：对抗训练与后处理（同义词替换、语义插入）水印；SBW 聚焦推理阶段统计方法，属于同一技术路线的效率优化分支。
- **Jovanović et al., 2024**：API 访问下的水印方案窃取；SBW 安全性建立在 CBRNG 逆推与 KGW PRF 等价的假设上，未做专门抗 spoofing 分析（论文自述为局限）。
- **Liu & Bu, 2024; Cai et al., 2026**：自适应 γ/δ 与统计框架；SBW 为后续自适应策略提供 O(1) 执行基础，可将 token 熵感知调度与状态less 结构结合。

## 局限性与未来方向
- 主要实验仅在一个 GPU（RTX 3090）与单一模型（Qwen3-8B）上完成；虽有 Falcon-7B/A6000 泛化验证（Appendix F），但跨模型族的大规模评估仍需补充。
- 鲁棒性仅覆盖 19 类非自适应攻击（字符/词删除、截断、paraphrase、MLM substitution）；自适应水印去除策略与 spoofing 攻击未系统评估（Appendix G 自述）。
- 当前依赖 `torch.compile` 生成的 Triton fused kernel；手写 CUDA kernel（显式内存合并、warp-level 优化）有望进一步降低小 batch 下的 latency。
- 状态less 架构理论上兼容分布式/张量并行推理（各设备基于共享密钥独立计算相同绿列表），但未在 tensor-parallel 或 pipeline-parallel 部署上实证。
- 论文承认 SynthID 可证明保留输出分布（non-distortionary），而 SBW 引入 δ 偏置；尽管实践中 PPL 影响微小（δ≤2），严格无失真性质未获证明。
- 哈希函数作为新轴的价值尚未在更多 seeding 方案、tokenizer 变体与多语言场景下验证。

## 研究启发与可借鉴点
- **将全局操作转化为局部独立判定**：KGW 的 `randperm` 瓶颈可通过 CBRNG + 阈值比较被重写为 pointwise 运算，这一"状态less + 单步判定"范式可推广至其他需要按上下文决定集合成员的推理优化场景。
- **哈希函数作为可优化设计变量**：论文以控制实验（SBW-ss-cpu）分离了 green-list 构造与 hash 函数的影响，揭示 Jenkins hash 的大输出范围与 ALU-only 特性可同时改善零假设校准与文本多样性；后续工作可在不同 seeding 上系统搜索 hash/PRNG 组合。
- **生产级消融实验设计**：用 SBW-ss-cpu（GPU Bernoulli + CPU perm hash）作为对照，严格归因分布差异来自 hash 而非伯努利构造；这种"单因子控制"策略值得在 watermark/decoder 效率论文中复用。
- **与 vLLM logits processor 生态对接**：SBW 作为纯 post-processing kernel 嵌入 `transformers.generate()` 与 vLLM，不对模型权重/前向计算作侵入式修改，为后续研究提供低摩擦的部署模板。
- **全词表 self-salt 的可行性重新评估**：SBW 证明当每候选代价降至 O(1) 时，放弃 top-k 近似反而能获得更强水印信号与更好检测校准；后续可探索"自适应 k"或"按 token 熵分配 δ"的全词表方案。

## 关键术语表
- **Stateless Bernoulli Watermarking (SBW)**：以独立逐 token 伯努利试验构造绿列表的统计水印，摒弃全局置换/锦标赛，实现 O(1) 单次比较与单内核融合。
- **CBRNG (Counter-Based Random Number Generator)**：给定种子与位置可在 O(1) 直接产出该位置随机值的 RNG；本文使用 Philox 4x32-10，支持 GPU 并行无依赖求值。
- **Green list / blue list**：KGW/SBW 将词表按伪随机规则划分为加偏置的"绿色"子集与不加偏置的"蓝色"子集，水印信号编码于绿色 token 的出现频率。
- **Self-salt seeding**：令候选 token 自身参与种子派生的 seeding 策略（如 selfhash），提升对编辑/删改的鲁棒性；SBW 首次支持全词表 self-salt。
- **Z-score detection**：通过绿 token 计数 W 与期望 Tγ 的标准化偏差 z = (W−Tγ)/√(Tγ(1−γ)) 判定文本是否含水印，H₀ 下服从 N(0,1)。
- **Fused kernel / logit post-processing**：将水印的绿/蓝判定与 logit 偏置合并为单次 CUDA kernel 启动，消除中间张量分配与多次 kernel launch 开销。
- **Jenkins integer hash**：Bob Jenkins 提出的整数哈希函数，可编译为纯 ALU 指令、输出范围大；本文用于替换 KGW 的置换表映射以改善均匀性。
- **Watermark tax**：水印在推理阶段引入的计算/延迟/内存开销；SBW 的目标是将该税收降至近零。

## 可复现要素
- **代码**：`pip install sbw`；完整实现与 vLLM logits processor 见 https://github.com/si-mon-jinn/sbw；水印评测库 waterpipe 见 https://github.com/si-mon-jinn/waterpipe；论文源码见 https://github.com/si-mon-jinn/flip-dont-shuffle。
- **模型**：Qwen3-8B（生成）、Qwen3.5-27B（PPL 评估）、Falcon-7B（泛化验证）。
- **数据集**：C4 validation split（500 prompts，截断至 30 tokens）；泛化实验使用 Alpaca。
- **硬件**：NVIDIA RTX 3090（主实验）、A6000（泛化）；CUDA 12.8、PyTorch 2.10.0、vLLM 0.19.1。
- **关键超参**：γ ∈ {0.25, 0.50}，δ ∈ {1, 2, 5, 10}，temperature=1.0（pure sampling）；self-salt 上下文宽 4 + 候选 token，top-k 近似时 k=40。
- **随机种子**：生成随机种子 42，hash key 15485863。
- **数据/权重是否公开**：代码开源；模型权重依原模型许可；数据集 C4 可公开获取。
