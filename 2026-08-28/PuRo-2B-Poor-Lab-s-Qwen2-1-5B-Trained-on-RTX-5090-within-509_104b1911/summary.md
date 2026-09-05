---
title: "PuRo-2B-Poor-Lab-s-Qwen2-1-5B-Trained-on-RTX-5090-within-509"
source: https://arxiv.org/pdf/2608.27370v1.pdf
model: agnes-2.5-flash
chunks: 5
summarized_at: "2026-09-05 01:51:01"
---

# 论文速读：PuRo-2B-Poor-Lab-s-Qwen2-1-5B-Trained-on-RTX-5090-within-509

## 一句话总结
本文面向学术与开源社区高昂的预训练成本痛点，提出一套全开源的低成本预训练配方：在消费级 RTX 5090 集群上仅耗资 **$6.9K**、历时 17.6 天，从 0 训练出 2B 模型（PuRo-2B），性能接近 Qwen2.5-1.5B，同时系统验证了课程学习、Blockwise FP8 与 MuonH 优化器在低预算场景下的协同增益。

## 研究问题与动机
1. **预训练成本居高不下**：开源配方虽多，但复现 Llama-3.2-3B 成本超 $1.5M，SmolLM3-3B 超 $700K，学术团队难以负担。
2. **消费级 GPU 的工程适配缺失**：RTX 5090 的 $/TFLOP 极具优势，但其 PCIe 带宽非对称、HBM 容量有限，缺乏针对性的并行与通信配置方案。
3. **课程学习与后期调度策略缺乏低成本验证**：现有课程学习多基于数据中心集群，其对消费级硬件上收敛效率、数据分配公平性的实际影响尚不明确。
4. **配方透明度不足**：多数工作仅公开最终权重，缺少数据集成分审计、计算等效拟合与代理能力评估协议，难以支撑后续迭代。

## 核心贡献（创新点）
1. **全开源低成本预训练配方**：公开数据、代码、权重与完整训练配置，以 $6.9K 在 RTX 5090 上复现 2B 模型，性能逼近 Qwen2-1.5B（$84K）。
2. **面向消费级 GPU 的并行与通信工程方案**：放弃高延迟敏感的 TP，采用 DP+PP 自定义 rank ordering，结合驱动级 PCIe P2P 与 GDR hack，将 8 GPU AllReduce 带宽提升至 27.34 GB/s。
3. **Blockwise FP8 在 72% 计算占比下的稳定性验证**：证明线性层 GEMM 使用 E4M3 FP8 仅造成 0.0031~0.0039 的 val loss 升高，等效保留 98.0% BF16 算力，质量调整加速达 1.34×。
4. **课程桶 + 多检查点平均的训练调度策略**：设计 376 个归一化排名桶，并在后期采用恒定 LR 延续与 6 检查点等权平均，相比均匀排序+直接平均提升 1.61 分。
5. **代理测量与 PCA 审计协议**：提出基于 Qwen3-0.6B 的数据成分反事实审计流程，通过四轴聚合与 correlation PCA 量化数据集对 Math/Code/Chinese/General 能力的边际贡献与权衡。

## 方法详解
- **硬件与网络配置**：Phase 1 使用 24×RTX 5090（3 节点），Phase 2 扩展至 96×RTX 5090（12 节点）；禁用 IOMMU/PCIe ACS，调整 NPS，启用 PCIe P2P 与 GPUDirect RDMA（GDR），搭配 400 Gbps InfiniBand。
- **并行策略**：全程仅用 DP+PP，规避 TP 的频繁 allreduce 开销。Phase 1 布局 (TP=1, PP=2, DP=12)，Phase 2 布局 (TP=1, PP=4, DP=24)；自定义 `pp-dp` rank ordering 使同 PP 组位于拓扑更近的 GPU。
- **混合精度**：线性层 GEMM 采用 blockwise E4M3 FP8，hidden/residual/LayerNorm/Embedding 保持 BF16，梯度与优化器状态累加使用 FP32。Transformer Engine 实现，FP8 占整体计算 72%。
- **优化器配置**：注意力+MLP 矩阵使用 MuonH（weight decay=0，LR=10×base）；Embedding/Norm/LM head 使用 AdamW（weight decay=0.1，LR=base）。Hyperball multiplier m=10，MuonH 矩阵有效 LR 固定为 4.08×10⁻⁴。
- **课程学习（Curriculum Bucket）**：将归一化排名范围划分为 376 个区间，每桶从各数据组件截取对应区间，约获 1/376 的各组件 token，每桶约 2.5B tokens；分数仅在自身来源内解释，避免跨数据集质量不可比。
- **后期训练策略**：比较直接平均、step 215,000 恒定 LR 延续、step 218,000 恒定 LR 延续三种方案；最优配置为 step 218,000 恢复、基础 LR 固定 4.08×10⁻⁵，每约 100 优化器步骤（0.63B tokens）保存一个检查点，最终 6 个检查点等权平均。
- **计算等效拟合**：采用 $C=6ND$ 计算理论 FLOPs，Loss 预测使用 $L_g(C)=L_\infty+A_g(C/C_0)^{-\alpha}$（共享 $L_\infty$ 与 $\alpha$）；水平效率因子 $\kappa_{v\to b}=(A_b/A_v)^{1/\alpha}$。
- **代理审计协议**：以 Qwen3-0.6B 为代理，在同一基础混合语料上继续训练 86B tokens，再追加约 8.4B continuation tokens（2,000 steps，GBS=1,024）；按分位数探查候选数据源，四轴聚合（Math=GSM8K+MATH，Code=MBPP+HumanEval，Chinese=C-Eval+CMMLU，General=其余 9 项），计算 population z-score 后进行 correlation PCA。

## 实验与结果
- **数据集**：Nemotron-CC、FineWeb-Edu-CN、MegaMath、SwallowMath-v2、CoderForge Trajectories 等，Phase 1 共 438.8B tokens，Phase 2 共 960B tokens，总计 1.4T。
- **评估基线**：Qwen2-1.5B、Qwen2.5-1.5B、Qwen3-1.7B-Base、Llama-3.2-3B、SmolLM3-3B-Base、Gemma-2-2B 等，采用 15 项 benchmark 无加权均值。
- **主要结果**：PuRo-2B 以 $6,891 成本、22,514 GPU-hours 取得 **57.81** 分；相比 Llama-3.2-3B（$1.5M，52.17 分）提升 **+5.64 分**，相比 Gemma-2-2B（$40,821，48.60 分）提升 **+9.21 分**，接近 Qwen2-1.5B（$84,302，55.14 分）。
- **消融结论**：
  - 课程排序 + 六检查点平均 vs 均匀排序 + 直接平均：**+1.61 分**（57.18 vs 55.57）。
  - 模型平均对均匀排序无益（−0.42 分），对课程排序中性（+0.01 分）。
  - Blockwise FP8 相对 BF16 val loss 仅高 0.0031~0.0039，有效算力保留率 **98.0%**。
  - MuonH 较调优后的 Muon 基线 compute-equivalent multiplier 为 **1.19×**，估计可节省约 16.1% FLOPs。
  - FP8 吞吐提升 1.36×，质量调整后净增益 1.34×。
- **最强结果**：PuRo-2B 在 $6.9K 预算下达到 57.81 分，为同价位最强基线，且以极低算力（1.4T tokens）在 15 项基准上实现均衡性能。

## 相关工作脉络
1. **Qwen2/2.5/3、Llama-3.2、Gemma 系列、DeepSeek-V3/V4**：主流商业/开源大模型，依赖数据中心集群与百万美元级预算，本文与其定位差异在于完全开放硬件选择与配方细节，强调可负担性。
2. **SmolLM3-3B、MiniCPM、MobileLLM/R1、LFM2/2.5**：侧重端侧与小模型推理效率，但缺乏完整的低成本预训练工程适配与配方开源，本文补充了从 scratch 训练的消费级硬件路径。
3. **Nemotron-CC / Nemotron 数学数据**：提供高质量指令与数学预训练数据，本文在其基础上验证课程桶划分与数据比例调整（如 MegaMath-Web-Pro 替换 FineMath）对 Math/Code 权衡的边际增益。
4. **MuonH / Hyperball 优化器前作**：原
