---
title: "SinkPruner-Sink-Free-Visual-Token-Pruning-for-Multimodal-Lar"
source: https://arxiv.org/pdf/2609.01004v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:41:52"
field: "多模态大模型高效推理"
keywords: ["multimodal large language models", "visual token pruning", "attention sink", "inference efficiency", "training-free pruning"]
innovations: ["发现高范数异常token在特征和空间维度高度冗余，提出scale-free top-ρ过滤规则", "级联视觉净化器+文本引导剪枝器的coarse-to-fine框架，显著降低attention sink和dispersion"]
benchmarks: ["LLaVA-1.5-7B", "Qwen2.5-VL-7B", "LLaVA-NeXT-7B", "MMBench", "MME", "POPE", "VideoMME", "MMStar"]
---

# 论文速读：SinkPruner: Sink-Free Visual Token Pruning for Multimodal Large Language Models

## 一句话总结
论文提出 SinkPruner，一种无需训练的级联视觉 token 剪枝框架，通过视觉净化器过滤高范数异常 token（在特征和空间维度高度冗余但被误认为重要），并结合文本引导剪枝器，在 LLaVA-1.5 和 Qwen2.5-VL 上实现 88.9%~94.4% 的 token 削减同时保留 91.8%~96.5% 的原始性能。

## 研究问题与动机
- **MLLM 推理效率瓶颈**：Vision encoder 输出的视觉 token 序列（如 LLaVA-1.5 的 576 token、LLaVA-NeXT 的 2880 token）远超文本 prompt，导致 Transformer 注意力机制的二次方计算开销，制约边缘设备部署。
- **视觉中心方法的"高范数陷阱"**：现有基于 CLS attention 的方法（VisionZip、Faster-VLM）优先保留高范数异常 token，这些 token 来自非信息性背景区域（天空、墙壁），在空间（与邻居相似度高）和特征（内部 cosine similarity 极高）维度均高度冗余，却因 norm 异常大而被错误保留。
- **文本引导方法的注意力汇聚（Attention Sink）与分散问题**：LLM decoder 中存在 massive activations（attention sinks），导致注意力过度锚定在少量无意义 token 上，同时文本-视觉注意力分散（entropy 高），使 query-aware 选择不可靠。
- **关键洞察**：高范数异常 token 是视觉冗余的核心来源，且它们同时驱动 attention sink；预过滤这些 token 可显著降低 decoder 中的 sink 比例（约 73%）并降低文本-视觉注意力 entropy，使下游剪枝更可靠。

## 核心贡献（创新点）
- **发现并量化高范数异常 token 的冗余性**：通过 norm 分布的双峰结构、空间邻居相似度和 pairwise cosine similarity 证明高范数 token 来自同质背景且特征空间坍缩，与现有方法盲目依赖 attention score 形成本质区别。
- **提出训练免费的级联剪枝框架 SinkPruner**：视觉净化器（high-norm 过滤+聚合为 sink token+基于 CLS attention 和多样性选择低范数 token）与文本引导剪枝器（early LLM layer 的 text-to-vision attention）级联协作，区别于单阶段 pruning 方法。
- **引入 scale-free top-ρ 高范数识别规则**：基于相对排名而非绝对阈值区分 outlier，无需逐模型校准，可泛化至 CLIP 和非 CLIP 编码器（如 DINOv2）。
- **验证 visual sanitizer 的可迁移性**：作为 plug-and-play 模块集成到 VisionZip 可稳定提升 0.9%~2.6%，证明高范数过滤是通用增强手段而非方法特有启发式。

## 方法详解
**Visual Sanitizer（视觉净化器）**：
1. **高范数异常识别**：计算每个 visual token 的 L2 norm $n_i = \|x_i\|_2$，按 top-ρ 规则（默认 ρ=1%）分离出高范数集合 $\mathbf{X}_{high}$ 和低范数候选集合 $\mathbf{X}_{low}$。
2. **高范数聚合**：对 $\mathbf{X}_{high}$ 做平均池化生成单一 proxy token $x_{sink} = \frac{1}{|\mathbf{X}_{high}|}\sum_{x \in \mathbf{X}_{high}} x$，压缩冗余同时保留全局信息。
3. **低范数 token 选择**：从 $\mathbf{X}_{low}$ 中选取 top-$k_{res}$ 个 CLS attention 最高的 token 构成 $\mathbf{X}_{res}$；再通过迭代 farthest-point 采样（或 batched 近似，每次选 b=16 个最不同的 token）构建多样性集合 $\mathbf{X}_{div}$，最终 purified representation $\mathbf{Z} = [x_{sink}, \mathbf{X}_{res}, \mathbf{X}_{div}]$。

**Text-Guided Pruner（文本引导剪枝器）**：
- 在 LLM decoder 的早期层（如 layers 2, 6, 15）计算 text-to-vision attention，对每个 visual token $z_j$ 聚合所有 text query 的 attention weight：$\tilde{p}_j = \frac{1}{L_t}\sum_{i=1}^{L_t} \text{Softmax}(\mathbf{Q}_{text}\cdot\mathbf{K}_{vis}^T)_{i,j}$，保留 top-K token。
- 支持 progressive pruning schedule（多阶段逐步削减）以平衡压缩率与性能。
- 对无 [CLS] token 的编码器（如 Qwen2.5-VL），使用 averaged received self-attention 替代 CLS attention 作为 salience score。

## 实验与结果
- **数据集**：12 个图像-语言基准（GQA、MMBench/MMB-CN、MME、POPE、SQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet）和 4 个视频-语言基准（MVBench、SEED-Bench、NextQA、VideoMME）。
- **模型**：LLaVA-1.5-7B/13B、Qwen2.5-VL-7B、LLaVA-NeXT-7B。
- **主要结果（LLaVA-1.5-7B）**：保留 64 token（剪枝 88.9%）时平均性能保留 96.5%，超越 VisionZip（+3.3%）和 HoloV（+1.8%）；保留 32 token（剪枝 94.4%）时保留 91.2%，超越 VisPruner（+4.0%）。
- **主要结果（Qwen2.5-VL-7B，动态分辨率）**：剪枝 66.7%/77.8%/88.9% 分别保留 98.6%/96.3%/91.8% 性能，在动态分辨率设定下显著优于 FastV、HoloV、VisionZip。
- **视频任务**：80% 剪枝率下平均保留 98.0%，超越 DART（96.6%）和 DivPrune（96.0%）。
- **难推理基准（32 token 预算）**：在 MM-Vet 和 AI2D 上 SinkPruner 达 29.36 和 53.63，远超 FastV（22.66/50.49）和 VisionZip（25.23/51.81）。
- **效率**：90% 剪枝率下总推理时间减少 33.3%，POPE 上 97.1% 准确率对比 VisionZip 的 91.8%。
- **消融**：移除 visual sanitizer 导致 MMB 下降 10.2%；移除 high-norm 过滤使 MME 从 1754.1 降至 1705.6；high-norm 聚合略优于直接删除。

## 相关工作脉络
- **Vision-centric pruning**：ToMe（token merging）、VisionZip/Faster-VLM（CLS attention 重要性排序）——本文指出这些方法被高范数 outlier 误导。
- **Text-guided pruning**：FastV/SparseVLM（decoder attention 选择 query-relevant token）——本文指出其受 attention sink 和 dispersion 干扰。
- **Progressive/Hierarchical pruning**：MustDrop/PDrop（渐进 dropping）、HoloV（全局上下文保留）、HiDrop（late injection/early exit）——本文采用 coarse-to-fine 级联设计，强调上游净化对下游选择的影响。
- **Token compression 通用方法**：PruMerge（adaptive merging）、AutoPrune（mutual information budgeting）、DivPrune（多样性驱动）——本文的 sanitizer 可作为正交模块增强这些方法。
- **Attention sink 现象**：Kang et al. (2025) 提出 visual attention sink 概念；本文进一步证明其根源在于 high-norm visual outliers 并给出预处理解决方案。
- **High-norm outlier 发现**：Darcet et al. (2024) 在 ViT 训练中用 register tokens 缓解 norm artifact；本文在 inference 时利用该现象进行剪枝。

## 局限性与未来方向
- 当前评估仅限 offline inference（预录制的固定长度输入），未涉及在线流式视频处理（如机器人连续感知场景）。
- 在高分辨率 tiled encoding 下，TextVQA 性能略低于 VisPruner；作者推测小 glyph patch 可能携带高 norm 并与背景 outlier 一起被聚合，提出未来可设计 text-aware sanitizer（如豁免高局部边缘密度的 token）。
- 方法为 training-free，未探索联合微调 sanitizer 与 pruner 的潜在收益。

## 研究启发与可借鉴点
- **高范数异常作为冗余信号**：将 feature norm 分布的双峰结构用于 token 筛选的思路可迁移至其他 vision encoder 架构（如 DINOv2）及不同模态（如 audio token pruning）。
- **Scale-free 阈值设计**：top-ρ 相对排名规则避免 per-model 校准，对跨架构泛化有参考价值，可推广至其他需要异常检测的预处理场景。
- **级联净化-选择架构**：上游去噪（sanitizer）+ 下游语义对齐（pruner）的粗到细设计，可启发多阶段推理加速方法（如 KV cache 压缩、长上下文裁剪）。
- **Attention sink 定量分析**：通过 sink ratio 和 attention entropy 评估剪枝效果，提供了可复用的诊断指标，可用于后续工作的 ablation study。
- **Plug-and-play 模块化设计**：visual sanitizer 作为正交通用模块增强现有方法（如 VisionZip），证明了基础洞察的工程价值，值得在其他 pruning 框架中验证。

## 关键术语表
- **High-norm outlier token**：视觉 encoder 输出的特征范数异常大的 token，通常来自同质背景区域，在特征和空间维度高度冗余。
- **Attention sink**：LLM decoder 中少量 token 吸引不成比例的大量注意力，导致其他 token 的注意力被稀释的现象。
- **Attention dispersion**：文本-视觉注意力分布过于分散（entropy 高），模型难以 confident 地定位 query-relevant 区域。
- **Visual sanitizer**：SinkPruner 的预处理模块，通过过滤和聚合高范数异常 token 来净化视觉序列。
- **Scale-free top-ρ rule**：基于相对排名而非绝对阈值识别高范数异常 token 的规则，无需逐模型校准。
- **Text-guided pruner**：利用 LLM decoder 中 text-to-vision attention 分数选择与文本查询语义对齐的视觉 token 的模块。
- **Progressive pruning schedule**：在多阶段 LLM layer 中逐步削减 visual token 数量的策略，平衡压缩率与推理质量。
- **Received self-attention**：无 [CLS] token 的编码器中，用所有 token 对某 token 的平均注意力替代 CLS attention 作为 salience score。

## 可复现要素
- **数据集**：GQA、MMBench、MME、POPE、SQA、VQA-v2、TextVQA、MMStar、MMMU、AI2D、MM-Vet、MVBench、SEED-Bench、NextQA、VideoMME（均为公开基准）。
- **代码**：已开源，https://github.com/LaVi-Lab/SinkPruner。
- **关键超参**：ρ=1%（高范数比例）、b=16（batched diversity selection 每轮数量）、pruning layers=(2,6,15)（LLaVA-1.5）、salience:diversity pool 比例=5:5（LLaVA）或 45%:5%/20%:5%（Qwen2.5-VL）。
- **硬件**：NVIDIA A800-SXM4-80GB GPU，Python 3.10 + PyTorch 2.1.2 + CUDA 12.1。
