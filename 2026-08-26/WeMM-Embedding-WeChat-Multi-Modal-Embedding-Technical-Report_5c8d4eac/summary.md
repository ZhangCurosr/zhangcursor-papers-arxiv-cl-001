---
title: "WeMM-Embedding-WeChat-Multi-Modal-Embedding-Technical-Report"
source: https://arxiv.org/pdf/2608.24053v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:30:26"
field: "多模态表示学习"
keywords: ["多模态嵌入", "Matryoshka Representation Learning", "对比学习", "知识蒸馏", "MMEB基准", "通用表示学习"]
innovations: ["基于Qwen3.5的两阶段训练策略（大规模对齐+精选精修）实现参数高效的多模态嵌入", "Bidirectional embedding distillation将大模型相似度结构知识传递给小模型", "MRL与对比/排序损失的统一结合，单模型支持64-2048维嵌套输出"]
benchmarks: ["MMEB-v2", "MMEB-v3", "MSCOCO", "Flickr30k", "VATEX", "MSR-VTT"]
---

# 论文速读：WeMM-Embedding

## 一句话总结
WeMM-Embedding 是腾讯微信团队提出的通用多模态嵌入模型家族（2B/4B/9B），基于 Qwen3.5 架构，采用"大规模多模态对齐 → 精修微调"两阶段训练策略，在文本、图像、视频、视觉文档及任意交错的 multimodal 输入上实现了领先性能，并已大规模部署于微信推荐与搜索系统。

## 研究问题与动机
- **异构内容统一表示的缺失**：早期 CLIP-style 双编码器方法无法自然支持文本-图像交错文档、复合多模态查询等混合模态输入的联合表示。
- **编码器方法的泛化性不足**：虽有工作将预训练视觉编码器作为 tokenzier 接入文本编码器以扩展 any-to-any 匹配能力，但随任务与输入组合日趋复杂，编码器范式已显局限。
- **MLLM 派生嵌入的潜力待充分挖掘**：多模态大语言模型天然支持任意交错的图文视频输入，但如何系统性地进行大规模对齐与细粒度相关性学习，仍需更完整的训练策略。

## 核心贡献（创新点）
1. **基于 Qwen3.5 的通用多模态嵌入模型家族**：构建了 2B/4B/9B 三档规模模型，支持文本、图像、视频、视觉文档及任意交错输入的统一编码，与已有工作本质区别在于原生支持任意模态组合而无需额外适配器。
2. **两阶段渐进式训练策略**：第一阶段在数亿级异质对上进行大规模多任务对齐；第二阶段用精选数据（语义平衡、更高质量、更具挑战性的 hard negatives）结合 richer relevance supervision 和跨尺度知识蒸馏进行精修，本质区别是"先广覆盖后精细化"的渐进范式。
3. **灵活的 Matryoshka 表示学习（MRL）**：单模型单次前向即可输出多种维度的嵌套表示（64–2048），实际应用中 256 维即可保留 98%+ 全维性能，区别于以往需多模型训练不同维度的做法。
4. **多维监督信号的系统性整合**：在同一框架内统一运用对比学习（InfoNCE + duplicate-aware masking）、分级相关性学习（score-gap-weighted CoSENT）、reranker 监督与嵌入蒸馏，本质区别是多种监督信号共存于统一 pair-based 格式而非单独训练。

## 方法详解
- **建模架构**：以 Qwen3.5 为骨干网络，输入序列由可选指令 $I_{\text{inst}}$ 和多模态段 $x_1, \dots, x_m$（文本/图像/视频）按序排列，末尾附加专用 `<embedding>` token，其最后一层隐藏状态经 L2 归一化后作为输出表示 $\mathbf{e}_{\mathcal{D}}$。支持在序列中插入多个 `<embedding>` token 以同时提取单模态与混合模态表示。
- **Matryoshka 表示**：对隐藏状态保留前 $d$ 维并重新 L2 归一化，得到任意维度 $d \in \mathcal{D}_{\text{MRL}}$ 的表示，单次前向即可获得所有支持维度。
- **统一 Pair-Based 数据格式**：$z_i = (I_i, q_i, c_i, \mathcal{N}_i, y_i)$，其中 $q_i$ 和 $c_i$ 可为任意模态组合，$\mathcal{N}_i$ 为显式 hard negatives，$y_i$ 为分级相关性分数。
- **Stage 1 — 大规模多模态对齐**：
  - **对比学习（InfoNCE + duplicate-aware masking）**：每个 batch 来自同一数据集，内 batch 负采样，对语义近似的 source/target 做掩码排除（公式 7-8）。
  - **分级相关性学习**：采用 score-gap-weighted CoSENT 目标（公式 10-12），对相关性差距更大的配对赋予更高权重。
  - **MRL 扩展**：在每个支持的 embedding 维度上独立计算上述损失（公式 13）。
- **Stage 2 — 精选数据精修与蒸馏**：
  - **精选数据构建**：通过 Semantic-ID-guided resampling（RQ-KMeans 量化降采样稀疏语义）、MLLM 质量过滤与事实修正、以及 hard-negative 构造（文本目标由 MLLM 生成，视觉目标由中间 checkpoint 检索）提升数据质量。
  - **Reranker 监督**：用 CoSENT 风格的 ranking 目标（公式 15），以 reranker 打分替换人工标注的相关性等级，仅在同 query 候选间构造比较。
  - **嵌入蒸馏**：以更大模型（9B→2B/4B）为 teacher，构造双向相似度分布（source→target 与 target→source）并用 KL 散度对齐（公式 16-20），不依赖离线标注。
  - **总目标**：$\mathcal{L}_{\text{Stage2}} = \mathcal{L}_{\text{Task}} + \lambda_{\text{Emb}} \mathcal{L}_{\text{Emb}}$（公式 21-22）。

## 实验与结果
- **数据集与评测基准**：
  - MMEB-v2：78 个数据集（图像 36、视频 18、视觉文档 24），覆盖分类/问答/检索/视觉定位等任务。
  - MMEB-v3：190 个任务（含 53 个文本、47 个 Agent、11 个音频任务）。
  - 12 数据集跨模态检索套件（MSCOCO、Flickr30k、DOCCI、VATEX、MSR-VTT 等）。
  - 微信内部 26 任务 benchmark（分类、搜索、跨域内容匹配、文章相关性、视频相关性）。
- **主要结果**：
  - **MMEB-v2**：2B 模型总分 **77.9**，超越 Qwen3-VL-Embedding-8B（77.8）；4B 达 79.2；9B 达 **80.6**，居公开榜单第一。
  - **MMEB-v3**：2B 模型 V3-All **56.0**（文本 45.3、Agent 45.1），4B 达 58.2，9B 达 59.5。
  - **跨模态检索（12 数据集）**：2B 均值 79.8，4B 80.8，9B 81.7，优于所有对比开源模型并与头部商业模型接近。
  - **内部 benchmark**：2B 模型平均 **72.0** vs. 基线 60.9（+11.1 点），五大类任务全面领先。
- **消融分析**：
  - Stage 1：task-consistent batching 影响最大（移除降 3.4 点），task-specific instructions 移除降 0.8 点。
  - Stage 2：精选数据、reranker 监督、嵌入蒸馏、扩展视觉输入预算逐级贡献，累计提升 2.2 点。
- **MRL 分析**：256 维嵌入在图像/视频任务上保留 98.7% 全维性能，512 维超 99%。
- **线上效果**：已在微信视频号、公众号、朋友圈、电商等推荐系统与搜索场景中部署，14 个线上 A/B 测试均获持续提升。

## 相关工作脉络
- **CLIP-style 双编码器**（Radford et al., 2021）：奠定大规模跨模态对齐基础，但缺乏对交错多模态输入的原生支持，本文以 MLLM 为骨干克服此局限。
- **VLM2Vec 系列**（Jiang et al., 2024; Meng et al., 2025）：探索 MLLM 隐藏态转为 embedding 的可行性，本文在两阶段训练策略与多监督信号整合上更进一步。
- **Qwen3-VL-Embedding**（Li et al., 2026）：同基于 MLLM 的 embedding 方法，本文 2B 规模即在 MMEB-v2 上超越其 8B 版本，强调参数效率。
- **GME**（Zhang et al., 2025）：利用 MLLM 改进跨模态检索，本文在视觉文档与 Agent 检索等更广泛任务上展现更强泛化。
- **E5-Omni / Omni-Embed-Nemotron**：专注 omni-modal（含音频）统一嵌入，本文当前不支持音频，未来方向之一。
- **Matryoshka Representation Learning**（Kusupati et al., 2022）：本文将其引入多模态 embedding 领域，实现单模型多粒度输出。

## 局限性与未来方向
- **不支持音频输入**：MMEB-v3 中 11 个音频任务得分为零，作者明确列为未来方向。
- **9B 模型通过 model merging 获得**：未直接训练 9B 版本，可能影响参数一致性。
- **精选数据构建依赖 MLLM 与 reranker**：引入了额外推理开销与潜在偏见。
- **未覆盖多语言场景**：当前数据以英文和中文为主，未系统评估多语言泛化。
- 未来方向：扩展至 omni-modal（含音频）、进一步扩大模型规模、改进数据精炼与细粒度相关性监督。

## 研究启发与可借鉴点
1. **"Semantic-ID-guided resampling" 数据平衡策略**：用 RQ-KMeans 量化语义分布并据此重采样，可作为大规模多模态训练数据去重的通用方案迁移至其他 embedding 研究。
2. **双向嵌入蒸馏（bidirectional embedding distillation）**：同时蒸馏 source→target 和 target→source 相似度分布，比单向蒸馏提供更丰富的语义结构信息，值得在其他小模型蒸馏任务中尝试。
3. **Matryoshka 表示在多模态中的系统性应用**：将 MRL 与对比学习、ranking 损失结合，在多个维度同步优化，对后续研究多粒度 embedding 有直接参考价值。
4. **两阶段"广覆盖→精细化"训练范式**：先大规模对齐建立通用表征空间，再用精选数据和多元监督精修，此范式可推广至其他多模态表示学习任务。
5. **duplicate-aware masking 的统一对比学习设计**：通过相似度阈值屏蔽近重复源/目标，适用于任何存在重复标签的 multi-task embedding 训练场景。

## 关键术语表
- **WeMM-Embedding**：腾讯微信团队提出的通用多模态嵌入模型家族（2B/4B/9B），支持文本、图像、视频及交错输入的 unified representation。
- **MMEB（Multimodal Multimetric Embedding Benchmark）**：面向多模态嵌入的综合评测基准，v2 涵盖图文视频，v3 进一步扩展至音频、复杂文本与 Agent 任务。
- **Matryoshka Representation Learning (MRL)**：一种使单模型输出嵌套式多粒度表示的技术，通过前缀截断即可得到不同维度的 embedding。
- **InfoNCE with Duplicate-Aware Masking**：在对比学习中对近重复的 source/target 进行掩码屏蔽，防止 false negative 污染训练信号。
- **Score-Gap-Weighted CoSENT**：以相关性分数差距为权重的 ranking 损失，差距越大权重越高，适用于多粒度相关性标注数据。
- **Embedding Distillation（双向）**：以大模型为 teacher，通过对称的 source→target 与 target→source 相似度分布的 KL 散度，将语义结构知识传递给小模型。
- **Semantic ID**：通过 RQ-KMeans 对 embedding 空间进行三层残差量化得到的离散标识符，用于衡量和引导训练数据的语义分布。

## 可复现要素
- **数据集**：论文未公开训练数据；部分评测数据集为公开基准（MMEB、MSCOCO、Flickr30k、VATEX 等）。
- **代码/权重**：模型权重已开源（https://huggingface.co/collections/tencent/wemm-embedding），代码已开源（https://github.com/Tencent/WeMM-Embedding）。
- **关键超参**：论文未详细列出学习率、batch size、训练步数等；温度参数 $\tau$ 为可学习参数；MRL 维度集合未明确列出具体数值；$\lambda_{\text{Emb}}$ 未给出具体值。
