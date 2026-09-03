---
title: "WeMM-Embedding-WeChat-Multi-Modal-Embedding-Technical-Report"
source: https://arxiv.org/pdf/2608.24053v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:30:30"
field: "多模态表示学习"
keywords: ["多模态嵌入", "Matryoshka Representation Learning", "对比学习", "嵌入蒸馏", "MMEB", "通用多模态表示", "两阶段训练", "RQ-KMeans"]
innovations: ["两阶段训练策略：大规模多模态对齐 + 精选数据细粒度精化与跨尺度知识蒸馏", "双向嵌入蒸馏：利用teacher模型batch-wise相似度分布做KL对齐，无需额外离线标注", "Semantic-ID引导的语义平衡重采样：基于RQ-KMeans量化自动去偏大规模训练数据"]
benchmarks: ["MMEB-v2", "MMEB-v3", "MSCOCO", "Flickr30k", "VATEX", "MSR-VTT", "ViDoRe V2", "微信内部26任务基准"]
---

# 论文速读：WeMM-Embedding: WeChat Multi-Modal Embedding Technical Report

## 一句话总结
WeMM-Embedding 是由腾讯微信视觉团队提出的通用多模态嵌入模型家族（2B/4B/9B），基于 Qwen3.5 骨干网络，通过"大规模多模态对齐 → 精选数据精化+跨尺度知识蒸馏"的两阶段训练策略，实现了图像、视频、视觉文档及任意交错多模态输入的统一稠密表示，在 MMEB-v2/v3 等公开基准上达到 SOTA，并已大规模部署于微信推荐与搜索系统中。

## 研究问题与动机
- **现有双编码器架构无法处理交错多模态输入**：早期 CLIP 类模型采用模态专属编码路径，难以对图文交错文档、组合式多模态查询、视频+ASR转录等混合输入建立统一表示。
- **MLLM-based 嵌入模型在细粒度相关性建模上仍有提升空间**：尽管 MLLM 天然支持任意模态组合，但直接将 hidden states 用于嵌入学习时，缺乏对细粒度相关性和难负样本的系统性监督。
- **参数效率与泛化能力之间亟待突破**：开源 8B 基线在 MMEB-v2 上的表现仍有差距，需要以更小参数实现更强泛化，以支撑微信海量内容的低延迟服务。
- **工业级多任务场景需要统一表征框架**：微信推荐、搜索、内容匹配等场景涉及分类、检索、跨域匹配等 26 类任务，需要单一模型支撑多样化下游应用。

## 核心贡献（创新点）
- **两阶段渐进式训练策略**：第一阶段在数亿级异构配对数据上进行大规模多模态对齐；第二阶段在精选数据上结合细粒度相关性监督与跨尺度知识蒸馏，使模型在保持广覆盖能力的同时持续提升细粒度匹配质量。（与先前仅依赖对比学习的单阶段训练方法本质不同）
- **统一配对格式（Unified Pair-Based Format）**：将所有异构任务（检索、分类、问答、 graded relevance 等）统一形式化为源-目标匹配对 $(q_i, c_i, \mathcal{N}_i, y_i)$，实现多任务混合训练。（相比各任务独立设计训练管道的做法，大幅简化了多任务学习框架）
- **嵌入蒸馏（Embedding Distillation）**：利用更大规模同系列模型在线生成 batch-wise 双向相似度分布作为软目标，无需额外离线标注即可向 compact 变体 transfer 语义结构。（区别于传统 reranker-based 硬标签蒸馏，保留了候选间相对相似度信息）
- **Semantic-ID 引导的语义平衡重采样**：借鉴 Semantic IDs 思想，通过三级残差 K-means 量化器为训练样本生成离散标识，按码本密度自适应重采样，缓解大规模数据的语义分布偏斜。（无需人工标注即可自动识别并补充语义稀疏区域）
- **MMEB-v2/v3 双榜 SOTA 与工业规模化部署验证**：2B 版本超越此前 8B 开源基线（MMEB-v2 77.9 vs 77.8），9B 版本达 80.6 登顶官方榜单；同时在 14 个线上 A/B 测试中持续获益并正式上线。（从公开 benchmark 到工业部署的全链路验证）

## 方法详解
- **模型架构**：基于 Qwen3.5 原生多模态 LLM 骨干，支持文本、图像、视频及任意交错组合。采用 last-token pooling：在序列末尾追加专用 `<embedding>` token，取其最后一层 hidden state 经 L2 归一化作为输出表示。支持在同一前向传播中插入多个 `<embedding>` token 以提取不同粒度的表示（如视频专属表示和视频+ASR联合表示）。
- **Matryoshka Representation Learning（MRL）**：支持灵活输出维度 $d \leq D$，推理时只需单次前向传播即可通过前缀截断+重新归一化获得所有目标维度的嵌入，便于不同场景下的效率-性能权衡。
- **Stage 1 — 大规模多模态对齐**：在数亿级异构配对数据上训练，每 batch 来自单一数据源，不同任务 batch 交错训练。核心损失：
  - **对比学习损失（InfoNCE + 重复感知掩码）**：同类任务 batch 内构造 in-batch negatives，引入源端/目标端相似度阈值 $\tau_{\mathrm{dup}}$ 的重复感知掩码 $M_{i,j,c}$ 排除近重复候选，避免分类任务中重复标签导致的假负样本。（公式 7-8）
  - **分级相关性学习（Score-gap-weighted CoSENT）**：对带手动相关性等级的样本，使用基于相关性 gap 加权的排序损失，鼓励大 gap 比较获得更大优化权重。（公式 10-12）
  - **MRL 多维扩展**：上述两种损失均在每个支持的嵌入维度上独立计算并加权求和。（公式 13）
- **Stage 2 — 精选微调与蒸馏**：在约 1/10 规模的精选数据上进一步训练，引入三重额外监督：
  - **Reranker 监督**：对同一 query 下的候选集，用专用多模态 reranker 给出排序分数替代人工标签，构造 $\mathcal{L}_{\mathrm{Rank}}$，仅在有稳定收益的任务子集上使用。（公式 14-15）
  - **嵌入蒸馏（双向 KL 散度）**：用冻结的 9B 模型作为 teacher，对学生模型生成 source→target 和 target→source 两个方向的关系分布做 KL 对齐，transfer batch 内相对相似度结构。（公式 16-20）
  - **整体损失**：$\mathcal{L}_{\mathrm{Stage2}} = \mathcal{L}_{\mathrm{Task}} + \lambda_{\mathrm{Emb}} \mathcal{L}_{\mathrm{Emb}}$，其中 $\mathcal{L}_{\mathrm{Task}}$ 根据 batch 类型选择对比/相关性/排序损失。（公式 21-22）
- **精选数据构建**：包含语义-ID 引导重采样（三级 RQ-KMeans 量化，低密度码本样本高保真保留）、MLLM 质量筛选与文本修正、以及针对文本/图像/视频目标的差异化难负样本构造（MLLM 生成 vs. 中间 checkpoint 检索 vs. reranker 评分）。

## 实验与结果
- **评测基准**：MMEB-v2（78 数据集，涵盖图像/视频/视觉文档的分类、QA、检索、视觉定位）、MMEB-v3（190 任务，新增 53 个复杂文本检索 + 47 个 Agent 任务）、Gemini Embedding 2 报告的 12 数据集跨模态检索套件、以及微信内部 26 任务基准。
- **MMEB-v2 主要结果**：WeMM-Embedding-2B 总分 **77.9**，超越 Qwen3-VL-Embedding-8B（77.8）和 DME-Small-2B（74.8）；4B 达 **79.2**；9B 达 **80.6**，全面超越所有列出开源及闭源基线（包括 DME-Large 80.2、Octen-VL-Large 80.1）。图像子集 81.9、视频子集 95.6、视觉文档子集 83.3 均为 SOTA。
- **MMEB-v3 主要结果**：2B 达 **56.0**（V3-All），超越所有基线；4B 达 **58.2**；9B 达 **59.5**。Text 组（45.3/47.9/48.8）和 Agent 组（45.1/49.0/51.0）均居首位。
- **跨模态检索（12 数据集）**：2B 均值 **79.8**，4B **80.8**，9B **81.7**，2B 已接近/超过部分闭源商业模型（如 Gemini Embedding 2 的 79.5）。
- **内部基准**：2B 模型内部 26 任务均值 **72.0**，较 Qwen3-VL-Embedding-2B（60.9）提升 **11.1 点**，五个类别（分类/搜索/跨域匹配/文章相关性/视频相关性）均全面超越。
- **线上 A/B 测试**：在微信视频号、公众号、朋友圈、电商等推荐和搜索系统中，14 个线上 A/B 测试均获得稳定收益并已全量上线。
- **MRL 分析**：256 维时图像/视频任务保留 98.7% 全维性能，512 维时达 99.2%/98.8%；分类任务对降维最不敏感，检索任务最敏感但 256 维仍可保留 >97%。

## 相关工作脉络
- **CLIP 类双编码器模型（Radford et al., 2021）**：模态专属编码路径，不支持交错多模态输入；WeMM-Embedding 基于原生多模态 LLM，天然支持任意模态组合。
- **VLM2Vec / VLM2Vec-V2（Jiang et al., 2024; Meng et al., 2025）**：将 MLLM hidden states 经持续对比训练转为嵌入；WeMM-Embedding 在此基础上引入分级相关性监督、reranker 监督和嵌入蒸馏，强化细粒度匹配。
- **Qwen3-VL-Embedding（Li et al., 2026）**：同样基于 Qwen3-VL 的两阶段多模态嵌入模型；WeMM-Embedding 在精选数据构建（Semantic-ID 重采样、硬负样本构造）和嵌入蒸馏方面做了系统扩展，2B 即超越其 8B 版本。
- **GME（Zhang et al., 2025）**：利用大规模配对数据合成与知识蒸馏改进多模态检索；WeMM-Embedding 的精选数据策略和双向嵌入蒸馏在监督信号丰富度和覆盖范围上更进一步。
- **DME（Douyin Multimodal Embedding, 2026）**：抖音视频-文本嵌入模型；WeMM-Embedding 在通用性（支持更多模态组合和任务类型）和参数效率上更具优势。
- **E5-Omni / Omni-Embed-Nemotron（2025-2026）**：支持音频等多模态的嵌入模型；WeMM-Embedding 当前不支持音频输入（MMEB-v3 音频任务得分为 0），但在此方向上有明确扩展计划。

## 局限性与未来方向
- **不支持音频输入**：当前模型对 MMEB-v3 中的 11 个音频任务得分均为 0，omni-modal 覆盖尚不完整。
- **9B 变体通过模型合并获得**：由于没有更大的 embedding teacher，9B 最终模型由多个 Stage 2 变体通过 TIES-Merging 合并而成，可能引入合并冲突或次优解。
- **Reranker 监督仅对有限任务子集有效**：论文自述对 embedding 模型自身检索结果的 reranking 并非在所有多模态任务上都能带来稳定增益。
- **精选数据规模约为大规模数据的 1/10**：相对于数亿级 Stage 1 数据，Stage 2 数据量有限，可能影响对极端长尾分布的覆盖。
- **未来方向**：扩展至全模态（含音频）、扩大模型规模、改进数据精选策略和细粒度相关性监督。

## 研究启发与可借鉴点
- **Semantic-ID 引导的语义平衡重采样**：用中间 checkpoint 表征 + RQ-KMeans 量化生成离散语义 ID，按码本密度自适应重采样，是一种无需人工标注的自动语义去偏方法，可迁移至任何大规模多模态/文本训练场景。
- **双向嵌入蒸馏（source→target + target→source）**：利用 teacher 模型的 batch-wise 相似度分布做双向 KL 对齐，比传统硬标签蒸馏保留更多相对结构信息，适合 compact 模型的跨尺度知识 transfer。
- **重复感知掩码（Duplicate-Aware Masking）**：在对比学习中通过相似度阈值屏蔽近重复源/目标，有效缓解分类等任务的假负样本问题，可广泛适用于多任务统一训练框架。
- **Stage 1 任务一致性 batching 的关键作用**：消融显示移除任务一致性 batching 导致 3.4 点大幅下降，说明 in-batch negatives 的 task-consistent 性质对统一多任务训练至关重要，值得在多任务 embedding 研究中重视。
- **MRL 与工业部署的结合**：单次前向传播支持多尺度嵌入输出，便于在推荐系统的不同阶段（召回/粗排/精排）复用同一模型的不同维度嵌入，具有良好的工程落地价值。

## 关键术语表
- **WeMM-Embedding**：腾讯微信团队提出的通用多模态嵌入模型家族，包含 2B/4B/9B 三个规模变体，支持文本、图像、视频及交错多模态输入。
- **Matryoshka Representation Learning (MRL)**：一种使模型单次前向传播即可输出多个嵌套维度嵌入表示的技术，通过前缀截断+重新归一化实现。
- **Semantic-ID**：通过残差 K-means 量化器将模型表征映射为离散标识，用于刻画训练数据的语义分布并以密度为导向进行重采样。
- **统一配对格式（Unified Pair-Based Format）**：将各类多模态任务统一表示为 $(I_i, q_i, c_i, \mathcal{N}_i, y_i)$ 的源-目标匹配对形式，支持多任务混合训练。
- **InfoNCE + 重复感知掩码**：标准对比学习损失辅以源/目标端近重复屏蔽机制，防止分类等任务中重复标签引入假负样本。
- **Score-gap-weighted CoSENT**：基于相关性等级 gap 加权的排序损失，对大 gap 比较赋予更高优化权重，适用于细粒度相关性学习。
- **嵌入蒸馏（Embedding Distillation）**：利用更大模型的 batch-wise 双向相似度分布作为软目标，通过 KL 散度将语义结构 transfer 给 student 模型。
- **MMEB（Multimodal Embedding Benchmark）**：评估多模态嵌入模型的综合性基准系列，v2 涵盖图像/视频/视觉文档，v3 新增文本、Agent、音频等任务。

## 可复现要素
- **数据集**：大规模训练数据来自公开数据集、网络弱监督源、任务导向合成数据及内部收集；精选数据为内部构建；部分数据集链接：MSCOCO、Flickr30k、DOCCI、TextCaps、VATEX、MSR-VTT、YouCook2、ViDoRe V2 等公开基准。**论文未明确说明全部训练数据是否公开**。
- **代码/权重**：模型权重已发布至 HuggingFace（https://huggingface.co/collections/tencent/wemm-embedding），代码开源至 https://github.com/Tencent/WeMM-Embedding。
- **关键超参**：论文未详细列出全部训练超参（如学习率、batch size、训练 step 数、温度参数 τ 的具体值等），仅提及 τ 为可学习温度参数、τ_dup 为重复感知掩码相似度阈值、γ 为 CoSENT 排序温度、λ_Emb 为蒸馏损失权重。
