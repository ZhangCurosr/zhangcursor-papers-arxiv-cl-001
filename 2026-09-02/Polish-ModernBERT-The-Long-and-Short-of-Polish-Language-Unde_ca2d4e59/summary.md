---
title: "Polish-ModernBERT-The-Long-and-Short-of-Polish-Language-Unde"
source: https://arxiv.org/pdf/2609.01379v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:19:02"
---

# 论文速读：Polish-ModernBERT-The-Long-and-Short-of-Polish-Language-Unde

## 一句话总结
本文面向波兰语构建 ModernBERT 编码器家族，通过分阶段预训练配方选择与 RoPE 范围渐进扩展，发布覆盖 Base/Large 规模与 512/8K 上下文长度的四款模型；同时提出 LongContext 五任务长文档基准，在 30 项下游任务中取得最优综合性能，长上下文任务较 Matched RoBERTa-8K 基线最高提升 9.68 分，且以更少参数实现更低推理延迟与显存占用。

## 研究问题与动机
- 现有波兰语编码器（HerBERT、Polish RoBERTa-v2/8K 等）仍基于早期 BERT/RoBERTa 架构，缺乏结合现代注意力设计与原生 8K 上下文的编码器家族。
- 尽管 ModernBERT 风格已扩展至部分语言与多语种场景，但形态丰富的波兰语在专用词表与单语预训练方面仍存在明显性能天花板。
- 长文档理解任务缺乏针对波兰语的设计基准，现有评测多依赖 512 token 截断，无法区分模型真实长上下文能力与长度外推能力。
- 工业部署需同时兼顾质量、延迟与显存开销，单语现代编码器在参数受限条件下仍有较大的效率优化空间。

## 核心贡献（创新点）
- 发布 Polish ModernBERT 编码器家族（Base/Large × 512/8K 共四款），填补波兰语现代架构与多尺度长上下文支持的空白，与既有工作的本质区别在于采用完整 ModernBERT 配方而非仅做长度外推或架构微调。
- 提出面向波兰语的分阶段预训练配方选择流程，通过逐阶段独立调整学习率调度、掩码目标/比例、语料配比与 RoPE 范围实现性能递进优化，区别于直接移植原始 ModernBERT Recipe 的做法。
- 构建 LongContext 五任务波兰语长文档基准（法律主题分类、意识形态方向预测、欧洲人权法院违规评估、文学事实一致性），专用于评估超出 512 token 窗口的文档级理解能力，弥补现有基准的领域与长度盲区。
- 在 30 项任务与 PIRB 检索基准上全面对比，证明 Polish ModernBERT-8K-Base 以 149M 参数（较 pl-RoBERTa-8K 的 190M 少 22%）在 LongContext 上提升 9.68 分，同时显著降低峰值显存与推理延迟。

## 方法详解
- **架构选型**：完全保留 ModernBERT 核心设计，包括 Rotary Positional Embeddings、GeGLU 前馈层、Pre-normalization 以及交替全局/局部滑动窗口注意力；Base 采用 50K SentencePiece Unigram 词表（补齐 7 个 token 使嵌入矩阵对齐硬件维度），Large 采用 128K 词表并引入 Byte fallback 保证任意输入可无损表征。
- **上下文扩展机制**：8K 变体由 Stage IV 512 版本初始化，仅将全局 RoPE base θ 从 10,000 提升至 160,000，局部 RoPE 保持 10,000 不变，共享架构、词表与全部预训练参数，避免全量重训。
- **预训练语料构成**：主语料 44.5B tokens，含精心策划波兰语文本（15.8B）、Common Crawl 波兰子集（16.3B，2019-10 snapshot）与 FineTranslations 波兰子集（12.4B）；后者保留 edu_score_raw > 1.24 的文档并按长度 > 2K token 额外补充以增强长文本比例。
- **四阶段 512-token 训练**：Stage I 使用 token-level MLM（masking 0.30）与完整语料；Stage II 切换至 Whole-Word Masking（WWM）0.25；Stage III 改用精选语料并降至 WWM 0.15；Stage IV 引入 annealing 语料进一步降至 WWM 0.08 并关闭 attention dropout。每阶段结束时重置优化器与 Cosine 调度状态。
- **第八千上下文持续预训练**：使用 context-extension 语料（精选语料+FineTranslations长文档），WWM 0.30，零 dropout，峰值 LR 1e-4，6% linear warmup + cosine decay，batch token capacity 约 491K。
- **工程实现**：基于官方 ModernBERT 仓库，启用 sequence unpadding、优化注意力核与 Megatron 风格参数初始化；优化器采用 Decoupled StableAdamW（β1=0.9, β2=0.98, wd=1e-6），全程 BF16 混合精度，分布式部署于 8× NVIDIA GH200 GPU。

## 实验与结果
- **评测体系**：涵盖 30 项任务（KLEJ 9、FinBench 6、Other Tasks 10、LongContext 5）与 PIRB 检索基准（41 个数据集），所有模型按统一微调协议（5 次随机种子取均值）评估。
- **Base 规模结果**：Pl-ModernBERT-512 短上下文均值 84.96，较 pl-RoBERTa-v2 提升 1.19 分；Pl-ModernBERT-8K 总体均值 83.99，LongContext 均值 77.15，较 pl-RoBERTa-8K（67.47）提升 9.68 分，参数 149M vs 190M。
- **Large 规模结果**：Pl-ModernBERT-8K-Large 总体均值 85.11，LongContext 均值 78.49，较 pl-RoBERTa-8K-Large（75.88）提升 2.61 分；512 版本短上下文均值 86.46 与 pl-RoBERTa-v2 持平。
- **效率对比**：512-token Base 延迟降低 26%（0.65→0.48 ms/sample）、峰值显存降低 54%（2344→1084 MB）；8K Base 显存降低 24%、延迟降低 6%，Large 显存降低 21%、延迟降低 25%。
- **检索性能**：Pl-ModernBERT-8K-Base 在 PIRB 上 NDCG@10 达 55.36，为 <300M 参数编码器中最佳，优于 pl-RoBERTa-8K-Base 1.26 分。
- **长上下文分析**：按输入长度分桶显示，Base 模型在 18/20 个任务-桶比较中领先，SCOTUS 与 ECtHR 任务随长度下降幅度最小，BookSummary 在极长桶出现小幅回落但仍保持优势。

## 相关工作脉络
- **BERT/RoBERTa 系**（Devlin 2019; Liu 2019）：奠定编码器架构与大规模 masked language modeling 范式，本文以其波兰语衍生版本（HerBERT、Polish RoBERTa-v2/8K）作为核心对比基线。
- **ModernBERT 架构**（Warner 2025）：融合滑动窗口注意力、GeGLU 与现代训练配方的新一代编码器，本文
