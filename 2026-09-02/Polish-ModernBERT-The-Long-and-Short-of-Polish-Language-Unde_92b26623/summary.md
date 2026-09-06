---
title: "Polish-ModernBERT-The-Long-and-Short-of-Polish-Language-Unde"
source: https://arxiv.org/pdf/2609.01379v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:18:44"
field: "多语种编码器预训练"
keywords: ["ModernBERT", "encoder-only Transformer", "Polish NLP", "长上下文理解", "预训练配方搜索", "long-context benchmark"]
innovations: ["将 ModernBERT 架构适配至波兰语，提出覆盖 Base/Large 与 512/8K 的四模型家族", "通过 compute-aware 分阶段实验确定最优预训练配方（遮蔽率递减+annealing语料）", "提出 LongContext 五任务波兰语长文档基准，专门评估超出 512-token 窗口的理解能力"]
benchmarks: ["KLEJ", "FinBench", "LongContext", "PIRB"]
---

# 论文速读：Polish ModernBERT: The Long and Short of Polish Language Understanding

## 一句话总结
本文针对波兰语引入了基于 ModernBERT 架构的编码器家族（Polish ModernBERT），覆盖 Base/Large 两种规模和 512-token/8K 上下文两种长度，并提出了新的长文档理解基准 LongContext；在 30 个下游任务上的评测表明，该模型在长文档理解上相比既有波兰语 RoBERTa-8K 基线取得了 9.68 分的显著提升，同时参数量更少、推理效率更高。

## 研究问题与动机
1. **现有波兰语编码器架构过时**：尽管 decoder-only LLM 快速发展，encoder-only Transformer 仍是判别任务和表示学习的高效选择，但波兰语专用编码器仍以早期 BERT/RoBERTa 风格为主，缺乏 ModernBERT 等新一代架构的支持。
2. **现代编码器能力在多语言间分布不均**：ModernBERT 风格架构虽已扩展至部分语言和 multilingual 设定（EuroBERT、mmBERT 等），但形态丰富的低资源语言（如波兰语）仍依赖旧架构，无法充分受益于更新的注意力机制、训练配方和实现优化。
3. **无统一的多规模多上下文波兰语编码器**：既有波兰语编码器（HerBERT、Polish RoBERTa-v2、Polish RoBERTa-8K）均未能同时覆盖 Base/Large 尺度与 512-token/8K 上下文两套设定，限制了长文档场景下的模型选型与比较。

## 核心贡献（创新点）
1. **引入 Polish ModernBERT 编码器家族**：提供四个公开检查点（Base/Large × 512/8K），填补了现代架构 + 多尺度多上下文长度波兰语编码器的空白；与既有 Polish RoBERTa 系模型的本质区别在于原生支持 8K 上下文窗口且采用 ModernBERT 架构而非 RoBERTa。
2. **适配 ModernBERT 分阶段训练配方**：通过对 Base 模型进行逐阶段选择性实验（学习率调度、遮蔽目标与比率、语料混合、峰值学习率），确定了优于直接迁移原始配方的四阶段 512-token 预训练 + 长上下文续训方案；与简单直迁的本质区别在于通过 compute-aware 逐步搜索而非一次调参完成配方适配。
3. **提出 LongContext 五任务长文档基准**：涵盖法律主题分类（SCOTUS-Dom/Dec）、欧洲人权法院判决预测（ECtHR-PL-AVA/VA）及事实一致性评估（BookSummary），专为区分标准与扩展上下文编码器性能而设计；与既有含长输入但非专为长文设计的评测集（如 FINBANK-LONG）的本质区别在于任务构建目标直接针对长文档理解能力。
4. **全面评测与效率分析**：在 30 个任务及 PIRB 检索基准上对比多种波兰语及多语种编码器，并报告推理延迟与显存占用；Base-8K 以 149M 参数超越 190M 参数的 Polish RoBERTa-8K，体现更优的质量-成本权衡。

## 方法详解

**架构设计**：
- 所有变体采用 ModernBERT 核心架构：Rotary Positional Embeddings（RoPE）、GeGLU 前馈层、Pre-normalization、交替全局/局部滑动窗口注意力（每 3 层全局注意力，局部窗口 128）。
- Base：149M 参数，22 层，hidden 768，12 头注意力，vocab 50,008（无 byte fallback）；Large：475M 参数，28 层，hidden 1024，16 头，vocab 128,256（含 byte fallback）。
- 上下文扩展时，全局 RoPE θ 从 10,000 增至 160,000，局部 RoPE θ 保持 10,000。

**预训练语料（共 44.5B tokens）**：
- Curated Polish corpus：15.8B tokens，70GB，涵盖百科、科学、教育、法律、议会、文学、问答、消费评论、新闻、论坛等。
- Common Crawl（CC-MAIN-2019-43）：16.3B tokens，70GB。
- FineTranslations pol_Latn 子集：12.4B tokens，57GB；按 edu_score_raw > 1.24 过滤（约 Top 25%），并额外保留长度 > 2000 tokens 的文档。
- 语料清洗流程：NLTK 句分割 → 标点/空白标准化 → URL 移除 → FastText 句级语言识别 → 基于随机森林的质量分类器（96% 验证准确率）→ KenLM 困惑度过滤 → 500 字符以下文档删除 → SHA-256 精确去重 → MinHash LSH 近重复删除（Jaccard > 0.7）。
- 预留约 0.1% 文档作为 held-out MLM 验证集。

**分阶段训练配方**（Base 模型选定，Large 按比例扩展）：
- **Stage I**（200K 步，峰值 LR 8×10⁻⁴）：token-level MLM，遮蔽率 0.30，全语料，cosine decay with 6% warmup 表现最佳。
- **Stage II**（200K 步，峰值 LR 2×10⁻⁴）：比较 MLM-0.30 / MLM-0.15 / WWM-0.25 / WWM-0.15，WWM-0.25 效果最佳。
- **Stage III**（100K 步，峰值 LR 1×10⁻⁴）：全语料 → 精选语料（Curated），同时降低 WWM 遮蔽率至 0.15。
- **Stage IV**（100K 步，峰值 LR 5×10⁻⁵）：使用 Annealing 语料（Wikipedia、法律文本、消费评论上采样），WWM 遮蔽率降至 0.08，attention dropout 设为 0.0。
- **8K 上下文续训**：从 Stage IV checkpoint 初始化，最大序列长度扩展至 8,192，全局 RoPE θ 升至 160,000，WWM 遮蔽率恢复至 0.30，0 dropout，峰值 LR 1×10⁻⁴，200K 步。
- 各阶段之间 optimizer 和 scheduler 状态重置（cosine restart），使用动态遮蔽（dynamic masking）和序列去填充（sequence unpadding）。
- 实现：官方 ModernBERT 代码，Composer 0.30.0，8× NVIDIA GH200 GPU，BF16 混合精度，StableAdamW 优化器。

## 实验与结果

**评测套件（30 个任务）**：
- KLEJ（9 任务）： sentiment、NER、有害内容检测、语义关系。
- FinBench（6 任务）：金融领域主题分类、意图检测、情感分析。
- Other Tasks（10 任务）：语义关系、主题分类、情感/情绪、有害内容、序列标注。
- LongContext（5 任务）：SCOTUS-Dom、SCOTUS-DEC、BookSummary、ECtHR-PL-AVA、ECtHR-PL-VA。

**主要结果（Base 规模）**：
- Polish ModernBERT-512：短上下文平均 84.96，超越 Polish RoBERTa-v2（83.77）+1.19。
- Polish ModernBERT-8K-Base：总体平均 83.99，超越 Polish RoBERTa-8K-Base（81.95）+2.04；LongContext 平均 77.15，超越 Polish RoBERTa-8K-Base（67.47）+9.68，且参数量少 22%（149M vs 190M）。
- 在 25 个非 LongContext 任务中，Base 模型在 17 个任务上取得最佳结果（含 6/9 KLEJ、4/6 FinBench、7/10 Other Tasks）。

**主要结果（Large 规模）**：
- Polish ModernBERT-8K-Large：总体平均 85.11，超越 Polish RoBERTa-8K-Large（84.89）+0.22；LongContext 平均 78.49，超越 Polish RoBERTa-8K-Large（75.88）+2.61。
- 512-token 变体 Short Context Avg 为 86.46，与 Polish RoBERTa-v2-Large 持平。

**推理效率**（单张 NVIDIA H100，BF16）：
- Base-512：相比 Polish RoBERTa-v2-Base，延迟降低 26%（0.48ms vs 0.65ms），峰值显存降低 54%（1,084MB vs 2,344MB）。
- Base-8K：相比 Polish RoBERTa-8K-Base，显存降低 24%，延迟降低 6%。
- Large-8K：相比 Polish RoBERTa-8K-Large，显存降低 21%（17,554MB vs 22,166MB），延迟降低 25%（21.69ms vs 28.97ms）。

**检索评测（PIRB，41 个数据集，NDCG@10）**：
- Base 规模：Polish ModernBERT-8K-Base 获得 55.36，在 <300M 参数编码器中最佳，超越 Polish RoBERTa-8K-Base（54.10）+1.26。

## 相关工作脉络
1. **BERT / RoBERTa**（Devlin et al., 2019; Liu et al., 2019）：encoder-only Transformer 的奠基工作；本文在此基础上引入 ModernBERT 架构改进。
2. **ModernBERT**（Warner et al., 2025）：本文直接采用的架构基础，融合更新的设计选择与优化实现，原生支持 8K 上下文；本文将其适配至波兰语。
3. **HerBERT / Polish RoBERTa-v2 / Polish RoBERTa-8K**（Mroczkowski et al., 2021; Dadas et al., 2020b, 2026）：既有波兰语编码器代表；本文在其基础上首次引入 ModernBERT 架构，并填补多尺度多上下文覆盖空白。
4. **EuroBERT / mmBERT**（Boizard et al., 2025; Marone et al., 2025）：现代化多语种编码器；本文强调单语种针对性预训练在形态丰富语言上仍具竞争力（Overall Avg 83.99/85.11 vs EuroBERT 76.24/80.46）。
5. **其他语种 ModernBERT 适配**（Modern-LiBERTa 乌克兰语、NeoDictaBERT 希伯来语、Finnish Modern-BERT、TabiBERT 土耳其语、NorBERTo 葡萄牙语）：证明单一语言 ModernBERT 适配路线的有效性；本文将其扩展至波兰语。
6. **MosaicBERT / NeoBERT**（Portes et al., 2023; Breton et al., 2025）：同期的现代编码器工作；本文通过分阶段配方搜索确定最优方案，而非简单迁移任一现有配方。

## 局限性与未来方向
1. **语言范围局限**：仅针对波兰语，分阶段预训练配方的有效性尚未在其他语言上验证，可能不适用于数据稀缺或形态差异较大的语言。
2. **评测任务类型偏窄**：30 个任务以单标签分类为主，缺少 reranking、span extraction、semantic search、结构化预测等任务的评估，无法全面反映模型能力。
3. **LongContext 基准依赖机器翻译与 LLM 生成数据**：五分之四的任务源自英文数据集的机器翻译，BookSummary 还使用了 LLM 生成的 claim；单 annotator 质量检查不足以消除翻译 artifacts 和潜在的数据污染风险。
4. **LongContext 领域覆盖不均**：偏向法律文档，缺少金融、科学、行政、新闻等长文档场景。
5. **Base/Large 使用不同 tokenizer**：规模比较同时反映了模型容量与分词差异，不能作为纯 scaling study 解读。
6. **未来方向**：跨语言泛化验证、增加多类型任务评测、补充非法律领域的长文档数据、开发原创波兰语长文档数据集。

## 研究启发与可借鉴点
1. **分阶段 compute-aware 配方搜索**：在每阶段固定多数超参、仅改变少量因子（LR 调度、遮蔽目标/比率、语料混合），以 KLEJ 验证集为指标选择最优配置并传递至下一阶段——这一策略可迁移至其他语言的编码器预训练中，避免盲调。
2. **渐进式 WWM 遮蔽率递减 + annealing 语料**：从 token-level MLM（0.30）→ WWM（0.25）→ WWM（0.15）→ annealing（0.08）的阶梯式下降，配合精选高质量语料，是提升预训练效果的有效策略。
3. **动态遮蔽（dynamic masking）+ 序列去填充（sequence unpadding）**：显著提升训练效率（512-token 阶段非 padding token 占比约 56%，8K 阶段约 16%），可在后续模型训练中直接采用。
4. **长上下文续训时仅调整全局 RoPE base**：从 10,000 提升至 160,000 而保持局部 RoPE 不变，是一种轻量且高效的上下文扩展方式，避免了从头预训练的昂贵开销。
5. **语料质量筛选组合策略**：KenLM 困惑度过滤 + 随机森林质量分类器 + MinHash LSH 去重，对构建高质量单语预训练语料具有通用参考价值。

## 关键术语表
- **ModernBERT**：一种更新架构的 encoder-only Transformer，融合 RoPE 位置编码、GeGLU 前馈层、pre-normalization 和交替全局/局部滑动窗口注意力，原生支持 8K 上下文。
- **LongContext 基准**：本文提出的五任务波兰语长文档理解评测集，涵盖法律分类、事实一致性评估等，专为检验 512-token 之外上下文能力设计。
- **Whole-Word Masking（WWM）**：对完整词汇（而非单个 token）进行遮蔽的 MLM 变体，适用于形态丰富的语言，避免将一个词的部分子词遮蔽导致信息泄露。
- **Annealing corpus**：在最终预训练阶段使用的上采样语料，增加 Wikipedia、法律文本和消费评论的比例，以提升百科、法律和观点类内容的表征质量。
- **Rotary Positional Embedding（RoPE）**：一种基于旋转矩阵的位置编码方式，本文在长上下文续训时将全局 RoPE base 从 10,000 升至 160,000 以支持更长序列。
- **Sliding-window attention**：局部注意力机制，每个 token 只关注窗口内相邻 token；本文采用每 3 层穿插全局注意力（访问全部 token）的模式平衡效率与表达能力。
- **Sequence unpadding**：训练时去除 padding token 以减少无效计算，本文 512-token 阶段非 padding token 占比约 56%，8K 阶段约 16%。
- **StableAdamW**：本文使用的优化器，具有解耦权重衰减和数值稳定性增强，适用于大规模混合精度预训练。

## 可复现要素
- **预训练语料**：Curated Polish corpus、Common Crawl（CC-MAIN-2019-43）、FineTranslations pol_Latn 子集；清洗和去重流程详见附录 A。
- **代码/权重**：使用官方 ModernBERT 实现（作者发布），四个检查点将在论文发表后公开；长期上下文续训使用自行构建的 context-extension 语料。
- **关键超参**：
  - Stage I：峰值 LR 8×10⁻⁴（Base）/ 5×10⁻⁴（Large），MLM 0.30，cosine + 6% warmup。
  - Stage II：峰值 LR 2×10⁻⁴，WWM 0.25。
  - Stage III：峰值 LR 1×10⁻⁴，WWM 0.15，Curated 语料。
  - Stage IV：峰值 LR 5×10⁻⁵，WWM 0.08，Annealing 语料，attention dropout=0。
  - 8K 续训：峰值 LR 1×10⁻⁴，WWM 0.30，全局 RoPE θ=160,000。
  - 优化器：StableAdamW，β₁=0.9，β₂=0.98，ε=10⁻⁶，weight decay=10⁻⁶。
  - 精度：BF16 混合精度。
  - 硬件：8× NVIDIA GH200 GPU（2 个 HPC 节点），Composer 0.30.0。
