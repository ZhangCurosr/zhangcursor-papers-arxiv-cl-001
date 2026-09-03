---
title: "HelaBERT-Enhancing-Sinhala-Language-Understanding-with-Dual"
source: https://arxiv.org/pdf/2608.22922v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:58:26"
---

# 论文速读：HelaBERT-Enhancing-Sinhala-Language-Understanding-with-Dual

## 一句话总结
论文提出了 HelaBERT，一个基于 BERT 架构的僧伽罗语（Sinhala）预训练语言模型家族（Small/Large），使用约 10 亿词元的多源语料从头训练，并配套专为黏着形态设计的 SentencePiece Unigram tokenizer；同时提出双池化分类头（dual pooling classification head），通过 [CLS] 与完整 token 序列的双向注意力交互，在四项僧伽罗语文本分类任务上取得优于现有单语/多语基线的性能，尤其在情感分析任务上带来显著增益。

## 研究问题与动机
- 僧伽罗语拥有约 1700 万使用者，但 NLP 资源极度匮乏：缺乏高质量的预训练模型、标注语料与标准化基准。
- 多语言模型（如 mBERT、XLM-R）虽覆盖该语言，但僧伽罗语仅占极小比例，参数容量被稀释，难以充分建模其黏着形态与复杂 abugida 书写系统。
- 现有专用模型 SinBERT 基于 RoBERTa+BPE 训练，预训练语料仅约 1.92 亿词元，且在分类头设计上沿用标准 `[CLS]`-linear 结构，未探索序列内部全局-局部信息的显式交互。
- 标准 `[CLS]` 汇总方式在输入较短或判别信号稀疏的任务中（如含否定/强调词的情感分析）容易丢失关键局部特征，需要更灵活的信息聚合机制。

## 核心贡献（创新点）
1. **HelaBERT 模型家族与专属 Tokenizer**：训练 HelaBERT-Small (~23.3M 参) 与 HelaBERT-Large (~110M 参)，采用从原始 Unicode 直接训练的 SentencePiece Unigram tokenizer（词表 32K，覆盖率 99.95%），跳过词边界假设，更适配僧伽罗语的黏着形态；与 SinBERT 在完全相同的评估协议下竞争，证明更大语料+更合适分词策略的有效性。
2. **双池化分类头（Dual Pooling Classification Head）**：设计双向 co-attention 机制，使 `[CLS]` 向量与全部 token 表征相互关注，融合全局摘要与局部序列信息；本质区别在于放弃单一 `[CLS]` 输出的线性映射，转而通过可学习的亲和力分数实现跨位置的信息交换。
3. **系统化的下游评测与适用边界分析**：在新闻分类、来源分类、情感分析、文体分类四个任务上进行 5 次独立种子评估；明确揭示该头在中等长度序列+中小规模模型上的优势，以及在极短输入或近天花板任务中的局限性，为分类头选型提供实证依据。
4. **开源资源与绿色计算透明化**：公开 HelaBERT 模型权重、SentencePiece tokenizer 及预训练代码，并详细记录训练硬件、功耗与 CO₂eq 排放，推动僧伽罗语 NLP 社区的可持续研究。

## 方法详解
- **预训练数据**：约 10 亿词元，来源于 MADLAD-400 僧伽罗语子集、CulturaX 僧伽罗语子集，以及自建语料（僧伽罗语维基百科、新闻文章、网页爬虫）。经 NFC 规范化、去除不可见 Unicode、过滤非僧伽罗字符、标准化标点与日期模式后，截取约 1.8 亿词元训练 tokenizer。
- **MLM 目标**：每个 step 随机选择 15% 非 padding token，其中 80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变；仅对掩码位置计算 cross-entropy 损失。
- **双池化分类头架构**：
  1. 输入 encoder 输出后
