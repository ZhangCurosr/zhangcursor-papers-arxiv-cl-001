---
title: "What-Tokens-are-Learned-when-Tokenization-is-Optimized-Joint"
source: https://arxiv.org/pdf/2608.17325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:34:23"
field: "分词与语言建模联合优化"
keywords: ["tokenization", "joint optimization", "SSLM", "H-Net", "morphological alignment", "cross-lingual"]
innovations: ["首次在多语言、多类型学视角下系统分析联合优化分词所学习的 token 结构", "揭示 SSLM 与 H-Net 在形态对齐与计算效率上的本质权衡", "证明无分词器方法作为 pretokenizer 可显著降低下游 BERT 困惑度并维持竞争力"]
benchmarks: ["MorphScore", "NewsCrawl", "NLLB"]
---

# 论文速读：What-Tokens-are-Learned-when-Tokenization-is-Optimized-Joint

## 一句话总结
本文系统研究了当分词与语言建模联合优化时，模型实际学到了哪些 token。通过对比无分词器架构（SSLM、H-Net）与固定分词器方法，发现联合优化会从根本上改变 token 结构，且行为因语言类型而异。

## 研究问题与动机
- 现有分词算法（BPE、ULM 等）通常在语言建模训练前独立学习并保持固定，无法随训练动态调整，可能限制模型性能。
- 不同语言的形态复杂度、类型学与文字系统差异巨大，单一固定分词策略难以泛化，尤其对黏着语、屈折语等非印欧语言。
- 近期无分词器架构（如 SSLM、H-Net）将分词边界预测与语言建模端到端联合优化，但其所学 token 的内生属性、语言学对齐程度以及与下游任务的关系尚不明确。
- 缺乏跨多语言、多类型学的大规模系统比较，现有研究多局限于少数语言或仅关注学习动态，未控制数据规模、模型大小等混淆因素。

## 核心贡献（创新点）
1. **首次在多语言、多类型学视角下系统分析联合优化分词所学习的 token 结构。** 与仅关注 3 种语言的先前研究（Meyer & Buys, 2025）不同，本文覆盖 18 种语言，控制数据规模，揭示 SSLM 与 H-Net 的本质差异。
2. **揭示两种无分词器架构在 token 学习上的权衡机制：SSLM 追求形态对齐与语境效率，H-Net 优先字节级计算效率。** 本质区别在于 SSLM 通过边际化所有分割方式学习子词，而 H-Net 通过数据依赖的边界预测直接学习 byte-level 序列。
3. **证明联合优化的分词能显著降低下游 BERT 预训练的困惑度，并保持有竞争力的下游任务性能。** 与固定分词器相比，SSLM 作为 pretokenizer 可实现最大 70.6% 的困惑度下降，且其学到的 token 虽与标准子词词表重叠低，仍能有效迁移至情感分析、词性标注、命名实体识别与依存句法分析任务。

## 方法详解
- **语言与数据**：选取 18 种形态类型多样、文字系统不同的语言（黏着语 9 种、屈折语 6 种、分析/内爆语 3 种），基于 NewsCrawl/NLLB 语料，使用 byte-premium（BP）调整句子数量以平衡数据规模。
- **无分词器方法**：
  - **SSLM**：基于 Transformer 的 subword segmental language model，对全部可能分割进行边际化，联合优化语言建模与分词，设最大 token 长度为 5，初始词表为 10,000 高频词。
  - **H-Net**：端到端分层网络，同时进行数据依赖边界预测与 byte-level 语言建模，不依赖预定义词表，模型约 3M 参数。
- **固定分词器基线**：BPE、ULM、WordPiece、SaGe、Morfessor、BPE-dropout、PathPiece、PickyBPE、MorphBPE/MorphULM/MorphWP、MYTE、SuperBPE、BoundlessBPE 等。
- **评估指标**：
  - 内在属性：有效词表大小、上下文指数（contextual exponence）、fertility、压缩率、Rényi 效率、型例比。
  - 语言学对齐：形态对齐 F1-score（基于 MorphScore 数据）。
  - 外部性能：在 12M 参数 BERT 上微调完成情感分析、词性标注、NER、依存句法分析。
- **训练设置**：SSLM 约 2M 参数，H-Net 约 3M 参数；固定分词器词表大小设为 5,000/10,000/20,000；下游评估限定英语、印地语、泰卢固语以控制计算成本。

## 实验与结果
- **数据集**：18 种语言，每语言约 25 万句子（经 BP 缩放），下游 BERT 预训练使用 1,000 万句子。
- **主要结果**：
  - SSLM 形态对齐 F1 在泰卢固语上约 0.26，印地语上约 0.65；H-Net 在所有语言上均低于 0.1。
  - H-Net 在非拉丁文字语言中平均 token 长度达 17–18.5 字符，远高于 SSLM 的 2.5–5.0 字符。
  - 与固定分词器相比，SSLM 的 Jaccard 重叠度最低（黏着语约 0.12–0.18），H-Net 几乎为零；但 BPE、BPE-dropout、PathPiece、MorphBPE 形成高重叠聚类（>0.80）。
  - 下游困惑度：使用 SSLM 作为 pretokenizer 可使 BERT 困惑度平均下降 45.2%–70.6%，且收敛更快。
  - 下游任务性能：SSLM-ULM 在泰卢固语 POS 标注上达 90.17% F1，SSLM-WPC 在印地语 NER 上达 91.41% F1；整体略低于 Morfessor 预分词，但显著优于多数修改版固定分词器。
  - 消融实验显示，随模型规模增大（2M→30M），SSLM 的优势逐渐饱和，但始终维持较低困惑度。

## 相关工作脉络
- **Meyer & Buys (2025)**：分析 SSLM 的学习动态，但仅覆盖 3 种语言，未控制数据规模；本文扩展至 18 种语言并引入 H-Net 对比。
- **Bostrom & Durrett (2020); Vemula et al. (2025)**：比较固定分词器，发现 ULM 在下游任务与形态对齐上优于 BPE；本文表明联合优化的 SSLM 可超越这些固定方法。
- **Uzan et al. (2024)**：强调推理策略的重要性；本文聚焦于联合优化对 token 结构的影响，而非推理策略。
- **Yehezkel & Pinter (2023) SaGe**：通过降低上下文指数改进分词；本文发现联合优化同样能降低该指数，且产生的 token 与 SaGe 重叠度较低。
- **Hwang et al. (2025) H-Net**：提出端到端分层网络；本文首次系统评估 H-Net 在多语言下的形态对齐与内在属性，揭示其与 SSLM 的本质差异。

## 局限性与未来方向
- 模型与数据规模受限：无分词器模型仅训练 25 万句子、约 2–3M 参数，下游 BERT 为 12M 参数，未探索大规模扩展下的性能。
- 下游评估仅限 3 种语言，无法全面反映 18 种语言的分词效果。
- 未对无分词器模型的超参数（如 SSLM 最大 token 长度）进行充分搜索，依赖先前研究的启发式设定。
- 未来工作可扩展至更大模型与更多语言，并探索联合优化分词在低资源语言、多模态场景中的应用。

## 研究启发与可借鉴点
- **控制数据规模的思路**：使用 byte-premium（BP）对不同语言进行数据量缩放，可在不均衡语料下保证公平比较，值得后续多语言研究借鉴。
- **多维评估框架**：结合内在属性（上下文指数、fertility、压缩率）与语言学对齐（形态对齐 F1）评估分词，为分词研究提供更全面的分析维度。
- **Pretokenizer 策略**：将无分词器方法作为下游固定分词器的 pretokenizer，可显著提升语言建模困惑度，且保留较好的下游性能，为流水线式 NLP 系统优化提供新思路。
- **类型学差异的发现**：黏着语在联合优化中呈现更动态的分割模式，提示未来研究应针对不同类型学语言设计差异化分词策略。

## 关键术语表
- **SSLM (Subword Segmental Language Model)**：一种将语言建模与所有可能子词分割边际化相结合的架构，通过联合优化学习形态对齐的子词。
- **H-Net**：端到端分层网络，同时进行数据依赖的边界预测与 byte-level 语言建模，不依赖预定义词表。
- **Morphological Alignment (形态对齐)**：评估分词结果与已知词素边界的吻合程度，常用 F1-score 衡量。
- **Contextual Exponence (上下文指数)**：每个 token 遇到的不同相邻 token 的数量，反映 token 的语境多样性。
- **Fertility (生育率)**：平均每个单词产生的 token 数量，衡量分词的粒度。
- **Byte-premium (BP)**：用于平衡多语言数据规模的缩放系数，基于字节长度调整句子数量。
- **Jaccard Overlap**：衡量两种分词方法所学 token 集合的重叠程度，取值 0–1。
- **Pretokenizer**：在标准分词器之前对文本进行初步分割，以提升下游语言建模或任务性能。

## 可复现要素
- **数据集**：NewsCrawl 与 NLLB 公开语料；MorphScore 形态对齐数据公开。
- **代码/权重**：论文未提及开源代码或模型权重。
- **关键超参**：SSLM 最大 token 长度 5，初始词表 10,000；H-Net 嵌入维度 256，FFN 维度 640，AdamW 学习率 3e-4，batch size 128，最大 70 epochs；固定分词器词表大小 5,000/10,000/20,000。
