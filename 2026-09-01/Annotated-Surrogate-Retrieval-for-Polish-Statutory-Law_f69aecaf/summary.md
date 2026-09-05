---
title: "Annotated-Surrogate-Retrieval-for-Polish-Statutory-Law"
source: https://arxiv.org/pdf/2608.30929v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:32:23"
---

# 论文速读：Annotated Surrogate Retrieval for Polish Statutory Law

## 一句话总结
本文针对波兰成文法条文检索任务，提出基于LLM生成文档代理（摘要/主题/概念集/假设问题）的三种检索架构（ASCR/ASCR-H/DTF），在300道法考真题上验证了“头部精准排序依赖LLM重排序、深度覆盖依赖确定性多路融合”的成本-质量权衡规律，并明确给出了多项生成式检索增强的负结果边界。

## 研究问题与动机
- **目标极窄的检索难题**：法律适用问答需在数万条成文法中定位唯一 governing provision，下游生成器对候选列表头部位置极度敏感，Hit@1与Hit@10表现迥异。
- **多语言法律检索空白**：现有基准（LegalBench、LEXTREME等）聚焦英语司法管辖区或分类/NER任务；波兰语七格屈折变化加剧词汇失配，成文法级检索资源匮乏。
- **密集检索在该任务中的反常失效**：开放域波兰检索中multilingual-e5-large显著优于BM25，但本题型因考题高度贴近法条原文，BM25以61.7% Hit@1碾压密集检索的52.3%。
- **生成式代理的独立价值未明**：文档扩展（Nogueira et al.）与假设文档生成（HyDE）在成文法场景下作为补充信号还是替代信号，缺乏系统对照实验。

## 核心贡献（创新点）
1. 提出ASCR、ASCR-H、DTF三种基于文档代理的检索设计，在单次生成预算下覆盖9倍延迟跨度，为法律RAG提供可定制的成本-质量前沿面。
2. 形式化“索引期一次性LLM标注（summary/theme/concept set/hypothetical questions）+ 查询期多路融合/重排序”的代理检索范式，明确其作为稀疏/密集路互补信号而非明文替代的定位。
3. 揭示检索优势的深度依赖性：重排序单独贡献27.6个百分点Hit@1提升；DTF在k≥20时以点估计反超，并以1/9延迟、不到一半成本达到oracle citation accuracy上限。
