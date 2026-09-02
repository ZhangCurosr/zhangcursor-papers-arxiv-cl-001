---
title: "Robustness-of-IR-Models-to-Collection-Growth"
source: https://arxiv.org/pdf/2608.23419v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:12:41"
---

# 论文速读：Robustness-of-IR-Models-to-Collection-Growth

## 一句话总结
本文形式化提出了“集合增长公理（CG Axiom）”以度量IR系统对无关文档添加的鲁棒性，并通过合并TREC-COVID与MS MARCO构建异构测试床，实证揭示无跨文档依赖的MDA检索器在集合扩张时更稳健，而MDA/MDD重排器表现相当，传统PRF模块反而会将主导集合偏见引入查询表示。

## 研究问题与动机
- **核心问题**：实际IR系统的文档集合持续动态增长，新增的无关文档是否应降低原有查询的检索效果？现有系统对此缺乏形式化定义与系统评测。
- **现有方法不足**：既有工作多聚焦领域外鲁棒性（如BEIR）、时间序列上的新查询+新文档演化，或对抗性添加，未将“无关文档增量”作为独立变量剥离评估；同时缺乏统一的理论刻画与可复现基准。
- **机制假设**：模型对集合中其他文档的依赖程度（如BM25的IDF统计、CDE的聚类邻域、Listwise CE的候选列表）可能是决定其对集合增长敏感性的关键因素。
- **动机**：建立可量化的鲁棒性公理与评测指标，厘清MDA/MDD架构在动态集合下的行为差异，为生产级持续扩张的检索系统设计提供依据。

## 核心贡献（创新点）
1. **提出CG Axiom与Collection Precision（CP）度量**：将“添加无关文档后检索性能不应显著下降”形式化为数学公理，并引入CP指标量化Top-k结果中源自原集合的比例，填补该方向的评测空白。
2. **构建MDA/MDD跨文档依赖分类体系**：首次按模型是否依赖集合内其他文档信息，系统划分Multi-Document-Agnostic与Multi-Document-Dependent两类，覆盖Bi-Encoder、CE、PRF、CDE、BM25等主流架构。
3. **揭示检索与重排阶段的差异化鲁棒性**：实证表明MDA检索器（SPLADE、RetroMAE）显著优于MDD检索器（BM25、CDE）；而MDA与MDD重排器性能几乎一致，说明依赖机制的影响随流水线阶段而异。
4. **构建开源异构基准与系统化评测**：将TREC-COVID与MS MARCO合并生成仅含1.9%新相关文档的Het集合，提供受控的集合增长测试床，代码与实验配置已公开。

## 方法详解
- **CG Axiom（性能不变性公理）**：设原集合$
