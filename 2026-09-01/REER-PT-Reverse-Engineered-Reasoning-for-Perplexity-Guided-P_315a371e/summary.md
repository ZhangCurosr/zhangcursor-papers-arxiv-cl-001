---
title: "REER-PT-Reverse-Engineered-Reasoning-for-Perplexity-Guided-P"
source: https://arxiv.org/pdf/2608.30627v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:28:35"
---

# 论文速读：REER-PT: Reverse-Engineered-Reasoning-for-Perplexity-Guided-P

## 一句话总结
论文提出 REER-PT，一种基于困惑度（Perplexity）引导的离线预训练数据增强框架，通过在原文高困惑度边界插入简明的“读书笔记式”推理注解，显式补全上下文与难以预测接续之间的逻辑链条，在保持标准 Next-Token Prediction 目标不变的前提下，显著提升 680M 参数模型的通用知识与复杂推理能力。

## 研究问题与动机
1. **高质量预训练数据日益成为瓶颈**：随着模型计算规模持续放大，原始文本的“量”已非唯一瓶颈，如何挖掘现有语料中的隐性逻辑关系成为关键。
2. **常规 NTP 缺乏显式推理监督**：标准预训练仅教导模型“续写什么”，却未解释“
