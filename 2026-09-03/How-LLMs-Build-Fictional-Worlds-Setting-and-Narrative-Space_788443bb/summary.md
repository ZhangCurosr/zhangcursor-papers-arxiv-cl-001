---
title: "How-LLMs-Build-Fictional-Worlds-Setting-and-Narrative-Space"
source: https://arxiv.org/pdf/2609.02482v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 21:00:42"
---

# 论文速读：How-LLMs-Build-Fictional-Worlds-Setting-and-Narrative-Space

## 一句话总结
本文基于叙事学“生活空间（lived space）”框架，将文本划分为五类叙事空间，对比分析 8,000 篇 LLM 生成故事与人类文学小说，发现 LLM 在构建虚构世界时系统性地过度依赖“感知空间（perceived space）”并弱化“行动空间（action space）”，该文体偏差在英/德双语、四种主流模型及全叙事时间轴上均保持稳定。

## 研究问题与动机
- **现有评估方法的盲区**：当前 LLM 创意写作研究多依赖 Torrance 等心理学创造力测试或人工风格打分，缺乏从叙事学理论出发的细粒度文本结构量化分析。
- **空间维度的缺失**：setting 是虚构世界建构的核心机制之一，但 AI 生成文本的空间叙事策略尚未被系统测量，尤其缺乏与人类文学基线的对照。
- **词法方法的局限**：词典/词频方法难以处理小说中的多义性与语境依赖（如同一空间词汇在不同句法中功能迥异），亟需上下文感知的分类器捕捉抽象叙事模式。
- **长文叙事时序规律未知**：现有工作多聚焦短篇或大纲，缺乏对数千词连贯生成文本中空间分布随叙事时间演化的实证刻画。

## 核心贡献（创新点）
1. **提出叙事空间五分类计算框架并实现自动标注**：将现象学“lived space”操作化为 action/perceived/visual/descriptive/no space 五类，区别于传统词典或人工评判，直接建模语境依赖的叙事空间模式。
2. **实证揭示 LLM 长文生成的系统性空间偏差**：证明所有 tested LLM 均显著高产出 perceived space 并低产出 action space，且偏差贯穿 10 段叙事时序，区别于仅评估整体可读性或短篇样本的既有工作。
3. **构建英德双语对齐的万词级叙事语料库与工具链**：发布 8,000 篇 AI 续写故事、匹配人类基线及开源英语分类器，填补多语言长文计算叙事学评测的数据与代码空白。
4. **验证空间分布作为可靠文体指纹**：表明仅凭各段空间比例即可高于随机水平识别生成模型（人类文本召回率 0.97），为 AI 生成检测与风格对齐提供可量化的新指标。

## 方法详解
- **叙事空间分类体系**：基于 phenomenological lived space 理论定义五类：`Action space`（角色移动与物体功能交互）、`Perceived space`（感官氛围与情绪沉浸）、`Visual space`（静态视觉观察）、`Descriptive space`（中立场景定位）、`No space`（无具体故事世界空间指涉）。
- **分类器构建与验证**：以 `RoBERTa-base` 为底座，沿用德语版标注规范在人工标注英语小说语料上微调 5 epochs（lr=1e-5），宏平均 F1 达 0.82；在 600 句 AI 生成文本黄金标准上验证泛化性（准确率 77.3%–85.3%，κ≈0.72–0.82，Mistral 3.2 德语除外）。
- **长文生成协议**：以 Project Gutenberg 人类小说首句为 seed，采用分章迭代提示策略生成 4 章长文（每章上限 3,000 tokens，总计约 10,000–12,000 tokens），temperature=1, top-p=1，固定随机 seed，覆盖 4 模型 × 2 语言。
- **多粒度分析设计**：微观层面截取前 15 句考察开篇；宏观层面将全文等分为 10 段追踪归一化频次随叙事时间的演化；另按章节内四分位重分箱以剥离分章提示引入的边界伪影。
- **统计建模**：使用 R `glmmTMB` 包拟合有序 β 回归 GLMM，公式为 `space ~ author × section + genre + (1 | story)`，作者因子含 Human、GPT-4.1、Gemma-3、Llama-3.3、Mistral-3.2，Genre 编码为 novel vs. other fiction，通过 Type III Wald χ² 检验与 Holm 校正事后对比评估显著性。

## 实验与结果
- **数据集与基线**：英语（1800–
