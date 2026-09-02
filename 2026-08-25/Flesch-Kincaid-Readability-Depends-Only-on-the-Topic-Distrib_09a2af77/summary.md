---
title: "Flesch-Kincaid-Readability-Depends-Only-on-the-Topic-Distrib"
source: https://arxiv.org/pdf/2608.23327v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:50:57"
---

# 论文速读：Flesch-Kincaid-Readability-Depends-Only-on-the-Topic-Distrib

## 一句话总结
本文在带显式句界符的Topic Model下严格证明：FKGL等Flesch–Kincaid可读性公式在长文本极限下几乎必然收敛至仅依赖于文档主题分布向量 θ 的确定性函数 Φ(θ)；实证表明该主题向量能较强预测FKGL，但额外增益在Brown语料上被“体裁+内容词平均音节数”基线完全吸收，在BNC上仅呈微弱且受拟合扰动影响的增量。

## 研究问题与动机
- FKGL/FRE公式因仅依赖句长与词音节统计、计算高效，已成为LLM可控生成与文本简化任务的主流自动评估信号，但其短文本不稳定性迫使研究者依赖长文本聚合值。
- 现有实践缺乏生成模型下的严格理论刻画：长文本分数稳定是否等价于对词汇组成（lexical composition）不变？公式结构本身决定了哪些文档变异会被保留、哪些会被丢弃？
- 公式正被越来越多地用作自动化奖励或内容过滤信号，但缺乏对其系统性偏倚（如特定词汇谱系/体裁/排版惯例被惩罚）的理论解释。
- 本文动机：在显式Topic Model框架下推导FKGL的长文本极限，解析其双标量因子化结构、不变集几何与收敛速率，并通过掩码、分半、体裁控制等实验审计主题向量对FKGL的实际可恢复性边界。

## 核心贡献（创新点）
1. **确立FKGL的长文本确定性极限与闭式表达**：证明在条件i.i.d. token的Topic Model下FKGL_N几乎
