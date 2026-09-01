---
title: "Assessing-Quality-of-Experience-in-Natural-Language-Generati"
source: https://arxiv.org/pdf/2608.18888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:05:55"
---

# 论文速读：Assessing-Quality-of-Experience-in-Natural-Language-Generati

## 一句话总结
本文构建了首个面向德语自然语言生成（NLG）的用户中心体验质量（QoE）评估数据集 TextQ-German，通过众包实验与因子分析实证挖掘出机器翻译（MT）与自动文本摘要（ATS）任务各具特色的四维感知质量构念，并提出了结合预训练语言模型嵌入与可解释语言学特征的混合自动预测架构，证明在多数设定下混合模型优于纯 Transformer 基线，且精选特征 alone 即可逼近大模型性能。

## 研究问题与动机
- 传统 NLG 自动评估指标（如 BLEU、ROUGE）仅依赖表层词汇重叠，忽略句法结构与语义上下文，与人类主观感知的相关性显著不足，难以真实反映生成文本在实际部署中的可用性。
- 通信与多媒体领域成熟的“体验质量（QoE）”用户中心视角尚未系统引入文本生成评估，且现有德语 NLG 质量研究多聚焦英语或单一可读性/复杂度属性，存在明显的语种与维度覆盖空白。
- 现有主观评价研究中质量维度定义碎片化、术语不一致（Howcroft et al., 2020），缺乏从德语母语者实际感知出发、数据驱动的维度提取与验证流程。
- 人类标注成本高昂且无法规模化部署，亟需开发既能逼近人类判断、又具备可解释性与计算轻量的自动化 QoE 预测模型，以支撑长期迭代评估。

## 核心贡献（创新点）
- **构建 TextQ-German 多子集数据集套件**：首次为德语 ATS 与 MT 提供细粒度感知维度与整体 QoE 评分的人机对照标注，并配套 LLM 生成扩展集与严格隔离的最终验证集；与以往依赖任务专用自动指标或单一主观维度的工作相比，本文以用户为中心建立了多维、跨生成范式的评估基准。
- **数据驱动的任务特异性质量维度挖掘**：通过语义差异量表与众包实验结合 EFA 独立派生出 MT 的 Precision/Complexity/Grammaticality/Transparency 与 ATS 的 Linguistic Logic/Complexity/Clarity/Predictability；与 SummEval 或标准化 MT 人工评估表格相比，本文维度完全由德语用户实际反馈提炼，避免先验术语偏差。
- **提出语言模型-语言学特征混合预测架构**：将预训练德语 Transformer 的 [CLS] 上下文嵌入与手工构建的可解释语言学特征（共 121 项候选）进行晚期融合（Hybrid Language Model 或 Hybrid SVM）；与
