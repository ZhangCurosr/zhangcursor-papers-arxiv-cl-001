---
title: "CoMerge-Conflict-Driven-Preference-Optimization-for-Multi-Ta"
source: https://arxiv.org/pdf/2609.02273v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:45:10"
---

# 论文速读：CoMerge-Conflict-Driven-Preference-Optimization-for-Multi-Ta

## 一句话总结
本文将多任务大模型合并重构为自监督偏好优化问题，利用朴素Task Arithmetic合并输出的行为缺陷作为硬负样本自动构建偏好对，仅优化1,445个张量级系数即可在MergeBench上取得0.9968的平均归一化性能，显著缓解参数干扰并逼近全参数微调效果。

## 研究问题与动机
1. **参数干扰瓶颈**：模型合并通过在参数空间整合多专家模型规避全量重训练，但不同任务向量方向冲突会导致严重的参数干扰（parameter interference），直接算术聚合往往引发性能退化。
2. **现有方法缺乏行为级负反馈**：数据自由方法（如TIES-Merging、DARE）依赖静态启发式规则，缺乏对跨层/跨任务冲突的自适应能力；数据驱动方法（如AdaMerging、Localize-and-Stitch）虽能学习合并系数或掩码，但仅依赖正样本信号或表征匹配，未直接利用朴素合并模型在实际生成中的失败模式。
3. **标注成本高**：传统偏好优化依赖人工标注或外部奖励模型排序，难以直接迁移至无标注的模型合并场景。
4. **核心诉求**：需要一种无需外部监督、能从合并缺陷中直接学习“应避免什么”的自监督框架，以显式规避参数冲突区域。

## 核心贡献（创新点）
1. **框架重构**：提出CoMerge，首次将多任务模型合并明确建模为自监督偏好优化问题，通过行为级负反馈显式缓解参数干扰；与以往仅依赖静态权重或正样本仿真的方法本质不同。
2. **自监督冲突驱动合成**：设计冲突驱动的偏好合成策略，直接以朴素Task Arithmetic合并的失败输出作为hard negative样本，零标注、零奖励模型即可构建偏好对；区别于依赖人工标签或reward
