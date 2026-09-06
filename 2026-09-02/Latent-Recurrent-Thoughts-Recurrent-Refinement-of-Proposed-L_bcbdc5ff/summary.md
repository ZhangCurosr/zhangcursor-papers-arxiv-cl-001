---
title: "Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-L"
source: https://arxiv.org/pdf/2609.01117v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:21:43"
---

# 论文速读：Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-L

## 一句话总结
本文提出 Latent Recurrent Thoughts（LRT）框架，通过任务专用编码器生成初始连续潜变量，并由轻量循环推理器进行多步有界残差修正，最终将精炼潜变量注入冻结的 LLM 完成解码；该方法在仅训练 11.2M 参数（<0.2% 主干）的前提下，显著优于同类冻结解码器方法，并在符号与自然语言推理任务上以更低推理算力超越零样本 CoT。

## 研究问题与动机
- 离散 token 空间的 Chain-of-Thought 存在错误传播、自回归生成成本高，且高质量推理迹的 elicitation 依赖模仿已有迹，在无迹监督场景下受限。
- 现有冻结解码器连续推理方法（SoftCoT、EBM-CoT）使用通用语言助手作为提案器，在符号等自然语言外生境下生成的潜变量严重偏离解码器输入流形，反而造成性能坍塌。
- 基于标量能量场的校准器仅沿固定梯度做单次微调，缺乏多步约束传播与迭代计算能力，难以胜任复杂组合搜索。
- 现有循环推理器（TRM、HRM、EqR 等）均为从零训练的单任务独立求解器，缺乏预训练 LLM 的通用序列建模与指令遵循能力，无法直接处理开放自然语言。

## 核心贡献（创新点）
- 提出 LRT 端到端管道，将“任务专用提案器 + 循环细化器”组合并严格对齐至同一冻结解码器接口，实现连续潜变量上的多步迭代推理。
- 以小型双向 Transformer 提案器替代通用助手，使初始潜变量紧贴解码器输入流形，从根本上解决符号
