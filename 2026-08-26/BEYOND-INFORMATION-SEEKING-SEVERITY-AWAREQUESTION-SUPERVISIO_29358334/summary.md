---
title: "BEYOND-INFORMATION-SEEKING-SEVERITY-AWAREQUESTION-SUPERVISIO"
source: https://arxiv.org/pdf/2608.24521v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:25:01"
---

# 论文速读：BEYOND INFORMATION SEEKING: SEVERITY-AWARE QUESTION SUPERVISION FOR PROACTIVE MEDICAL DIALOGUE

## 一句话总结
本文提出 Expected-Severity-Risk (ESR)，一种面向主动医疗对话的后果感知提问监督目标，通过在提问时刻边际化未观测答案、评估候选问题对“严重度加权终端风险”的期望降低量来指导证据获取；在 DDxPlus 九病种子集上以 Qwen3-4B 为底座蒸馏训练，使高严重度漏诊率降低 29.5%，诊断准确率提升至 93.2%，仅额外增加 0.14 个问题/对话。

## 研究问题与动机
- **纯信息准则忽略医学后果非对称性**：现有主动提问方法多以后验熵减或信息增益为选问标准，但临床诊断中漏诊严重疾病与混淆两种良性疾病的风险并不等价，纯不确定性优化会系统性低估高危证据的排查价值。
- **选择时刻答案不可观测导致价值评估困难**：提问必须在患者回答之前完成，直接依赖真实答案的 rollout 评估会引入高方差与部署开销，缺乏可在决策时刻计算的轻量替代信号。
- **RL/奖励设计易造成目标耦合**：基于强化学习的对话策略将提问价值与策略优化、奖励设计深度纠缠，样本效率低且难与下游诊断器保持一致性；亟需一种解耦、可直接监督 LLM 的策略学习方法。
- **部署效率约束**：临床场景要求系统在仅依赖可观测对话前缀时完成选问，教师端的复杂风险计算不应拖慢推理。

##
