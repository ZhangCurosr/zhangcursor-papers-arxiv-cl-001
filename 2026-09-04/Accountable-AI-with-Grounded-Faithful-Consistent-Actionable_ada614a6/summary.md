---
title: "Accountable-AI-with-Grounded-Faithful-Consistent-Actionable"
source: https://arxiv.org/pdf/2609.03366v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-07 11:01:48"
---

# 论文速读：Accountable-AI-with-Grounded-Faithful-Consistent-Actionable

## 一句话总结
论文提出 VERDICT 系统，首次将 SMT/MAXSMT 求解器与大语言模型解耦结合，用于临床试验患者-试验匹配任务；通过将自然语言纳入/排除标准编译为形式化约束，求解器输出确定性决策、完整推理链、显式假设及最小翻转条件，在准确率、医生偏好与反事实自一致性上显著优于现有神经与神经符号基线。

## 研究问题与动机
- LLM 在高风险临床决策中易生成流畅但缺乏依据、不完整且不忠实于实际推理过程的解释，导致问责性
