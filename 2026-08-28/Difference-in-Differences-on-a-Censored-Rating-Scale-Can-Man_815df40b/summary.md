---
title: "Difference-in-Differences-on-a-Censored-Rating-Scale-Can-Man"
source: https://arxiv.org/pdf/2608.27309v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:29:42"
---

# 论文速读：Difference-in-Differences-on-a-Censored-Rating-Scale-Can-Man

## 一句话总结
本文证明在有限制评分量表（bounded rating scale）上构建的双重差分（DiD）审计终点在测量学上是不可识别的：当自变量引发两个候选回复发生同等程度的绝对评分偏移（severity shift）时，由于两极距离量表上下界的原始距离不同，截断衰减会产生不对称性，从而在统计上“制造”出一个名义显著的交互效应；作者通过一项严格预注册的冻结版教育学 LLM 评委审计给出了该机制的实证反例。

## 研究问题与动机
- **审计终点的设计假设存在盲区**：当前 LLM-as-a-judge 偏见审计普遍采用“固定刺激+双条件对比+双重差分”设计，期望抵消共同偏移；现有工作隐含假设“有界量表的截断只会衰减真实效应，不会凭空制造效应”（即认为 censoring 是 conservative 的）。
- **短刻度与高质量刺激物的致命组合**：当候选回复区分度高、分别贴近量表中不同边界时，DiD 的两项差值各自被不同程度地截断，其差值实际度量的是“衰减差异”而非“偏好差异”，导致交互效应不可识别。
- **问题在教育/垂直领域审计中尤为突出**：pedagogy judge、rubric-based evaluator 等常使用 1–5 或 1–10 短量规，且随着被评估 tutor 系统质量提升，两极更易饱和至天花板/底端，虚假交互风险随之放大。
- **预注册审计是检验测量假设的理想实验室**：材料、终点、分析代码在评分前全部冻结，能够排除数据窥探偏差，从而 cleanly 分离“方法缺陷”与“真实行为”。

## 核心贡献（创新点）
- **有界 DiD 端点的可识别性形式化分析**：将计量经济学中受限因变量（limited-dependent-variable）的经典结论引入 LLM 审计设定，证明在共同偏移假设下观测端点退化为 `(κ_H − κ_L)δ`，并在某一极钉界时严格退化为一极对比，填补了审计测量学的代数缺口。
- **预注册审计中的首个实证反例**：对冻结 pedagogy judge（Claude Opus 4.8）进行 990 次调用审计，主终点为零（+0.085, p=
