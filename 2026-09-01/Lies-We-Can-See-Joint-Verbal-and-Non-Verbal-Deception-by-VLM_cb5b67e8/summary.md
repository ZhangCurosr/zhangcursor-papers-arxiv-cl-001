---
title: "Lies-We-Can-See-Joint-Verbal-and-Non-Verbal-Deception-by-VLM"
source: https://arxiv.org/pdf/2608.30428v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-05 13:49:26"
---

# 论文速读：Lies-We-Can-See-Joint-Verbal-and-Non-Verbal-Deception-by-VLM

## 一句话总结
本文在首个 3D 多模态 Among Us 沙盒 MINEAMONGUS 中系统研究 VLM 代理的“言语+非言语”联合欺骗行为，发现 Harness 架构配置对欺诈胜率的影响可与 VLM 骨干能力相匹敌甚至更大，且顶尖模型可通过非言语伪装主导或言语虚假主导两条差异化路径获胜。

## 研究问题与动机
- **核心问题**：VLM 代理在具身社交互动中如何协同利用言语与非言语渠道实现策略性欺骗？模型能力与代理架构（Harness）何者起决定性作用？
- **现有方法不足**：
  1. 现有 VLM 多智能体基准多聚焦合作/竞争，缺乏支持联合欺骗细粒度拆解的社交博弈环境。
  2. 欺骗研究多依赖纯文本或单一模态，忽略了具身场景中空间导航、行为隐匿等非言语线索的关键作用。
  3. 代理设计常将“模型能力”与“架构配置”混为一谈，缺乏正交消融以厘清二者相对贡献。
  4. 缺乏可解释的欺骗行为量化标准，难以建立具体原子行为与胜率之间的统计关联。

## 核心贡献（创新点）
1. **提出 MINEAM
