---
title: "Plan-Pointers-and-Record-Directive-Form-in-Budgeted-Verifica"
source: https://arxiv.org/pdf/2609.03450v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-07 20:20:59"
field: "大模型可解释性与对齐"
keywords: ["字节控制", "指针响应", "提示工程", "注册复现", "POINTER-STRONG", "跨域传输"]
innovations: ["提出字节控制提示框架系统操控指针响应", "构建模型家族敏感性谱系揭示稳定性差异", "通过注册复现验证POINTER-STRONG结论"]
benchmarks: ["Study B'/G/I 六臂对照测试", "6模型家族指针响应率"]
---

# 论文速读：Plan-Pointers-and-Record-Directive-Form-in-Budgeted-Verifica

## 一句话总结
本文系统性研究**字节控制（byte-controlled）提示工程**对大模型指针（pointer）/归属（credit）响应行为的操控效应，揭示了不同模型家族在措辞敏感性上的巨大差异，并通过注册复现验证了 POINTER-STRONG 结论的稳健性。

## 研究问题与动机
- **核心问题**：提示措辞的系统性变化是否能可靠地操控模型对指针分配和归属判断的响应？
- **现有方法不足**：以往研究缺乏跨模型家族的对比分析，且提示工程的迁移性（transport）未被系统检验。
- **动机**：为安全对齐、模型可解释性提供量化证据，区分"模型本征行为"与"提示诱导行为"。

## 核心贡献（创新点）
- **字节控制提示框架**：提出基于细粒度字节层级的提示扰动方法，实现对模型指针响应的系统性操控。
- **模型家族敏感性谱系**：揭示 GPT-5.6 系列（极端稳定）与 Fable 系列（极端敏感）在提示工程响应上的根本差异。
- **注册复现验证（Study B'）**：通过 1,200 次 episode 的预注册实验，确认 share ≈ 0.957 的 POINTER-STRONG 结论可复现。
- **跨域传输对比（transport contrast）**：首次量化提示行为在不同 world 间的迁移程度，证明多数模型不可直接跨 world 迁移。
- **六臂对照实验设计（Study G）**：构建 crit / both_IL / crit_pad / crit_other / both_fresh / both_inherited 六组条件，实现精细的对照分析。

## 方法详解
- **字节控制（Byte-Controlled Prompting）**：通过在提示中引入特定后缀（如 +id）或替换命名指针（如 memory_44），控制模型对归属关系的响应模式。
- **POINTER-STRONG 判定标准**：当 share = Δ_plan / Δ_bridge 接近 1 且超出预注册 margin 时，判定为 POINTER-STRONG，表明模型对指针指令高度敏感。
- **注册主量（Registered Quantities）**：
  - Δ_plan：pricing vs onboarding 条件下的 V73 概率差
  - Δ_bridge：bridge 版条件下的概率差
  - Δ_attract：pricing vs none
  - Δ_suppress：none vs onboarding
  - share：比率指标，核心判据
- **六臂测试矩阵**：每个模型在 6 种措辞条件下各测试 n=40 次，统计指针响应比例。
- **跨域传输对比**：计算同一模型在 valid world 与 superseded world 下的响应差异，差异显著非零则判定为"不可迁移"。

## 实验与结果
- **数据集**：6 大主流模型家族（GPT-5.6 Sol/Terra/Luna、Opus 5、Fable 5/5.1、Sonnet 5），共 1,200+ episodes。
- **评估基线**：bare 条件（无指针指令）、crit 条件（批判性措辞）、both_IL（双向指令加载）。
- **主要结果**：
  | 模型 | crit | both_IL | crit_other | GPT-5.6 Sol 全套 |
  |---|---|---|---|---|
  | Opus 5 | 40/40 | 3/40 | 0/40 | — |
  | Fable 5 | 40/40 | 0/40 | 0/40 | — |
  | GPT-5.6 Sol | 40/40 | 40/40 | 40/40 | **100% 稳定** |
  | Sonnet 5 | 40/40 | 40/40 | 40/40 | **全部5种措辞±id均稳定** |
- **Study B' 核心数字**：share = 0.957 [0.895, 1.029]，Δ_plan = +81.7 [78.3, 85.0]，Δ_bridge = +85.3 [79.3, 90.7]，全部 Exceeds Margin。
- **最强结果**：GPT-5.6 全系列在所有测试条件下达到 40/40（100%）一致行为；Sonnet 5 是唯一对所有 5 种措辞 ±id 后缀均保持稳定的模型。
- **传输对比**：GPT-5.6 系列传输差为零（完全可迁移），其余模型传输差显著非零。

## 相关工作脉络
- **POINTER-STRONG 前作**：本文 Study B 锁定 share ≈ 0.929，本文 Study B' 扩展至 0.957，确认并强化了原结论。
- **提示工程脆弱性研究**：Opus 5 和 Fable 系列对 +id 后缀的剧烈响应（如 P0_id=2/40, P1_id=0/40）延续了"提示敏感"文献的发现。
- **模型可解释性与对齐**：通过字节层级控制指针响应，为模型内部表征的可操纵性提供实证依据。
- **跨 world 迁移研究**：transport contrast 概念与现有"领域自适应"研究形成对照，强调提示诱导行为往往不可跨情境迁移。
- **注册复现方法论**：Study B' 采用预注册、大样本（1,200 episodes）设计，呼应开放科学运动对可重复性的要求。

## 局限性与未来方向
- **模型覆盖有限**：仅测试 6 大模型家族，未涵盖开源小模型或多模态架构。
- **措辞空间受限**：仅探索 crit / both_IL 等少数几种提示变体，未系统扫描全措辞空间。
- **因果机制未明**：虽然观察到行为差异，但未深入揭示模型内部的因果路径或表征变化。
- **未来方向**：可扩展至更多模型架构、探索自动化的提示敏感性分类器、研究如何通过字节控制实现安全的模型行为塑形。

## 研究启发与可借鉴点
- **六臂对照实验设计**可复用于其他模型行为测试，提供精细的对照分析范式。
- **注册复现框架**（Study B'）为 AI 安全研究的可重复性提供了操作模板。
- **transport contrast 指标**可用于评估模型在不同应用场景下的行为一致性。
- **模型家族敏感性谱系**的发现提示后续研究需按家族分层评估，而非假设"模型统一行为"。
- **字节控制方法**可与现有对齐技术结合，探索更精细的行为操控手段。

## 关键术语表
- **字节控制（byte-controlled）**：通过在提示文本中精确控制字符/字节级扰动来操控模型响应的方法。
- **指针（pointer）**：提示中用于引导模型进行归属判断的命名实体或指令标记。
- **POINTER-STRONG**：当 share ≈ 1 且超出预注册 margin 时，判定模型对指针指令高度敏感的结论。
- **share**：Δ_plan / Δ_bridge 的比率，衡量指针指令对最终决策的相对影响强度。
- **跨域传输（transport contrast）**：同一提示在不同 world 条件下引发的行为差异，衡量可迁移性。
- **注册复现（registered replication）**：预先声明分析计划后重复实验，增强结论可信度。
- **crit / bare / both_IL**：六种提示条件之一，分别代表批判性措辞、无指令基线、双向指令加载等。
- **Exceeds Margin**：估计值超出预注册 tolerance margin，判定为统计显著。

## 可复现要素
- **数据集**：论文未明确公开原始 episode 数据，但提供了聚合统计数字。
- **代码**：未提及开源代码仓库。
- **权重**：未提供自训练权重，使用现成商业模型。
- **关键超参**：每条件 n=40（Study G）、总 episodes=1,200（Study B'）、margin 参数见 Table 14。
