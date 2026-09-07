---
title: "Plan-Pointers-and-Record-Directive-Form-in-Budgeted-Verifica"
source: https://arxiv.org/pdf/2609.03450v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-07 20:21:18"
---

# 论文速读：Plan-Pointers-and-Record-Directive-Form-in-Budgeted-Verifica

## 一句话总结
本文针对预算化验证场景中的计划指针（Plan Pointers）与记录指令（Record Directives）形式化问题，构建了多条件对照的实验框架，并通过预先注册分析（Pre-registered Analysis）系统评测了多款大语言模型在计划遵循、记录覆盖与跨条件迁移上的决策表现。

## 研究问题与动机
- 预算验证任务要求模型同时遵循“计划指针”与“记录指令”，但现有评估缺乏对多条件形式（如批判性条件、交叉指令、新旧记录继承等）的细粒度隔离与对照。
- 不同模型在预设决策规则下的命中率与条件敏感性差异尚不明确，亟需大规模、可复现、预注册的基准测试以支撑方法选型。
- 传统 LLM 评测多依赖事后统计分析，缺乏对计划遵循（Δ_plan）、桥接能力（Δ_bridge）、记录覆盖率（share）等核心指标的预先注册与边际检验，易受选择性报告偏差影响。

## 核心贡献（创新点）
1. 提出涵盖 `crit`、`both_IL`、`crit_pad`、`crit_other`、`both_fresh`、`both_inherited` 六类控制臂的条件化评测框架，系统化解耦计划指针与记录指令的交互形式。  
   *与已有工作区别：* 突破单一指令遵循评测范式，首次在同一基准上显式分离多种条件组合对模型决策路径的影响。
2. 引入预先注册分析协议，对 `Δ_plan`、`Δ_bridge`、`share` 等核心指标实施 Wilson 95% CI 与预设边际（Exceeds Margin）检验，提升统计严谨性与结果可复现性。  
   *与已有工作区别：* 将假设检验前置并强制报告三类判定标签（Exceeds Margin / Negative / Unresolved），有效规避事后 p-hacking。
3. 完成六款主流模型（Opus 5、Sonnet 5、Haiku 4.5、Fable 5/5.1、GPT-5.6 Sol/Terra/Luna）在多研究（G、B′、I、H1）中的横向对比，揭示 `crit` 条件对高命中率的核心驱动作用及 `valid`/`superseded` 双世界差异。  
   *与已有工作区别：* 首次在同一预算验证基准上对齐多个前沿模型，并明确区分记录生命周期状态对决策稳健性的影响。

## 方法详解
- **条件控制臂设计**：设置六个实验条件（`crit`、`both_IL`、`crit_pad`、`crit_other`、`both_fresh`、`both_inherited`），用于分离计划指令、交叉约束、新旧记录继承等不同机制。
- **核心量化指标**：
  - `Δ_plan`（计划遵循提升率）、`Δ_bridge`（桥接能力提升率）、`share`（记录覆盖率）；
  - `Y₁`：模型在决策端点严格遵循当前档案记录的命中率；
  - `V₇₃`、`V₄₄`、`V_c2`：特定任务/细胞（cells）级别命中率，以 40 cells/model 为抽样单位。
- **统计与注册判定**：采用 Wilson 95% 置信区间，设定 `Exceeds Margin`、`Negative, Exceeds Margin`、`Unresolved` 三类判定标签；Study B′ 作为预先注册复制，整体判定为 `POINTER-STRONG; REPLICATED`。
- **世界状态控制**：引入 `valid`（记录有效）与 `superseded`（记录被替代）两种状态，检验模型在不同记录生命周期下的决策一致性。

## 实验与结果
- **数据集与环境**：预算验证基准（含 Study G、B′、I、H1），Study B′ 规模 1,200 episodes，其余研究以 40 cells/model 为单元。
- **基线与模型**：Opus 5、Sonnet 5、Haiku 4.5、Fable 5、Fable 5.1、GPT-5.6 Sol/Terra/Luna。
- **Study G（V₇₃ 命中率）**：`GPT-5.6 Sol` 全条件均达 **40/40 [91.2%, 100%]**；`Opus 5` 在 `crit` 与 `crit_pad` 下为 40/40，`both_IL`=3/40，`both_fresh`=13/40；`Fable 5` 仅 `crit`=40/40，其余五条件均为 **0/40**；`Fable 5.1` 仅 `crit`=14/40。
- **Table 13/14（V₄₄ 与 δ=10 注册内对比）**：`GPT-5.6 Sol` 在 `crit_other` 条件下仍达 **40/40**；`both_IL−crit` 对比显示 Opus 5/Fable 5 显著负向（−92.5% / −100.0%），`crit_other−crit` 所有模型均为 **−100.0%**。
- **Study B′（预先注册复制）**：`Δ_plan` = **+81.7%**、`Δ_bridge` = **+85.3%**、`share` = **0.957 [0.895, 1.029]**；七项主量度中 `Δ_attract`=+27.3%、`Δ_suppress`=+54.3%、`Δ_mirror`=+81.7% 均 Exceeds Margin，判定 `REPLICATED`。
- **Study I（Y₁ 决策遵循）**：`GPT-5.6 Sol` 在 valid 世界 `crit−none` 达 **+100.0%**；Opus 5/Sonnet 5/Haiku 4.5 的 `crit−none` 显著正向（+44%~+100%）；`Fable 5.1` 在 valid/superseded 下均呈负向（`crit−none` = −4.0% / −32.0%，`both_IL−none` = −48.0%）。
- **最强结果**：`GPT-5.6 Sol` 在多数条件下稳定达 40/40 命中，`crit` 条件是其核心优势来源；`Fable 5/5.1` 对非 `crit` 条件极度敏感，表现断崖式下降。

## 相关工作脉络
- **指令遵循与 LLM 评测**：本文聚焦计划指针与记录指令的形式化交互，区别于传统仅考察零样本/少样本指令跟随的通用 benchmark 评测。
- **预注册与可复现 AI 研究**：沿袭计算社会科学/行为经济学中的预注册范式，将假设检验前置；相较传统事后报告，提供更高的统计严谨性与抗选择性偏差能力。
- **多条件对照实验设计**：借鉴对照组实验思想（`crit` vs `both_IL`、`valid` vs `superseded`），区别于单任务单一指标的评测路线。
- **动态记录系统建模**：针对 `superseded` 记录处理机制，补充了现有 LLM 评测中较少关注的“记录生命周期管理”与状态转换敏感性维度。

## 局限性与未来方向
- **模型覆盖有限**：仅评测六款模型，部分模型（如 Fable 系列）在非 `crit` 条件下完全失效，泛化性与鲁棒性有待进一步验证。
- **条件特异性强**：`crit` 条件
