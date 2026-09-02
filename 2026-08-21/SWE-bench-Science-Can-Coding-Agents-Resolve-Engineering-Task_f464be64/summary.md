---
title: "SWE-bench-Science-Can-Coding-Agents-Resolve-Engineering-Task"
source: https://arxiv.org/pdf/2608.19799v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-02 12:37:07"
field: "AI for Science / 科学计算软件评测"
keywords: ["SWE-bench-Science", "科学软件修复", "编码智能体", "Chain-of-Evidence Protocol", "代码修复基准", "AI for Science"]
innovations: ["提出 SWE-bench-Science 仓库级基准，覆盖 20 个科学领域 98 个 GitHub 仓库的 119 个 Issue/PR 修复任务", "设计 Chain-of-Evidence Protocol 与 Scientific Auxiliary Information Separation，首次量化科学领域知识的边际贡献", "提出四种科学失败机制分类框架并量化 8 个前沿模型的错误分布"]
benchmarks: ["SWE-bench-Science", "SWE-bench", "SWE-bench Verified"]
---

# 论文速读：SWE-bench-Science-Can-Coding-Agents-Resolve-Engineering-Task

## 一句话总结
本文提出 **SWE-bench-Science**，一个面向科学软件修复的仓库级基准测试，包含 119 个来自 20 个科学领域、98 个 GitHub 仓库的真实 Issue/PR 修复任务，通过 Chain-of-Evidence Protocol 和多维度评估指标揭示前沿编码智能体在科学计算软件错误修复中的能力边界与失败机制。

---

## 研究问题与动机

1. **现有基准缺乏科学软件专项评估**：主流编程基准（如 SWE-bench）聚焦通用软件工程任务，未覆盖科学计算软件中特有的数值一致性、物理语义正确性、单位/坐标系统一、边界条件处理、稀疏/稠密路径等价性等关键要求。
2. **公开测试通过 ≠ 真实修复正确**：现有评估难以区分"可见测试表现"与"完整私有测试正确性"，导致对智能体科学软件修复能力的评估存在虚高。
3. **科学辅助信息的边际贡献不明确**：外部领域知识对模型修复性能的影响机制尚不清楚，缺乏可控的分离实验设计。
4. **科学软件错误的系统性机制未被刻画**：当前研究未对科学软件修复中的错误类型进行系统分类与定量分析。

---

## 核心贡献（创新点）

1. **SWE-bench-Science 基准构建**：收录 119 个来自 20 个科学领域（气候、卫星测高、GNSS、分子建模、量子化学等）、98 个 GitHub 仓库的真实 Issue/PR 修复任务，填补科学计算软件修复评测空白。
2. **Scientific Auxiliary Information Separation 设计**：在 119 个任务中分离出 91 个可分离科学推理支持的任务，首次提供可控实验以量化外部领域知识的边际贡献。
3. **Chain-of-Evidence Protocol**：结合分离的公开/私有测试与修复进展、精确成功、回归保护等多指标，区分"可见测试表现"与"完整私有测试正确性"，并通过隐藏验证器（替代执行路径、状态重置测试、模块间行为契约测试）确保架构级集成正确性。
4. **四种科学失败机制分类框架**：首次系统刻画科学软件修复中的 Knowledge/Abstraction Deficit、Exploration/Surface Repair、Repair Coverage/System Integration、Scientific Generalization Failure 四类失败模式，并量化各模型的错误分布。
5. **多模型全面评测与 token 效率分析**：在 8 个前沿编码智能体配置上评测，揭示无单一模型在所有指标上领先的格局，以及 Claude-Opus-5 在 token 效率上的最优权衡。

---

## 方法详解

**1. SWE-bench-Science 数据集构建**
- 任务来源：98 个 GitHub 仓库的真实 Issue/PR，涵盖气候/气象（CLIMADA）、卫星测高（Satpy）、GNSS（gnss_lib_py）、分子建模（deepmd-kit）、Autografs、PyBaMM（电池建模）、polyply（聚合物）、复合材料（ABD 矩阵）、有限体积法、RDKit（化学信息学）、SymPy（符号计算）、皱纹场（wrinkleFE）、截面属性（section-properties）、细长体动力学（PyElastica）、混合截面塑性、太阳能资源（pvlib）、储能系统（PyPSA）、结构分析（Pynite）、分子力学（tblite）、量子化学（QCEngine）等领域。
- 条目编号 096–119，涉及 Issue/PR 编号如 #831/#839、#5663/#5854、#3182/#3217 等。

**2. Scientific Auxiliary Information Separation**
- 在 119 个任务中，91 个任务允许将科学推理支持（scientific rationale support）从材料中分离，形成"有/无科学信息"的对照实验，用于评估外部领域知识的边际贡献。

**3. Chain-of-Evidence Protocol**
- 评估维度包括：
  - **PublicScore**：公开测试通过率
  - **PrivateScore**：私有测试通过率（反映真实修复正确性）
  - **Fail2Pass**：修复失败私有测试的比例（负向指标，越低越好）
  - **Pass2Pass**：保持已通过的私有测试比例（回归保护，越高越好）
  - **Pass@1**：全部私有测试通过的二值指标
- 隐藏验证器设计：包含替代执行路径测试、状态重置测试、模块间行为契约测试，确保修复满足架构级集成要求而非仅通过单个公开测试。

**4. 科学失败机制分类**
- **Knowledge/Abstraction Deficit**：科学对象、数学定义或领域抽象错误/不完整
- **Exploration/Surface Repair**：仅修复可见症状，未追溯根本科学契约
- **Repair Coverage/System Integration**：局部修复未满足完整软件系统需求
- **Scientific Generalization Failure**：修复适用于观察到的案例但未推广到 unseen conditions

**5. 科学计算软件一致性约束要求**
- 数值一致性、物理语义正确性、单位/坐标系统一、边界条件处理、稀疏/稠密路径等价性等，修复后的 Issue/PR 聚焦于消除单位混用、边界条件错配、稀疏/稠密表示不等价、物理量报告歧义、对称性/手性破坏、数值不稳定等问题。

---

## 实验与结果

**评估基线（8 个编码智能体配置）**

| 模型 | Harness |
|---|---|
| GPT-5.6-sol | Codex (max) |
| Claude-Opus-5 | Claude Code (max) |
| DeepSeek-V4-Pro | Claude Code (max) |
| Kimi-K3 | Kimi Code (max) |
| GLM-5.2 | Codex (max) |
| Nex N2 | Codex |
| DeepSeek-V4-flash | Claude Code (max) |
| Qwen3.5-397B | Codex |

**主要结果（Table 2）**

| 模型 | Public Score | Private Score | Fail2Pass | Pass2Pass | Overall Pass@1 |
|---|---|---|---|---|---|
| GPT-5.6-sol | 99.16% | **78.82%** | **72.30%** | **97.66%** | 46.22% |
| Claude-Opus-5 | 96.64% | 75.11% | 68.60% | 97.37% | **47.90%** |
| DeepSeek-V4-Pro | **100.00%** | 73.16% | 65.77% | 96.58% | 42.02% |
| Kimi-K3 | 98.32% | 66.34% | 57.55% | 94.94% | 35.29% |
| GLM-5.2 | 94.12% | 63.61% | 53.81% | 97.53% | 31.93% |
| Nex N2 | 93.28% | 61.89% | 51.09% | 94.92% | 24.37% |
| DeepSeek-V4-flash | 98.32% | 61.41% | 52.34% | 95.74% | 23.53% |
| Qwen3.5-397B | 96.64% | 51.79% | 38.33% | 95.16% | 14.29% |

- **无单一模型在所有指标上领先**：GPT-5.6-sol 私有分数最强（78.82%）；Claude-Opus-5 整体 Pass@1 最高（47.90%）且在 Issue-driven 和 Expert-exploratory 任务上领先；DeepSeek-V4-Pro 公开分数满分（100.00%）且在 Engineering-integration 任务上最佳。
- **即使前沿模型 Pass@1 也低于 50%**，凸显科学软件工程的难度。

**Token 效率（Figure 5）**
- Claude-Opus-5 以中等 token 预算实现最高 Pass@1；
- GPT-5.6-sol 以较短输出达到相似水平；
- Nex N2（<400B 模型）在较小模型中 token 效率最佳。

**错误机制分析（Table 3）**

| 模型 | 总错误 | Knowledge | Surface Repair | Integration | Generalization |
|---|---|---|---|---|---|
| GPT-5.6-sol | 64 | 18 | 10 | 22 | 14 |
| Claude-Opus-5 | **58** | 24 | **2** | 21 | 11 |
| DeepSeek-V4-Pro | 69 | **15** | 12 | **19** | 23 |
| DeepSeek-V4-flash | 91 | 23 | 14 | 48 | **6** |
| Qwen3.5-397B | **102** | 26 | 15 | 33 | 28 |

- Claude-Opus-5 科学错误最少（58），误导探索/表面修复最少（2）；
- DeepSeek-V4-Pro 在知识抽象缺陷和系统集成不足上最少；
- DeepSeek-V4-flash 科学泛化错误最少（6）。

**科学信息对性能的影响（91 任务子集）**

| 模型 | 条件 | Public Score | Private Score | Pass@1 | Input Tokens | Output Tokens |
|---|---|---|---|---|---|---|
| GPT-5.6-sol | 无科学信息 | 96.70% | 73.23% | 36.26% | 3.86M | 43.75K |
| GPT-5.6-sol | 有科学信息 | 97.80% | 74.06% | 31.87% ↓ | 3.70M | 40.25K |
| DeepSeek-V4-flash | 无科学信息 | 98.90% | 61.21% | 16.4% | — | — |

- 科学信息的引入对部分模型（如 GPT-5.6-sol）的 Pass@1 产生负面影响（36.26% → 31.87%），表明当前模型可能难以有效利用或正确整合领域知识。

---

## 相关工作脉络

1. **SWE-bench / SWE-bench Verified**：通用软件工程修复基准，本文在此基础上面向科学计算软件领域扩展，覆盖领域多样性和科学语义约束是本质差异。
2. **SciCode**：其他科学编程基准，主要关注代码生成任务，而非真实仓库级 Issue/PR 修复，本文聚焦于修复场景。
3. **现有科学代码基准**：覆盖范围有限、缺乏仓库级规模和科学领域多样性，本文 98 个仓库 20 个领域的规模显著扩展。
4. **科学计算软件验证方法**：传统方法依赖人工设计和单元测试，本文引入隐藏验证器（替代路径、状态重置、模块契约）实现更严格的架构级验证。
5. **AI 编码智能体研究**：本文在 8 个前沿模型上进行系统评测，揭示错误分布规律，为后续模型改进提供诊断依据。

---

## 局限性与未来方向

1. **任务规模有限**：119 个任务对于全面评估科学软件修复能力仍显不足，需要扩展到更多领域和仓库。
2. **科学信息利用机制不明**：科学辅助信息的引入对部分模型产生负面影响，其内在机制有待深入研究。
3. **错误分类的主观性**：四种科学失败机制的分类依赖人工标注，可能存在主观偏差，需要更客观的评估方法。
4. **领域覆盖不均衡**：虽然涵盖 20 个科学领域，但部分领域（如量子化学、符号计算）的任务数量较少。
5. **未来方向**：探索如何更有效地将科学领域知识整合到模型训练中；开发更高效的科学软件修复策略；扩大基准覆盖范围和规模。

---

## 研究启发与可借鉴点

1. **Chain-of-Evidence Protocol 的多指标评估设计**：结合 PublicScore、PrivateScore、Fail2Pass、Pass2Pass、Pass@1 等多维度指标，同时评估修复正确性和回归保护，可作为后续科学软件评测的参考框架。
2. **Scientific Auxiliary Information Separation 实验设计**：通过分离科学推理支持构建"有/无信息"对照实验，为量化领域知识边际贡献提供可复用的实验范式。
3. **隐藏验证器设计思路**：替代执行路径、状态重置测试、模块间行为契约测试等机制，可用于构建更严格的软件验证协议，提升评测可靠性。
4. **科学失败机制分类框架**：四类失败机制（Knowledge Deficit、Surface Repair、Integration Gap、Generalization Failure）为系统分析 AI 在科学任务中的错误模式提供了可迁移的分析工具。
5. **Token 效率与模型选择的权衡分析**：Claude-Opus-5 在 token 效率上的最优表现提示后续研究需综合考虑性能与成本，而非仅关注 Pass@1 等单一指标。

---

## 关键术语表

- **SWE-bench-Science**：面向科学软件修复的仓库级基准测试，包含 119 个来自 20 个科学领域、98 个 GitHub 仓库的真实 Issue/PR 修复任务。
- **Chain-of-Evidence Protocol**：结合公开/私有测试与修复进展、精确成功、回归保护等多指标的综合评估协议，用于区分可见测试表现与完整私有测试正确性。
- **Scientific Auxiliary Information Separation**：将科学推理支持从任务材料中分离的设计，用于量化外部领域知识对模型修复性能的边际贡献。
- **PublicScore / PrivateScore**：分别表示公开测试通过率和私有测试通过率，后者反映真实修复正确性。
- **Fail2Pass / Pass2Pass**：负向指标（修复失败私有测试比例，越低越好）和回归保护指标（保持已通过私有测试比例，越高越好）。
- **Pass@1**：全部私有测试通过的二值指标，衡量模型一次尝试完成完整修复的能力。
- **隐藏验证器（Hidden Validators）**：包含替代执行路径、状态重置测试、模块间行为契约测试的验证机制，确保架构级集成正确性。
- **Scientific Generalization Failure**：修复仅适用于观察到的案例但未推广到 unseen conditions 的科学失败机制。

---

## 可复现要素

- **数据集**：SWE-bench-Science，119 个任务，来自 98 个 GitHub 仓库，论文声明可公开获取（具体链接见原文）。
- **代码**：论文未明确声明代码开源状态，需查看原文补充说明。
- **权重**：评测涉及 8 个商业/开源模型（GPT-5.6-sol、Claude-Opus-5、DeepSeek-V4-Pro、Kimi-K3、GLM-5.2、Nex N2、DeepSeek-V4-flash、Qwen3.5-397B），模型本身权重由对应提供方维护。
- **关键超参**：各模型的 Harness 配置（Codex max / Claude Code max / Kimi Code max），论文未详细列出具体 token 限制、temperature 等超参，需查阅原文或各 Harness 文档。

---
