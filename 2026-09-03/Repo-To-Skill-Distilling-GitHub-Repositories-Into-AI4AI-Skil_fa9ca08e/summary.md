---
title: "Repo-To-Skill-Distilling-GitHub-Repositories-Into-AI4AI-Skil"
source: https://arxiv.org/pdf/2609.02749v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-06 22:39:41"
---

# 论文速读：Repo-To-Skill-Distilling-GitHub-Repositories-Into-AI4AI-Skil

## 一句话总结
本文提出 DisCo 框架与 AREX-Skill Library，将 1,000 个 GitHub ML 仓库及前沿论文蒸馏为 5,353 条可组合的“技能”（操作知识单元），在不动模型骨干与 harness 的前提下，通过渐进式披露与两级路由，在 MLE-bench、PaperBench、FrontierCS、PassNet 四个基准上实现大幅且一致的性能提升。

## 研究问题与动机
- **自主 ML 智能体缺失操作知识层**：现有工作聚焦模型骨干或研究 harness，但智能体每次启动都需重新探索包行为、配置陷阱与实现细节，导致探索预算耗尽且无法跨任务复用发现。
- **全量知识库与上下文窗口的矛盾**：传统静态知识库全量加载会撑爆上下文，而按任务临时检索又易偏离主题；缺乏“接口-底物-执行”分层的多粒度知识管理方案。
- **蒸馏成本与泛化性瓶颈**：手工积累操作知识耗时极长（如 PassNet 基线完整求解需 >20 天），现有竞赛/代码基线（Famou-Agent、AIBuildAI 等）在复杂任务上仍显著落后。
- **任务无关与任务导向的割裂**：已有系统多侧重单一任务形式（纯竞赛或纯论文复现），缺乏统一框架同时支持长期通用库构建与按任务按需蒸馏。

## 核心贡献（创新点）
- **提出 DisCo 双模式科研智能体框架**：首次在模型骨干与研究 harness 之上独立构建“操作知识层”，不改写底层模型即可实现跨任务技能复用。
- **设计技能三元结构与渐进式披露机制**：`SKILL.md`（接口）+ `references/`（底物）+ `scripts/`（执行），使智能体可持有数千技能而仅按需加载任务所需部分，避免上下文过载。
- **构建 AREX-Skill Library 与两级路由系统**：基于 1,000 个 ML 仓库蒸馏出 5,353 条技能，建立 20 areas / 178 capability families 分类体系与自动生成 router，实现从宏观领域到具体仓库图的精准收敛。
- **验证四阶段蒸馏协议与严格门禁**：提出 `scope → ground → construct → verify` 流程，任何未经验证技能不得入库，遗留缺口记录于构建记录 R，确保技能质量与可追溯性。

## 方法详解
- **DisCo 双模式架构**：Creator 模式一次性支付蒸馏成本，预构建长期可复用技能图；Researcher 模式按任务 τ 实际打开的知识内容付费，实现计算预算的精细化分配。
- **技能三元结构**（Eq. 6）：
  - `SKILL.md`：知识接口，包含目标、关键概念、工具用法、示例、已知失败模式；
  - `references/`：知识底物，按需加载的 API 文档、算法细节、参数配置；
  - `scripts/`：执行接口，封装可执行脚本，智能体直接调用而非重写代码。
- **技能图（skill graph）**（Eq. 7）：单个源码可产出多个关联技能，通过路由边、依赖边、组合边连接，支持模块化复用与组合推理。
- **蒸馏四阶段**（Eq. 8）：`scope`（界定证据边界，排除生成文件/构建产物/缓存）→ `ground`（私有检查环境安装验证 import/版本/CLI entry points）→ `construct`（生成仓库级入口 skill 并路由至数据准备、训练、评估、故障恢复等组件 skill）→ `verify`（运行时 skill 与检查严格分离；静态门禁审核元数据、链接完整性、本地路径泄露等）。
- **AREX-Skill 路由机制**：两级能力分类法（20 areas / 178 families）+ 自动生成 router，从 area → family → repository graph 逐步收窄；明确拒绝仅关键词匹配、仅依赖、仅可选集成、仅示例的误归属。
- **任务导向蒸馏变体**：
  - **MLE-bench**：两阶段协议，探索阶段（≤24 GPU-hours）积累广义描述型技能池（含诊断/选择/策略/检查），运行阶段（≤24 GPU-hours）Codex 自主组合提交，屏蔽原始竞赛网页防泄漏。
  - **PaperBench**：每任务选 ≤10 篇相关论文，每篇拆解为 3–5 个模块级 skill，隔离测试 + 组合恢复验证（仅用论文与生成 skill，不读原仓库）。
  - **FrontierCS**：单一恢复导向技能图，覆盖 Agent Track 全部设置；初始证据来自两份人工编写资源（常见算法/数据结构目录、启发式搜索量化指导）；图谱为 9 节点 42 边，基于配对试验迭代保留。
  - **PassNet**：screening-and-dispatch 工作流缩放；Skill v2 几何均值 0.6941，Match 失败率 2%，平均求解 22 min；发现 Skill v1 保守警告“不重实现 vendor 优化算子”过度阻断合法简化，修订为可废止先验。

## 实验与结果
- **数据集与基准**：MLE-bench（75 项竞赛，分 Low/Medium/High，网页源码排除）、PaperBench（20 篇论文，排除目标论文自身及已发布代码）、FrontierCS（188 个 Agent Track 任务，5 小时共享预算）、PassNet（200 样本，A100-SXM4-40GB）。
- **主实验结果**：
  - **MLE-bench**：Codex w/o skills 基线 Any-Medal **31.11%**，加 AREX-Skill 后达 **72.89%**（+41.78pp，**+134.3%**）；High 难度 13.33% → **62.22%**（**+366.8%**）。
  - **PaperBench**：基线平均 **29.45%** → **39.59%**（+10.14pp，**+34.4%**）。
  - **FrontierCS**：基线 Score **70.63** → **77.14**（+6.51pp，**+9.22%**）；以 4.47M tokens 超越 Claude Code (Opus 4.8: 74.5/14.72M tok; Qwen3.7 Max: 61.9/13.85M tok) 及 Gemini CLI (60.2/2.00M tok)，Pareto 占优。
  - **PassNet**：AS Score 1.343 → **1.5313**（+14.0%），Correctness 81.35% → **90.76%**，Failed samples 14 → **5**（-64.3%），超越 TorchInductor 默认编译器基线 (1.419)。
- **公开基线对比**（Table 1 MLE-bench）：Famou-Agent 2.0 (64.44%)、AIBuildAI (63.11%)、CAIR MARS+ (62.67%)、MLEvolve/PiEvolve (61.33%)、Thesis (48.44%)、R&D-Agent (35.11%)，Codex+Skills 全面领先。
- **结论**：增益随任务难度递增，证实复杂任务中操作知识价值更高；少数任务回退（PaperBench 上 `sample-specific-masks` −5.07、`stay-on-topic` −4.52）归因于技能检索精度，需改进路由或 fallback。

## 相关工作脉络
- **Famou-Agent 2.0 / AIBuildAI / CAIR MARS+ / MLEvolve / PiEvolve**：同为 MLE-bench 竞赛智能体，但依赖通用 harness 与无状态重复探索；本文通过操作知识蒸馏打破“每次重探索”瓶颈，实现跨任务复用。
- **Thesis / R&D
