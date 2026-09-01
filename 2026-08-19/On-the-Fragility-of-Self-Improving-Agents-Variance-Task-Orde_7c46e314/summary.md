---
title: "On-the-Fragility-of-Self-Improving-Agents-Variance-Task-Orde"
source: https://arxiv.org/pdf/2608.18066v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:04:38"
---

# 论文速读：On-the-Fragility-of-Self-Improving-Agents-Variance-Task-Orde

## 一句话总结
本文对两类基于文本记忆的自改进智能体（AWM 与 ReasoningBank）进行了系统性重评估，通过多轮运行与随机任务顺序揭示其在强基线下的显著脆弱性：方差普遍放大（71% 的情况下）且性能高度依赖隐式任务序（随机打乱后反而下降 4.5%）；人工归因表明“环境与任务 underspecification”是核心诱因，注入额外规范信息可弥合 31% 的性能退化。

## 研究问题与动机
- **可靠性评估缺失**：现有记忆型自改进智能体研究普遍只报告单次运行结果，缺乏对评估噪声与稳定性的量化，难以支撑高 stakes 的实际部署。
- **隐式任务序掩盖缺陷**：Prior 工作采用默认任务 ID 顺序，该顺序隐含 easy-to-hard 课程；真实场景中任务无序到达，该方法在随机序下可能完全失效。
- **强基线下增益边际化**：随着 GPT-5-mini 等更强 backbone 与更优 harness 的出现，自改进方法在原有弱基线下的显著增益是否依然成立尚未验证。
- **记忆生成的 underspecification**：记忆构建模块缺乏对平台约束（如仅支持浏览器操作）与任务 rubric 的明确规范，易产生“合理但不可执行”的错误记忆并引发连锁退化。

## 核心贡献（创新点）
1. **首次系统量化记忆型自改进 agent 的多运行方差**：发现 71% 的情况下引入自改进会放大方差，best-worst gap 最高达 8.2%，揭示了单次运行评估的严重误导性。
2. **揭示任务顺序对自改进效果的强依赖性**：默认顺序带来的 +1.5% 增益在随机打乱后逆转为 -4.5%，证明 prior 成功部分依赖隐式 curriculum 而非方法本质鲁棒。
3. **定位 underspecification 为脆弱性核心成因**：通过人工审查内存条目，指出环境约束缺失与任务歧义会导致 agent 沉淀无效策略（如推荐 API 调用、滥用 Haversine 公式）。
4. **提出可复用的 underspecification 干预方案**：注入 rubric、环境反馈与约束型 prompt 修改（+All），可弥合 31% 的随机序性能退化，为后续 memory validation 提供实证基线。

## 方法详解
- **重评估协议**：在 WebArena、VisualWebArena、SCUBA 三个 Web 浏览基准上，使用 GPT-5-mini 作为 backbone 与 memory construction 模型；对每个实验设置重复 3 次独立运行，记录 pass@1 均值、标准差与 best-worst gap；同时引入 Shuffle-1 与 Shuffle-2 两种随机任务顺序。
- **基线升级**：采用最新版 WALT harness（WebArena/VWA）与 SCUBA harness，以更强无记忆模型作为 self-improvement 起点，确保对 prior 工作（如 Gemini-2.5-Pro/Claude-3.5-Sonnet 基线）的全面超越。
- **Underspecification 干预设计**：在 RBank 的 memory construction 阶段注入三类额外上下文：
  1. **+Rub（Rubrics and Scores）**：任务结束后提供评估器使用的函数型 rubric（must_include/exact_match/fuzzy_match）与得分，减少歧义任务导致的错误记忆。
  2. **+Env（Environment Feedback）**：将 UI 交互错误日志（如 `input_text failed`）纳入轨迹回顾，防止 agent 重复执行无效操作。
  3. **+PMod（Prompt Modification）**：显式禁止建议 API/外部网站/人工确认等环境不支持的策略，并引导提取程序性知识与站点导航结构。
  4. **+All**：上述三项组合，评估其对 Shuffle 序下性能的修复幅度。
- **统计检验**：使用 unpaired t-test 计算多 run 均值的 p-value，严格区分统计显著增益与随机波动。

## 实验与结果
- **数据集**：WebArena (812 tasks, 6 domains), VisualWebArena (910 tasks, 3 domains), SCUBA (267 tasks, 3 roles)。
- **主要结果**：
  - Baseline (GPT-5-mini) 在 WebArena 达 54.8%，VWA 54.9%，SCUBA 49.6%，已超越 Prior RBank (53.9%) 与 AWM (32.7%)。
  - 默认顺序下 RBank 平均提升 +1.5%，AWM 为 -0.7%；但 RBank 增益 p=0.23，**统计不显著**。
  - 随机打乱顺序（Shuffle-1）下，RBank 从 +1.5% 逆转为 **-4.5%**（56.3% → 49.8%），AWM 同样显著下降。
  - 方差分析：71% 的案例方差上升，11 个案例相对增幅 >50%；Map 域 best-worst gap 扩至 8.2%，GitLab 达 7.8%。
- **干预效果**：+All 在 Shuffle-1 下将 RBank 从 49.8% 提升至 52.7%（+2.9%），**弥合了 31% 的退化**；Default 顺序性能保持。Shuffle-2 下仍有 1.1% 提升，但剩余 69% 差距未完全修复。
- **最强结果与提升幅度**：强基线下 RBank 在默认序取得 +1.5% 均值增益；加入三类规范信息后，+All 在乱序条件下实现 +2.9% 绝对提升（相对原始 RBank 乱序结果），是当前最优干预成效。

## 相关工作脉络
1. **AWM [11] / ReasoningBank [12]**：本文直接重评估的代表方法；Prior 工作在弱基线与单一运行下报告显著增益，本文揭示其在强基线与随机任务流下的性能衰退。
2. **WALT [20] / SCUBA harness [15]**：本文采用的无记忆 baseline 框架；相比 prior 显著提升了初始 pass rate，重新校准了自改进方法的收益基准。
3. **WebArena [13] / VisualWebArena [14]**：核心评测基准；本文指出其默认任务排序隐含 curriculum，呼吁在评估协议中加入随机序与多 run 统计。
4. **Agent Evaluation & Reliability [10, 50, 51, 52]**：Rabanser 等提出将 reliability 作为 agent 评估的一级指标；本文将其从单次会话扩展至多会话自改进场景，强调方差与顺序敏感性。
5. **Continual Learning Stability [2, 57, 58, 60]**：传统 fine-tuning 与 meta-learning 领域早已关注数据顺序与训练波动；本文指出 agent 的 stateful 记忆机制会进一步放大此类不稳定性，使评估更具挑战性。

## 局限性与未来方向
- **局限**：仅聚焦基于文本记忆的 Web 浏览智能体，未覆盖代码/数学领域或参数化/混合记忆方法；内存审查为定性抽样，未做全量自动化分析；当前干预仅解决部分 underspecification，剩余退化成因未知。
- **未来方向**：开发可扩展的内存自动化审查与过滤机制（memory validation）；设计支持人类及时干预的 oversight 接口；建立涵盖多 run 方差与随机任务序的标准化评估协议；探索任务流无序到达场景下的鲁棒自改进架构。

## 研究启发与可借鉴点
1. **评估协议必须升级**：自改进 agent 论文应强制报告多 run 统计（均值±标准差/best-worst gap）与随机任务序结果，单次运行 claims 已失去可靠性依据。
2. **Underspecification 注入范式可迁移**：将环境约束、任务 rubric、失败反馈显式写入 memory generation prompt，可作为通用技巧迁移至任意持续学习 agent 系统
