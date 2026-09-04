---
title: "Zero-Shot-Self-Orchestration-with-Ledger-Based-Control-for-I"
source: https://arxiv.org/pdf/2608.26480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:47:35"
---

# 论文速读：Zero-Shot Self-Orchestration with Ledger-Based Control for Improved LLM Coding Performance

## 一句话总结
本文提出了一种零样本、免训练的 Manager-Worker 编排框架，通过在共享文件系统账本上动态分解任务与迭代验证，在固定底层模型与评测集的前提下严格评估了多智能体协作对 LLM 编程能力的真实收益。实验表明该框架可显著提升中低参数及关闭推理模型的得分（最高 +42 个百分点），且以约三倍 Token 成本换取的准确率提升，往往比直接调用更昂贵的旗舰模型更具成本效益。

## 研究问题与动机
- **混杂变量导致结论模糊**：现有 LLM 多智能体系统效果参差不齐，多数对比实验同时改变了 Token 预算、工具调用、Prompt 结构与检索策略， aggregate gain 无法归因于“编排本身”。
- **缺乏严格的消融对照**：学界亟需在控制模型、温度、评测集完全一致的前提下，量化“仅引入编排层”对单轮调用的边际贡献。
- **长上下文生成的脆弱性**：大模型单次长输出易遭遇截断（truncation）、上下文溢出或陷入无效循环，单一 Pass 难以可靠交付完整解法。
- **成本效益的工程权衡**：实际部署中，通过轻量编排提升中等模型性能，是否比直接升级至更大参数模型更具性价比，尚未有系统化实证。

## 核心贡献（创新点）
- **提出零样本自编排框架（Zero-Shot Self-Orchestration）**：使用固定 Prompt 的 Manager 动态维护任务队列、决策终止时机，配合 Worker 与 Verifier 在共享文件账本上协作，全程无需任何训练或 benchmark 特定调优。与 Conductor、Fugu 等训练型编排器的本质区别在于完全剥离了学习到的路由策略，仅依赖提示词与规则驱动。
- **设计了控制混杂变量的严格对照协议**：固定同一模型、同一 LiveCodeBench 100 题集与采样温度，仅对比“单次调用”与“Manager-Worker 循环”两种条件，确保观测到的性能差异纯粹来源于编排结构本身。
- **揭示了编排收益的条件性机制**：发现增益高度依赖模型规模与推理模式——对小模型或关闭推理（reasoning off）的模型提升最大（如 Kimi-K3 +42），对已具备强单步推理的大模型增益有限甚至为负（Qwen3.6-35B -9）；并通过转录分析定位“上下文管理”与“问题分解”为核心增益通道。
- **提供了系统的成本-性能权衡分析**：量化证明编排框架虽使 Token 消耗增至约 3 倍，但单位准确率提升的边际成本远低于直接扩容模型（如 GPT-5.6-Terra+Manager 仅需 Fable 5 单次价格的约五分之一即可达到相近准确率）。

## 方法详解
- **共享账本工作区（Shared Filesystem Workspace）**：所有角色共享一组持久化文件（`task.md`, `plan.md`, `tasks.json`, `notes.md`, `solution.py`），计算状态跨调用保持，避免单一上下文窗口无限膨胀或截断丢失关键信息。
- **Manager-Worker 动态控制流**：
  1. **Manager 规划**：读取题目，输出 3–6 句整体策略与 3–6 个种子子任务写入 `plan.md`。
  2. **Worker 头脑风暴**：首个 Worker 不写代码，仅识别核心难点、候选方案与陷阱，追加至 `notes.md`。
  3. **Manager 调度循环**：折叠计划与笔记，去重/合并/补充任务后写入 `tasks.json`，若判定完成则终止，否则派发下一个最高价值子任务。
  4. **Worker 执行**：新 Worker 读取当前 `solution.py` 与任务列表，执行单个子任务并重写代码与笔记。
  5. **Verifier 验证（v2 新增）**：将候选代码作为独立子进程运行，against 题目公开的 stdin 样例测试，将 pass/fail  verdict 反馈给 Manager 作为不可覆盖的真值信号。
  6. **Finalizer**：若预算耗尽或 Manager 无新任务，触发最终化 Worker 输出确定性解。
- **守卫与温度设置**：最大迭代轮次 `MAX_ITERS=10`；无进展守卫（重复派发相同任务即停）；截断摘要器（对超上限 Worker 输出做简短摘要防信息丢失）。Manager 规划/头脑风暴温度 0.3/0.4，任务执行/调度温度 0.2。
- **零样本设定**：所有角色使用同一基础模型，独立角色提示词固定，无任何训练或 per-benchmark 调参。

## 实验与结果
- **数据集与基线**：LiveCodeBench `release_v6` 硬分片最新 100 题；基线为同一模型单次调用（Single），对照组为引入编排框架（Manager）。
- **核心指标（128k cap, reasoning ON, 5 passes）**：
  - Qwen3.8-27B：63.0 → 86.4（Δ = +23.4, p < 10⁻⁴）
  - GPT-5.6-Luna：67.2 → 77.8（Δ = +10.6）
  - GPT-5.6-Terra：77.0 → 85.0（Δ = +8.0）
  - Opus-5：85 → 91（Δ = +6，单 pass）
- **Reasoning OFF 条件（16k cap, 5 passes）**：Kimi-K3 +30.4，Minimax-M3 +11.0，Qwen3.5-9B +7.2；Qwen3.6-35B 无显著变化（-1.2）。128k 下 Kimi-K3 增益达 +42。
- **截断与空输出缓解**：Qwen3.8-27B 单次调用 500 problem-passes 中有 35 个无代码输出，Manager 挽回 25 个（贡献约 5.0 个百分点，占总增益的 1/5）。GPT-5.6 系列与 Fable 5 几乎无截断，增益完全来自逻辑分解。
- **成本分析**：编排使单次 Pass 账单增至约 3 倍。但性价比更优：GPT-5.6-Terra+Manager（$11.71/pass）性能与 Fable 5 单次（$61.11/pass）无统计差异（p = 0.59），节省约 80% 费用；Qwen3.8-27B+Manager（$51.75）逼近 Fable 5 得分且支持本地自托管。
- **评测器缺陷修正**：发现 LiveCodeBench 原生 harness 中 `sys.stdin.buffer.readline()` mock 实现存在 bug，修正后对所有存档生成重新计分，相关数字已同步更新。

## 相关工作脉络
- **训练型编排器（Learned Orchestrators）**：如 Fugu、Conductor 通过 RL/偏好数据学习协调策略。本文定位为其零样本对照基线，旨在剥离学习因素后评估纯提示词编排的独立价值。
- **固定流水线工作流**：如 MetaGPT、ChatDev、AgentCoder 采用静态 SOP 流转。本文差异在于抛弃固定管道，由 Manager 动态重编任务列表并自主决策终止，适应性更强。
- **共享黑板/去中心化架构**：如 ARIADNE（MCTS+黑板）、LbMAS、Mixture-of-Agents。本文工作区功能类似黑板，但去除了 MCTS/奖励模型/投票聚合，仅依赖单 Manager 的 plain task-list loop，结构更轻量。
- **预算公平对比研究**：如 Wang et al.、Tran & Kiela 主张等 Token 预算下比较。本文不追求等计算量，而是论证“额外计算预算购买准确率的性价比”优于直接扩容模型。

## 局限性与未来方向
- **服务提供方可靠性噪声**：部分实验依赖 OpenRouter 网关，路由抖动、中断响应、输出意外截断等基础设施噪声可能影响绝对分数可复现性（作者已因此改用 Pinned Backend 强化核心实验）。
- **单一评测维度**：仅验证于竞技编程代码生成；数学（AIME/MATH-500）与知识类基准因旗舰模型已接近天花板而未充分展开，跨域泛
