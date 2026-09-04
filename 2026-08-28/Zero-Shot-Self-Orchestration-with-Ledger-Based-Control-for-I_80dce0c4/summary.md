---
title: "Zero-Shot-Self-Orchestration-with-Ledger-Based-Control-for-I"
source: https://arxiv.org/pdf/2608.26480v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:47:55"
---

# 论文速读：Zero-Shot-Self-Orchestration-with-Ledger-Based-Control-for-I

## 一句话总结
该论文在严格固定底层模型与试题集的前提下，零样本、免微调地验证了基于共享文件系统工作空间的 Manager-Worker 编排框架对 LLM 编程能力的增益；结果表明该架构能显著提升中小规模模型及未启用深度推理的模型（如 Qwen3.8-27B 提升 +23.4 分），但可能干扰部分强推理模型的固有策略，且开销约为单次的 3 倍，却仍比直接升级更大模型更具成本效益。

## 研究问题与动机
- **多智能体增益证据混杂**：现有研究普遍声称多智能体系统优于单模型，但对比实验常同时改变 token 预算、工具调用、Prompt 与检索策略，难以剥离“编排架构本身”的真实贡献。
- **训练型协调器成本高且可复现性差**：如 Fugu、Conductor 等需对协调策略进行 RL/GD 训练，依赖任务特定微调，工程落地门槛高。
- **固定流水线缺乏动态适应性**：MetaGPT、AutoGen、AgentCoder 等预设静态 SOP 或固定角色分工，无法根据中间进展实时重排任务或终止冗余循环。
- **长上下文截断与状态丢失问题**：单次长输出易触发 API 或上下文上限，导致空响应或思路中断，亟需一种跨调用持久化状态的低成本机制。

## 核心贡献（创新点）
- **提出零样本自编排（Zero-Shot Self-Orchestration）框架**：仅依靠固定 Prompt 与共享文件系统即可实现 Manager-Worker 动态协同，无需任何训练或基准调参，直接测量编排架构本身的纯增益。
- **设计 Ledger-Based 共享工作空间机制**：通过 `plan.md`、`tasks.json`、`notes.md`、`solution.py` 等结构化文件将计算状态跨 Agent 调用持久化，有效替代易溢出的对话历史，降低单轮上下文长度与截断风险。
- **引入带样本测试门控的六步循环控制流**：Manager 每轮审阅进度并重排任务队列，Worker 专注执行单一子任务，新增 Verifier 环节以公开样例运行结果作为强制 Ground Truth 反馈，形成闭环纠错。
- **提供严格控变量的多模型对照实验**：在 9 个模型（5 开源/4 闭源）的 LiveCodeBench 硬集上对比单次调用与脚手架编排，量化了增益的条件性、截断救援效应与成本效益曲线。
- **发现并修复 LiveCodeBench 评估器的隐藏缺陷**：定位到 `sys.stdin.buffer.readline()` mock 实现的二进制视图 bug，重测全部生成结果，确保排行榜数据的准确性。

## 方法详解
- **共享文件系统工作空间**：所有角色调用同一基础模型，但每次请求为独立上下文。工作空间包含题目文件、宏观计划、任务列表（含 id/desc/status/result）、累积笔记与当前最优代码，状态跨调用持久化且不依赖单一 Agent 的上下文窗口。
- **六步控制流（v2 脚手架）**：
  1. **Manager-Plan**：读取题目，输出 3–6 句策略与 3–6 个种子子任务。
  2. **Worker-Brainstorm**：首名 Worker 不写代码，仅分析核心难点、候选方案与陷阱，追加至 `notes.md`。
  3. **Manager-Loop**：合并重复项、标记已完成、选取下一个最高价值任务，或宣告问题解决。
  4. **Worker-Do**：新鲜 Worker 执行单一任务，重写 `solution.py` 并提议后续步骤。
  5. **Verifier-Test**（v2 新增）：将候选程序作为真实子进程运行公开样例测试，pass/fail  verdict 作为不可覆盖的外部信号反馈给 Manager。
  6. **Finalizer**：当预算耗尽或 Manager 陷入死循环时，由 Finalizer 输出确定性最终答案。
- **安全守卫与温度设置**：最大循环次数 `MAX_ITERS=10`；无进展守卫（重复下发相同任务即停）；截断总结器（Worker 超限时由短调用总结残存思路）。Manager 规划/头脑风暴温度 0.3/0.4，任务执行与列表整理温度 0.2；基线单次调用温度同为 0.2。
- **基线对照设计**：Single-call baseline 使用完全相同的模型、温度与题目，仅一次性输出，无工作空间、无循环、无工具调用，确保唯一变量为“是否接入编排脚手架”。

## 实验与结果
- **数据集与评估**：LiveCodeBench `release_v6` 硬分级最新 100 题，使用官方隐藏测试 Evaluator。核心实验为固定后端 v2 脚手架、128k 输出上限、推理开启、5 次独立重复。
- **主要精度结果（128k, Reasoning ON, 5 passes）**：
  - Q
