---
title: "Prime-Agent-A-Self-Improving-RLM-Harness"
source: https://arxiv.org/pdf/2608.23552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:12:40"
---

# 论文速读：Prime-Agent-A-Self-Improving-RLM-Harness

## 一句话总结
Prime Agent 是一个面向长周期评估与编程智能体工作流的开源框架，通过持久化 IPython REPL、递归子智能体异步通信与 Continual Harness 在线精炼机制，将执行环境与模型策略彻底解耦。该框架显著提升了模型在长程交互任务中的实际表现（ARC-AGI-3 RHAE Best@1 从 30% 跃升至 95.5%），并验证了其在自主研究、系统构建与多日持久任务中的工程有效性与可扩展性。

## 研究问题与动机
- **模型计算边界受限**：LLM 本质是有界顺序处理器，其决策仅依赖权重与当前活动上下文，难以独立完成需要大量外部状态、工具调用与测试时计算（test-time compute）的长周期任务。
- **Harness 与模型能力混淆**：现有框架多采用固定工作流图或无状态工具调用，容易因状态丢失、资源计量不准、过早终止或接口摩擦导致“假性失败”，无法准确剥离执行器瓶颈与模型真实上限。
- **长程信息管理能力缺失**：复杂任务需要跨轮次的上下文压缩、程序化数据聚合、持久化记忆与可复用技能，但现有方案缺乏统一的轨迹留存与在线状态演化机制。
- **评估基础设施不标准**：不同研究间的性能对比常因底层执行协议、补偿策略与成本计量不一致而失真，亟需一套标准化、可审计的长周期评估基座。

## 核心贡献（创新点）
1. **持久化 REPL 与 RLM 异步原语**：为每个会话提供独立的 IPython 环境，模型通过 `await rlm(...)` 创建子会话并异步持有句柄，实现测试时计算的程序化调度。（区别于传统固定工作流或无状态工具调用，将上下文处理与执行语义完全分离，支持并行委派与结果延迟聚合。）
2. **Continual Harness 在线精炼机制**：支持从执行轨迹中提取可版本化的 Prompt 提示、事实记忆、可执行技能与子智能体规格，实现不更新权重的“自我改进”。（区别于静态 Prompt 工程或单次记忆检索，将历史证据转化为可持续复用、支持回滚的结构化状态。）
3. **Daemon 托管的直接 Agent-to-Agent 通信拓扑**：基于守护进程管理会话生命周期，通过家族作用域异步队列实现父/子/兄弟节点间消息传递，并暴露可视化 Human-Agent 干预接口。（区别于中心化编排器或纯共享内存架构，支持动态递归树扩展与低侵入式人工介入。）
4. **标准化长周期控制与资源审计语义**：引入自主模式（Autonomous Mode）、目标保持（Goal Retention）与心跳调度（Heartbeats），统一聚合根与子会话的 token、时间与成本消耗，明确区分 Harness 故障与模型失败。（区别于各 benchmark 自定义运行协议，提供可复现、可审计的评估基线。）
5. **跨多类长程任务的显著性能提升**：在 ARC-AGI-3、EmulatorBench、Factorio、MazeBench 等基准上大幅超越或持平原生/Harness 方案，验证了框架的工程有效性与泛化潜力。

## 方法详解
- **L0–L3 状态层级架构**：系统将信息状态划分为四层：L0（模型权重， Learned Computation）、L1（活动上下文， Token-Visible Working State）、L2（持久化 REPL 与子智能体， Code/Tools/Retained Values）、L3（磁盘备份， History/Artifacts/Memories/Skills/Prompts）。跨层信息流动由程序化操作驱动：L2 值序列化进入 L1，压缩（Compaction）将长前缀替换为摘要并归档至 L3，Revisable Memories 按需注入补充提示。L2 层采用 **Agentic Garbage Collection**，由模型按需创建、摘要或删除中间值与子会话，防止上下文膨胀。
- **RLM 异步递归执行**：每个会话拥有独立 IPython 内核，中间计算结果跨轮次保留且默认不进入上下文，仅在显式序列化后才注入 L1。模型通过 `rlm` 原语异步调度子会话，父进程继续本地计算，结果通过 daemon 队列延迟抵达；支持并行多个 `rlm` 调用与后续状态恢复（compaction/restart 后仍可复用句柄）。
- **Continual Harness 精炼闭环**：轨迹证据通过 `/refine` 命令或后台模型调用触发，在轮次边界应用版本化编辑；每条编辑记录触发源、预期效果与时间戳，支持全量回滚。未修改基础系统提示，仅追加补充状态。常用计算固化为 Skills，重复协调模式固化为 Subagent Specs，错误假设修正为 Memories。
- **会话生命周期与 Agent View**：Root 与子会话均由 Daemon 托管，状态分为 Running（执行中）、Idle（加载无活动回合）、Inactive（卸载但可恢复）。客户端 detach 不中断执行，稳定 ID 保持递归拓扑一致。Agent View 提供可视化树状结构，支持只读观察（agent-observe）、定向消息发送（agent-message）与动态附加/分离。
- **长周期控制三机制**：① Autonomous Mode 在显式预算内持续回合并通过任务指定测试条件判定结束；② Goal 模式跨续期保持单一目标直至智能体显式标记完成；③ Heartbeat 支持 cron/定时触发回合。所有资源消耗在根与后代会话间统一聚合，事件历史（模型调用、工具、消息、干预、重试、验证器输出、harness 编辑）全程可审计。

## 实验与结果
- **ARC-AGI-3 交互推理（RQ1）**：RHAE Best@1 从 30% 提升至 **95.5%**，显著高于官方 Claude Code / Codex 报告值；额外 output tokens 与 API 成本可高效转化为进度，证明低摩擦接口使测试时扩展更有效。
- **长上下文
