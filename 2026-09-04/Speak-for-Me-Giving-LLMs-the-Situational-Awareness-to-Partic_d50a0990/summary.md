---
title: "Speak-for-Me-Giving-LLMs-the-Situational-Awareness-to-Partic"
source: https://arxiv.org/pdf/2609.03923v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-08 01:51:10"
---

# 论文速读：Speak-for-Me-Giving-LLMs-the-Situational-Awareness-to-Partic

## 一句话总结
本文提出 CAPA（Collaborative Agent Predictive Architecture），一种通过显式维护会议状态（追踪议题、立场、发言权等）的感知-行动-校准闭环架构，解决 LLM 代理在在线会议委托中因“失语”而无法把握介入时机的问题；在 AMI 语料库上，该架构将沉默率从 51.4% 降至 2.5%，可信恢复率翻倍，且幻觉率仅 0.6%。

## 研究问题与动机
- **核心问题**：LLM 代理作为缺席参会者的在线会议委托（meeting delegation）时，难以持续判断何时介入对话，导致大量实质性发言机会被遗漏。
- **现有方法不足**：纯 Prompt-based 代理直接依赖原始对话历史生成，缺乏对参会者立场、未决事项、前置覆盖度、发言权（floor）等结构化状态的追踪，引发“沉默弃权”（51.4%）、话题错位、重复发言和时机错误。
- **上下文堆叠无效**：单纯拉长输入窗口无法解决长程多轮对话中的推理缺陷，状态漂移（contextual drift）是根本原因，扩展 context 反而可能加剧“迷失中间”现象。
- **评估手段缺失**：传统生成指标仅衡量逐轮文本重叠，无法评估介入时机（timing）与命题级贡献（proposition-level）的匹配度，缺乏面向 episode 的统一协议。

## 核心贡献（创新点）
- **提出 CAPA 架构**：构建感知-行动-校准（perceive-act-recalibrate）循环，通过显式会议状态 `m_t` 替代原始对话历史作为决策基底，将多主体委托从“上下文阅读”转变为“持续状态维护”。
- **解耦策略决策与语言生成**：Controller 先 commitment 离散命题 `z_t`，再由 Generator 按参与者风格实现，使 SILENT 成为一等公民动作，从根本上缓解幻觉与时机错位。
- **引入双裁判状态校准机制**：Predictor 预演下一轮意图，Environment Judge 与 Delegate Judge 分别评估预测与实际转向、代理动作与实际转向的匹配度，校准信号回流至状态更新而非输出重写。
- **设计 episode-level 评估协议**：以参会者实际 idea units 为锚点，程序化评分是否介入、何时介入、内容是否匹配，并验证 LLM judges 与人工标注一致性达 Cohen’s κ = 0.71。

## 方法详解
- **共享内存（Shared Memory）**：按生命周期正交划分四类存储：长期记忆（稳定角色/风格 `p_u*`）、会话简报（会议特定目标 `b_u*`）、工作记忆（动态状态 `m_t`）、情节记忆（历史动作轨迹），避免混合生命周期导致的冗余与过滤开销。
- **Perceiver（状态估计）**：每轮接收新 utterance 与旧状态 `m_{t-1}`，通过 schema-constrained LLM 调用更新六维状态：active topic、decisions/open questions、prior coverage、participant stances、floor，将对话状态追踪从任务槽位扩展至交互变量。
- **Act 模块（Controller + Generator）**：Controller 读取 `m_t` 和 `c_u*`，调用三个 Helper（候选策展人按相关性排序简报项、覆盖评估器判断是否冗余、发言评分器校验 floor 动态）决定 SILENT/SPEAK；SPEAK 时先 Commit 命题 `z_t`，再由 Generator 以目标风格生成自然语言，遵循 plan-then-realize 经典 NLG 范式。
- **Recalibrator（状态校准）**：行动前 Predictor 生成意图级预测；行动后 Environment Judge 比对预测与实际接续（隔离感知误差），Delegate Judge 比对代理动作与实际接续（隔离行动误差）；两者判决融合后更新 `m_t`，实现自我修正。
- **因果回放协议**：评估时严格限定因果性，delegate 仅可见当前及之前 turn，目标 idea unit 和 future anchor 被隐藏，确保线上场景保真；状态更新仅在对应 ground-truth turn 被因果观测后才消费。

## 实验与结果
- **数据集**：AMI Meeting Corpus（scenario portion），137 场会议，四人角色（PM, ID, UI, Marketing）设计遥控器场景。
- **基线**：Transcript-only delegate（Hu et al., 2025 重实现，配 50-turn 原始上下文）、Reflexion-style baseline（滚动程序记忆但无显式状态/Helper/状态校准）、State-ablated CAPA（去状态+长窗口）、No-recalibration CAPA。
- **主要结果**：CAPA 将沉默率从 51.4% 降至 2.5%；Loose recall 从 26.1% 提升至 52.2%（翻倍）；Strict recall 从 10.7% 提升至 25.1%；Decision F1 提升 24.9 个点（38.1 → 63.0）；幻觉率保持 0.6%，冗余率 0.0%。
- **时序表现**：93.8% 的可信匹配早于或对齐人类 anchor 时刻，中位提前 1 个 turn；两组系统在 timing 分布上一致，差异仅在于是否主动介入。
- **错误转移**：失败模式由 opaque omission 转为可归因的 selection near-miss（不完整匹配、不同命题、晚期匹配、安全过滤），便于模块级诊断。
- **跨模型/跨语料**：在 Gemini-2.5-Pro、Llama-3.3-70B、Qwen3.6-27B 上 Decision F1 稳定在 57.9–69.2；ICSI 语料（真实研究组讨论，平均 6 人）验证沉默率 1.2%，Loose recall 73.6%，F1 64.8%，幻觉 1.8%，证明架构可迁移性。
- **消融结论**：移除会议状态导致 loose recall 下降 22.4%，F1 下降 24.7%，冗余上升 10.7%；移除 Recalibrator 仅使 loose recall 下降 3.9%、uncredited 上升 4.6%，F1 与 grounding 无显著变化，说明状态负责“是否发言”，校准负责“发言质量”。

## 相关工作脉络
- **Meeting Agents（Mao et al., 2024; Alsobay et al., 2025 等）**：聚焦会后摘要、引导与问答，属被动观察者；本文转向在线实时干预与特定利益相关者代表，填补持续参与空白。
- **MEETING DELEGATE 基准（Hu et al., 2025）**：首次系统记录 prompt-only 委托的失败模式；本文在其发现的 silient abstention / cue policy / content policy 缺陷基础上提出架构级解决方案，
