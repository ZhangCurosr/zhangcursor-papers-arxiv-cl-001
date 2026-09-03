---
title: "Plans-You-Can-Check-Verifier-Grounded-Learning-of-an-Open-We"
source: https://arxiv.org/pdf/2608.25622v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:44:41"
---

# 论文速读：Plans You Can Check: Verifier-Grounded Learning of an Open-Weight Planner for Executable Video-Editing

## 一句话总结
针对视频编辑的决策层规划问题，本文提出 RefineCut 框架，通过显式约束账本与确定性校验器对多教师轨迹进行规范化重放与仲裁，再经 Rubric-Verifier 联合打分的自改进 DPO 训练出轻量开放权重规划器；最终 8B 模型在 RefineCut-Bench 上 VES 达 0.924，在完全相同的闭环协议下匹配或超越 GPT-5.4、Qwen3-Max 等前沿闭源教师。

## 研究问题与动机
1. **编辑本质是约束规划而非像素生成**：实际视频编辑需将脚本、片段池、音乐元数据与硬约束转化为可执行的时间线，而现有系统要么直接生成/变换像素（Text-to-Video / Instruction-Editing），要么将冻结的前馈模型套入手工脚手架工作流，规划策略永远无法针对任务简报的硬约束进行优化。
2. **编辑决策具备“可验证性”**：与开放域生成不同，编辑输出可通过显式约束账本（Constraint Ledger）进行逐条确定性交验，这为执行级监督（Execution-based Supervision）提供了天然信号，无需依赖不稳定的 LLM Judge。
3. **多教师轨迹天然含噪声**：同一剪辑简报允许多种合法方案，教师的“首选分支”仅是猜测而非 Ground Truth；直接模仿异构轨迹会继承 Schema 不一致与执行失败，需要确定性仲裁机制过滤噪声。
4. **缺乏规划层基准与可复现协议**：现有视频编辑基准多评估渲染后的像素质量或仅做离线打分，缺少将真实片段/音乐元数据、显式约束、多教师轨迹与确定性校验绑定的规划层评测体系。

## 核心贡献（创新点）
1. **形式化可执行视频编辑规划并开源 RefineCut-Bench**：首次发布包含 3,578 规范任务、7,971 带_caption 片段、499 首音乐轨道、显式约束账本与多教师轨迹的规划层基准，配套确定性校验器与统一评估协议。
2. **校验器重放蒸馏（Verifier-Replayed Distillation）**：将异构教师轨迹规范化为 Canonical Patch Trajectory，逐分支重放于确定性校验器计算 6 维执行信号，以校验器最优分支替代教师原始首选分支作为 SFT 目标与混合粒度偏好对，从根本上过滤执行噪声。
3. **RefineCut-Evo 自改进阶段**：学生在训练状态上采样候选修复，经确定性校验器与七维任务评分表（ER1–ER7）联合打分后，仅保留高边际偏好对执行离线 DPO，实现推理时完全脱离教师调用的闭环规划。
4. **跨骨干可迁移性与前沿竞争力**：校验器重放监督在 Qwen3-8B、Llama-3.1-8B、GLM-4-9B 上均显著提升；最终 8B 规划器在相同闭环协议下 VES 达 0.924，显著高于 GPT-5.4（+0.030）与 Qwen3-Max（+0.150），与 DeepSeek-V4-Pro 统计持平，且推理延迟仅 11.7 s/task。

## 方法详解
1. **任务与状态定义**：任务实例 $x = (b, C, M, s_0, L)$，其中 $b$ 为自然语言简报，$C$ 为带 caption 与视觉元数据的片段池，$M$ 为可选音乐节拍/能量元数据，$s_0$ 为初始时间线状态，$L$ 为显式约束账本。规划器 $\pi_\theta$ 在步骤 $t$ 输出 RFC 6902 风格的 RefinePatch $p_t$，校验器执行 $s_{t+1} = \text{Apply}(s_t, p_t)$ 并重算 $L_{t+1} = \text{Verify}(s_{t+1})$，目标是在小修复预算内使 $L$ 全部满足。
2. **Stage 1 校验器重放蒸馏**：
   - 采集 GPT-5.4、Qwen3-Max、DeepSeek-V4-Pro 的多分支轨迹，统一规范化为按 `(task_id, teacher_id)` 索引的 Patch Trajectory，校验 clip 引用是否在 task-local alias 范围内。
   - 每步保留最多 4 个 Jaccard-distinct 分支，逐分支重放计算六维加权分 $V(b) = 0.35\Delta\text{CSR} + 0.20\text{TargetedRepair} + 0.20\text{ReqClipRecall} + 0.10\text{PASR} + 0.10\text{NoRegression} + 0.05\text{Locality}$。
   - 选取 $V$ 最高的分支作为 Verified SFT 目标（标准 next-token CE，1,200 steps，LR=5e-5）；同时构造步级与轨迹级混合偏好对，用离线 DPO 训练得到 Mixed-Pref 8B（800 steps，LR=2e-6，$\beta=0.1$
