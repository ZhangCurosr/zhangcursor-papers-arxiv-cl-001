---
title: "Plans-You-Can-Check-Verifier-Grounded-Learning-of-an-Open-We"
source: https://arxiv.org/pdf/2608.25622v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:44:39"
---

# 论文速读：Plans-You-Can-Check-Verifier-Grounded-Learning-of-an-Open-We

## 一句话总结
论文将视频剪辑建模为**可执行的约束规划问题**，提出 RefineCut 框架：通过确定性验证器对多教师轨迹重放打分，替代直接模仿以构建 SFT 与偏好对，并引入 Rubric-结构化自改进 DPO 阶段；最终 8B 开放权重规划器在相同闭环协议下 VES 达 0.924，性能匹敌或超越 GPT-5.4、Qwen3-Max 等前沿闭源模型。

## 研究问题与动机
1. **规划策略不可学习**：现有工作流型视频创作系统（DIRECT、UniVA 等）仅将冻结的前沿模型嵌入脚手架，编辑策略随提示词固定，无法针对具体约束账本进行优化，也无法开源复用。
2. **缺乏可验证的剪辑决策层**：视频编辑本质是“先规划后渲染”的决策过程，但社区缺少带显式约束规格、结构化输出接口与机器可执行校验的规划级基准与协议。
3. **多教师轨迹噪声难直接模仿**：同一剪辑 brief 允许多种合法方案，教师首选项仅是猜测而非 ground-truth，直接 SFT 会继承分歧与错误分支。
4. **闭环评估与离线训练的分布偏移**：已有评测多依赖 rendered pixel 质量或 LLM judge，难以反映规划器在 Apply/Verify 循环中的真实决策能力。

## 核心贡献（创新点）
1. **形式化可执行视频编辑规划并提出 RefineCut-Bench**：首次将真实片段/音乐元数据、显式约束账本、多教师轨迹与确定性验证器评测协议统一于规划层面，填补了“决策层基准”空白。
2. **验证器重放蒸馏（Verifier-Replayed Distillation）**：通过规范化 + 确定性重放将噪声多教师分支转化为一致的 SFT 目标与混合粒度偏好对，使监督信号从“教师首选”变为“执行最优”。
3. **RefineCut-Evo 自改进阶段**：学生模型以验证器与固定任务评分规则（ER1–ER7）联合打分候选修复，筛选高边际硬负样本进行 DPO，实现训练后期向闭环自我进化的迁移。
4. **跨架构可迁移的规划训练范式**：验证器重放信号在 Qwen3-8B、Llama-3.1-8B、GLM-4-9B 上均带来显著增益；最终 8B 规划器在同协议下超越/持平其蒸馏来源的前沿模型。

## 方法详解
- **任务形式化**：实例 $x=(b, C, M, s_0, L)$，其中 $L$ 为显式约束账本（duration/transition/music_sync/clip inclusion/exclusion/repeat limit/pacing 七类字段），每项为 `(item_id, type, spec, satisfied, evidence)`。规划器 $\pi_\theta$ 在步骤 $t$ 输出 RefinePatch（RFC 6902 JSON Patch），验证器执行 $s_{t+1}=\text{Apply}(s_t,p_t)$ 并重算 $L_{t+1}$。
- **轨迹收集与规范化**：从 GPT-5.4、Qwen3-Max、DeepSeek-V4-Pro 收集 refine 轨迹，统一 JSON Pointer 路径至规范命名空间，校验 clip 引用是否落在 task-local alias 内，剔除 schema 非法项。
- **验证器重放与分支仲裁**：每步保留最多 4 个 Jaccard 不同的候选分支，由确定性验证器打分：$V(b)=0.35\Delta\text{CSR}+0.20\text{TargetedRepair}+0.20\text{ReqClipRecall}+0.10\text{PASR}+0.10\text{NoRegression}+0.05\text{Locality}$。取最高分分支作为监督目标。
- **Stage 1：Mixed-Pref 蒸馏**：Verified SFT（3,317 条通过过滤器的高质 patch，标准 next-token CE）+ 混合粒度 DPO（step-level 与 trajectory-level 偏好对共 2,380 对，$\beta=0.1$）。
- **Stage 2：RefineCut-Evo**：在 1,500 个训练状态采样 $K=4$ 学生候选，联合分数 $S(c)=0.65V(c)+0.35R(c)$（$R$ 为 ER1–ER7 加权 rubric），筛选边际 $\geq\tau$ 的硬负对，以 $\beta=0.05$ 进行 DPO，step 300 在 dev100 选优。
- **闭环推理**：测试时 Planner 读取当前 $s_t$ 与未满足 ledger 条目，输出单步 RefinePatch；验证器 Apply/Verify 后循环至多 $T=3$ 步，全程无教师 API 调用。

## 实验与结果
- **数据集与划分**：RefineCut-Bench 含 3,578 规范任务、7,971 条 caption 片段、499 首音乐轨；主测 Common-100，另设 canonical-clean (N=92) 与 Human50 自由指令集。
- **评估协议**：VES = 0.30 FCSR + 0.15 HardPass + 0.15 PASR + 0.15 ReqClipRecall + 0.10 DurPass + 0.10 TimelineValidity + 0.05 NoRegression；配套 Converged@3、失败分类学与人工盲评。
- **主结果（Common-100）**：Prompted 0.594 → Raw 0.620 → Verified 0.858 → Mixed-Pref 0.864 → **RefineCut-Evo 0.924**（相对 Mixed-Pref +0.060）；HARDPASS 0.670→0.820，CONVERGED@3 0.800→0.950。
- **跨骨干迁移**：同数据/协议下 Llama-3.1-8B Verified−Raw +0.153，GLM-4-9B +0.079；Required-Clip Recall 提升至 0.98/0.92/0.99。
- **同
