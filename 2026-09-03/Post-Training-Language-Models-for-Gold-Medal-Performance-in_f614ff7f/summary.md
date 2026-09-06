---
title: "Post-Training-Language-Models-for-Gold-Medal-Performance-in"
source: https://arxiv.org/pdf/2609.02849v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 22:36:49"
field: "代码生成与推理能力评测"
keywords: ["competitive programming", "post-training", "SFT", "reinforcement learning", "test-time compute", "GenCorrect", "IOI", "LLM code generation"]
innovations: ["端到端流水线：合成推理轨迹+SFT+可执行奖励RL+测试时多轮GenCorrect", "以累计子任务反馈与多样性中心选择为核心的迭代精化策略", "在IOI 2026实时同条件下首次超越人类最高得分"]
benchmarks: ["IOI 2025", "IOI 2026", "ICPC 2025", "LiveCodeBench Pro"]
---

# 论文速读：Post-Training-Language-Models-for-Gold-Medal-Performance-in-Coding-Competitions

## 一句话总结
本文提出一套从数据筛选、合成推理轨迹生成、监督微调（SFT）、强化学习（RL）到测试时迭代精化（GenCorrect）的端到端专门化流水线，并在 IOI 2025/2026 上实现超越人类金牌选手的竞技编程表现（如 Ultra-CC 在 IOI 2026 实时评测中拿到 535.4/600）。

## 研究问题与动机
- 竞技编程被视为检验 LLM 推理与编码能力的强指标，但已有系统多为封闭式或混合多种改动，导致各组件贡献难以隔离。
- 现有 Medal-level 结果多依赖专有模型或大量专用工程，缺少可复现、可归因的开源化全流程与实证分解。
- 仅靠模型规模或单阶段训练仍不足以稳定达到并超越竞赛阈值，需要结合训练与测试时计算的系统设计。
- 需要在贴近人类参赛约束（时间、提交次数、无网络）的条件下验证 AI 系统的真实竞争力。

## 核心贡献（创新点）
- 提出端到端竞技编程流水线，覆盖大规模问题筛选、合成推理/自我改进轨迹生成、长上下文 SFT、可执行奖励 RL 以及测试时多轮 GenCorrect。与已有工作相比，本文强调对各阶段贡献的可分离实证分析与开源复现路径。
- 引入 GenCorrect：以提交预算为约束的多轮生成—多样性选择—执行反馈—条件精化闭环，避免将问题机械划分为子任务，而是让模型自主决定优先攻克的子任务。相比仅靠单次采样或简单自修改的方案，它更强调执行落地与多样性控制。
- 给出 Nano-CC（3B 活跃参数）与 Ultra-CC（55B 活跃参数）两套体系，量化 SFT、RL、模型规模与测试时计算的相对贡献。与仅提升某单阶段的方案相比，本文更强调“小模型强后训练+测试时放大”和“大模型轻后训练”两种可行路线。
- 在 IOI 2026 实时评测中，以与人类相同的时长、提交限制与无网络约束取得 535.4/600，超越当届人类最高分。与赛后回顾性评测相比，本文证明了系统在真实竞赛环境中的可用性。
- 提供针对实时竞赛的工程适配方案（更大最终轮候选池、NVFP4 量化以提升吞吐、GLM-5.2 教师数据选取），为高约束场景下的系统部署提供参考。

## 方法详解
- 数据筛选：从近二十年 16 类区域/国际竞赛及在线平台收集 22,000 题，构建含题目、约束、测试、辅助文件与参考解的可执行环境；剔除评估集（IOI 2025/ICPC 2025/LiveCodeBench Pro）并去重；RL 阶段额外过滤执行过慢的题目。
- SFT：用 DeepSeek-V4-Flash 为 Nano 生成 1.2M、为 Ultra 生成 477,642 条推理轨迹，难题分配更多生成，并包含“教师基于旧解自我改进”的轨迹以让模型学到迭代精化行为；Nano 训练 3 个 epoch，Ultra 训练 1 个 epoch，序列打包至 262K tokens。
- RL（仅 Nano-CC）：使用 3,219 题（2,847 训练/372 验证），采用 GRPO，每步 64 prompt、每 prompt 16 rollouts、温度 1.0；C++17 程序编译执行后以是否满分作为二元终态奖励（1/0），采用 token-level clipped policy gradient，无 reference-policy KL 惩罚，按验证集性能选取 checkpoint。
- GenCorrect（至多 5 轮，IOI 每轮提交 10 次，共 50 次）：
  - 生成：首轮仅依赖题干；后续轮同时使用历史解答与评估反馈。
  - 多样性选择：先以无分启发式初始化中心集合 C，再迭代选取与现有中心最远距离的候选：
    $$c_{\mathrm{next}} \in \arg\max_{c \notin C} \min_{z \in C} [1 - \sin(c, z)]$$
    其中 $\sin(c,z)$ 为结构化/词袋类相似度；选出 c=10 个中心后把候选分配到最近中心，并在每个簇中选 $Q(c)$ 最高的代表。
  - 执行：提交 10 个代表，IOI 反馈为各子任务得分。
  - 精化：累积每子任务最优得分：
    $$A_r(t) = \max\left(A_{r-1}(t), \max_{c \in S_r} s_t(c)\right),\quad A_0(t)=0$$
    下一轮以累计子任务得分向量及三条互补参考（最强整体、最大剩余缺口、覆盖面最广）为条件再生成。
- 竞赛专项适配：最后一轮将候选池扩至 1,000 并按 GenCluster 式执行筛选排名；使用 NVFP4 + FP8 KV cache + MTP=5 提升吞吐；选用 GLM-5.2 作为 Ultra SFT 教师数据以提升同预算下单位时间生成质量与长度控制。

## 实验与结果
- 数据集与基准：IOI 2025、ICPC 2025、LiveCodeBench Pro；IOI 2026 为赛前实时 prospective 评测。
- 基线：gpt-oss-120b、Qwen3.6-35B-A3B、Nemotron-Cascade-2-30B-A3B、Nemotron-3 Nano/Ultra base、DeepSeek-V4-Flash/Pro、GLM-5.2 等；所有竞赛结果由统一评测 harness 获得。
- 主要结果（单样本 Score@1/Pass@1）：
  - Nano-CC：IOI 2025 48.5%（≥ 多数基线，仅次于 DSV4 Flash/Pro、GLM-5.2），ICPC 2025 51.0%，LCB Pro 71.6%。
  - Ultra-CC：IOI 2025 50.7%，ICPC 2025 57.4%，LCB Pro 74.5%，为本文模型中绝对最高。
- 分量贡献：
  - SFT 对 Nano-CC 贡献最大：IOI 21.7%→47.3%，ICPC 16.9%→46.7%，LCB Pro 17.6%→70.7%。
  - RL 锦上添花：IOI 46.7%→48.5%，ICPC 47.3%→51.0%，LCB Pro 70.7%→71.6%；直接从 base 启动 RL 仅到 24.9%，无法替代 SFT。
  - 测试时 GenCorrect：Nano-CC 在 IOI 2025 由 360.6 提升至 468.2（+107.6 分）；Ultra-CC 由 343.9 提升至 502.0（+158.1 分），并在三轮后超越 IOI 2025 金牌线。
- 竞赛实战：Ultra-CC 在 IOI 2026 实时条件下获得 535.4/600，超金牌线 174.3 分、超当届人类最高分 37.1 分；通用五轮 GenCorrect 离线均值为 521.72（495.0–545.8）。

## 相关工作脉络
- AlphaCode/AlphaCode 2 奠定基础的大规模采样—过滤—行为聚类范式，本文在此基础上以执行反馈与子任务级积累进行持续精化。
- OpenAI o1-ioi/o3、Google DeepMind ICPC 系统等多报道 medal 表现但偏专有或工程化，本文强调可复现的开源化流水线与分量归因。
- GenCluster 利用大规模生成与执行筛选达成 IOI 金牌；本文进一步引入多样中心选取与多轮反馈条件生成，并将策略扩展到实时竞赛工程部署。
- OpenCodeReasoning/OpenCodeReasoning-II 探索合成推理与自批判；本文把这些思想并入 SFT 轨迹（含自我改进样例）并在 RL 与测试时双重落地。
- DeepSeek-R1/DAPO 等确立可验证奖励 RL 的可扩展配方；本文采用 GRPO 与二进制终态奖励，强调长轨迹与稀疏相对奖励下的稳定优化。
- 评测体系方面，LiveCodeBench/Pro 提供防污染时序评测；本文在此基础上加入 IOI/ICPC 的真实竞赛规则并与人类成绩对照。

## 局限性与未来方向
- 训练与测试时算力需求较大，实时结果更适合“系统级同条件对比”，而非与人类完全同资源对比。
- Ultra 规模未做 RL，且未能对全部模型尺度与训练阶段做穷举消融；SFT/RL/测试时计算的联合最优配比仍有探索空间。
- 经验来自竞技编程，跨到其他代码/推理场景的泛化性需进一步验证。
- 训练语料因第三方 Redistribution 限制无法完整公开，只能提供详尽流程与部分组件开源。
- 二元终态奖励与长 horizon 信用分配使 RL 收益相对克制，未来可探索更细粒度的中间奖励或子任务完成信号。

## 研究启发与可借鉴点
- “轻后训练+重测试时计算”路线在高可用算力和更长推理窗口下同样有效，适合对训练成本敏感、但需要冲刺更高上限的场景。
- SFT 中加入“教师基于旧解自我改进”的轨迹，能有效把推理期的迭代精化行为蒸馏进模型，便于后续 GenCorrect 式多轮升级。
- 以“累计子任务得分 + 互补参考”作为条件生成信号，比强行拆题更有效；可迁移到带部分分的多阶段评测任务。
- 最后一轮大幅扩展候选池并结合执行筛选（类似 GenCluster）是实用收益较高的工程技巧，能在有限提交预算内最大化命中率。
- 量化/吞吐配置（NVFP4、FP8 KV、MTP）对竞赛型高并发推理影响显著，训练与部署联合优化是落地关键。

## 关键术语表
- **GenCorrect**：多轮测试时精化策略，按轮生成多样性候选、执行打分并以累计子任务反馈指导下一轮条件生成。
- **Score@k / Pass@k**：IOI 以每子任务历史最优求和计分并跨多次运行平均；ICPC/LCB 以通过全部隐藏用例的比例统计。
- **GRPO**：Group Relative Policy Optimization，以组内相对优势进行策略梯度优化的 RL 方法。
- **NVFP4**：NVIDIA 混合精度量化配方，可在保持较低性能损耗的同时显著提升推理吞吐。
- **MTP**：Multi-Token Prediction，一次性预测多个 token 以提升生成吞吐。
- **GLM-5.2 / DeepSeek-V4-Flash**：本文用作 SFT 教师或对比基线的强语言/代码模型。
- **IOI / ICPC**：国际信息学奥林匹克与程序设计大赛，分别允许部分分与按题计分的不同赛制。
- **LiveCodeBench Pro**：面向代码与推理能力的防污染时序评测基准。

## 可复现要素
- 数据集：16 类竞赛与平台共 22,000 题；IOI 2025/ICPC 2025/LiveCodeBench Pro 已从训练中去重；IOI 2026 为赛前实时评测。完整训练语料因第三方限制无法全部公开，论文提供了详细构造与过滤流程。
- 代码/权重：计划通过 NeMo-Skills 发布 Competition Ultra-CC checkpoint 及可运行的推理与评测 recipe；NeMo RL 与 NVIDIA Model Optimizer 为开源工具。
- 关键超参（摘要）：SFT 序列长度 262K、batch 64、Nano 3 epoch/1.2M 样例、Ultra 1 epoch/477,642 样例；RL 64 prompt × 16 rollouts=1024、温度 1.0、最大生成 255,144 tokens、学习率 3e-6、无 KL 惩罚；GenCorrect 每轮 200 生成选 10 提交，最后轮可扩至 1,000。
