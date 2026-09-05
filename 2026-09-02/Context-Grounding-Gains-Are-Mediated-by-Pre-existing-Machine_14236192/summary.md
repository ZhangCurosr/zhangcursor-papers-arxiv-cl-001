---
title: "Context-Grounding-Gains-Are-Mediated-by-Pre-existing-Machine"
source: https://arxiv.org/pdf/2609.00925v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 22:31:14"
field: "大语言模型对齐与可解释性"
keywords: ["context grounding", "post-training", "GRPO", "SFT", "DPO", "mechanism reuse", "activation steering"]
innovations: ["系统比较 GRPO/SFT/DPO 在起始模型上的 grounding 增益差异", "通过起始模型方向干预验证机制复用", "提出冻结已知集与单独报告非决定性比例的评价规范"]
benchmarks: ["CounterFact", "ConFiQA", "FaithEval"]
---

# 论文速读：Context-Grounding Gains Are Mediated by Pre-existing Machinery

## 一句话总结
本文系统对比 GRPO、SFT、DPO 三种后训练方案在知识冲突情境下的上下文 grounding 增益，发现 grounding 能力的提升主要来自起始模型中已存在的因果机制，而非训练新建的机制。

## 研究问题与动机
- 语言模型在提示证据与参数记忆冲突时往往忽略提示证据，后训练能否可靠改善该问题？
- 当后训练改善 grounding 时，内部机制是新建的还是强化了已有机制？
- 不同训练方案（GRPO/SFT/DPO）的 grounding 增益是否存在系统性差异？
- 现有方法未在同一模型、同一评估协议下对多种后训练配方进行机制级对照审计。

## 核心贡献（创新点）
- 提出并验证“机制复用”解释：grounding 提升主要由起始模型中已有的因果结构承载。
- 在同一检查点比较 GRPO、SFT、DPO 九类后训练配方，明确其 grounding 增益差异。
- 通过因果 head 定位与方向干预（加减起始方向）建立干预层面的因果证据。
- 提供“冻结已知集+独立记录非决定性比例”的评估报告规范，降低干预带来的分母偏差。
- 给出 GRPO 在 grounding 上增益有限的经验与机制解释边界。

## 方法详解
- 冲突评估协议：采用 CounterFact 双-pass 协议，先用闭卷测试冻结“已知正确项”，再引入反事实上下文，记录 follow-ctx / follow-mem，以 update rate 为主指标。
- 训练配方：基于 Qwen2.5-1.5B/3B/7B、Llama-3.2-3B、Phi-3.5-mini 等指令微调检查点；GRPO 200步、lr=3e-6、KL β=0.02、group=8；SFT 2 epochs；DPO β=0.5、3 epochs、4500对。
- 方向估计：在起始模型第21层（1.5B）计算 follow-ctx 与 follow-mem 残差均差（DiM）作为 grounding 方向。
- 因果 head 定位：按 arm 独立进行 per-head knockout，统计与起始模型 top-8 因果 head 的重叠。
- 干预验证：在生成末尾位置注入/减去 α·d̂，与等范随机方向对照；以冻结已知集的 exclusive follow-ctx 率报告主效应，并单独报告非决定性比例。
- 稳定性追踪：DPO 训练过程中每步计算 DiM 方向，测量与起始方向的余弦相似度。

## 实验与结果
- 数据集：CounterFact（主）、ConFiQA、FaithEval；HotpotQA 用于辅助 F1 指标。
- 主要结果：GRPO 各变体 grounding 增益小（4种子 CI 含零，等价检验限定效应 < ±.044）；Conflict-SFT 提升 +.044–+.063；DPO 在 CounterFact/ConFiQA 上提升最大，Qwen2.5-3B ConFiQA-QA 提升 +.599，多模型多族均接近天花板。
- 最强结果：DPO 在 Qwen2.5-3B ConFiQA-QA 达到 +.599（base .362 → .961）；DPO 在五个模型上均实现显著增益。
- 机制证据：审计 arms 独立发现的 top-8 因果 head 与起始模型重叠 7–8/8；起始方向在各训练 arm 中余弦对齐 ≥ .915；减去起始方向显著抑制 DPO/Conflict-SFT 增益；注入起始方向可在起始模型恢复约 35%（最大合规剂量 α=20）的 DPO 增益。
- 训练动态：DPO 至 step 160/800 时 grounding 已达最终值的约 90%，方向余弦 ≥ .968。
- GRPO 解释：监督 warm start 提升 hit@8 并改善 grounding +.023，但再执行同一 GRPO 配方仅加 +.001（p=.91）。

## 相关工作脉络
- Context-grounding 与后训练工作：Li et al. (2023)、Bi et al. (2025) 等证明后可改善 grounding；本文定位在于跨配方系统比较与机制审计。
- 机制复用研究：Prakash et al. (2024)、Lee et al. (2024) 显示后训练可保留/利用已有机制；本文将该观点推进到 grounding 行为与因果 head 层面。
- 激活导向（Activation steering）：Rimsky et al. (2024)、Arditi et al. (2024) 等；本文以定向干预作因果检验，而非替代训练。
- RLVR/GRPO 研究：Yue et al. (2025)、Chen et al. (2026)、Tamo et al. (2026) 探索强化学习对 grounding/推理的影响；本文表明在特定奖励设计下 GRPO 的 grounding 增益有限。
- 机制解释与电路审计：Ortu et al. (2024)、Minder et al. (2025)、Jin et al. (2024) 定位冲突相关机制；本文强调“起始模型中即可识别方向”。

## 局限性与未来方向
- 评估指标依赖词汇包含，与 LLM judge 一致性中等（κ=.507），绝对率解读受限。
- 机制审计仅覆盖部分 arms，干预为单种子实验；因果 head 集合未声称完整电路。
- “预存在”指指令微调起始检查点中存在，未必在原始预训练模型中存在。
- GRPO 结论限于当前奖励家族与低上下文答案 rollout 覆盖的设置；合成数据或冻结骨干门控配方不在此范围。
- 未来可扩展到更大规模、更多模型族、更多评估协议与更系统的多轴方向控制。

## 研究启发与可借鉴点
- 以“起始方向+加减干预”作为机制复用的因果证据范式，可迁移到其他对齐/可控生成任务。
- 冻结已知集并单独报告非决定性比例，能减少干预导致的评价偏置，适合部署干预类实验。
- 训练早期即达大部分行为增益、方向保持稳定，提示可优先在早期阶段预算内验证方向有效性。
- GRPO 在 grounding 上的有限增益提示：若需显著提升，需配套增强上下文答案覆盖或设计直接针对证据使用的奖励。
- 跨模型族的 norm 校准与匹配随机对照，有助于公平比较干预强度。

## 关键术语表
- **Context Grounding**：模型在提示提供证据时遵循上下文信息而非依赖参数记忆的行为能力。
- **Knowledge Conflict**：提示中的反事实或冲突证据与模型参数化知识不一致的情形。
- **GRPO**：Group Relative Policy Optimization，基于组相对优势的强化学习策略优化方法。
- **SFT**：Supervised Fine-Tuning，监督微调，使用标注数据微调模型。
- **DPO**：Direct Preference Optimization，直接偏好优化，无需显式奖励模型的对齐训练方法。
- **Causal Head**：对特定行为具有因果贡献的 Attention head，可通过 knockout 实验定位。
- **Activation Steering**：通过向模型激活向量中注入/减去特定方向来改变模型行为的干预技术。
- **Difference-in-Means Direction**：以两类样本残差均值之差估计的表示方向。

## 可复现要素
- 数据集：CounterFact、ConFiQA、FaithEval、HotpotQA（公开）；已知集按模型计算并冻结。
- 代码/权重：论文声明代码、逐条数据、估计方向向量及复现脚本将在发表时开源。
- 关键超参：GRPO 200步、lr=3e-6、KL β=0.02、group=8；SFT 2 epochs；DPO β=0.5、lr=5e-6、3 epochs、4500对；GRPO 温度1.0、max prompt length 1536。
