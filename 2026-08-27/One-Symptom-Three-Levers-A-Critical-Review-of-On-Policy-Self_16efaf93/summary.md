---
title: "One-Symptom-Three-Levers-A-Critical-Review-of-On-Policy-Self"
source: https://arxiv.org/pdf/2608.25936v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:43:40"
field: "大语言模型推理训练"
keywords: ["on-policy self-distillation", "privileged information", "model collapse", "reasoning LLMs", "knowledge distillation", "diversity collapse"]
innovations: ["提出'一症状三杠杆'框架统一分析OPSD的collapse现象", "建立特权信息的可重构性安全准则和风险排序", "澄清forward KL与reverse KL在OPSD中的适用边界"]
benchmarks: ["AIME25", "C-Eval", "LiveCodeBench v6", "MathArena"]
---

# 论文速读：One-Symptom-Three-Levers-A-Critical-Review-of-On-Policy-Self

## 一句话总结
本文是对 On-Policy Self-Distillation (OPSD) 方法的批判性综述，指出该领域共同的症状是 collapse（推理多样性崩塌），并提出三个可调节的杠杆：信号几何、特权信息的选择、教师动态时序，用于系统性地理解与控制 OPSD 的训练行为。

## 研究问题与动机
- **OPSD 的核心矛盾**：OPSD 通过让教师模型使用 privileged information（特权信息，如参考答案）为学生生成 dense token-level 信号，消除了对外部大模型的依赖，但同一不对称性也导致信号有偏，引发 collapse。
- **collapse 的隐蔽性**：平均准确率上升可能掩盖 functional diversity 的下降，仅靠 pass@k、G-Pass@k 等指标才能揭示多样性损失，而现有文献评估标准不统一。
- **benchmark 测量脆弱性**：小模型在数学推理 benchmark 上的得分受方差、数据污染、模型族特异性三重影响，导致结果不可靠，需要严格的评估协议。
- **方法论缺乏统一框架**：OPSD 衍生出的两百余篇工作对同一失败模式使用了不同术语，缺乏共享词汇和系统分类，阻碍方法比较与改进。

## 核心贡献（创新点）
- **提出"一症状三杠杆"的结构性分析框架**，将 collapse 视为症状，用信号几何（A）、特权信息（B）、循环稳定性（C）三个维度统一描述不同工作。
- **建立特权信息的风险排序体系**，按"对学生测试时可重构性"从高风险（最终答案）到低风险（错误对齐反馈）进行排序，并给出可迁移的信号选择原则。
- **澄清 forward KL 与 reverse KL 的选择逻辑**，指出 OPSD 采用 forward KL 是合理的，但密度不等于更好，selective density 才是关键。
- **连接自监督视觉领域的教师更新经验到 LLM**，提出 EMA、gated refresh、trust region 等机制在 LLM 场景下的适用性与待验证问题。
- **确立"特权信息仅在学生可在测试时重构它时才有益"的安全准则**，将 Vapnik 的 privileged information 学习理论与 LLM 蒸馏工作建立理论联系。

## 方法详解
**OPSD 机制核心流程**：
1. **生成 rollout**：学生模型 $p_S$ 在 prompt $x$ 下自回归生成 rollout $\hat{y} = (\hat{y}_1, \dots, \hat{y}_{|\hat{y}|})$，长度上限 1,024 tokens。
2. **教师评分**：教师（同一模型，但附加特权信息 $y^\star$）对 rollout 每个位置 $n$ 计算条件分布 $p_T(\cdot|x, y^\star, \hat{y}_{<n})$，与学生分布 $p_S(\cdot|x, \hat{y}_{<n})$ 并行计算。
3. **损失计算**：在每个位置计算 forward KL 散度 $D_n = \text{KL}(p_T \| p_S)$，经 clipping（每维 capped at $\tau$）后对所有位置取平均得 $\mathcal{L}(x, y^\star)$，再对 batch 取平均得 $\mathcal{L}_{\text{OPSD}}(\theta)$。
4. **反向传播**：仅对学生模型反向传播，教师权重固定。

**SDPO 变体**：与 OPSD 同族，但特权信息为执行反馈 $f$（如代码错误信息），教师可通过 EMA 或插值正则化保持稳定。

**三个杠杆的设计空间**：
- **Lever A（信号几何）**：散度方向（forward/reverse/JSD）、token 权重策略（entropy-aware、alignment-based、problem-level anchoring）。
- **Lever B（特权信息）**：从高风险（最终答案、完整 solution）到低风险（plan、rubric、error-aligned critique、execution feedback）。
- **Lever C（循环稳定性）**：教师更新规则（frozen、EMA、gated refresh by reward gain、proximal/trust region）、特权信息衰减调度（prefix length reduction、adaptive masking）。

## 实验与结果
本文为综述论文，**未报告新实验**，但汇总了多篇关键工作的实验结论：

**性能对比**：
- OPSD 在数学推理上达到与 GRPO 相当的性能，但生成 token 数大幅减少（单 rollout × 1,024 tokens vs GRPO 的 8 rollouts × 16k tokens）。
- AIME25 上 OPSD 从 step 50 达到 43.9（forward from 36.7）。
- 同等计算预算下，OPSD 一步约需 GRPO 两步的成本（20.6s vs 11.2s on Qwen3-8B, 8×H100）。

**collapse 量化结果**：
- Kaur et al. [24]：对 thinking models 使用完整 solution 作为特权信息，avg@16 相对下降高达 −17%。
- Nicolicioiu et al. [37]：Qwen3-8B 上 pass@1 从 71.9 升至 73.4，但 pass@16 从 83.6 降至 78.5；且 self-distilled 模型的 token entropy 高于 GRPO 训练模型，说明 entropy 不是多样性的有效代理。
- DPH-RL [28]：reverse KL 加速 diversity collapse，pass@1 上升但 pass@k 下降。

**特权信息比较**：
- Yu et al. [56]：五类特权信息对比，step-wise hints without execution 最佳（71.3 on C-Eval），final answer only 最差（59.5 vs 63.0 no-privilege baseline）。
- Kara & Ersoy [23]：step-aligned critique 优于标准 OPSD +5.27、优于 GRPO +16.11（avg@12）。
- AR-OPD [60]：anchor-only 策略使 shortcut events 减少超过 20%。
- DemoPSD [30]：disagreement-modulated 方法在性能和多样性上同时优于 GRPO 和 SDPO。

**最强结果**：step-aligned critique（Kara & Ersoy [23]）和 step-wise hints without execution（Yu et al. [56]）在严格自蒸馏设置下表现最佳，分别获得 +5.27 和 +8.3 的相对提升。

## 相关工作脉络
- **GKD [1]**：On-policy distillation 的形式化奠基，引入 forward/reverse KL 选项，本文视其为 OPSD 的直接祖先。
- **MiniLLM [16]**：推广 reverse KL 用于 LLM 蒸馏，本文指出其对多样性抑制的风险。
- **SDPO [21]**：与 OPSD 同时独立提出，使用执行反馈而非参考答案，本文将其归入同族但强调 PI 性质的差异。
- **DeepSeek-R1 [11] / GRPO [39]**：RLVR 路线的代表，OPSD 作为其 dense signal 替代方案被提出，本文在效率对比中多次参照。
- **Vapnik & Vashist [48]**：2009 年提出的 privileged information 学习理论，本文将其作为 OPSD 的理论根源重新引入。
- **SimSiam [8] / BYOL [14] / Mean Teacher [47]**：视觉自监督中的教师更新机制，本文主张将其移植到 LLM 蒸馏场景验证。

## 局限性与未来方向
- **领域局限**：仅覆盖数学推理，未涉及 multimodal learning 和 tool-using agents 两个最大分支。
- **模型族局限**：绝大多数工作基于 Qwen 系列，存在模型族特异性结果（spurious rewards），缺乏跨族复现。
- **规模局限**：实验多在数亿到数十亿参数模型上进行， founding paper 自身也承认 compute 约束。
- **评估协议待统一**：缺少在同一模型、同一 benchmark 上比较多种 privileged information 的严格控制实验。
- **教师更新机制待系统化比较**：frozen、EMA、gated refresh、proximal teacher 四者尚未在相同条件下对比。
- **forgetting 排序仍有争议**：dense self-distillation 与 sparse RL 的遗忘程度排序在不同工作中结论不一致。

## 研究启发与可借鉴点
- **"可重构性"作为特权信息选择的判断标准**：privileged information 应满足学生在测试时能从自身观测中重构，否则将引入 shortcut 依赖。这一原则可直接迁移到其他蒸馏场景。
- **Entropy-aware token weighting 的工程价值**：在高熵 token 上改用 forward KL、低熵 token 用 reverse KL 的策略仅需 +4.5% 额外计算，可有效保护推理分叉点的多样性。
- **Progress-measured gated refresh**：CGTR [17] 提出的"仅在奖励提升时刷新教师"机制，比固定间隔或 EMA 更稳定，可作为教师动态更新的实用基线。
- **Residual guidance 的信号分离思想**：AR-OPD [60] 将教师信号分解为 anchor（可学习部分）和 residual（shortcut 部分），仅蒸馏锚定分量，这一思路可迁移到任何存在信息泄漏风险的对齐场景。
- **pass@k 应成为 OPSD 类方法的标配评估**：mean score 和 entropy 均无法可靠反映 diversity collapse，必须在报告中包含 pass@k 和 G-Pass@k。

## 关键术语表
**On-Policy Self-Distillation (OPSD)**：学生与教师为同一模型，教师通过 privileged information（如参考答案）获得比学生更多的信息，生成 dense token-level 蒸馏信号。

**Privileged Information (PI)**：仅教师可见的学生不可见的辅助信息，如参考答案、执行反馈、计划等；其选择直接影响蒸馏信号的质量。

**Collapse**：OPSD 中的主要失败模式，表现为推理路径多样性逐步收窄，pass@k 下降而 pass@1 可能上升，entropy 不一定同步降低。

**Pointwise Mutual Information (PMI) 机制**：教师条件于参考答案时，对解中已蕴含的 token（如连接词）给予正 PMI 奖励，对 deliberation token（如"wait"、"maybe"）给予惩罚，从而抑制学生的探索能力。

**Signal Geometry**：指蒸馏信号的散度方向（forward/reverse/JSD）和密度分布（uniform/selective），决定信号是覆盖多模还是聚焦单模。

**Exposure Mismatch**：教师在训练时始终看到完整参考答案，而学生在测试时无此信息，造成训练-推理分布不一致。

**Rich-get-richer Effect**：示范中最常出现的解法被反复强化，导致稀有但正确的策略信号衰减甚至消失。

**Forward KL vs Reverse KL**：forward KL 为 mass-covering，要求学生覆盖教师的所有 mode；reverse KL 为 mode-seeking，要求学生集中于教师的高概率 mode。

## 可复现要素
- **数据集**：AIME25、C-Eval、LiveCodeBench v6、MathArena 等数学/代码推理 benchmark（论文未提及是否公开，实际这些均为公开 benchmark）
- **模型**：Qwen3 1.7B/4B/8B（主要实验平台）、其他模型族提及但未详细报告
- **代码**：OPSD founding paper [61] 有开源实现；相关方法代码分散在各论文中；本文未提供统一代码库
- **关键超参**：rollout 长度 1,024 tokens、vocabulary size ≈ 150,000（Qwen3）、clipping bound τ、AR-OPD 的 λ=0.6、EMA momentum ρ（未校准于 LLM）
