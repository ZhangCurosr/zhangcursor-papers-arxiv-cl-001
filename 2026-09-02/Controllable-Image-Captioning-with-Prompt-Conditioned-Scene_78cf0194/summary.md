---
title: "Controllable-Image-Captioning-with-Prompt-Conditioned-Scene"
source: https://arxiv.org/pdf/2609.00709v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 22:32:08"
---

# 论文速读：Controllable-Image-Captioning-with-Prompt-Conditioned-Scene

## 一句话总结
本文提出 FOCUS 方法，通过提示词条件化的场景图组件正负加权目标，结合 GRPO 强化学习训练，使 VLM 能够依据自然语言控制指令精准强调属性/关系/前景/背景，并主动抑制超纲内容；同时构建 SCOPE 基准，以对比式 Include/Avoid 约束定量评估细粒度可控性。

## 研究问题与动机
- **语义控制缺失**：现有 LVLM 图像描述虽流畅，但面对同一图像只能输出单一的“通用最佳描述”，无法可靠响应用户对特定语义重点（如只强调属性或空间关系）的请求。
- **零样本提示不可靠**：直接添加自然语言提示往往导致模型漂移回高概率的通用内容，难以实现细粒度的“强调目标+抑制非目标”双重行为。
- **结构化输入门槛高**：现有可控描述方法通常要求在推理时提供长度 token、边界框或形式语义图等结构化侧输入，交互成本高且不符合自然语言控制范式。
- **评测指标空白**：CIDEr、SPICE 等标准指标仅衡量整体质量，无法量化“覆盖目标内容与排除超纲内容”的对比式可控行为。

## 核心贡献（创新点）
1. **提出 FOCUS 提示词条件化控制目标**：将场景图分解转化为核心的训练奖励信号，通过针对不同语义类别设计的符号权重（正权重奖励目标、负权重惩罚超纲），实现无需架构修改的可控描述生成。
2. **提升场景图 Reward 的信噪比**：引入严格的对象互最佳匹配有效性阈值（τ=0.5）与基于 CoT 的 Qwen3-30B-A3B 属性/关系推理评分，显著降低 Embedding 漂移与 LLM  judge 宽松打分带来的奖励噪声。
3. **构建 SCOPE 可控性评测基准**：提出包含对比式 Include/Avoid 原子事实列表的评测协议，从 Coverage（召回）、Adherence（抑制）、Faithfulness（事实一致性）三维汇总 Overall Score，并验证其与人工偏好高度一致（Spearman ρ=0.5704）。

## 方法详解
- **场景图对齐组件得分**：使用 spaCy 解析生成描述，提取候选对象集合 $\mathcal{C}$ 与 Ground-truth 对象集合 $\mathcal{G}$。通过 Sentence-BERT 计算余弦相似度矩阵，经互最佳匹配与阈值 τ=0.5 过滤后计算对象得分 $S_{obj}$；对已匹配对象调用 Qwen3-30B-A3B-Instruct 结合 CoT 提示输出 0–5 分的属性对齐分 $S_{attr}$ 与关系方向分 $S_{rel}$；前景/背景得分按固定权重组合对应子集的三组件分。所有得分线性归一化至 $[0,1]$。
- **提示词条件化控制目标**：定义奖励 $R(y|p,z^*) = \sum_{k \in \mathcal{K}} w_k(p) \tilde{S}_k$。通用提示沿用 CompreCap 统一权重；属性聚焦设为 $R_{attr}=0.1\tilde{S}_{obj}+0.9\tilde{S}_{attr}-1.0\tilde{S}_{rel}$，关系聚焦设为 $R_{rel}=0.1\tilde{S}_{obj}-1.0\tilde{S}_{attr}+0.9\tilde{S}_{rel}$；前景/背景聚焦采用 $R_{fg/bg}=\tilde{S}_{fg/bg}-\tilde{S}_{bg/fg}$，负权重显式惩罚非目标组件。
- **GRPO 优化**：采用两阶段 SFT+GRPO 流程，优化目标为 $\mathbb{E}[R - \beta D_{KL}(\pi_\theta||\pi_{ref})]$。每组 prompt 采样多条描述计算组内相对优势，五个提示类别联合训练以保持跨场景鲁棒性。
- **SCOPE 构建与评测**：从 COCO/CompreCap/DOCCI 手动筛选 189 张图像，利用 Gemini-3-Flash 迭代生成并验证四类特化描述，抽离原子事实形成 Include/Avoid 列表。评测时由 LLM Judge 验证描述是否命中 Include 事实（Coverage）、是否泄露 Avoid 事实（Adherence=1-泄露率）、是否与 Include 事实矛盾（Faithfulness），三者调和平均得 Overall Score。

## 实验与结果
- **实验设置**：骨干网络为 Qwen2.5-VL-3B 与 InternVL3-2B；基线含 Zero-shot、SFT、SFT+CLIP、SFT+CompreCap。
- **SCOPE 可控性**：SFT+FOCUS 在两类骨干上全面领先。相对 Zero-shot，Qwen2.5-VL-3B 整体提升 **+16.08**，InternVL3-2B 提升 **+11.23**；属性聚焦增益最大（+24.51 / +28.19）。即使用强化推理提示（Table 2），FOCUS 仍保持最优，表明训练目标优于纯提示工程。
- **CompreCap 细粒度对齐**：SFT+FOCUS 同样取得最高 Overall（Qwen2.5-VL-3B +7.74，InternVL3-2B +8.86），且各分项（General/Attribute/Relation/Foreground/Background）均稳步提升。
- **通用描述保持**：在 DOCCI 测试集（5000 张）上，FOCUS 维持 CIDEr、METEOR、ROUGE-L 及 CAPTURE 等标准指标，证明可控性增强未牺牲通用质量。
- **关键消融**：引入对象有效性阈值带来小幅增益（+0.53），而启用 CoT 验证与更强 Judge 分别带来 **+4.79** 与 **+5.06** 的大幅度提升；三者联合使 InternVL3-2B 整体提升至 36.86（相对 baseline 提升 26.0%）。惩罚权重过大易引发拒绝生成，过小则导致超纲漂移，需适度校准。

## 相关工作脉络
- **LVLM 详细描述生成**（ShareGPT4V、DOCCI 等）：本文聚焦于其控制能力短板，区别于以往仅追求整体流畅度与覆盖率的工作。
- **可控图像描述**（长度/区域/AMR 图控制）：现有方法依赖推理时结构化侧输入；本文仅凭自然语言 prompt 即可实现同等细粒度控制，免去额外标注与解码头修改。
- **场景图评测与训练信号**（SPICE、CAPTURE、CompreCap）：以往将场景图作为事后评测或固定权重监督；本文将其升级为提示词条件化的实时 Reward，并引入负权重实现对比抑制。
- **偏好优化算法**（SCST、CLIPScore、DPO、GRPO）：本文采用 GRPO 优化复杂分项 Reward，区别于基于全局 CLIP 相似度或简单 CIDEr 的优化目标。
- **评测基准设计**：标准指标无法刻画“Include vs Avoid”行为；SCOPE 填补了这一评测空白，其对比式原子事实提取范式可迁移至其他可控生成任务。

## 局限性与未来方向
- 训练阶段依赖场景图解析与 LLM 打分，计算开销较大，且解析错误可能污染 Reward 信号。
- 仅在 2B–3B 级别 VLM 上验证，向更大参数规模模型的泛化能力尚未检验。
- SCOPE 依赖 LLM Judge 进行事实验证，存在潜在的系统性偏差；尽管 MTurk 人类对齐验证了合理性，但仍需更鲁棒的自动化评测方案。
- 未来可探索轻量化解析器、替代 LLM 的专用打分器、或与 SAM/OWLv2 等视觉 grounding 工具结合以进一步提升空间控制精度。

## 研究启发与可借鉴点
- **符号加权对比优化范式**：同时设置正权重（鼓励目标）与负权重（惩罚非目标）的奖励设计，为多目标可控生成提供了简洁且高效的优化框架，可直接迁移至视频描述、图表解读等场景。
- **CoT 细粒度打分替代启发式匹配**：将属性/关系评估从 Embedding 相似度升级为思维链推理打分，显著提升了 RL Reward 的信噪比，对任何依赖复杂语义对齐的偏好优化任务均有借鉴价值。
- **Include/Avoid 对比评测构建模板**：SCOPE 的“生成-验证-原子事实抽取-互补列表对比”流水线，为评测生成模型的“选择性输出”能力提供了可复用的基准建设方法。
