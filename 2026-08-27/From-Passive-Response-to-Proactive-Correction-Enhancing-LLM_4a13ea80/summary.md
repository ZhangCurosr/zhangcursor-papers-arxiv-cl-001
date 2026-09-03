---
title: "From-Passive-Response-to-Proactive-Correction-Enhancing-LLM"
source: https://arxiv.org/pdf/2608.25894v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:33:20"
field: "大语言模型幻觉与事实核查"
keywords: ["hallucination mitigation", "fact perturbation", "LLM robustness", "input-side correction", "multi-perspective deliberation", "MisFactQA"]
innovations: ["提出Detect-Devise-Correct三阶段框架将LLM从被动响应转为主动纠错", "设计Generator-Reviewer-Arbiter多视角审议机制避免自我强化偏差", "构建MisFactQA数据集并提出MR/CR/CS三维细粒度评估指标"]
benchmarks: ["TruthfulQA", "FalseQA", "MisFactQA"]
---

# 论文速读：From-Passive-Response-to-Proactive-Correction-Enhancing-LLM

## 一句话总结
本文提出 DEDUCE 框架，通过将 LLM 从被动响应者转变为主动纠错者，解决用户输入包含事实性错误时引发的幻觉问题。三阶段"检测-制定-纠正"流程在多个模型家族上显著降低误导率并提升纠错能力。

## 研究问题与动机
- 现有 hallucination 缓解研究几乎全部聚焦模型端能力建设，忽视真实用户查询中常见的事实性错误对模型推理的主动误导作用
- 即使模型具备正确参数知识，仅向输入注入少量事实错误即可使准确率下降 30%-60%（Figure 2）
- 现有基准多聚焦单错误类型且仅以最终答案准确率评估，无法刻画"被误导/部分纠正/完整澄清"等细粒度行为差异
- 传统 self-correction 方法在错误输入下容易陷入自我强化偏差，难以跳出输入设定的推理框架

## 核心贡献（创新点）
1. 提出"检测-制定-纠正"（Detect-Devise-Correct）三阶段框架 DEDUCE，将 LLM 从被动应答转为主动纠错
   - 与已有工作的本质区别：现有方法假设输入可靠，仅在输出端校正；本文在生成前对输入进行事实预审，从根本上阻断错误前提的传播
2. 设计原子事实分解机制与多视角策略审议（Generator-Reviewer-Arbiter）
   - 与已有工作的本质区别：区别于单模型自我反思（Self-Refine/CRITIC），三角色协作避免自我强化偏差，且在策略生成阶段引入 adversarial critique
3. 构建 MisFactQA 数据集并提出 MR/CR/CS 三维细粒度评估指标
   - 与已有工作的本质区别：突破单一 Accuracy 评估，同时度量"被误导程度/纠错能力/澄清质量"，覆盖单错误、矛盾描述、复合错误三种复杂度

## 方法详解
DEDUCE 由三个协同模块组成：

**1. Detect（输入检测与分析）**
- 原子事实分解：φ(Q) = {AC₁, AC₂, ..., ACₙ}，将查询拆分为最小独立可验证事实单元
- 双重检查：真实性检查 F(ACᵢ) 识别事实错误集 E_fact； pairwise 一致性检查 C(ACᵢ, ACⱼ) 识别矛盾集 E_conflict
- 扰动指示器：ε(Q) = 1[|E_fact| + |E_conflict| > 0]
- 输出结构化诊断摘要 MisSum(Q) = Summarize(E_fact ∪ E_conflict)

**2. Devise（多视角策略审议）**
- Generator：s⁽⁰⁾ = G(Q, MisSum(Q))，生成初始纠正策略
- Reviewer：r = R(s⁽⁰⁾, MisSum(Q))，从完备性、准确性、可靠性三维度 critique
- Arbiter：π*(Q) = A(s⁽⁰⁾, r)，仅在 r ≠ ∅ 时介入，综合双方观点产出最终策略
- 角色分离有效缓解单一模型的自我强化偏差

**3. Correct（纠正与响应生成）**
- 按步骤顺序执行：①错误识别 → ②带理由的纠正 → ③基于纠正前提的可靠回答
- 确保所有已识别错误被系统性处理，最终响应忠实于验证后的策略

**两种实现策略：**
- DEDUCE-Prompting：多阶段结构化提示，无需训练，即可部署
- DEDUCE-Tuning：两阶段 SFT，Stage 1 学习错误检测（用 GPT-4o 生成训练数据），Stage 2 整合 Devise+Correct（用 Qwen2.5-14B 生成）
- 损失函数：L_Detect = -Σ log P_θ(y_error, MisSum(Q)|Q)；L_Devise+Correct = -Σ log P_θ(y_correct|Q)

## 实验与结果
- **数据集**：TruthfulQA-MC（684题）、FalseQA（3274题）、MisFactQA（1140题）
- **基线**：Original、ICL、CoT、LoRA SFT、IAQ-FA
- **模型**：Qwen2.5-7B、Llama-3.1-8B、Gemma3-12B，以及 GPT-4o-mini、DeepSeek-V3

**主要结果：**
- Gemma3-12B 在 FalseQA 上，DEDUCE-T 较最佳基线 SFT（53.60%）提升 25.99 个百分点至 79.59%（Table 2）
- Qwen2.5-7B DEDUCE-T：FalseQA Acc 77.84%、MR 19.72%、CR 78.56%；MisFactQA Acc 64.65%、MR 21.48%、CR 75.20%
- CoT 在 TruthfulQA 上反而低于原始模型（如 Qwen2.5-7B：CoT 61.84% vs Original 67.11%，Table 3），验证错误输入下链式推理会放大偏差
- DEDUCE-T 在更强模型 GPT-4o-mini（82.9%）和 DeepSeek-V3（84.6%）的 MisFactQA 上仍带来显著提升（Table 5）
- 消融实验（Table 4）：移除 Strategy 模块导致 Llama-3.1-8B 在 MisFactQA 上 Acc 从 70.01% 骤降至 44.75%，表明策略制定贡献最大
- 多轮审议实验（Appendix D）：2-3 轮交互即可收敛至最优，过多轮次可能引入噪声

## 相关工作脉络
1. **IAQ-FA (Wang & Blanco, 2025)**：可解释虚假假设检测，但止步于检测不生成纠正策略；DEDUCE 进一步完成纠正与回答
2. **Self-Refine / CRITIC**：自纠正机制聚焦输出文本验证；DEDUCE 将 critique 机制前置到输入分析与策略生成阶段
3. **RAG-based 事实验证（Check Your Facts 等）**：依赖外部知识库校正输出；DEDUCE 完全基于模型内部知识完成输入预审，无需检索
4. **Prize 数据集 (Yuan et al., 2024)**：仅关注单错误类型与最终准确率；本文扩展为三种错误复杂度并引入细粒度行为指标
5. **ConflictBank (Su et al., 2024)**：评估知识冲突影响；本文提供端到端检测-纠正框架而不仅是冲突分析

## 局限性与未来方向
- 实验主要在 7B-12B 中等规模模型上验证，尚未在更大参数模型（如 70B+）上系统检验
- 仅针对事实性错误/矛盾，未探索模糊查询、价值冲突或对抗性 prompt 等其它不可靠输入形式
- 多轮审议深度与效率的权衡需根据任务难度动态调整，当前最佳轮次为经验性结论

## 研究启发与可借鉴点
1. **原子事实分解范式**：将复杂查询拆解为最小可验证单元（subject-relation-object 三元组思路），可迁移至医疗、法律等高可靠性领域的事实核查
2. **Generator-Reviewer-Arbiter 角色分离设计**：通过职责分工打破单模型自我强化闭环，该机制可用于需要高可信决策的任何生成任务
3. **输入端幻觉缓解视角**：从"输出纠错"转向"输入预审"的范式转换，启发团队关注用户侧输入质量对下游任务可靠性的系统性影响
4. **三维评估指标体系（MR/CR/CS）**：相比单一 Accuracy，能更全面刻画模型在误导输入下的行为光谱，适合用于后续工作的基准对比

## 关键术语表
- **Fact Perturbation**：用户输入中的事实性错误扰动，包含虚假前提、事实矛盾、复合错误三种形式
- **Atomic Fact**：从查询中提取的最小独立可验证事实声明单元
- **Clarification Score**：1-5 分量表，评估模型对错误输入的纠正完整度（1=完全被误导，5=完整澄清并提供正确答案）
- **Multi-Perspective Deliberation**：Generator-Reviewer-Arbiter 三角色协作的策略审议机制
- **FPQA (Fact-Perturbed Question Answering)**：面向含事实扰动输入的任务定义，要求模型检测错误、纠正并提供可靠回答
- **Sycophancy**：模型为最大化"帮助性"感知而迎合用户错误观点的行为倾向

## 可复现要素
- **数据集**：TruthfulQA、FalseQA 为公开数据集；MisFactQA 由 Prize 子集、EchoMist 及 hallucinations-dpo 等构建，论文未明确声明开源
- **代码/权重**：论文未提及代码或权重开源
- **关键超参**：temperature=0.0；SFT epoch=2，batch_size=32，learning_rate=2×10⁻⁵，warmup_ratio=0.1，3×A40 GPU
- **评估工具**：o3-mini-medium 作为 automated judge，与人工标注 Cohen's κ=0.936，Clarification Score Pearson r=0.916
