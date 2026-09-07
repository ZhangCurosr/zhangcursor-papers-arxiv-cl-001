---
title: "A-Circuit-for-Plural-Reference-How-LLMs-Represent-and-Retrie"
source: https://arxiv.org/pdf/2609.03687v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:34:41"
field: "机制可解释性"
keywords: ["mechanistic interpretability", "plural reference", "coreference resolution", "activation patching", "attention heads", "LLM circuits"]
innovations: ["首次揭示 LLM 复数指代的三头协作电路（Selection/Formation/Interpretation）", "通过因果干预与注意力相关性分析量化复数偏好机制", "在 LLM 中复现心理语言学关于 ontological similarity 与 conjunction 的复数指代约束"]
benchmarks: ["合成 pronoun prediction task (D_pl / D_sg)"]
---

# 论文速读：A Circuit for Plural Reference: How LLMs Represent and Retrieve Singular and Plural Entities

## 一句话总结
本文利用机制可解释性方法（activation/path patching）与注意力模式分析，揭示了 LLM 在处理**复数指代**（plural reference）时的内部电路：模型通过一组专门化的注意力头来编码实体信息、识别构成复数实体的实体对，并最终选择单数或复数代词；同时验证了 LLM 在复数指代偏好上与心理语言学人类实验结论一致（倾向于 ontological similarity 高且由 `and` 连接的实体组合）。

## 研究问题与动机
- **核心问题**：LLMs 如何在内部表示复数指代（plural entity），并在预测代词时检索这些复数实体？
- **现有工作不足**：此前对 coreference resolution 的评估多为**行为层面**，且主要关注单数指代；关于现代语言模型如何追踪和处理 discourse 中复数实体的**内在机制**几乎未知。
- **理论背景缺口**：心理语言学已证明人类处理复数指代比单数更复杂，受 ontological similarity、连接词类型等因素影响，但 LLMs 是否具备类似偏好、其内部是如何实现这一过程尚不清楚。
- **方法论挑战**：需要在保持句子结构相似的前提下，通过因果干预隔离出对复数/单数代词预测有直接/间接贡献的具体组件（attention heads）。

## 核心贡献（创新点）
1. **首次揭示 LLM 中复数指代的机制电路**：定位并命名了三组关键 attention heads（Pronoun Selection Heads、Plurality Formation Heads、Pronoun Interpretation Heads），明确各自在复数指代表示与检索中的分工。
   - 与以往仅做行为评测或针对单数/coreference 整体电路的工作不同，本文首次将**复数指代**作为独立认知过程进行电路级拆解。
2. **提出并验证一个“复数指代电路”（Plural Reference Circuit）**：通过 activation patching 与 path patching 相结合，证明了信息流从中间层（表示/构建复数实体）到末层（选择代词）的传递路径。
   - 本质区别在于：以往 MI 工作多聚焦单一功能（如 factual retrieval、IOI），本文构建了一个包含**多阶段、多 heads 协作**的完整推理电路。
3. **在 LLM 中复现了心理语言学关于复数指代偏好的关键约束**：发现 LLMs 同样偏好 ontological similarity 高（如同为 proper name）且由 `and` 而非 `with` 连接的实体组合形成复数指代。
   - 与以往仅用行为测试验证 LLM 是否“知道”某些语言事实不同，本文通过**注意力模式分析**展示了这种偏好如何体现在内部表征中。
4. **建立了 attention weight 差异与代词概率差异之间的量化关联**：发现 Pronoun Selection heads 对第二个实体（e₂）相对于第一个实体（e₁）的注意力优势程度，与模型输出复数 vs. 单数代词的概率差显著正相关（r ≈ .78）。
   - 这一发现将抽象的模型内部权重变化与具体行为概率直接挂钩，提供了可解释、可测量的桥梁。

## 方法详解
- **任务设计**：构造了两个合成数据集 `D_pl`（期望复数代词）与 `D_sg`（期望单数代词），句式均为 `When [s] saw [e1] [c] [e2], [s] [v] at`，其中 `D_sg` 通过将 e2 替换为非生命实体（如 her bike）并配合不兼容动词，使得复数指代不被偏好。
- **因果干预技术**：
  - **Activation patching**：在输入 `s_pl` 的前向传播中，将特定组件 C 的激活值替换为来自 `s_sg` 的对应值，测量对复数代词概率 `P_pl` 的间接影响。
  - **Path patching**：在上述基础上，进一步恢复所有下游组件的原始激活值，从而隔离组件 C 的**直接效应**。若直接效应导致 `P_pl` 下降，则可归因该组件为关键节点。
- **干预位置与污染提示构造**：仅在**最后一个 token 位置**与**被污染的 token（e₂ 或连接词 c）位置**进行干预；构建两种 corrupted prompts：`(e1, c, e2')` 与 `(e1, c', e2)`，分别操纵实体 ontological similarity 与连接词类型。
- **评估指标 M**：`M = ((P_pl,intervened - P_pl,original) / P_pl,original) × 100`，负值表示干预抑制了复数信号。
- **注意力模式分析**：可视化关键 heads 在不同 prompt 类型（`D_pl,a`、`D_sg,a`、`D_pl,w`）下的 attention weight 分布，以推断其语义/语用功能。

## 实验与结果
- **模型**：Qwen3-0.6B、Qwen3-1.7B、GPT2-medium；主文聚焦 Qwen3-1.7B，附录验证跨模型泛化。
- **统计建模**：对 349 个 prompt 拟合 Linear Mixed-Effects Regression（REML），验证 ontological similarity 与 conjunction 的主效应显著（`p < .001`），`with` 相比 `and` 显著降低复数代词偏好（β = -0.669）。
- **电路定位**：
  - **Layer 23** 的 MHSA 模块是唯一对 `P_pl` 有直接效应的层。
  - **L23H12、L23H13（Pronoun Selection Heads）**：在 `D_pl` 下末 token 对 e₁ 和 e₂ 均有高注意力，且对 e₂ 的注意力更强；在 `D_sg` 下仅对 e₁ 有高注意力。与复数概率差呈强正相关（r ≈ .78）。
  - **L17H9（Plurality Formation Head）**：在 `and` 连接下注意力覆盖整个复数构造（e₁、c、e₂）；在 `with` 连接下仅对 e₁ 和 c 有注意力，e₂ 几乎无注意力；直接通过 K/V 向量影响 Pronoun Selection heads 的 Value 表示。
  - **L13H6（Pronoun Interpretation Head）**：作为 coreference map，代词 token 会回头关注其先行词（如 `her→Mary`、`he→John`）；当出现复数代词 `them` 时，会同时关注整个复数构造。
- **泛化验证**：电路结构在 Qwen3-0.6B 与 GPT2-medium 中保持一致；更换句式模板（Var. 1）后 attention pattern 亦稳定。

## 相关工作脉络
1. **Wang et al. (2022) – Indirect Object Identification (IOI) circuit**：本文方法直接沿用并扩展了其 activation/path patching 范式，但将研究对象从单一语法关系（IO）拓展至更复杂的 discourse-level 复数指代。
2. **Meng et al. (2022) – Factual knowledge retrieval**：两者均采用 causal mediation 定位关键组件，但本文关注的是**指代选择**而非事实检索，且强调了多阶段协作电路。
3. **Clark et al. (2019) / Tenney et al. (2019) – BERT attention head typology**：本文发现的 Pronoun Interpretation Head 与之前 BERT 研究中识别的“coreference heads”现象一致，但本文首次在**自回归 LLM**中通过干预验证其功能，并纳入复数上下文。
4. **Dai et al. (2024, 2026) – Entity tracking in LLMs**：工作同属 mechanistic interpretability 下的实体表征研究，但本文聚焦**复数集合的构建与选择**，而非单一实体的追踪。
5. **心理语言学文献（Koh & Clifton, 2002; Moxey et al., 2012）**：本文的实验设计与统计验证直接呼应这些人类行为研究，验证了 LLMs 在复数指代偏好上是否与人类对齐。

## 局限性与未来方向
- **复数构造的复杂性受限**：仅研究最简单的 conjoined noun phrases（两个元素由 `and`/`with` 连接），未涉及 split-antecedent（分散先行词）或 mereological reference（部分-整体指代）等更复杂情况。
- **电路的非唯一性**：未充分验证该电路是否专属于 coreference 任务，可能与其他句法/语义过程共享组件。
- **MLP 模块功能未明**：虽观察到晚期 MLP 干预对 `P_pl` 有强影响，但未深入剖析其在电路中的具体作用。
- **模型规模与架构泛化**：仅在 Qwen3 与 GPT2 系列中验证，其他架构（如 MoE、不同注意力变体）下的电路尚未探索。

## 研究启发与可借鉴点
1. **“污染提示”构造策略**：通过系统性操纵单一变量（ontological similarity 或 conjunction）来破坏/增强特定信号，可广泛用于定位功能特异性 heads。
2. **注意力差异与概率差异的相关性分析**：将 head 层面的注意力权重差（α_diff）与输出概率差（P_diff）做 Pearson 相关，为电路功能验证提供了**量化、可复现**的补充证据。
3. **跨模型电路迁移性验证**：在多个尺寸/家族的模型中复现同一电路，增强了发现的可信度与通用性，可作为后续机制研究的 baseline 范式。
4. **与心理语言学对话的实验设计**：先通过行为统计（Linear Mixed Effects）确认人类-like 偏好，再以此指导 patching 实验的设计（选择中间态、非极端 prompt），使 MI 分析与认知科学假设紧密耦合。

## 关键术语表
**Plural Reference**：用复数名词短语指代一个或多个实体的语言现象，如 "John and Mary" 被 "they" 指代。
**Activation Patching**：机制可解释性技术，通过将模型中某组件的激活值替换为来自另一输入的激活值，以因果方式估计该组件对输出的影响。
**Path Patching**：在 activation patching 基础上恢复下游所有激活值的原始状态，用于分离组件的**直接效应**与间接效应。
**Pronoun Selection Heads**：位于网络末层（L23）的注意力头，负责根据当前语境选择候选先行词并直接影响单数/复数代词预测。
**Plurality Formation Heads**：位于中间层（如 L17）的注意力头，负责识别构成复数实体的实体对，并将该信息传递给 Pronoun Selection Heads。
**Pronoun Interpretation Heads**：编码先行词与代词之间 coreference 关系的注意力头，充当 discourse 层面的“指代地图”。
**Ontological Similarity**：指复数构造中两个实体在语义类别/ animacy 上的相似程度，是影响复数指代偏好的关键因素。
**Conjunction Effect**：连接词类型（`and` vs. `with`）对复数实体可及性的影响，`and` 更倾向于促成复数指代。

## 可复现要素
- **数据集**：合成数据集（`D_pl`、`D_sg`），共 300+300 个 sentence prefixes，**未公开**（论文附录 Table 3/4 提供了模板与统计样本说明）。
- **代码/权重**：未提及代码开源；使用标准开源模型 Qwen3-0.6B/1.7B、GPT2-medium。
- **关键超参**：未详细报告训练超参（使用预训练模型）；patching 实验在多个位置（last token、corrupted token）进行；统计模型使用 R `lme4` 包、REML 估计。
