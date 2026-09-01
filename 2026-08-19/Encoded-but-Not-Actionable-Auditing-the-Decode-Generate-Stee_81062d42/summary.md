---
title: "Encoded-but-Not-Actionable-Auditing-the-Decode-Generate-Stee"
source: https://arxiv.org/pdf/2608.17843v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 06:51:16"
field: "几何推理与大语言模型表示分析"
keywords: ["几何约束", "大语言模型", "表示审计", "线性探测", "激活干预", "可解释性"]
innovations: ["提出解码-生成-干预四维度统一审计框架，区分编码能力与行为表达能力", "发现局部约束预训练增益显著高于全局DOF状态的编码不对称性", "证明线性可解码信息未必转化为生成输出或可通过单方向steering控制"]
benchmarks: ["SketchGraphs", "Fusion 360 Gallery"]
---

# 论文速读：Encoded but Not Actionable: Auditing the Decode-Generate-Steer Gap in Frozen LLMs for Geometric Constraints

## 一句话总结
本文对六个冻结通用大语言模型进行了四部分审计（线性解码、强制生成、激活干预、行为控制），揭示了冻结LLM中"可解码的几何约束信息"与"可行动的行为表达"之间存在系统性鸿沟。

## 研究问题与动机
- **核心问题**：冻结LLM究竟编码了什么几何约束信息？这些信息如何与模型的实际行为（生成、干预敏感性、可控性）关联？
- **现有方法不足**：当前CAD生成研究多关注输出级评估（如任务级成功率），但无法诊断成功/失败源于内部表征缺失还是行为表达能力不足。
- **局部vs全局不对称**：参数化CAD中局部成对约束与全局草图自由度(DOF)状态可能存在不同的编码机制，现有工作缺乏分离两者的评估框架。
- **解释性方法的局限**：线性探测高准确率常被直接解读为"模型掌握了知识"，但缺乏对生成表达、激活影响与行为控制的多维验证。

## 核心贡献（创新点）
1. **提出四部分审计框架**：将线性可解码性(P1/P2)、强制选择生成(P3)、激活恢复、均值差异steering整合为统一评测协议，首次系统区分"编码能力"与"行为表达"。
2. **发现局部-全局编码不对称**：引入解离指数(DI)，证明预训练对成对约束解码(P1)的提升(0.127–0.185)远大于对DOF状态解码(P2)的提升(0.037–0.048)，且该现象跨6个模型稳定成立。
3. **揭示解码-生成鸿沟**：在相同held-out数据上，P1探针F1达0.714–0.734，而P3生成F1仅0.025–0.259，Gap达0.46–0.70，证明线性可解信息未必被模型生成能力捕获。
4. **激活影响力与时空分离**：激活恢复效应集中于早期层(layer 4)后消失，而可解码性持续至晚期层，表明信息可能已路由到其他token位置或分布式表示中。
5. **控制手段的失效边界**：mean-difference steering在最强恢复层完全不产生目标类翻转，证明单方向干预无法可靠控制输出，区分了"有影响"与"可控制"的界限。

## 方法详解
- **数据与标签**：从SketchGraphs训练集派生两类标签——P1(8类成对几何关系：COINCIDENT/PARALLEL/PERPENDICULAR/TANGENT/EQUAL/MIDPOINT/CONCENTRIC/NOCONSTRAINT)与P2(3类DOF状态：under/well/over-constrained)。输入仅含几何序列化文本，排除所有EdgeOp标注。
- **三匹配任务**：
  - P1：拼接实体表示 $\mathbf{x}_{ij} = [\mathbf{h}_i; \mathbf{h}_j]$ 后8类分类
  - P2：均值池化 $\bar{\mathbf{h}} = |\mathcal{E}|^{-1}\sum_{e}\mathbf{h}_e$ 后3类分类
  - P3：零样本强制选择生成 Constraint(Ei,Ej)=...
- **探针与控制**：每层训练独立L2正则逻辑回归，class-stratified 75/25划分；对比预训练 vs 同架构随机初始化 vs pure-input baseline；shuffle-entity order控制位置线索；selectivity = $F_{1,\text{task}} - F_{1,\text{shuffled}}$ 控制探针记忆。
- **解离指数DI**：
  $$\text{DI} = (F_{1,\text{pre}}^{P1}(\ell^*) - F_{1,\text{rand}}^{P1}(\ell^*)) - (F_{1,\text{pre}}^{P2}(\ell^*) - F_{1,\text{rand}}^{P2}(\ell^*))$$
  其中$\ell^*$为预训练模型P1峰值层，所有四项在该固定层取值。
- **激活干预**：对Qwen2.5-3B与Llama-3.1-8B，在entity i位置注入高斯噪声后恢复干净激活；测量restoration rate与distractor specificity；在layer 4施加mean-difference向量，$\alpha \in \{0.5,1,2,4,8\}$，评估flip-to-target率。
- **跨数据集验证**：在Fusion 360 Gallery上复现P1探针，peak F1=0.643(layer 26)，确认层wise模式一致性。

## 实验与结果
- **模型覆盖**：Qwen2.5-0.5B/1.5B/3B/7B、Llama-3.1-8B、Mistral-7B，各配同构随机初始化对照。
- **P1解码结果**：预训练peak macro-F1=0.714–0.734(远超0.125随机基线)，selectivity=0.593–0.606；随机初始化已达0.549–0.598；纯输入仅0.359。Shuffle order降0.10–0.13，但预训练仍优于随机0.026–0.075。
- **P2解码结果**：预训练peak=0.719–0.732，但随机初始化已达0.679–0.695，预训练净增益仅0.037–0.048；仅用entity数量回归可达0.419。
- **DI稳定性**：所有模型DI>0，范围0.106–0.167，95% CI均不含零；chance-normalized DI同样为正(0.107–0.178)。
- **P1-P3鸿沟**：Gap=0.460–0.700；Mistral-7B几乎全预测Coincident(99.8%)，Qwen2.5-7B最优仅F1=0.259；four-shot仅提升至0.138±0.013，Gap仍>0.57。
- **激活恢复**：Qwen2.5-3B在layer 4达峰值0.781后于layer 16归零(Llama-3.1-8B在layer 12归零)，而P1解码在layer 21/14达峰并保持平台。
- **Steering失效**：mean-difference方向在任意$\alpha$下均无目标类翻转；Qwen2.5-3B仅改变约4倍于随机的标签数但方向错误。
- **最强结果**：P1预训练probe macro-F1最高0.734(Llama-3.1-8B)；P3生成最高0.259(Qwen2.5-7B)；P2随机初始化已达0.695。

## 相关工作脉络
1. **SketchGraphs vs Fusion 360 Gallery**：前者建模显式成对约束，后者通过构建历史序列表示CAD；本文沿用SketchGraphs因标签结构清晰，并辅以Fusion 360 Gallery跨数据集验证P1泛化。
2. **Vitruvion/DeepCAD/SkexGen**：生成模型分别针对约束草图或构造序列；本文不生成CAD，而是审计通用LLM内部的约束表征质量。
3. **Text2CAD/CAD-Llama/STEP-LLM等**：近期工作利用LLM生成CAD序列或可执行代码；本文指出高输出级性能未必源于内部约束编码，需审计中间表征。
4. **线性探针与线性表征假设**：既往工作证明游戏状态、时空信息可线性解码；本文补充证明"可解码≠可用"，需结合生成与干预验证。
5. **Activation patching (Vig et al. 2020; Meng et al. 2022)** 与 **Representation steering (Turner et al. 2024; Zou et al. 2025)**：本文整合两类干预，首次在同一框架下比较"恢复敏感性"与"方向控制"的差异。
6. **Hewitt & Liang (2019)的控制任务设计**：本文采用shuffle-label与random-init对照，并扩展至几何约束领域，验证信息是否来自预训练而非架构偏置。

## 局限性与未来方向
- **输入限定**：仅评估geometry-only序列化输入，不含EdgeOp标注，现实CAD系统中模型可能依赖额外符号信息。
- **P2标签可靠性**：DOF标签基于启发式Grübler公式计算而非solver验证，冗余约束集可能被错误标记。
- **干预范围有限**：仅测试两个backbone(Qwen2.5-3B/Llama-3.1-8B)，未覆盖更多架构或token位置；residual-stream干预未定位到具体attention head/MLP/neuron。
- **P3提示形式单一**：仅评估一种forced-choice模板与四种few-shot设置，其他verbalizer或prompt策略可能有不同表现。
- **跨数据集不完整**：Fusion 360 Gallery缺乏匹配的P2三分类标签，全局推理的跨集验证受限。
- **未来方向**：测试更多几何语料与CAD原生模型(Vitruvion)；结合solver-derived validity signals改进P2标签；扩展干预到更多backbone与direction类型；开展circuit-level分析追踪信息路由机制。

## 研究启发与可借鉴点
1. **四维度审计范式可迁移**：线性解码+生成+激活干预+steering的组合框架可复用于其他结构化推理领域(如程序合成、科学规划)，诊断"知识存在但无法调用"的失败模式。
2. **解离指数DI的设计思路**：固定层位对比不同任务预训练增益的差值，有效剥离架构/维度混杂因素，可推广至其他任务的表征效率评估。
3. **Positional shortcut控制**：entity-order shuffling是量化位置线索贡献的简洁方法，适用于任何序列化输入的结构化推理任务。
4. **分层对比的价值**：激活恢复早衰而可解码性持久，提示后期层可能已进行信息路由而非本地编码；这一发现指导未来可针对性分析cross-token信息传递机制。
5. **混合评估策略**：P3的four-shot提示仅缩小部分Gap，证明微调/提示工程不足以弥补内部表征与输出能力的断裂，为后续研究提供"为何需要结构化监督"的实证依据。

## 关键术语表
**Linear Decodability**：通过线性探针从冻结LLM隐藏状态中恢复标签的能力，衡量表征中信息的可提取性。
**Dissociation Index (DI)**：预训练对局部(P1)与全局(P2)解码增益之差，正DI表明预训练更有效地编码局部约束而非全局状态。
**Activation Patching**：将噪声注入某位置输入后，用干净激活替换中间层表示，测试该位置对预测的因果影响力。
**Mean-Difference Steering**：在特定层添加两类均值之差向量以引导输出，测试表征方向是否支持定向行为控制。
**SketchGraphs**：大规模2D参数化CAD草图数据集，提供实体类型、几何参数与成对约束标签。
**DOF (Degrees of Freedom)**：草图中未被约束的自由度数量，负/零/正值分别对应over/well/under-constrained状态。
**Selectivity**：$F_{1,\text{task}} - F_{1,\text{shuffled}}$，衡量探针性能中真正反映任务结构而非标签记忆的部分。
**Restoration Rate**：激活恢复实验中，patching成功还原clean预测的比例，反映该位置激活的预测敏感性。

## 可复现要素
- **数据集**：SketchGraphs (Seff et al. 2020)训练集；Fusion 360 Gallery r1.0.1（Appendix F）
- **模型**：Qwen2.5-0.5B/1.5B/3B/7B、Llama-3.1-8B、Mistral-7B及其同构随机初始化版本
- **代码/权重**：论文未明确声明开源代码库；使用公开模型权重
- **关键超参**：probe正则强度ℓ、划分比例75/25、shuffle重复次数、steering系数α∈{0.5,1,2,4,8}、采样种子固定复用
- **数据划分**：P1按entity-pair切分，appendix D验证5-seed sketch-level GroupShuffleSplit得可比结果
- **评估统计**：macro-F1、selectivity、95% CI(1000次bootstrap)、DI不确定性以quadrature合成
