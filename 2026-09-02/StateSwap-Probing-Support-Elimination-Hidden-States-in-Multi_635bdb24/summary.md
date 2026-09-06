---
title: "StateSwap-Probing-Support-Elimination-Hidden-States-in-Multi"
source: https://arxiv.org/pdf/2609.01081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:20:24"
field: "大语言模型机制可解释性"
keywords: ["activation substitution", "prompt framing", "mechanistic interpretability", "multiple-choice reasoning", "state intervention", "contrastive steering"]
innovations: ["提出[STATE]token作为未训练残差流接口，实现跨框架激活替换", "发现支持/消除框架在中间层诱导可分离表示并证明其行为相关性", "均值差异导向比对比激活添加具有更稳定的层间响应"]
benchmarks: ["MMLU-17", "MedQA-CH"]
---

# 论文速读：StateSwap: Probing Support–Elimination Hidden States in Multiple-Choice Questions

## 一句话总结
论文研究了LLM在处理逻辑等价但框架不同（支持导向SUP vs. 消除导向ELIM）的多选题提示时，内部表示是否会产生系统性差异。通过引入未训练的[STATE] token作为残差流接口，论文证明了两种框架在中间层诱导的激活是可分离的，且跨框架替换这些激活能够系统性地改变模型预测并提升跨框架决策一致性。

## 研究问题与动机
- **核心矛盾**：现有研究表明过程消除（PoE）策略可提高LLM多选题性能，但也有研究指出消除式提示可能损害准确性；若SUP（"哪个选项正确"）和ELIM（"哪个选项错误"）编码相同的任务目标与语义内容，理论上不应产生不同预测。
- **未解问题**：两种提示框架是否会在模型内部诱导可区分的表示？这些表示差异是否具有行为相关性？
- **实验隔离需求**：需控制解码随机性，采用确定性贪婪解码与配对提示评估，以分离表示层面效应与采样方差。
- **干预可行性**：能否通过实例级激活替换实现跨框架决策对齐，并为框架敏感性提供因果证据？

## 核心贡献（创新点）
1. **可分离表示的发现**：首次识别出SUP与ELIM提示诱导的[STATE]表征在中间层呈现显著可分性，而早期和晚期层混合度较高。
   *本质区别*：不同于以往仅关注输出差异的工作，本文从内部表示几何角度揭示了同一决策问题的框架依赖性编码。

2. **激活替换的行为干预**：证明了跨框架替换[STATE]激活能够系统性地改变模型预测，并在多个设置下提升单框架准确率与跨框架Jaccard重叠度。
   *本质区别*：与简单的提示集成或投票聚合不同，激活替换直接修改了内部计算路径，提供了干预层面的因果证据。

3. **导向方法的稳定性比较**：发现基于双框架对比的均值差异导向（mean-difference steering）比匹配的对比激活添加（CAA）方向具有更低的层间变异性与更有界的响应。
   *本质区别*：CAA对比的是指令固定下的正确/错误选项，而本文对比的是选项固定下的支持/消除指令，后者产生更稳健的层wise干预效果。

## 方法详解
- **[STATE]接口设计**：向词表注册一个未训练的special token `[STATE]`，其embedding保持初始化值，无预训练词汇语义。对于问题q，构建配对提示：
  $$\mathbf{x}_{I}(q) = \text{Concat}(I, q, [\text{STATE}]), \quad I \in \{\text{ELIM, SUP}\}$$
  通过padding指令段使两提示中[STATE]占据相同token索引$t_S = T$。

- **激活替换操作**：选定连续干预窗口$W \subseteq \{1,\dots,L\}$后，对框架$I$的提示执行：
  $$\mathbf{s}^{(k)}(I, q) \leftarrow \mathbf{s}^{(k)}_{\text{cache}}(\bar{I}, q), \quad k \in W$$
  即用互补框架的缓存状态覆盖中间层$k$的残差流输出，后续自回归生成正常进行。

- **窗口定位诊断**：扫描特征窗口$[j:j+w)$，计算配对Cohen's d统计量。两种诊断方式：
  - **随机方向诊断**：采样随机单位向量投影，得$d_l^{\text{rand}}(j,w)$。
  - **均值差异诊断**：构造$\mathbf{v}_l(j,w) = \frac{1}{N}\sum_i(\mathbf{s}_{\text{ELIM},i,[j:j+w)}^{(l)} - \mathbf{s}_{\text{SUP},i,[j:j+w)}^{(l)})$，标准化后投影得$d_l^{\text{md}}(j,w)$。
  按Top-K均值聚合层强度，选取$\tilde{d}_l \geq 0.8 \cdot \max_{l'}\tilde{d}_{l'}$的连续区间作为$W$。

- **量化评估指标**：准确率ACC、跨框架Jaccard重叠度$J = \frac{|C_{\text{Sup}} \cap C_{\text{Elim}}|}{|C_{\text{Sup}} \cup C_{\text{Elim}}|}$、长度感知余弦相似度（LAC）、BERTScore-F1。

## 实验与结果
- **数据集**：MMLU-17（推理导向子集，4,607题，英语）与MedQA-CH（中文医学基准，1,015题），共5,622题。50个校准样本用于定位窗口，不进入测试。
- **模型**：Qwen-2.5-7B-Instruct与GLM-4-9B，zero-shot设定，贪婪解码（max_new_tokens=1024）。
- **干预窗口**：Qwen选层11–20，GLM选层21–40。
- **主要结果**：
  | 模型 | 数据集 | ACC(SUP) | ACC(ELIM) | Jaccard |
  |------|--------|----------|-----------|---------|
  | Qwen-2.5-7B | MMLU-17 (Sub) | 66.31 | 74.36 | 78.10 |
  | Qwen-2.5-7B | MedQA-CH (Sub) | 80.26 | 77.45 | **81.58** |
  | GLM-4-9B | MMLU-17 (Sub) | 68.74 | 69.21 | 78.21 |
  | GLM-4-9B | MedQA-CH (Sub) | 76.82 | 77.92 | 82.48 |
- **最强结果**：GLM-4-9B在MedQA-CH上达82.48% Jaccard；Qwen-2.5-7B在MedQA-CH上达80.26% SUP准确率。
- **消融验证**：
  - 随机标签控制：可分离性显著高于随机基线。
  - 接口特异性：[STATE]替换优于最终内容token替换（LAC中位数0.8766 vs. 0.9780）。
  - 内容特异性：交叉框架替换保持高相似度，而零向量/随机替换显著降低稳定性。
  - 位置敏感性：非末尾位置放置[STATE]时>95%生成失效。

## 相关工作脉络
- **Activation Patching**（Heimersheim & Nanda, 2024）：提供测试局部组件行为相关性的框架，本文将其推广至跨框架表示替换。
- **Contrastive Activation Addition (CAA)**（Rimsky et al., 2024）：通过正确/错误选项对比构造导向，本文对比指令差异而非答案差异，获得更稳定的层间响应。
- **Inference-time Intervention**（Li et al., 2023）：证明高维行为可通过激活空间操纵影响，本文在MCQ框架场景下验证了这一观点。
- **Process of Elimination (PoE)**（Ma & Du, 2023）：动机来源，但本文从表示层面解释为何消除策略效果不一致。
- **Representation Engineering**（Zou et al., 2023）：自顶向下透明性方法，本文使用未训练token作为最小化读写接口，区别于学习的continuous prompt。
- **Sparse Autoencoder for CoT**（Chen et al., 2025）：刻画链式思维激活模式，本文聚焦框架敏感性而非推理过程本身。

## 局限性与未来方向
- **解码限制**：仅在确定性贪婪解码下验证，未评估采样解码或多步生成场景。
- **任务范围**：方法绑定多选题结构，扩展至开放-ended任务需设计共享评估目标的配对提示与新度量。
- **接口依赖**：未训练[STATE] token虽对初始化鲁棒，但仍可能引入细微分布偏移，效应独立性未完全证明。
- **白盒限制**：需访问中间激活，黑盒模型适用性待检验。
- **细粒度状态混淆**：SUP/ELIM类内可能混杂领域、不确定性、推理策略等因子，未做分层解耦分析。

## 研究启发与可借鉴点
1. **未训练token作为干预接口**：注册无预训练语义的special token并置于序列末尾，可作为最小侵入式的残差流读写口，避免learning-based prompt tuning的优化开销。
2. **成对提示控制设计**：保持题目、选项、语言、解码不变，仅改变指令框架，有效隔离表示差异来源，为因果诊断提供干净实验设置。
3. **窗口定位的Top-K聚合策略**：使用Cohen's d结合Top-K均值聚合层强度，无需调优即能稳定定位表征可分离区域，可迁移至其他表示探针任务。
4. **均值差异导向 vs. CAA**：固定答案对比指令的变化比固定指令对比答案的变化产生更有界层响应，为activation steering的方向构造提供新视角。

## 关键术语表
- **Support-Oriented (SUP) / Elimination-Oriented (ELIM)**：两种逻辑等价的提示框架，前者要求识别正确选项，后者要求识别错误选项。
- **[STATE] Token**：新增的未训练special token，作为残差流的固定读写接口，其上下文表示聚合 preceding prompt 信息。
- **State Substitution**：在选定层窗口内，将一种框架的[STATE]激活替换为互补框架的缓存激活，保持文本输入不变。
- **Mean-Difference Steering**：基于配对提示激活差的均值方向进行激活偏移，相比CAA具有更低的层间变异性。
- **Jaccard Overlap**：跨框架决策一致性感知指标，定义为两框架正确回答题号集合的交并比。
- **Intervention Window W**：通过可分离性诊断选定的连续Transformer层区间，干预仅在此范围内执行。
- **Length-Aware Cosine Similarity (LAC)**：衡量生成文本语义相似度的指标，通过长度比惩罚过度偏离的生成。

## 可复现要素
- **数据集**：MMLU-17与MedQA-CH均为公开基准，使用官方test split；50个校准样本来自训练split。
- **代码/权重**：代码已开源（https://github.com/Cha0Ga0/SWAPSTATE），模型为开源Qwen-2.5-7B-Instruct与GLM-4-9B。
- **关键超参**：贪婪解码（do_sample=False, max_new_tokens=1024），随机种子固定为123，浮点精度float16，KV缓存启用。
- **窗口参数**：Qwen-2.5-7B干预层11–20，GLM-4-9B干预层21–40；特征窗口w=6。
- **指令模板**：每组框架含两个paraphrase变体，各框架2次确定性运行，严格多数投票。
