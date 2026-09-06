---
title: "StateSwap-Probing-Support-Elimination-Hidden-States-in-Multi"
source: https://arxiv.org/pdf/2609.01081v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:20:32"
field: "大语言模型可解释性与推理时干预"
keywords: ["mechanistic interpretability", "activation intervention", "prompt framing", "multiple-choice reasoning", "state substitution", "contrastive activation steering"]
innovations: ["提出未训练[STATE]token作为推理时残差流读写接口", "通过交换SUP/ELIM配对提示的中间层激活系统性改变模型预测", "揭示双提示均值差方向比CAA方向层间波动更小"]
benchmarks: ["MMLU-17", "MedQA-CH"]
---

# 论文速读：StateSwap: Probing Support–Elimination Hidden States in Multiple-Choice Questions

## 一句话总结
本文提出 StateSwap，一种无需训练的推理时干预框架，通过在选择题提示末尾插入未训练 token [STATE] 作为残差流读写接口，交换支持导向（SUP）与消除导向（ELIM）两种逻辑等价提示在不同中间层产生的可分离激活，系统性改变模型预测并提升跨提示一致性与准确率。

## 研究问题与动机
- **核心问题**：SUP 提示（"哪个选项正确？"）与 ELIM 提示（"哪个选项错误？"）在逻辑上指向同一答案，但实证研究表明两者可能诱导不同的内部表示，导致模型预测不一致；这种不一致是否源于表征层面的差异？
- **现有方法不足**：
  - Process-of-elimination (PoE) 策略在某些研究中提升 LLM 性能，但在另一些研究中反而降低准确率，矛盾现象缺乏统一解释。
  - 传统提示工程（如多提示集成、链式推理）仅关注输出聚合，未触及决策过程中表征结构的因果机制。
  - 现有激活干预方法（如 CAA）多依赖数据集级统计方向，缺乏与具体决策语义对齐的实例级干预接口。

## 核心贡献（创新点）
- **发现提示框架诱导可分离的中间层表征**：SUP 与 ELIM 提示在 [STATE] 位置产生线性可分的隐藏状态，且该可分离性集中于中间层（非均匀分布于全网络深度），为提示框架效应提供了表征层面的因果证据。
- **提出实例级激活替换干预**：通过交换配对提示的 [STATE] 激活，可在保持文本输入不变的前提下系统性改变模型预测；SWAP 不仅提升单提示准确率，还显著提高跨提示 Jaccard 一致性。
- **揭示均值差 steering 方向的结构化优势**：基于双提示对比的均值差方向在层间响应上比匹配的对比激活加法（CAA）方向更稳定、波动更小，表明提示语义对比比选项对比能产生更可控的干预方向。

## 方法详解
- **[STATE] 接口设计**：在 tokenizer 中注册未训练特殊 token [STATE]，嵌入矩阵随 vocabulary 扩展而扩容，但保持初始化值不变；所有提示以 [STATE] 作为最后一位，且通过指令段 padding 确保两个提示中 [STATE] 占据相同 token 索引 $t_S = T$。
- **双提示构造**：对每道题 $q$，构建配对提示 $\mathbf{x}_{\text{SUP}}(q) = \text{Concat}(\text{SUP}, q, [\text{STATE}])$ 与 $\mathbf{x}_{\text{ELIM}}(q) = \text{Concat}(\text{ELIM}, q, [\text{STATE}])$，仅指令部分不同。
- **可分离性诊断**：扫描连续特征窗口 $[j : j+w)$，计算配对 Cohen's d 统计量；使用两种标量化：随机方向投影与均值差方向投影（$\mathbf{v}_l(j,w) = \frac{1}{N}\sum_i(\mathbf{s}_{\text{ELIM},i}^{(l)} - \mathbf{s}_{\text{SUP},i}^{(l)})$），选取层强度超过最大值的 80% 作为候选层。
- **激活替换规则**：在选定层窗口 $W$ 内，将 $\mathbf{s}^{(k)}(\text{SUP}, q) \leftarrow \mathbf{s}^{(k)}_{\text{cache}}(\text{ELIM}, q)$（反向同理），替换仅作用于 $W$ 内的 post-block 残差状态，下游层通过残差传播自然更新。
- **无 [STATE] 的均值差 steering 扩展**：移除 [STATE]，在答案 token 位置构建双提示对比方向 $\mathbf{v}_l^{\text{DF}} = \frac{1}{N}\sum_i[h_l(p_i^{\text{SUP}}, y_i) - h_l(p_i^{\text{ELIM}}, y_i)]$，与标准 CAA 方向 $\mathbf{v}_l^{\text{CAA}} = \frac{1}{N}\sum_i[h_l(p_i^{\text{SUP}}, y_i) - h_l(p_i^{\text{SUP}}, \tilde{y}_i)]$ 进行逐层系数扫描对比。

## 实验与结果
- **数据集**：MMLU-17（17 个推理导向子集，4,607 题）与 MedQA-CH（中文医学基准，1,015 题），共 5,622 题；50 个校准样本用于窗口定位，不在测试集使用。
- **模型与解码**：Qwen-2.5-7B-Instruct 与 GLM-4-9B，zero-shot，确定性 greedy 解码，固定 chat template 与上下文长度。
- **主要结果**（Table 1，均基于两提示实例严格多数投票）：

| 模型 | 基准 ACC(SUP) | Sub ACC(SUP) | 基准 ACC(ELIM) | Sub ACC(ELIM) | 基准 Jaccard | Sub Jaccard |
|---|---|---|---|---|---|---|
| Qwen-2.5-7B / MMLU-17 | 65.52 | 66.31 | 65.38 | 74.36 | 66.53 | 78.10 |
| Qwen-2.5-7B / MedQA-CH | 77.12 | 80.26 | 77.45 | 81.58 | 76.63 | 81.58 |
| GLM-4-9B / MMLU-17 | 66.47 | 68.74 | 69.21 | 72.36 | 72.96 | 78.21 |
| GLM-4-9B / MedQA-CH | 73.54 | 76.82 | 77.92 | 82.48 | 78.67 | 82.48 |

- **核心结论**：
  - StateSwap 在全部设置下均提升单提示准确率与跨提示 Jaccard；
  - GLM-4-9B 在 MedQA-CH 上 ELIM Sub 达到最高 82.48%，相对基准提升 4.56pp；
  - 双 StateSwap 集成（Table C.1/C.2）显著优于朴素 SUP+ELIM 集成，证明增益非单纯投票效应；
  - 消融实验（Figure 6/Appendix E）表明：[STATE] 接口 specificity、结构化替换内容必要性、最终位置依赖性；
  - 无 [STATE] 的均值差 steering（Figure 7/Table C.5）在 $a = -1$ 时层间波动范围从 CAA 的 60.89 降至 28.08（Qwen）/ 19.90（GLM）。

## 相关工作脉络
- **PoE 提示与选择题推理**：Ma & Du (2023) 提出 PoE 策略，部分工作验证其提升（Zhu et al., 2025; Fu et al., 2025），另有工作指出 ELIM 可能损害准确率（Balepur et al., 2024）；本文从表征层面解释该矛盾。
- **激活 patching 与因果追踪**：Meng et al. (2022) 建立事实编辑的因果追踪框架，Heimersheim & Nanda (2024) 提供 activation patching 解读指南；本文沿用该思路但聚焦于提示框架效应的表征隔离。
- **推理时激活干预**：Li et al. (2023)、Turner et al. (2024)、Zou et al. (2023) 展示激活方向操控可影响模型行为；Rimsky et al. (2024) 提出 CAA；本文以实例级配对替换替代数据集级统计方向。
- **参数高效接口**：Prompt tuning / prefix tuning（Lester et al., 2021; Li & Liang, 2021）学习连续 embedding；本文采用未训练 token 作为最小接口，避免任何微调开销。
- **状态追踪与机械可解释性**：Zhang et al. (2025) 证明 transformer 可发展出 MLP 神经元级别的隐状态追踪机制；本文发现框架敏感表征集中分布于中间层，支持该观点。

## 局限性与未来方向
- **任务与解码限制**：仅在四选项选择题与 greedy 解码下验证，未评估采样解码、多步生成或开放生成场景；
- **输出解析依赖 MCQ 结构**：当前方法要求输出可解析为 Correct/Wrong 选项集合，扩展至开放域需重新设计评估指标；
- **[STATE] 接口引入潜在偏移**：未训练 token 可能引入细微分布漂移，虽消融显示影响可忽略，但未证明完全独立于接口设计；
- **白盒访问假设**：依赖完整激活访问，对黑盒模型不适用；
- **帧内异质性未分解**：SUP/ELIM 类别内部可能包含领域、置信度、推理策略等细粒度状态，当前分析对此做边际化处理。

## 研究启发与可借鉴点
- **最小接口设计范式**：使用未训练特殊 token 作为残差流读写接口，无需参数更新即可实现实例级干预，可迁移至其他推理时干预任务；
- **双提示对照协议**：保持题目与选项不变、仅切换指令语义（支持 vs 消除）的配对设计，可有效隔离"框架"这一单一混淆变量；
- **均值差方向 vs CAA 方向的对比分析**：提示对比（保持答案固定、切换指令）产生的 steering 方向比选项对比（保持指令固定、切换答案）层间波动更小，为 future 的 prompt engineering 提供了方法论参考；
- **分层可分离性诊断流程**：从 random-direction 到 mean-difference 的双通道诊断 + Top-K mean 聚合 + 80% 阈值选取，构成可复用的层窗口定位 pipeline。

## 关键术语表
- **[STATE]**：论文新增注册的未训练特殊 token，作为残差流读写接口，嵌入值保持初始化，不参与参数更新。
- **SUP / ELIM 提示**：支持导向（"选择正确选项"）与消除导向（"选择错误选项"）两种逻辑等价的提示框架。
- **激活替换（State Substitution）**：在选定层窗口 $W$ 内，将某一提示的 [STATE] 残差状态替换为配对提示的同层状态，保持文本输入不变。
- **Cohen's d 可分离性诊断**：基于配对样本差值的标准化效应量，用于量化两提示在特征窗口上的表征差异强度。
- **Mean-difference steering**：对配对样本的 [STATE] 差值取均值后归一化得到的干预方向。
- **CAA（Contrastive Activation Addition）**：Rimsky et al. (2024) 提出的数据集级对比激活加法，通过正确与错误选项的激活差构造 steering 方向。
- **Jaccard 指数**：度量两个提示下正确回答集合的交集与并集之比，反映跨提示决策一致性。
- **LAC（Length-Aware Cosine Similarity）**：句子级余弦相似度乘以长度修正因子，用于评估干预前后生成的语义稳定性。

## 可复现要素
- **数据集**：MMLU（公开）、MedQA-CH（公开中文子集）；官方 test split 用于评测，50 个训练样本校准窗口定位；
- **代码**：已开源，地址 https://github.com/Cha0Ga0/SWAPSTATE；
- **模型**：Qwen-2.5-7B-Instruct、GLM-4-9B（均为开源权重）；
- **关键超参**：特征窗口大小 $w$ 与起始 $j$ 由诊断选取；层窗口 $W$ 阈值设为最大强度的 80%；Qwen 选取 11–20 层，GLM 选取 21–40 层；greedy 解码，max_new_tokens=1024；steering 系数 $a \in \{-2, -1, 1, 2\}$；
- **随机种子**：全部实验使用 seed=123；[STATE] 随机初始化鲁棒性测试覆盖 seed {1, 2, 3, 12, 123}。
