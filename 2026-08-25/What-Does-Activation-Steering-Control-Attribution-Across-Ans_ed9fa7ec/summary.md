---
title: "What-Does-Activation-Steering-Control-Attribution-Across-Ans"
source: https://arxiv.org/pdf/2608.22985v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:04:23"
field: "大语言模型可解释性与机制解释"
keywords: ["activation steering", "interpretability", "representation engineering", "attribution", "contrastive activation addition", "LLM interpretability"]
innovations: ["提出跨编码评估框架，固定干预方向反事实重构答案编码以归因导向实际控制的变量", "发现extraction-index following现象并定位至输出敏感的低秩子空间（15.4%范数保留96.3%效应）"]
benchmarks: ["NormBank", "MNLI", "Social Chemistry 101 (SC101)"]
---

# 论文速读：What-Does-Activation-Steering-Control-Attribution-Across-Ans

## 一句话总结
论文提出**跨编码导向评估（Cross-Encoding Steering Evaluation）**框架，固定干预方向并重构答案编码，用于归因激活导向（如CAA）实际控制的对象；研究发现导向增益往往追踪的是提取时的**选项索引（extraction index）**而非语义标签，且该效应集中于输出敏感的低秩子空间中。

## 研究问题与动机
1. **归因模糊性**：现有激活导向方法（CAA、ITI等）通常在用于构建方向的answer encoding下评估，报告的性能提升可能仅反映对构建时答案标识符（如A/B/C）的兼容性，而非真正的语义控制。
2. **混淆变量难分离**：一个指向C选项的增益可能来自语义标签"expected"、选项标识符"C"本身、或其展示行位置，现有评估无法区分这三种解释。
3. **缺乏系统性审计**：已有工作多关注导向是否有效，但未系统回答"导向到底控制了啥"这一基本归因问题。
4. **多粒度评估差异未被重视**：多选题（MCQ）得分与开放式生成行为可能给出矛盾结论，需要统一的归因框架。

## 核心贡献（创新点）
1. **提出跨编码评估框架**：固定干预方向与超参，仅改变答案编码（标签→标识符映射、词汇表、行顺序），实现对语义标签跟随、提取索引跟随、提取行跟随三者的反事实分离。
2. **发现extraction-index following现象**：在NormBank上，CAA的增益主要来自追踪提取时的选项索引而非当前语义标签，该结论在所有五种重映射下稳健成立。
3. **定位效应来源的深度与子空间**：extraction-index效应出现在较深层（75%–87.5%深度），且集中于输出敏感的低秩子空间（仅含15.4%方向范数却保留96.3%效应）。
4. **跨方法/任务验证归因差异**：ITI复现了NormBank上的索引优先模式；MNLI整体支持索引跟随，但SC101反转偏好语义标签跟随，证明归因依赖模型、任务与评估粒度。
5. **揭示MCQ与开放式评估的结论分歧**：公开CAA方向在多选题与开放生成中可能对同一行为（如refusal/sycophancy）给出相反判断。

## 方法详解
1. **对比激活添加（CAA）**：从正负样本对的隐藏状态差中抽取方向 $\mathbf{v}_{\mathrm{raw}}^{(c, p_{\mathrm{ext}})} = \frac{1}{N_c}\sum_i(h(x_i^+) - h(x_i^-))$，在推理时于注入位置 $p_{\mathrm{inj}}$ 添加 $\alpha \mathbf{v}$。
2. **跨编码评估**：固定 $\mathbf{v}$、层、位置、强度 $\alpha$，仅改变答案编码 $f$（包括标签-标识符映射 $\pi$、词汇表、行顺序、完成格式），以每个编码内的平均token对数似然边界 $m_f(x) = s_f(y^+|x) - s_f(y^-|x)$ 衡量效果。
3. **三种跟随定义**：
   - **Semantic-label following**：跟随当前编码下目标语义标签对应的标识符。
   - **Extraction-index following**：跟随提取时目标标签所分配的标识符索引（如A=1, B=2, C=3），跨词汇表对齐为A/X/1、B/Y/2、C/Z/3。
   - **Extraction-row following**：跟随提取时目标标签所在展示行的当前标识符。
4. **指数优势**：$A_{\mathrm{id}\mathrm{x},\pi}(x) = \Delta m_{\mathrm{id}\mathrm{x},\pi}(x) - \Delta m_{\mathrm{sem},\pi}(x)$，正值表示索引效应超过语义标签效应。
5. **因子审计**：交叉6种语义映射×3种词汇表×6种行顺序=108种条件，分离索引与行位置的影响。
6. **输出敏感子空间定位**：计算选项对logit差对预答案位置隐藏状态的梯度 $\mathbf{g}_{x,a,b} = \nabla_h[z(o_a) - z(o_b)]$，构建梯度矩阵 $\mathbf{G}$ 的SVD，取前$r$个右奇异向量构成 $\mathbf{U}_r$，将CAA方向分解为投影 $\mathbf{v}_\parallel$ 与残差 $\mathbf{v}_\perp$，比较两者的效应保留率与能量占比。

## 实验与结果
- **数据集**：NormBank（主要审计，T/N/E三标签，上下文匹配）、MNLI（entailment/neutral/contradiction）、SC101（bad/ok/good，行动级配对）。
- **模型**：Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、Mistral-7B-Instruct-v0.3、Gemma-2-9B-IT。
- **NormBank核心结果**（表1）：所有五种重映射下extraction-index效应均为正（3.00–4.13），且均超过semantic-label效应，指数优势为1.95–4.04。
- **因子审计**（图2、表14）：extraction-index效应超过extraction-row效应在所有四模型成立（+0.32至+1.86）；Llama/Mistral/Gemma优先索引，Qwen优先语义标签（-0.655）。
- **深度定位**（图3）：范数匹配后，extraction-index效应在50%/62.5%/75%/87.5%深度分别为0.065/0.130/0.215/0.160，优势从-0.061升至+0.098/+0.236/+0.208，75%处达峰。
- **位置定位**（表19）：从预答案位置移至scenario-end提取使指数效应从0.218骤降至0.004（-98.3%）。
- **子空间集中性**（表21–24）：投影子空间含15.4%方向平方范数，保留96.3%指数效应；残差含84.6%范数仅保留1.0%效应。跨词汇表迁移时投影仍主导（X/Y/Z保留95.0%，1/2/3保留83.3%）。
- **ITI复现**（表27–28）：在三模型上指数优势全为正，与CAA排序一致。
- **跨任务**（表2）：MNLI整体5/5映射指数优势为正（Qwen例外）；SC101反转，0/5映射指数优势为正。
- **开放生成**（表3）：hallucination在MCQ与开放生成中均正向；refusal MCQ高增益但开放无显著变化；sycophancy MCQ非正但开放显著正向。

## 相关工作脉络
1. **Representation Engineering / CAA**（Zou et al., 2023; Rimsky et al., 2024）：本文与其区别在于不依赖answer encoding一致性来声称语义控制，而是通过冻结干预反事实重构编码来归因。
2. **Inference-Time Intervention (ITI)**（Li et al., 2023）：本文用ITI验证NormBank归因模式的可迁移性，发现其与CAA在索引优先排序上一致。
3. **Output-sensitive subspaces / DecodeShare**（Shao et al., 2026）：DecodeShare识别任务共享解码子空间并因果分解方向；本文采用类似投影-干预逻辑但目标不同——将CAA方向分解到identifier-readout局部子空间以检验extraction-index效应集中度。
4. **Evaluation reliability**（Pres et al., 2024; Tan et al., 2024）：已有工作指出导向对输入敏感、分布外脆弱；本文进一步证明即使在同一任务内，答案编码变化也会导致归因结论翻转。
5. **Verbalizer/manipulation robustness**（Zheng et al., 2023; Li et al., 2024）：多项选择题标签词鲁棒性研究；本文扩展到导向干预场景，证明固定方向的效应随编码变化而系统性偏移。
6. **Behavioral granularity**（Xu et al., 2026）：不同行为粒度评估可能冲突；本文通过公开CAA方向在MCQ与开放生成中的分歧提供了实证案例。

## 局限性与未来方向
1. **六映射审计覆盖有限**：仅测试三种三标签任务和四个指令微调模型族，未涵盖更多标签结构或模型架构。
2. **子空间为功能性定位**：梯度定义的子空间在tested层和位置上将效应局部化，但未识别完整的因果回路。
3. **层间范数匹配局限**：匹配方向$L_2$范数但未匹配各层特定的模型敏感性，深度差异可能混杂架构因素。
4. **位置定位仅单点干预**：预答案与scenario-end的对比使用单一注入位置，未能完整刻画效应沿序列的传播路径。
5. **开放生成评估依赖协议特定剂量**：公开CAA复现使用原论文protocol-dose，跨协议比较需谨慎。

## 研究启发与可借鉴点
1. **跨编码审计作为归因标准流程**：任何声称控制某语义概念的导向方向，应通过反事实答案编码重映射来检验其真正追踪的变量，避免将索引/行效应误读为语义控制。
2. **输出敏感子空间分解技术**：通过option-logit梯度构建局部identifier-readout子空间并投影方向，可有效分离"有效成分"与"噪声成分"，适用于其他导向方法的责任分配分析。
3. **多粒度评估联合验证**：MCQ得分与开放生成行为需独立评估，二者结论不一致时不应简单以MCQ为准；建议建立claim-evidence匹配表（如论文Table 4）。
4. **方向构建的映射平衡控制**：将六种语义映射下抽取的方向平均后再范数匹配，可显著降低extraction-index优势（表38），为设计更"语义忠实"的导向方向提供了构造策略。
5. **跨词汇表迁移测试**：冻结A/B/C导出的子空间并在X/Y/Z、1/2/3下评估，可检验输出敏感成分的通用性，该方法可直接迁移至其他identity-tracking归因任务。

## 关键术语表
**Cross-Encoding Steering Evaluation**：固定干预方向与超参，对同一测试项反事实地重构答案编码（标签映射、词汇表、行顺序），以分离导向实际追踪的变量。

**Extraction-index following**：导向效应追踪的是提取时目标标签所分配的标识符索引（如A/B/C中的第1/2/3位），而非当前编码下该语义标签所对应的标识符。

**Semantic-label following**：导向效应追踪当前测试编码下目标语义标签所直接对应的标识符。

**Output-sensitive subspace**：由option-logit差对隐藏状态的梯度张成的低维子空间，反映局部 identifier-readout 敏感性。

**Factorial audit**：交叉变换语义映射（6种）、标识符词汇表（3种）和展示行顺序（6种）的完整因子实验，用于分离索引、行、标签三类效应的独立贡献。

**Extraction-index advantage ($A_{\mathrm{id}\mathrm{x}}$)**：extraction-index效应减去semantic-label效应之差，正值表示导向更倾向于追踪提取索引。

**Mapping-balanced direction**：对全部六种语义映射下抽取的CAA方向求平均并范数匹配，作为减少答案编码依赖性的构造控制。

**Random-adjusted effect**：从原始效应中减去$L_2$匹配的随机方向效应的均值，用于控制对残差扰动的通用编码敏感性。

## 可复现要素
- **数据集**：NormBank（Ziems et al., 2023）、MNLI（Williams et al., 2018）、SC101（Forbes et al., 2020）均为公开数据集。
- **代码/权重**：论文配套提供评估代码与数据隔离审计（Appendix A），模型使用公开指令微调版本（Qwen2.5-7B、Llama-3.1-8B、Mistral-7B、Gemma-2-9B）。
- **关键超参**：CAA注入强度 $\alpha = 0.8$（NormBank/MNLI）或 $\alpha = 1.0$（SC101）；层位置为各模型75%深度（Qwen 20、Llama 23、Mistral 23、Gemma 31）；随机控制方向数 $K=5$；子空间秩在{2,4,8,16}中选择满足验证梯度平方范数捕获≥90%的最小值。
