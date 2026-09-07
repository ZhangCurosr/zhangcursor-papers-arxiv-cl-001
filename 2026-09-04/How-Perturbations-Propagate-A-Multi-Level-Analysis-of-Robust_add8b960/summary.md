---
title: "How-Perturbations-Propagate-A-Multi-Level-Analysis-of-Robust"
source: https://arxiv.org/pdf/2609.03322v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:03:30"
field: "大语言模型鲁棒性与可解释性"
keywords: ["robustness", "perturbation propagation", "mechanistic interpretability", "activation patching", "intrinsic dimension", "CKA", "adversarial robustness", "attention heads"]
innovations: ["提出输出-几何-组件三级联动扰动传播评估框架", "HotFlip对抗扰动与rate-matched随机替换的配对对照设计", "发现copying head分数与activation patching恢复高度相关（r≈0.7）"]
benchmarks: ["WikiText-2", "GPT-2系列（base/medium/large/xl）", "Qwen2.5（0.5B/1.5B）"]
---

# 论文速读：How-Perturbations-Propagate-A-Multi-Level-Analysis-of-Robust

## 一句话总结
本文从输出行为、隐藏状态几何和注意力头功能三个层面，系统分析了六种自然与合成输入扰动在解码器语言模型中的传播机制，发现单一行为或表示指标可能误导鲁棒性评估结论。

## 研究问题与动机
1. 现有LLM鲁棒性评估主要依赖输出行为指标（如NLL、生成文本变化），无法揭示扰动如何改变模型内部计算过程。
2. 两种扰动可能产生相似的行为退化，却破坏了不同的表示或计算组件；反之，微小的输出变化可能掩盖了隐藏状态的显著偏移。
3. 当前研究缺乏对扰动效应是否因腐蚀类型而异、是否定位到可识别机制、是否跨模型规模和架构族泛化的系统回答。
4. 对抗性扰动（如HotFlip）与自然/合成扰动在内部层面的差异尚未被严格控制对比——尤其是控制了编辑率和位置后。

## 核心贡献（创新点）
1. 提供了六种文本扰动类型的多层次实证分析，联合测量输出行为、表示几何和注意力头响应；与既有工作相比，突破了单一行为指标评估框架。
2. 测试了扰动效应是否与功能表征的注意力头相关联（通过注意力熵变化和激活修补）；与先前仅研究clean输入上头功能的工作形成对比。
3. 评估了扰动签名在GPT-2多尺度（base→xl）与Qwen2.5系列间的稳定性；跨架构族的对比在鲁棒性分析中较为罕见。
4. 将梯度引导的HotFlip对抗扰动与精确匹配的随机token替换进行对照，分离了对抗优化效应与单纯编辑率的贡献；这一配对实验设计是本文的独特之处。

## 方法详解
**输入与扰动设置**：使用WikiText-2 Raw数据集，保留长度≥128的序列，固定seed采样300条；扰动强度p∈{5%, 30%, 50%}。六种扰动类型覆盖字符级（uniform character substitution、keyboard typo）、词级（word substitution、synonym substitution via WordNet）、token级（random token substitution）和结构级（token shuffling，保留token身份但打乱局部顺序）。

**模型**：GPT-2系列（gpt2/gpt2-medium/gpt2-large/gpt2-xl）和Qwen2.5（0.5B/1.5B）。行为比较使用全部六个checkpoint；注意力头分析仅在GPT-2上进行。

**行为指标**：(1) 序列级负对数似然（NLL）；(2) 生成输出的normalized Levenshtein距离（output divergence）。

**表示指标**：(1) Centered Kernel Alignment（CKA）逐层比较clean与perturbed激活矩阵的相似度，剔除前5个高方差维度以降低离群敏感性；(2) TwoNN估计器计算逐层本征维度变化，衡量激活流形的局部复杂度偏移。CKA仅适用于保持token数量的扰动（token shuffle），其余扰动因re-tokenization导致位置不对齐。

**注意力头功能与响应**：计算四类head-function score：previous-token、duplicate、induction、copying（基于OV电路的C矩阵行top-k覆盖率）。扰动响应通过：(1) 30%扰动下归一化注意力熵变化；(2) clean activation patching——将perturbed输入的head h激活替换为clean激活，计算NLL和output divergence的恢复率Δ%，仅适用于token-substitution和shuffle（保持token数）。

**对抗对比**：HotFlip梯度引导攻击，在选定位置计算梯度并构造词汇表top-50候选短列表，贪心选择使NLL增量最大的token；每个对抗替换配对一个随机token控制（同位置、同速率），用相同度量对比。

## 实验与结果
**RQ1（跨层传播）**：字符替换产生最大output divergence（50%扰动下0.932），键盘误触次之（0.852）；token和word替换的output divergence相近（0.669 vs 0.661），但NLL差异显著（10.24 vs 8.56），说明生成文本变化和预测置信度并不一致。Shuffling的破坏最渐进。CKA和本征维度指标给出与行为层不同但互补的排序：字符/typo扰动在浅层降低本征维度，而token substitution产生最大的本征维度正偏移。

**RQ2（注意力头响应）**：Induction score与熵变化的负相关最强（多数扰动下−0.27至−0.609）；shuffle下previous-token score呈强正相关（0.670）。Copying score与激活修补恢复高度相关：token substitution下Δ%NLL相关系数0.726、Δ%OutDiv为0.652；shuffle下分别为0.622和0.348。Copying score与熵变化弱相关，说明注意力重分布和功能重要性是head响应的不同维度。

**RQ3（跨模型泛化）**：GPT-2与Qwen2.5在NLL基准上相差0.592，需基线调整后下结论。Output divergence跨家族差异整体较小，shuffle在30%/50%下差距最大（+0.071/+0.059）。本征维度上，token substitution在GPT系列引发更大变化，而synonym substitution几乎无变化；字符/typo扰动在浅层Qwen比GPT更敏感（可能源于re-tokenization噪声）。跨家族差异因指标和扰动类型而异，不能简单归因于架构。

**RQ4（对抗vs随机）**：HotFlip在所有六个checkpoint的5%和30%扰动下，均产生更高的NLL和output divergence（Table 5），且GPT-2的30%扰动下CKA更低、本征维度增量更大（Figure 4）。对抗优化的破坏效应在行为和内部层面均一致且跨模型稳健。

## 相关工作脉络
1. Belinkov & Bisk (2018)、Pruthi et al. (2019)：证明字符/词级自然噪声显著降解模型性能；本文扩展至多层评估并引入对抗vs随机配对设计。
2. Ebrahimi et al. (2018) HotFlip：梯度引导文本对抗攻击；本文沿用其方法但新增与rate-matched随机替换的对照及内部表示分析。
3. Kornblith et al. (2019) CKA；Davari et al. (2023) 指出CKA对离群维度敏感；本文在CKA计算前剔除高方差维度并联合本征维度交叉验证。
4. Aghajanyan et al. (2021)、Yin et al. (2024)、Razzhigaev et al. (2024) 将本征维度应用于LM微调、真实性等分析；本文首次将本征维度变化作为扰动响应指标。
5. Olsson et al. (2022)、Wang et al. (2023) 识别induction和copying头等功能机制；本文将其与扰动传播关联，验证clean-task头功能对perturbed-input的预测力。
6. Heimersheim & Nanda (2024) 讨论activation patching的解释依赖；本文限定仅用于token-count-preserving扰动并谨慎解释恢复结果。

## 局限性与未来方向
1. CKA要求激活矩阵形状匹配，四个扰动类型（char/typo/word/synonym）经re-tokenization后token数可能变化，导致逐层位置对齐不可靠，这些扰动的CKA分析受限。
2. 注意力头分析仅建立相关性，未确立因果链接；head function score与扰动传播之间是否存在因果机制仍需干预实验验证。
3. GPT-2与Qwen2.5的跨家族差异可能源于GQA、RoPE、词表大小等架构因素，但本文未做消融隔离，属于未来方向。
4. 全部实验仅在WikiText-2一个数据集和GPT/Qwen两个模型族上进行，结论外推到其他文本域和架构需谨慎。
5. 对抗扰动框架可扩展至其他扰动类型，但本文仅在token substitution上完成配对比较。

## 研究启发与可借鉴点
1. **多层次指标收敛准则**：当多个独立指标（行为、几何、组件级）指向同一结论时，鲁棒性 claim 的可信度显著提升；单一指标结论应持审慎态度。这对团队设计评估协议有直接借鉴价值。
2. **配对控制实验设计**：HotFlip与rate-matched random substitution的对照范式可有效分离"对抗优化效应"与"编辑率效应"，可迁移到其他对抗攻击或数据增广方法的评估中。
3. **Copying head作为扰动恢复标志物**：Copying score与activation patching恢复的高度关联（Spearman r≈0.7）提示，头功能分析可辅助解释扰动传播机制，值得在团队的可解释性工作中探索其因果角色。
4. **本征维度作为扰动敏感性指标**：TwoNN估计的本征维度变化可捕捉CKA未能反映的表示结构偏移（如token substitution在GPT中引发更大维度跳跃），丰富了扰动分析的度量工具箱。
5. **跨架构族交叉验证的必要性**：GPT-2与Qwen2.5在同一扰动下行为层可能一致但内部层分化，提示任何鲁棒性结论都需要在多架构中交叉验证，避免指标或架构特异性的误判。

## 关键术语表
**Centered Kernel Alignment (CKA)**：衡量两组神经网络激活矩阵之间线性不变相似度的指标，通过对矩阵去中心化并计算Hilbert-Schmidt独立系数来评估表示对齐程度。
**Intrinsic Dimension（本征维度）**：通过TwoNN等最近邻估计算法测量的激活流形局部有效维度，反映表示空间的复杂度；扰动引起本征维度变化指示表示结构的偏移。
**Activation Patching**：一种机制可解释性干预方法，将扰动输入的某层/某head激活替换为clean输入对应激活，通过输出变化量评估该激活对目标行为的因果贡献。
**Copying Score**：基于注意力头OV电路（C=E·W_v·W_o·U）计算的函数指标，衡量头将输入token映射到输出 vocab top-k位置的倾向，表征copy/name-mover功能。
**HotFlip**：基于梯度的白盒文本对抗攻击方法，计算损失关于token embedding的梯度，在top-k候选词中贪心选择使序列NLL增幅最大的替换。
**Output Divergence**：clean与perturbed输入下生成序列之间的normalized Levenshtein距离，量化扰动对生成文本的破坏程度。
**Induction Head**： transformer中支持序列续写的注意力头，在重复前缀出现后关注前缀末尾以预测后续token，是in-context learning的关键电路组件。
**Centered Kernel Alignment (CKA) 离群修正**：本文在CKA计算前剔除clean激活中方差最大的5个维度，以降低CKA对高方差离群维度的敏感性。

## 可复现要素
- 数据集：WikiText-2 Raw（公开），论文使用固定seed采样300条长度≥128的序列。
- 模型权重：Hugging Face公开checkpoint（openai-community/gpt2、gpt2-medium、gpt2-large、gpt2-xl、Qwen/Qwen2.5-0.5B、Qwen/Qwen2.5-1.5B）。
- 代码：论文未明确提及代码仓库开源信息。
- 关键超参：扰动强度p∈{5%, 30%, 50%}；HotFlip候选短列表大小|S|=50（基于附录图5的ablation选择）；CKA剔除top-5高方差维度；两NN本征维度估计。
