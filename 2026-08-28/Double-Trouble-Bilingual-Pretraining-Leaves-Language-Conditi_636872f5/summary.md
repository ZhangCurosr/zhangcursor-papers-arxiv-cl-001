---
title: "Double-Trouble-Bilingual-Pretraining-Leaves-Language-Conditi"
source: https://arxiv.org/pdf/2608.26576v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:30:20"
---

# 论文速读：Double-Trouble-Bilingual-Pretraining-Leaves-Language-Conditioned-Effects-in-Shared-Language-Representations

## 一句话总结
本文证明了在控制所有混杂变量后，为单语模型额外加入第二语言进行双语预训练，会改变模型对共享语言（英语）的内部表征几何结构；这种"隐藏状态不匹配"在 token embedding 对齐后几乎消失，但在用于预测的深层隐藏状态中显著存在，且随 Transformer 层深在中层达到峰值——作者将仅依赖 embedding 对齐来推断模型等价性的做法称为"对齐谬误"（alignment fallacy）。

## 研究问题与动机
1. **核心问题**：在跨模型对比、可解释性分析和跨语言迁移研究中，研究者常假设经过 embedding 对齐后，两个模型对共享语言的理解是等价的——这个假设是否成立？
2. **现有方法的不足**：主流的双语词向量对齐方法（如 Orthogonal Procrustes）仅在输入层（token embedding）上拟合变换矩阵，无法检验 Transformer 深层用于 next-token prediction 的上下文隐藏状态是否真正一致。
3. **已有研究的空白**：既往工作（Pires et al., 2019; Artetxe et al., 2020; Libovický et al., 2020）研究的是"天生就多语言"的模型内部结构，从未有人**隔离**"给一个原本是单语的模型增加第二语言"这一操作本身的影响。
4. **混淆变量难以控制**：双语模型与单语模型的差异可能来自三个混杂因素——不同的英语训练步数、不同的总优化步数、以及共享的英语文档，现有工作很少同时控制这三者。

## 核心贡献（创新点）
1. **对齐谬误的发现**：首次系统地证明，仅靠 token embedding 线性对齐无法保证两个模型在预测层具有等价的共享语言表征，embedding 层面的"接近一致"会掩盖深层隐藏状态的实质差异。
2. **严格的对照实验设计**：设计了四种控制设置（C1–C4），分别独立控制英语暴露量、总训练步数、文档重叠度，并在八个语言对上验证——这是多语言表征研究中罕见的完全隔离变量的对照实验。
3. **语义轴探测框架的跨模型迁移**：将原本用于单模型内部方向性探测（axis probing）的技术扩展到跨模型配对比较，基于 Hofstede/Schwartz/Inglehart-Welzel/GLOBE 跨文化调查框架构造了 50 个语义轴，使表征差异可被语义可解释地量化。
4. **种子变异基线（seed-variation baseline）**：引入六组不同随机种子的单语模型作为零假设基线，将双语 vs 单语的 mismatch 与单语 vs 单语的随机初始化差异做减法，从而区分"真实语言效应"与"正常训练噪声"。

## 方法详解

**实验架构**：预训练 16 对 310M 参数 decoder-only 模型（12 层 Llama-style，RMSNorm，hidden size 768，128,256 token BPE tokenizer），在 BabyBabelLM 数据集上以 100M token 预算覆盖八个语言（ZH, FR, FAS, NLD, UKR, BUL, IND, DEU）。

**四种控制设置（Table 1）**：
- **C1（Equal-English, shared docs）**：EN 训练 1500 步，EN+L2 训练 3000 步（1500 EN + 1500 L2），共享相同英语文档——固定英语暴露量。
- **C2（Equal-English, disjoint docs）**：同上但使用不重叠的英语文档——额外去除文档重叠。
- **C3（Equal-compute, shared docs）**：EN 训练 3000 步，EN+L2 训练 3000 步（1500 EN + 1500 L2），共享英语文档——固定总计算量。
- **C4（Equal-compute, disjoint docs）**：C3 条件但不共享英语文档——同时固定两者。

**对齐与评估协议**：
- 用 3000 个高频英语词作为 anchor，用正交 Procrustes 拟合对齐映射：
$$W^\star = \arg\min_{W^\top W = I} \|X_\mathcal{N}^a W - X_\mathcal{N}^b\|_F$$
- 在 1000 个 held-out probe 词上评估，probe 均匀分布在 10 个语义域（价值观/家庭/宗教/食物/节日/服饰/符号/治理/社会身份/日常习俗）。
- 主指标 **轴投影不 disagreements（$D_{Axis}$）**：
$$D_{Axis,i} = \frac{1}{|\mathcal{C}|}\sum_{w \in \mathcal{C}} |\langle x_w^a, \hat{u}_i^a\rangle - \langle x_w^b, \hat{u}_i^b\rangle|$$
- 辅以近邻不匹配 $D_{NN}$（k=25）和结构不匹配 $D_{Struct}$（余弦相似度）。
- 最终报告经种子变异基线减除后的 $\Delta m = m(M_{EN}, M_{EN+L2}) - \bar{m}_{seed-var}$。

**额外鲁棒性检验**：
- 逐层提取 12 个 Transformer 层，每层独立重新拟合对齐。
- 使用仿射对齐（affine Procrustes）替代正交对齐。
- 直接在上下文 anchor 上拟合对齐映射（而非仅在 embedding 上）。
- 每隔 500 步做密集 checkpoint 评估（500–3000 步）。
- Row-L2 归一化和 neutral-anchor z-scoring 控制向量范数效应。

## 实验与结果

**核心数字**：

| 设置 | 中位数隐藏状态不匹配（$D_{Axis}$，已减种子基线） |
|---|---|
| C1（Equal-English, shared docs）| 0.16 |
| C2（Equal-English, disjoint docs）| 0.18 |
| C3（Equal-compute, shared docs）| 0.53 |
| C4（Equal-compute, disjoint docs）| 0.32 |

- **Embedding 层面**：所有设置下 embedding $D_{Axis}$ 均在种子变异基线附近（~0.014），即对齐后 token embedding 看起来几乎相同。
- **Contextual 层面**：所有设置下 contextual $D_{Axis}$ 均显著高于种子基线，即使在 C2（不共享英语文档）下中位数仍为 0.18。
- **逐层轨迹**：不匹配从输入层开始很小，在中层（第 3–7 层）逐渐增大并达到峰值，最终层有所回落但仍远高于 embedding 水平（EN-100M vs EN+BUL 峰值 2775.9，最终层 3.472）。
- **早期出现**：500 步时 contextual $D_{Axis}$ 已达 ~2.8–3.2，embedding $D_{Axis}$ 仅 ~0.07–0.08，说明效应很早就出现而非训练后期 artifacts。
- **对齐方法鲁棒性**：正交 vs 仿射对齐、embedding anchor vs contextual anchor 对齐，contextual $D_{Axis}$ 均在 0.386–0.580 范围内，未消除。
- **目标语言验证**：以非英语探针在目标语言空间评测（Table 17），contextual/embedding 放大比（ratio）仍高达 17–36 倍，排除英语特异性假说。
- **最强结果**：EN-100M vs EN+UKR 在 C3 设置下 contextual $D_{Axis}$ 达 0.940，是所有语言对中最大的不匹配。

## 相关工作脉络

1. **Mikolov et al. (2013); Smith et al. (2017); Artetxe et al. (2018)**：经典双语词典诱导工作，用正交映射对齐词向量空间——本文指出这类方法仅作用于输入层，不能推断深层表征一致性。
2. **Pires et al. (2019); Conneau et al. (2020)**：证明多语言模型存在共享跨语言结构——但研究对象是从一开始就多语言训练的模型，与本文"先单语再双语"的对照设计有本质区别。
3. **Libovický et al. (2020); Wendler et al. (2024)**：证明多语言模型内部不同语言占据不同子空间——本文补充指出，即使是共享语言，双语训练也会在深层改变其几何表征。
4. **Chi et al. (2020); Hartmann & Søgaard (2018); Vulić et al. (2020)**：探讨对齐方法的局限（非等距性、域偏移、频率不匹配）——本文将这些局限置于对照实验框架下量化。
5. **Bolukbasi et al. (2016); Garg et al. (2018); Arora et al. (2023)**：方向性探测（axis probing）在单模型内测偏差和语义漂移——本文将其扩展至跨模型配对比较，并结合跨文化 survey 框架。
6. **Dufter & Schütze (2020); Dodge et al. (2021)**：指出共享词汇/文档会夸大跨模型一致性——本文通过 C2/C4 的 disjoint-docs 设置直接控制了这一混淆。

## 局限性与未来方向

1. **模型规模受限**：仅在 310M 参数的单一 decoder-only 架构上验证，未能回答该效应在更大模型（如 7B+）中是否同样存在或减弱。
2. **机制未隔离**：论文承认"共享词汇叠加（superposition）"或"双语多义性"可能是原因，但实验设计无法唯一确定具体机制。
3. **50 个语义轴和翻译管道**的结果可能不完全代表所有可能的表征维度；非英语探针依赖 NLLB 翻译和 COMETKiwi 质检，翻译质量可能引入额外噪声（尽管作者做了分层分析排除此影响）。
4. **仅测量了几何表征差异**，尚未回答这些差异是否最终导致下游行为差异（如同一 prompt 下的不同输出分布）。
5. **混合语言文档比例极低**（仅 0.088% 的文档含目标语言+英语），但作者也承认这种稀疏混合可能间接贡献效应。

## 研究启发与可借鉴点

1. **对照实验设计范式可迁移**：四种 control setting（C1–C4）的"分离混杂变量→逐一排除"思路，可用于其他表示学习研究中隔离特定训练变量的影响（如数据去重、课程学习等）。
2. **种子变异基线的引入**：将 "difference between A and B" 与 "difference between two seeds of A" 做差，是一种低成本但有效的统计校准方法，值得推广到更多表征比较场景。
3. **方向性探测的跨模型迁移**：将 axis probing 从单模型内分析扩展到配对比较，结合跨文化 survey 框架（Hofstede/Schwartz 等）构造语义轴，是一种兼具理论根基和实证可操作性的探测方法。
4. **分层对齐评估协议**：在每一 Transformer 层独立重新拟合对齐而非传递单一旋转矩阵，是一种更严谨的逐层表征比较方法，可推广到 encoder-only 或多模型对比研究。
5. **与团队方向的结合机会**：若团队从事跨语言迁移或多语言微调研究，可借鉴本文协议验证"微调语言 X 是否会改变模型对语言 Y 的内部表征"，或在 LoRA/Adapter 等参数高效微调方法的表征稳定性评估中引入种子基线对照。

## 关键术语表

**Alignment Fallacy（对齐谬误）**：仅凭 token embedding 对齐后相近，就推断两个模型在预测层对共享语言有等价的内部表征——本文证明这是一个错误的结论。

**Seed-Variation Baseline（种子变异基线）**：用同架构不同随机种子的单语模型之间的平均不匹配作为零假设，用于区分双语效应的真实信号与正常训练随机性。

**Axis Projection Disagreement（$D_{Axis}$）**：衡量对齐后两个模型在同一语义轴上对同一词汇投影位置的绝对差值，是本文的主指标。

**Procrustes Alignment（普罗克修斯对齐）**：经典的线性对齐方法，寻找最优正交（或仿射）矩阵使两个词向量集合的 Frobenius 距离最小化。

**Contextual Hidden States（上下文隐藏状态）**：Transformer 各层在输出投影前产生的中间表示，承载了模型对词汇的上下文化理解，区别于静态的 token embedding。

**Semantic Axes（语义轴）**：由一对对立端点词定义的语义方向向量（如 individualism vs collectivism），用于将高维表征差异转化为可解释的标量度量。

**Mixed-Language Document（混合语言文档）**：在同一文档中同时包含两种或以上语言的文本片段；本文通过 fastText 审计发现此类文档占比极低（~0.088%）。

**BabyBabelLM**：本文使用的多语言预训练数据集，包含 8 种 Tier-1 非英语语言的发展适宜性文本（儿童-照护者对话、教育材料、儿童书籍等）。

## 可复现要素

- **数据集**：BabyBabelLM（Jumelet et al., 2026）——论文提供了详细的 split protocol 和数据统计（Table 9, A.3），但数据集本身需通过原论文获取。
- **代码开源情况**：论文正文及附录均未明确声明代码开源仓库；相关实现细节见 Appendix A.1。
- **模型权重**：论文未声明开源预训练权重。
- **关键超参**：310M 参数，12 层 pre-norm decoder，RMSNorm，hidden size 768，FFN size 3072，12 attention heads，vocab size 128,256；batch size 8，gradient accumulation 8，seq length 512，throughput 32,768 tokens/step；训练 1500 步（≈50M tokens）和 3000 步（≈100M tokens）；3000 个 anchor 词 + 1000 个 probe 词 + 50 个语义轴；Adam 优化器（论文未明确 lr/weight_decay，见附录 A.1）。
