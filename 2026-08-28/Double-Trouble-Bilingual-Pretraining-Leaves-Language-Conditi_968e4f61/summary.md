---
title: "Double-Trouble-Bilingual-Pretraining-Leaves-Language-Conditi"
source: https://arxiv.org/pdf/2608.26576v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:30:36"
---

# 论文速读：Double-Trouble-Bilingual-Pretraining-Leaves-Language-Conditi

## 一句话总结
本文揭示了多语言模型表征比较中的“对齐谬误”：仅对词嵌入空间进行线性对齐无法证明双语模型与单语模型在共享语言上的内部表征一致；通过严格控制英语暴露量、总计算量与文档重叠，证明双语预训练虽不改变输入层对齐程度，却会显著重塑用于 next-token prediction 的深层隐状态几何结构。

## 研究问题与动机
- **核心问题**：研究者常通过对齐词嵌入空间来 probing、做可解释性分析或评估跨语言迁移，并默认对齐后共享语言的表征是可比的，该假设对 Decoder-only 模型是否成立？
- **现有方法不足**：经典双语词嵌入对齐（如正交 Procrustes）仅在输入层拟合线性映射，若双语训练改变了 Transformer 中间层的上下文处理，嵌入对齐会掩盖隐状态的实质性差异，产生“对齐幻觉”。
- **既有研究盲区**：Prior work 多聚焦“天生多语”模型的共享结构或语言子空间分离，缺乏严格控制架构、数据域、超参后隔离“增加第二语言”本身对共享语言表征影响的实验。
- **研究动机**：建立一套可控的诊断协议，验证双语暴露是否会在对齐后的共享语言空间留下可检测的内部表征变化，避免后续跨模型比较研究误将嵌入对齐等同于上下文等价。

## 核心贡献（创新点）
- **提出并实证“对齐谬误”**：证明单靠嵌入对齐不能确立上下文隐状态的一致性，双语预训练会重塑共享语言的深层几何，而传统嵌入级诊断会遗漏该差异。
- **设计严格受控的配对预训练实验**：在 8 种类型学语言（ZH/FR/FAS/NLD/UKR/BUL/IND/DEU）上预训练 310M 配对模型，分别控制英语暴露、总优化步数与文档重叠，唯一变量为第二语言。
- **明确区分输入嵌入与预测隐状态**：发现对齐后 token embeddings 差异退化为随机种子水平，但用于预测的深层 hidden states 显著偏离种子基线，且失配在中层 Transformer 达峰。
- **强鲁棒性验证**：该隐状态失配在切换非重叠英语文档、改用仿射对齐（affine）、以及密集训练检查点（500 步起）下均保持显著，排除数据重叠、对齐族选择与晚期训练 artifact 的解释。

## 方法详解
- **模型与数据控制**：使用 310M 参数 Llama-style prenorm decoder（12 层，hidden 768，FFN 3072，12 attention heads，untied token/output matrices，vocab 128,256），基于 BabyBabelLM 语料。配对比较四种混淆控制设置：C1 同英语暴露+共享文档、C2 同暴露+ disjoint 文档、C3 同总计算+共享文档、C4 同计算+ disjoint 文档。
- **表示对齐**：抽取 3000 个高频共享英语锚点词，构建矩阵 $X^a, X^b \in \mathbb{R}^{V \times d}$，通过正交 Procrustes 拟合映射：$W^\star = \arg\min_{W^\top W = I} \|X_\mathcal{N}^a W - X_\mathcal{N}^b\|_F$。评估集合与对齐集合严格分离。
- **探针与语义轴**：1000 个 held-out probes 覆盖 10 个语义域（价值观/家庭/宗教/饮食/节日/服饰/符号颜色/治理/社会身份/日常习俗），由 50 个跨文化调查框架（Hofstede, Schwartz, Inglehart-Welzel, GLOBE）派生的对立词对定义语义轴；另设 100 个负控制词（身体部位等） bounding 基线差异。
- **核心度量**：主指标为轴投影失配 $D_{Axis,i} = \frac{1}{|\mathcal{C}|}\sum_{w \in \mathcal{C}} |\langle x_w^a, \hat{u}_i^a \rangle - \langle x_w^b, \hat{u}_i^b \rangle|$，对方向反转敏感。附加 $D_{NN}$（k=25 近邻失配）与 $D_{Struct}$（ pairwise cosine 失配）作补充。
- **零假设基线**：$\Delta m = m(M_{EN}, M_{EN+L2}) - \bar{m}_{seed-var}$，其中 $\bar{m}_{seed-var}$ 来自 6 对不同随机种子的纯 EN-only 模型配对。$\Delta m > 0$ 表示双语效应超出随机初始化波动。
- **量纲控制**：采用 Row-L2 归一化与 neutral-anchor z-scoring 排除向量范数差异导致的假阳性。

## 实验与结果
- **数据集**：BabyBabelLM，8 种 Tier-1 非英语语言，每对模型约 100M tokens 预算（1500/3000 步对应约 50M/100M tokens）。
- **评估基线**：随机种子变异基线、嵌入级 NN/结构/轴失配、隐状态级对应指标、非重叠文档条件、仿射对齐条件。
- **主要结果数字**：
  - 所有控制设置下，隐状态失配持续显著高于种子基线，嵌入失配始终贴近种子水平。C1/C2 中位数隐状态失配分别为 0.16 / 0.18；C3/C4 为 0.53 / 0.32。
  - 移除文档重叠（C1→C2）不消除失配，仅轻微扰动个别探针值，符号方向不一致，表明共享文档非主因。
  - 逐层分析显示：输入层失配小，随层数加深在中层（约第 4-7 层）达到峰值，末层略有回落但仍远高于嵌入基线；Row-L2 与 neutral-anchor z-score 均不消除该峰值。
  - 密集检查点（500 步起）显示失配在训练极早期即出现并持续至 3000 步，非后期训练 artifact。
  - 对齐方法敏感性：正交对齐与仿射对齐下，held-out contextual $D_{Axis}$ 均值分别为 0.386 与 0.580；直接以隐状态锚点对齐亦无法消除失配（Table 2）。
- **最强结果与提升**：EN-centered 100M 检查点下，EN+UKR  contextual $D_{Axis}$ 达 0.940，EN+FAS 达 0.726，显著高于种子基线（~0.014-0.035）；L2-centered 非英语对照同样复现隐状态>嵌入的规律，排除英语特异性。

## 相关工作脉络
- **双语词嵌入对齐（BLI）**：Mikolov et al. (2013), Smith et
