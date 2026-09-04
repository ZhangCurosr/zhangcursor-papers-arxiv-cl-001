---
title: "Difference-in-Differences-on-a-Censored-Rating-Scale-Can-Man"
source: https://arxiv.org/pdf/2608.27309v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:29:49"
field: "LLM评估与审计"
keywords: ["LLM-as-judge", "pre-registered audit", "difference-in-differences", "censored rating scale", "bounded ordinal scale", "manufactured effect", "identifiability", "educational evaluation"]
innovations: ["证明有界量表上双重差异端点不可识别且可退化为单极对比", "提出零差分校准构造作为可计算的截断诊断工具", "在预注册审计中从内部数据实证复现虚假交互效应（79-85%可截断解释）"]
benchmarks: ["Fan et al. 2026 pedagogy judge corpus", "acid-mixture word problems (6 items)", "5-point pedagogy rubric (4 sub-scores + overall)"]
---

# 论文速读：Difference-in-Differences-on-a-Censored-Rating-Scale-Can-Manufacture-an-Effect

## 一句话总结
本文从内部审计视角揭示：在有限制评分量表（bounded rating scale）上进行的"双重差异"（difference-in-differences）端点存在**不可识别性**——当两个候选响应距离量表的上下限不等时，共同的严重程度偏移会被截断效应转化为虚假的交互效应；在一项预注册审计中，一个名义显著的 +0.378 点交互效应中 79–85% 可由零差分校准的截断模型复现。

## 研究问题与动机
- **现有审计设计的共同缺陷**：LLM-Judge 审计普遍采用"匹配条件下对比"设计，更强设计进一步做双重差异（同一项目内两个候选响应的对比，再在不同操纵属性间差分），以抵消加法噪声。但这种设计隐含假设：有界量表上的截断只会衰减效应、不会制造效应——该假设在双重差异端点下**不成立**。
- **截断不对称性导致虚假交互**：双重差异的每一项都受自身截断程度影响；当两个响应距离量表的上下限不等（好材料尤其如此），共同的严重程度偏移会以不同比例被截断，从而产生非零交互项。
- **预注册保护失效的风险**：预注册绑定了分析计划，但无法保证注册的"解释性保证"（如"截断不会制造效应"）在数学上正确，需要可计算的识别性检验。
- **教育场景中的具体危害**： pedagogical rubric 通常为短尺度（如 1–5 分），高质量材料更容易使两极靠近量表中界，放大截断不对称性，导致更多已报告的交互效应实为"衰减差异"而非真实偏好。

## 核心贡献（创新点）
1. **双重差异端点在有限制量表上的不可识别性分析**：将有限因变量计量经济结果（limited-dependent-variable results）引入 LLM-Judge 审计设定，证明当两极距离边界不等时，端点退化为单极对比——这与已有审计工作直接处理交互效应不同。
2. **预注册审计的"反例级"证据**：对冻结 pedagogy judge（Claude Opus 4.8）进行 990 次评分的严格预注册审计，主端点为零（+0.085, p=0.684），但其次级分析提供一个活生生的虚假效应反例，直接证伪预注册中的解释性假设。
3. **可从审计自有评分中计算的可操作性识别性检验**：基于公式 $PAG = (\kappa_H - \kappa_L) \cdot \delta$，提出一种可通过零差分校准构造复现效应量的诊断方法，并报告每个子维度的衰减份额，这是已有审计工作中缺失的识别性检查步骤。
4. **揭示"截断不保守"机制**：证明天花板（ceiling）同样会掩蔽效应下降，双向非单调性使得去截断后端点移动方向不可预测，推翻预注册中"截断仅保守低估"的错误假设。

## 方法详解
- **实验设计**：55 个预注册刺激（来自 acid-mixture 酸混合物应用题），分为 weak（30）和 strong（25）两个 stratum；每个刺激配对高脚手架响应 $R_H$ 和低脚手架响应 $R_L$，在三个 arm 下评分：无 profile（D）、新手 profile（$P_{nov}$）、高级 profile（$P_{adv}$），每单元 3 次随机重复取均值，共 990 次 judge 调用。
- **核心端点定义**：脚手架偏好 $\Delta_s(a) = S_s(R_H|a) - S_s(R_L|a)$，主端点 Profile Anchoring Gap（弱 stratum）为：
  $$PAG_s = \Delta_s(P_{nov}) - \Delta_s(P_{adv}) = a_{L,s} - a_{H,s}$$
  其中 $a_{j,s} = S_s(R_j|P_{adv}) - S_s(R_j|P_{nov})$ 为每极偏移。
- **不可识别性推导**：设真实潜变量为 $Y_j^*$，观测评分 $S = c(Y^*) = \min(\max(Y^*, \ell), u)$；若共同偏移 $\delta$ 作用于两极且无差分校准，则观测端点 $PAG = (\kappa_H - \kappa_L)\cdot\delta$，其中 $\kappa_j = 1 - a_j/\delta$ 为截断衰减系数；当一极被固定于界时（$\kappa=1$），端点退化为另一极的纯偏移 $-a_H$，成为单极对比而非真正的双重差异。
- **识别性检验构造**：将观测到的高极偏移 $a_{H,s}$ 平移到低极，裁剪到 $[1,5]$ 区间后重新平均，复现的 $PAG$ 即为截断可制造的效应量；残差 $= PAG_{obs} - PAG_{reproduced}$ 为低极预测误差，提供效应是否被充分截断的诊断。
- **统计推断**：源 tutoring run 为聚类单位，主检验用精确 Wilcoxon 符号秩检验（two-sided），BCa bootstrap 区间（10,000 次重抽样），报告 rank-biserial 相关系数；无多重比较校正。

## 实验与结果
- **数据集**：55 个刺激（30 weak / 25 strong），23 个源 tutoring run（18 weak / 10 strong / 5 shared），6 个 acid-mixture 酸混合物应用题，使用 Fan et al. [2026] 发布的 pedagogy judge（Claude Opus 4.8 实例）的冻结版本。
- **主要结果数字**：
  - 主端点（weak stratum）：$PAG_{weak} = +0.085$，95% BCa $[-0.167, +0.353]$，exact signed-rank $p = 0.684$，power 在 +0.353 处仅 0.709，需效应量 >0.40 才能达到 80% power。
  - 高级 profile 对绝对分的影响：high-pole 偏移 $-0.238$（$p=0.0625$），low-pole 偏移 $-0.153$（$p=0.00781$），方向一致但幅度不同。
  - **唯一名义显著交互**：productive-struggle 维度 $PAG = +0.378$（$p=0.00195$），其中高极贡献 72–96%，$a_H = -0.395$，$a_L = -0.017$。
- **截断复现验证**：零差分校准构造复现 $PAG = +0.321$（85% 原始效应量）；若将偏移分数先 round 到整数再裁剪，复现 79%；残余 +0.057 为预测误差，在 5 个非零 cluster 上无法达到 $p<0.05$。
- **强 stratum 结果**：$PAG_{strong} = +0.106$，$[-0.194, +0.572]$，$p=0.914$，与弱 stratum 方向一致但不显著。
- **天花板效应**：17/30 weak 刺激的 productive-struggle 低极在三个 arm 下均被钉在 floor（评分=1），使该维度端点恒等退化为一极对比。

## 相关工作脉络
- **LLM Judge 审计设计**（Wang et al., 2024; Howell et al., 2025; Maltbie & Raval, 2026）：这些工作通过匹配条件对比来认证偏差，但均直接报告交互效应而未做识别性检查；本文指出其算术基础在有界量表上存在结构性缺陷。
- **有限因变量与截断计量**（Tobin, 1958; McKelvey & Zavoina, 1975; Puhani, 2012）：已知 latent interaction 与 observed cross-difference 不同，非线性 DiD 中交叉差不是处理效应；本文首次将此结果带入 LLM-Judge 审计语境。
- **有序量表度量的分析陷阱**（Liddell & Kruschke, 2018; Amidei et al., 2019; Rohrer & Arslan, 2021）：已有文献警告将 Likert 量表当作区间数据会导致虚假交互，但 NLP 领域尚未系统采纳；本文展示即使远离边界的度量也会出问题，且好材料反而使情况更糟。
- **社会人口学 prompt 敏感性**（Beck et al., 2024; Hu & Collier, 2024）： persona 变量通常解释较小方差；本文借用此发现校准预期效应应较小，聚焦于端点识别性问题而非效应大小。
- **Pre-registration in ML**（Van Miltenburg et al., 2021; Hofman et al., 2023）：本文作为警示案例表明：预注册绑定分析流程并不等于绑定分析假设的正确性，注册的保护是窄的。

## 局限性与未来方向
- **单一 judge 与单一 rubric**：仅审计 Claude Opus 4.8 实例及四条 pedagogy principle 量表，结果的外推性受限；不同 judge 模型的截断敏感性可能不同。
- **刺激材料内部差异**：两极响应不仅脚手架水平不同，还包含问号频率（37/55 vs 1/55）、boxed answers（18/55 vs 0）、文本长度等混淆特征，虽在加法假设下抵消但非加性交互未完全排除。
- **stratum 间系统性差异**：weak/strong 与 tutor model family（pedagogy-tuned vs conversational）高度混淆，且 single-message contexts 集中在 strong stratum，导致行为证据厚度不同。
- **无"无对话+有 profile"对照 arm**：此 arm 会打破 prompt 同构性，故未设置；无法分离 profile 对纯评分 vs 对评分+上下文交互的影响。
- **截断处理为诊断性而非纠正性**：本文证明效应不可识别但未提出去截断估计量；恢复 latent 端点需潜在变量模型或 ordinal response threshold 显式建模，依赖额外假设。

## 研究启发与可借鉴点
- **任何使用有界量表（Likert/ Rubric）的双重差异审计均须报告识别性诊断**：计算每极的衰减系数 $\kappa_j$ 和 pole-pinning 比例，作为结果可信度的必要补充信息。
- **零差分校准构造法（zero-differential-preference construction）可作为标准诊断工具**：将一极偏移平移至另一极并裁剪后复现端点，可量化截断贡献份额；建议未来审计将其作为主结果的附录分析。
- **预注册应区分"分析计划注册"与"解释性假设注册"**：本文表明后者可能因数学错误而失效，预注册本身不验证假设正确性。
- **可与本团队方向结合**：在 LLM-as-judge 评估 pipeline 中引入截断敏感性检验，或在设计新 rubric 时主动避免使用过短量表（<7点）以避免两极同时趋近边界；对已有的 audit 结果做 post-hoc 截断分解可作为有价值的复现性研究。
- **度量经济学框架可直接迁移**：Ai & Norton（2003）、Puhani（2012）、Athey & Imbens（2006）的工具箱（latent interaction vs observed cross-difference、monotone rescaling identifiability）可系统应用于各类 LLM evaluation benchmark 的交互效应报告规范。

## 关键术语表
- **Difference-in-Differences (DiD) on Bounded Scale**：在有限制评分量表上进行的两次差分（先同项目内双响应对比，再跨条件差分），本文证明此端点在有界量表上不可识别。
- **Profile Anchoring Gap (PAG)**：注册的主端点，衡量 stated learner profile 从 novice 到 advanced 变化时 judge 脚手架偏好的改变量，正值为高级 profile 抑制偏好。
- **Censoring / Truncation（截断）**：评分被 clip 到量表界值 $[\ell, u]$ 的现象；每项差的截断程度不同导致共同偏移被差分化产生虚假交互。
- **Attenuation Coefficient $\kappa_j$**：第 $j$ 极对共同偏移 $\delta$ 的观测衰减份额，$\kappa_j = 1 - a_j/\delta \in [0,1]$；一极被钉在界上时 $\kappa_j = 1$。
- **One-pole Contrast Degeneracy**：当一极在所有 arm 下均被钉在界上时，双重差异端点退化为单极偏移的负值（$PAG \equiv -a_H$），丧失双重差分的因果解释。
- **Zero-Differential-Preference Construction**：零差分校准构造，假设真实无偏好差异，仅将观测到高极偏移平移至低极并裁剪，用于量化截断可制造的效应量上限。
- **BCa Bootstrap Interval**：偏差校正加速 bootstrap 置信区间，本文用于主端点和纯 profile 效应的主推断，10,000 次重抽样。
- **Exact Signed-Rank Test（精确符号秩检验）**：基于 Wilcoxon 符号秩的精确版本，适用于小样本聚类数据，报告中报告完整枚举 p 值而非渐近近似。

## 可复现要素
- **数据集**：55 个刺激材料来自 Fan et al. [2026] 发布的 pedagogy judge corpus（arXiv: 2607.28128），公开可用；profile 文本（novice/advanced）和 judge system prompt 在附录 A 完整提供。
- **代码**：预注册分析代码 frozen 并 hashed，随 pre-registration 一起锁定；附录中有 specification status table 区分注册与探索性分析。
- **权重**：judge 为冻结的 Claude Opus 4.8 实例，alias 身份未公布具体 snapshot；test-retest 复现显示 Pearson $r = 0.979$，mean drift = +0.012。
- **关键超参**：3 次随机重复取均值；no temperature specified（provider default）；512-token output cap；每单元输入 token 数固定（D 为基础，$P_{nov}$ 多 122 tokens，$P_{adv}$ 多 121 tokens）。
- **显著性水平**：$\alpha = 0.05$，主检验为 two-sided；无 multiplicity correction 注册或声称。
