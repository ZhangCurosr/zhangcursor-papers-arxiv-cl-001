---
title: "Terrain-signatures-in-Welsh-settlement-names"
source: https://arxiv.org/pdf/2608.26978v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 23:45:08"
field: "环境地名学与空间计算社会科学"
keywords: ["environmental toponymy", "Welsh place names", "spatial cross-validation", "preregistered analysis", "terrain polarity", "geographically held-out prediction"]
innovations: ["冻结词汇分类+预注册假设检验+地理留出预测的可复现框架", "高地/低地地形极性在 Welsh 定居点中保留局部地形预测信息（24.4 m, p=0.00137）", "10/25/50 km 空间阻塞交叉验证证明地形名称极性提供 4.63%–7.30% MSE 改善"]
benchmarks: ["OS Open Names 威尔士定居点库（N=3,757）", "OS Terrain 50 / Copernicus GLO-90 高程数据", "CORINE 2018 土地利用数据"]
---

# 论文速读：Terrain-signatures-in-Welsh-settlement-names

## 一句话总结
本研究通过预注册假设检验框架，验证了威尔士定居点名称中的高地/低地地形词素（bryn/mynydd vs cwm/pant）是否能携带可测量的当代局部地形信息；结果显示高地形名称定居点平均比低地形名称定居点高出24.4米（相对2公里邻域地形），且在地理留出的空间交叉验证中仍能减少4.63%–7.30%的预测误差。

## 研究问题与动机
- 地名作为文化景观中持久的空间信息载体，其形式与当代环境条件之间是否存在可测量的对应关系？现有文献多为探索性描述，缺乏预注册、可复现的确认性检验框架。
- 传统地名与环境关联研究常将现代表面拼写直接等同于词源解释或历史环境记录，但 Welsh 等地名具有分层语言史、双语轨迹和地域特异性，需明确区分"词汇暴露定义"与"个体词源验证"。
- 地理结构偏差：高地/低地词素可能在威尔士不同区域偏好分布，传统训练-测试随机拆分可能让近邻观测共享地理上下文，导致虚假关联；需地理结构化的验证策略。
- 先前地名-环境研究中，空间自相关残差常被忽略或处理为次要问题，而非测量并报告；需要明确的空间诊断与受限制随机化程序。

## 核心贡献（创新点）
- 提出"冻结词汇分类+预注册假设检验+地理留出预测"的可复现框架，将词汇暴露定义与确认性环境对比明确分离，区别于以往探索性地名学研究。
- 首次通过预注册的高地vs低地地形极性对比，证明选定 Welsh 地形词素在控制地理和定居点基线后仍保留局部地形位置的预测信息（24.4 m 差异，Holm-adjusted p = 0.00137）。
- 采用 10/25/50 km 三层空间阻塞交叉验证，证明地形名称极性在地理留出的 Welsh 定居点上仍能降低 MSE（4.63%/6.22%/7.30%），且该增益主要由预注册的地形极性驱动而非通用词汇信号。
- 引入受限制的空间随机化检验（1,999 permutations，按25-km空间块、定居点类型、语言状态、名称长度分层层），在保留局部地理结构的同时检验词素-地形对应是否超出零假设。
- 通过独立高程源（Copernicus GLO-90 vs OS Terrain 50）、邻域尺度敏感性（1/2/5 km）、重复名称权重控制等多重稳健性分析，收敛于同一地形区分结论，提升结果可信度。

## 方法详解
- **分析框架**：基于 Ordnance Survey Open Names 的 3,757 个威尔士定居点，使用预注册的 24 元素 Welsh 地名词素库（ETDE v1 + TETO v1），在确认性分析前冻结分类规则。
- **假设定义**：
  - H1（河流）：afon/nant vs 其他，结局为 log(1 + river distance)（距 OS Open Rivers 距离）。
  - H2（地形极性）：bryn/mynydd（高地形）vs cwm/pant（低地形），排除同时含两类词素的记录，结局为 settlement elevation − 2 km 邻域均值 elevation。
  - H3（林地）：coed/llwyn vs 其他，结局为 1 km 缓冲区内 CORINE 2018 林地覆盖比例；预注册 fractional-logit 模型不收敛，视为 non-estimable。
- **统计模型**：H1/H2 使用 Gaussian 线性模型，H3 使用 fractional-logit quasi-likelihood 模型；均调整定居点类型、名称语言状态、规范化名称字符数、token 数、非目标词汇检测、British National Grid 的 tensor-product cubic spline 空间项。
- **推断策略**：使用 normalized-name-group cluster-robust 标准误；H1–H3 构成确认性多重比较家族，采用 Holm 校正；H3 因模型不收敛赋 p = 1 仅用于保守多重校正，不代表推断结果。
- **空间诊断**：对模型残差计算 Moran's I（有向十近邻权重 + 999 permutations）评估残余空间自相关。
- **地理留出预测**：预注册后修订，使用 ridge-regularised spatial cross-validation（10/25/50 km 五折阻塞），M0（基线）含定居点协变量 + 惩罚空间样条；M2 = M0 + H2 地形极性指示变量；MSE 改善率 = 100 × (MSE_M0 − MSE_M2) / MSE_M0。ridge 正则化仅应用于空间样条系数。
- **稳健性分析**：邻域尺度（1/2/5 km）、独立高程源（Copernicus GLO-90）、单记录去重、等权重重复名称、坐标聚类去重、受限制空间置换检验。

## 实验与结果
- **数据集**：3,757 个威尔士定居点（OS Open Names），H2 分析子集 240 个（高地形 101 + 低地形 139）；H1 子集 3,757（暴露 49 vs 对照 3,708）；H3 子集 2,366（暴露 43 vs 对照 2,323）。
- **H2 主要结果**：高地形组比低地形组局部地形位置高 24.43 m（95% CI: 10.77–38.08 m；Holm-adjusted p = 0.00137）。
- **稳健性**：1 km 邻域 +19.23 m、5 km 邻域 +25.89 m、Copernicus GLO-90 +24.07 m、单记录 +21.25 m、等权重 +23.59 m；受限制空间置换 p = 0.002。
- **H1 次要结果**：afon/nant 定居点距河流更近，log(1+river distance) 乘性比率 0.649（95% CI: 0.459–0.917；Holm p = 0.0284），方向符合预期但证据较弱。
- **H3 结果**：预注册 fractional-logit 模型不收敛，non-estimable，不作正反证据解读。
- **地理留出预测**：10 km 阻塞 MSE 降低 4.63%（3/5 折改善）、25 km 降低 6.22%（3/5 折）、50 km 降低 7.30%（4/5 折）；增益由 H2 极性驱动，非目标词汇在 10 km 几乎无贡献、在更大尺度反而恶化预测。
- **最强结果**：H2 地形极性对比的 24.43 m 调整效应（p = 0.00137）及 50 km 地理留出的 7.30% MSE 改善。

## 相关工作脉络
- Fagúndez & Izco (2016) 将植物地名多样性与环境/社会因素关联；本文延续"地名承载环境信息"思路，但严格区分词汇暴露与词源验证，采用预注册确认性框架。
- Valkó et al. (2023) 发现地名多样性有助于植被自然性保护；本文进一步证明特定词素类别（地形极性）在控制地理基线后仍保留局地地形预测信息。
- Conedera et al. (2007) 用"brüsáda"地名重建历史土地利用，需结合档案/地形/环境证据；本文仅测试当代地形对应，不声称历史记忆或因果命名过程。
- Roberts et al. (2017)、Valavi et al. (2019) 强调时空结构数据的交叉验证策略；本文采用 10/25/50 km 空间阻塞 CV 与受限制置换检验，直接应对地名研究的地理结构偏差。
- Dormann et al. (2007)、Gaspard et al. (2019) 综述空间自相关处理方法；本文明确报告残余 Moran's I 诊断，承认未完全去混杂，避免因果解读。
- Owen (2015)、Parry (2023) 指出 Welsh 地名分层语言史与双语轨迹；本文使用源审计的 24 词素库并记录语言状态作为协变量，但不声称解决个体词素形态/语义消歧。

## 局限性与未来方向
- 残余空间自相关未被完全消除，调整后的关联与地理留出预测不应解释为因果效应或完整空间去混杂结果。
- 多数分析用定居点名称缺乏显式 Welsh 语言标签（"unresolved"），规则型词汇筛选器未经外部验证，非形态解析器或词源分类器。
- OS Terrain 50（DTM）与 Copernicus GLO-90（DSM）的类型差异意味着两者一致性仅支持来源稳健性，而非产品可互换性。
- 所有环境结局为当代代表，无法证明历史环境记忆；需 dated name attestations + independently reconstructed 历史环境数据。
- 地理留出验证仅在 Wales 内部进行，不构成外部验证；跨语言/跨地域泛化需独立参考数据与外部重复。
- H3 林地假设模型不收敛，暴露组过小（n = 43），未来需扩大样本或改进模型设定。

## 研究启发与可借鉴点
- **预注册+冻结词汇分类**的研究设计可直接迁移至其他语言的地名-环境关联研究，确保探索性与确认性分析边界清晰。
- **地理阻塞交叉验证**（10/25/50 km 多尺度）与**受限制空间随机化**的结合，为处理空间自相关的地名研究提供了可复用的评估范式。
- **词素极性对比**（高地 vs 低地、河流 vs 非河流）比泛化词汇分类更具解释力；本文的 lexical decomposition 分析表明非目标词汇甚至可能损害预测，提示未来研究应聚焦预定义的语义极性而非全量词素。
- **多源高程稳健性检验**（OS Terrain 50 vs Copernicus GLO-90）可作为地名-地形研究的标配 sensitivity analysis。
- 本团队可借鉴该框架研究中国地名中的地形/水文词素（如"山/岭"vs"谷/洼"、"河/江"vs 无水词素）与当代 DEM/河网数据的对应关系，并扩展至历史地名演变追踪。

## 关键术语表
**Toponymy（地名学）**：研究地名起源、意义、分布及其与语言、历史、地理关系的学科。
**Preregistered confirmatory analysis（预注册确认性分析）**：在收集/分析数据前公开假设与模型规格，区分探索性与确认性推断，减少 p-hacking 风险。
**Local topographic position（局部地形位置）**：定居点高程减去邻域（如 2 km 半径）栅格均值高程，反映点在局地地形中的相对高低而非绝对海拔。
**Spatial blocking cross-validation（空间阻塞交叉验证）**：按地理距离将样本划分为 k 个阻塞块，训练/测试集空间分离，评估模型在地理留出区域的泛化能力。
**Restricted spatial randomisation（受限制空间随机化）**：在保留局部地理结构（如空间块、协变量层）的约束下置换暴露标签，构建零分布检验观察效应是否异常。
**Cluster-robust standard errors（聚类稳健标准误）**：允许同一聚类（如规范化名称组）内观测相关性，提供保守的推断标准误。
**Fractional-logit model（分数对数模型）**：适用于 0–1 连续比例结局的 quasi-likelihood 模型，本文 H3 林地覆盖率建模使用，但未收敛。
**ETDE v1 / TETO v1**：论文冻结的 Welsh 地名词素暴露定义（24 元素库）与源审计注册表的版本标识，非机器学习模型。

## 可复现要素
- **数据集**：OS Open Names（3,757 定居点）、OS Open Rivers、OS Terrain 50、Copernicus DEM GLO-90、CORINE Land Cover 2018；原始第三方空间数据未重新分发。
- **公开状态**： publication-safe 源数据与数值结果已归档于 Zenodo（DOI: 10.5281/zenodo.22079109）；代码与可复现材料开源于 GitHub（https://github.com/oktaykarakus/templar-wales-reproducibility）；预注册方案归档于 Cardiff 研究数据库。
- **关键超参**：邻域半径 1/2/5 km；空间阻塞尺度 10/25/50 km、五折；ridge 正则化仅作用于空间样条系数；置换次数 1,999；Moran's I 使用十近邻权重与 999 permutations；Holm 校正 across H1–H3。
