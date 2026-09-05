---
title: "Language-Proficiency-Assessment-from-Eye-Movements-in-Natura"
source: https://arxiv.org/pdf/2608.30583v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:42:57"
field: "计算语言学与眼动分析"
keywords: ["eye-tracking", "language proficiency assessment", "L2 reading", "EyeScore", "L1 bias", "psycholinguistics", "naturalistic reading"]
innovations: ["首次系统性复现并扩展眼动语言评估至段落阅读和多数据集", "提出基于L1语言距离的两阶段残差去偏方法EyeScore^db", "首次报告眼动能力分数的心理测量信度并证实优于传统测试"]
benchmarks: ["MECO L2", "OneStopL2", "LexTALE", "Michigan Placement Test", "Composite Proficiency"]
---

# 论文速读：Language-Proficiency-Assessment-from-Eye-Movements-in-Natura

## 一句话总结
本文验证并扩展了基于眼动轨迹的L2英语语言能力评估方法，在真实语境段落阅读中复现了EyeScore框架，发现其存在母语（L1）语言距离偏差但可通过校正消除，且信度高于传统标准化测试。

## 研究问题与动机
- **传统测试的瓶颈**：标准化语言测试（如TOEFL、IELTS）开发成本高、使用频率低、材料不公开，且仅能捕捉离线答题结果，无法追踪实时认知加工过程。
- **眼动方法的潜力与空白**：Berzak et al. (2018) 提出从单句阅读的眼动预测L2水平，但未在更自然的段落阅读、不同 proficiency 指标和阅读模式下验证其泛化性。
- **两个开放问题**：(1) EyeScore 是否因读者L1与英语的语言学接近度而产生评分偏差？(2) 眼动分数是否具有足够的心理测量信度？
- **应用场景扩展**：验证在"信息搜寻"（information seeking）阅读模式下的有效性，以及 Fixed Text vs. Any Text 两种数据可用场景。

## 核心贡献（创新点）
1. **首个系统性复现与扩展**：在MECO L2（1,106名L2参与者）和OneStopL2（278名L2参与者）两个大规模数据集上验证眼动能力评估方法，覆盖词汇/语法/阅读多种外部基准。
   *→ 与Berzak et al. (2018)的单句CELER数据本质区别：扩展到真实语境段落阅读和更丰富的能力指标。*

2. **首次量化并校正L1偏差**：发现EyeScore系统性偏袒L1与英语语言学距离近的读者（β_dist < 0, p < 10⁻⁷），提出基于外部测试残差的去偏方法EyeScore^db。
   *→ 填补了眼动语言评估公平性研究的空白，提供了可操作的偏差校正流程。*

3. **首次报告眼动分数的心理测量信度**：计算Cronbach's α和split-half reliability，发现EyeScore的信度（平均α=0.98）高于Michigan Placement Test（α=0.91）、IELTS（α=0.91）和TOEFL-iBT（α=0.90）。
   *→ 回答了眼动测试实用性的关键质疑，为其实际应用提供证据支撑。*

4. **系统评估多特征集与多模型**：比较6种眼动特征集（Avg. Fix., S-Clusters, WP-Coefs, Transitions, Word Fix., WPM）与3种预测模型（Ridge, LightGBM, TabPFN-3），发现WP-Coefs和Word Fix.表现最佳。
   *→ 为后续研究提供了特征选择和方法对比的全面基准。*

## 方法详解

### EyeScore（独立眼动评分）
- **核心思想**：将L2读者的眼动特征向量与L1读者平均原型向量的余弦相似度作为 proficiency 分数。
- **公式**：$\mathrm{EyeScore}_p = \frac{\tilde{v}_p \cdot \bar{v}_{L1}}{\|\tilde{v}_p\| \|\bar{v}_{L1}\|}$，其中 $\tilde{v}_p$ 是在L2群体上z-score化的特征向量。
- **两阶段流程**：先在L2训练集上拟合z-scaler，再计算L1原型向量，最后对测试L2参与者计算相似度。

### 外部测试分数预测
- 使用Ridge回归、LightGBM和TabPFN-3直接从眼动特征预测LexTALE、Michigan Placement Test和Composite Proficiency分数。
- 比较Fixed Text（测试文本的眼动数据来自其他参与者）与Any Text（测试文本无眼动数据）两种评估设置。

### L1偏差校正（EyeScore^db）
1. 拟合残差模型：$\mathrm{res}_{p, \mathrm{Test}} = \mathrm{EyeScore}_p - (\hat{\beta}_0 + \hat{\beta}_{\mathrm{Test}} \cdot \mathrm{Test}_p)$
2. 拟合L1距离效应：$\widehat{\mathrm{res}}_{p} = \hat{\beta}_0 + \hat{\beta}_{dist} \cdot d(\mathrm{L1}_p, \mathrm{English})$
3. 去偏分数：$\mathrm{EyeScore}_p^{\mathrm{db}} = \mathrm{EyeScore}_p - \widehat{\mathrm{res}}_{p}$
- **L1距离度量**：基于URIEL+的173个语言特征（103句法+28语音+42书写系统），使用角距离计算。

### 眼动特征表示
- **Avg. Fix.**：6项平均注视指标（FF, GD, TF, GP, SR, RP）
- **S-Clusters**：按43个POS标签分组平均的6项指标，共258维
- **WP-Coefs**：词长、词频、surprisal对注视时长的线性回归系数，共24维
- **Transitions**：词对间的眼动转移次数（高维，适用于固定文本）
- **Word Fix.**：逐词注视指标（GD, TF），高维稀疏

## 实验与结果

### 数据集
- **MECO L2**：1,106名L2参与者（18种L1）、95名L1英语读者，12篇信息文本，1007人有LexTALE和Composite分数。
- **OneStopL2**：278名L2参与者（10种L1），30篇新闻报道，154名普通阅读+124名信息搜寻模式，全部有LexTALE和Michigan分数。
- **OneStop**：360名L1英语读者，作为L1参考群体。

### 主要结果
**EyeScore相关性（Pearson r）**：
- 最佳：MECO L2 Any Text WP-Coefs vs. LexTALE r=0.54***；OneStopL2 Any Text WP-Coefs vs. Michigan r=0.52
- WPM基线r≈0.48，眼动特征稳定超越阅读速度
- Any Text与Fixed Text表现相近，表明方法对文本类型鲁棒

**外部测试预测（Ridge回归）**：
- MECO L2 Fixed Word Fix. vs. LexTALE：r=0.62, MAE=7.59（最强）
- OneStopL2 Any S-Clusters vs. LexTALE：r=0.57, MAE=8.67
- TabPFN-3在MECO L2上优于Ridge，但在小样本OneStopL2 Fixed Text上较差

**L1偏差**：
- MECO L2 WP-Coefs vs. LexTALE：β_dist = -0.52, r = -0.18, p < 10⁻⁷
- 去偏后β_dist降至-0.02（Seen L1），偏差几乎消除

**信度分析**：
- EyeScore平均Cronbach's α = 0.98，split-half r = 0.93
- 高于Michigan（α=0.91）、IELTS阅读（α=0.92）、TOEFL-iBT（α=0.90）

## 相关工作脉络
1. **Berzak et al. (2018)**：首次提出从眼动预测L2语言能力的框架（EyeScore和外部测试预测），本文在其基础上扩展到段落阅读和更多评估维度。
2. **Cop et al. (2015a), Berzak & Levy (2023)**：证明L1/L2读者在标准眼动指标上存在系统性差异，为本方法的心理学基础提供支持。
3. **Shubi et al. (2025b) EyeBench**：最近的基于眼动的读者特征预测benchmark，本文与其共享方法论但聚焦于语言能力的标准化评估。
4. **Reich et al. (2022), Skerath et al. (2023)**：从眼动解码L1的研究，本文扩展至L1语言距离对能力评分的潜在偏差。
5. **Ahm et al. (2020), Mézière et al. (2023)**：用眼动预测阅读理解，本文区分了"阅读理解表现"与"外部标准化语言能力"的差异。
6. **Standardized tests (IELTS, TOEFL, Michigan)**：作为外部效标，本文通过与之相关性和信度对比建立眼动方法的实用价值。

## 局限性与未来方向
- **能力覆盖有限**：眼动方法只能评估阅读理解能力，无法评估口语和书面表达等产出性技能。
- **L1偏差校正的假设**：依赖外部测试本身无L1偏差的假设，若测试也有偏差则会低估需校正量；且仅适用于单L1读者，不适用于双语者。
- **数据局限性**：当前数据集仅限英语、两种文本领域（信息类/新闻），L1背景有限，缺乏儿童和更多年龄层数据。
- **生态效度**：数据来自实验室高精度眼动仪（EyeLink 1000），需验证在低质量设备和真实场景下的性能。
- **反作弊与稳定性**：尚未研究读者是否能通过顶部策略（如skimming）作弊，以及跨测试会话的信度。
- **未来方向**：扩展至其他目标语言、更多L1背景、不同文本类型，以及开发在线/移动端部署方案。

## 研究启发与可借鉴点
1. **L1偏差校正框架可迁移**：两阶段残差回归（Test→L1距离）的去偏思路可应用于其他基于行为特征的公平性评估场景。
2. **Any Text regime的实用性**：证明在无测试文本眼动先验的情况下仍能可靠评估，降低了部署门槛。
3. **WP-Coefs特征的稳定性**：基于语言学词属性（surprisal/frequency/length）系数的特征集在多个数据集和设置下 consistently 表现最优，值得后续研究借鉴。
4. **信度评估的标准**：采用Cronbach's α和split-half reliability双指标评估眼动分数，为心理测量学严谨性树立标杆。
5. **多模型对比的价值**：同时报告Ridge、LightGBM、TabPFN-3，揭示复杂模型在小样本下的局限，为方法选择提供指导。

## 关键术语表
- **EyeScore**：基于L2读者眼动特征与L1原型余弦相似度的独立语言能力评分。
- **Fixed Text regime**：测试文本的眼动数据来自其他参与者的评估设置。
- **Any Text regime**：测试文本无眼动先验、仅用其他文本数据训练的更通用设置。
- **WP-Coefs**：词属性系数特征集，反映个体对不同语言学刺激（词长、频率、surprisal）的注视响应敏感度。
- **L1 distance**：基于URIEL+语言特征库的母语与英语之间的角距离度量。
- **Information seeking regime**：先呈现问题再阅读的主动信息搜寻模式。
- **Cronbach's α**：衡量测验内部一致性的信度系数，取值0-1，越高表示项目间一致性越强。
- **Split-half reliability**：将测验随机分半后计算两半得分的相关系数，估计重测信度。

## 可复现要素
- **数据集**：MECO L2 v2.1、OneStopL2 v0.1、OneStop v1.0.3，均为公开数据集（IRB批准）。
- **代码**：论文未明确提供开源代码仓库链接，但提及使用psycholing-metrics v1.1.9。
- **关键超参**：Ridge正则化在log-space [10⁻³, 10⁴]内10值网格搜索；LightGBM最大叶节点{15,31,63}、学习率{0.03,0.1}、特征采样{0.3,1.0}；TabPFN-3使用预训练模型无调参。
- **复现难点**：需获取三个眼动数据集授权，L1距离计算需URIEL+特征库，surprisal需GPT-2模型。
