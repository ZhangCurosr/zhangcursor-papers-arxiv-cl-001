---
title: "Language-Proficiency-Assessment-from-Eye-Movements-in-Natura"
source: https://arxiv.org/pdf/2608.30583v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:25:49"
field: "计算心理语言学 / 语言测评"
keywords: ["eye-tracking", "language proficiency assessment", "L2 reading", "EyeScore", "L1 bias debiasing", "reliability"]
innovations: ["提出并验证了基于外部测试残差消除 EyeScore 母语（L1）偏差的简单有效去偏方法", "证明基于眼动的熟练度评分（EyeScore）在段落阅读范式下具有比 TOEFL/IELTS/Michigan 等传统测试更高的心理测量信度"]
benchmarks: ["LexTALE", "Michigan Placement Test", "Composite Proficiency (MECO L2)", "IELTS", "TOEFL-iBT"]
---

# 论文速读：Language-Proficiency-Assessment-from-Eye-Movements-in-Natura

## 一句话总结
本文在自然段落阅读范式下验证并扩展了基于眼动的L2英语熟练度评估方法，发现并纠正了其受母语（L1）与英语距离影响的评分偏差，且证明该方法信度高于传统标准化测试。

## 研究问题与动机
- 传统语言熟练度测试（如TOEFL、IELTS）开发成本高、耗时久，且依赖离线的单项选择题或写作等静态信号，无法捕捉实时语言处理认知过程。
- 之前的眼动基于熟练度评估（Berzak et al., 2018）仅在单句阅读数据上验证，其泛化到自然段落阅读、不同熟练度指标、不同预测模型及信息搜寻阅读模式的有效性尚不明确。
- 实际应用部署前，需澄清两个关键开放问题：① EyeScore 是否因读者 L1 与英语的语言学距离不同而产生系统性评分偏差（威胁效度）；② 该方法的测量信度如何，是否满足实际测试要求。

## 核心贡献（创新点）
- **多范式复制与扩展验证**：在 1,265 名 L2 参与者的两个大规模段落阅读眼动语料库（MECO L2、OneStopL2）上，验证了 EyeScore 和外部测试分数预测方法，覆盖了不同熟练度指标（LexTALE、Michigan、Composite）、多种预测模型（Ridge、LightGBM、TabPFN-3）及常规理解阅读与信息搜寻两种阅读模式。与Berzak et al. (2018)的本质区别在于从单句扩展到自然段落，并系统检验了L1偏差与信度这两个现实部署关键属性。
- **L1偏差的发现与量化**：首次系统揭示 EyeScore 显著偏向 L1 与英语语言学距离更近的 L2 学习者（$\hat{\beta}_{dist}$ 为显著的负值），并通过残差回归量化了该偏差强度。与已有研究（如 Berzak et al., 2017; Reich et al., 2022 仅证明可从眼动解码L1）的本质区别在于，本文聚焦于这种L1接近性如何污染“熟练度”评分的效度，而非仅仅分类L1。
- **有效的去偏方法（EyeScore$^{db}$）**：提出基于外部参考测试残差的简单校正流程，从 EyeScore 中减去由 L1-英语距离预测的系统偏差分量，在 Seen L1 设置下几乎完全消除了偏差，且对交叉测试（LexTALE/Composite/Michigan）保持鲁棒。与简单剔除L1特征或使用更复杂对抗去偏方法（如 Adversarial Debiasing）的本质区别在于，本文采用可解释的、基于心理测量参考的残差减法，实现成本极低且有效。
- **高信度证明**：首次系统报告 EyeScore 的心理测量信度，发现其内部一致性（平均 Cronbach’s $\alpha$ = 0.98）和分半信度（平均 Pearson r = 0.93）均高于 Michigan Placement Test（$\alpha$=0.91）、IELTS（$\alpha$≈0.91）和 TOEFL-iBT（整体 0.90）。这是首次证明眼动熟练度测量在可靠性维度上不输于甚至在某些方面优于成熟标准测试。

## 方法详解
- **EyeScore 计算**：对每位参与者 $p$，提取眼动特征向量 $v_p \in \mathbb{R}^d$，在 L2 参与者集合 $D_{L2}$ 上进行逐特征 z-score 归一化得 $\tilde{v}_p$。计算 L1 原型向量 $\bar{v}_{L1} = \frac{1}{|D_{L1}|} \sum_{p \in D_{L1}} \tilde{v}_p$。EyeScore 为归一化后个体向量与 L1 原型的余弦相似度：$\text{EyeScore}_p = \frac{\tilde{v}_p \cdot \bar{v}_{L1}}{\|\tilde{v}_p\| \|\bar{v}_{L1}\|}$。
- **外部测试分数预测**：使用眼动特征向量通过 Ridge 回归（与 Berzak et al., 2018 保持一致以便对比）、LightGBM 和 TabPFN-3 直接预测 LexTALE、Michigan 或 Composite 分数。超参数方面，Ridge 正则化参数在 $10^{-3}$ 到 $10^4$ 的 10 个对数间隔值中通过训练集交叉验证选取；LightGBM 搜索最大叶子数 {15, 31, 63}、学习率 {0.03, 0.1} 和特征采样 fraction {0.3, 1.0}；TabPFN-3 直接使用预训练模型不进行调参。
- **眼动特征集**：① Average Fixation Metrics (Avg. Fix.)：FF、GD、TF、GP 的全词平均值，以及 Skip Rate (SR) 和 Regression Probability (RP)。② Syntactic Clusters (S-Clusters)：上述 6 项指标按 43 种词性标签（14 UPOS + 29 PTB）平均，共 258 维。③ Word Property Coefficients (WP-Coefs)：对每位参与者拟合 RT $\sim$ length + surprisal + frequency 的线性模型系数（surprisal 来自 GPT-2，词频来自 Wordfreq），共 24 维。④ Transitions：每对词之间各方向扫视次数（仅适用于 Fixed Text）。⑤ Word Fixation Metrics (Word Fix.)：每个词片的 GD 和 TF（高维，仅适用于 Fixed Text）。
- **L1 偏差度量与去偏**：首先拟合 $\text{EyeScore}_p \sim \text{Test}_p$ 回归得到残差 $\text{res}_{p,\text{Test}}$，再拟合 $\text{res}_{p,\text{Test}} \sim d(\text{L1}_p, \text{English})$ 得到偏差预测 $\hat{\text{res}}_{p,\text{Test}}$，其中 $d(\cdot)$ 是基于 URIEL+ 数据库中 103 项句法、28 项音系、42 项书写系统特征的角距离。去偏分数为 $\text{EyeScore}^{db}_p = \text{EyeScore}_p - \hat{\text{res}}_{p,\text{Test}}$。去偏模型通过 k 折交叉验证（k 为 L1 种类数）拟合，分为 Seen L1（训练/测试均包含所有 L1）和 New L1（测试集留出一种 L1）两种严格程度。

## 实验与结果
- **数据集**：MECO L2（1,106 L2 参与者来自 18 种 L1，95 L1 参与者，12 篇信息类文章，1,007 人有 LexTALE，1,007 人有 Composite）；OneStopL2（278 L2 参与者来自 10 种 L1，154 人常规阅读，124 人信息搜寻，全部有 LexTALE 和 Michigan 分数）；OneStop（360 L1 参与者，作为 OneStopL2 分析的 L1 参照）。
- **基线**：阅读速度（WPM）、平均训练集分数（Avg. Train）。
- **EyeScore 结果**：在所有设置下 EyeScore 系统性地显著优于 WPM 基线。在 MECO L2 上，WP-Coefs 和 Word Fix. 表现最佳，与 LexTALE 的 Pearson r 达 0.54***（Fixed 和 Any 文本 regime）。在 OneStopL2 上，Word Fix. 与 Michigan 的相关性达 0.55**。固定文本与任意文本 regime 下结果相近，表明特征集对文本内容具有鲁棒性。
- **外部分数预测结果**：直接预测模型在多数情况下优于 EyeScore 和 WPM。在 MECO L2 LexTALE Fixed Text 上，Word Fix. + Ridge 达到最高 $r=0.62^{***}$、MAE=7.59；LightGBM 进一步将 MAE 降至 7.45。TabPFN-3 在 MECO L2 上系统性优于 Ridge（LexTALE Fixed r=0.63, MAE=7.40）。OneStopL2 Fixed Text 结果偏低，归因于该设置下训练样本少（平均仅 24 L2 训练者）。
- **与标准测试间相关性的对比**：LexTALE 与 Composite 在 MECO L2 中相关为 0.55，LexTALE 与 Michigan 在 OneStopL2 中为 0.61；文献报道的主流标准测试间相关通常在 0.61–0.85 区间。EyeScore 及相关预测结果处于该范围低端，但仍具实质参考价值。
- **L1 偏差结果**：在 MECO L2 Any Text WP-Coefs 中，$\hat{\beta}_{dist} = -0.52$ ($r=-0.18, p<10^{-7}$)，表明确实存在显著偏差。去偏后（Seen L1），几乎所有特征集在 LexTALE 上的 $\hat{\beta}_{dist}$ 降至接近 0（如 WP-Coefs 从 -0.52*** 降至 -0.02），偏差几乎完全消除。即使评估指标换为 Composite 或 Michigan，去偏后偏差也大幅降低（如 WP-Coefs 对 Composite 从 -0.71*** 降至 -0.19*）。去偏对原始相关系数的影响极小（平均 $\Delta r_{db}$ 约为 ±0.01），EyeScore 与 EyeScore$^{db}$ 的平均相关性高达 0.99。
- **信度结果**：EyeScore 在 Any Text  regime 下的平均 Cronbach’s $\alpha$ 为 0.98（WPM 0.98, Avg.Fix. 0.98, S-Clusters 0.98, WP-Coefs 0.91），平均分半信度 r 为 0.93。在 Fixed Text regime 和信息搜寻 regime 下结果类似（$\alpha$ 0.98–1.00，分半 r 0.91–0.99）。相比之下，Michigan Placement Test 在 OneStopL2 中的整体 $\alpha$=0.91、分半 r=0.92；IELTS 阅读/听力平均 $\alpha$=0.91；TOEFL-iBT 整体可靠性 0.90。EyeScore 信度显著高于这些标准测试。

## 相关工作脉络
- **Berzak et al. (2018)**：开创性提出从单句阅读眼动预测 L2 熟练度的 EyeScore 和回归预测框架，本文的核心基础工作，主要区别在于本文扩展到段落阅读、新语料库、新指标、并研究偏差与信度。
- **WPM (Reading Speed) 基线**：作为最简单且无需眼动设备的对照，本文系统证明眼动特征在控制阅读速度后仍提供显著增量信息。
- **LexTALE / Michigan Placement Test / TOEFL / IELTS**：作为外部效度参考标准和信度对比基准。本文表明眼动方法虽在效度相关上处于标准测试相关范围的偏低端，但在信度上已超越或部分媲美这些成熟测试。
- **Berzak et al. (2017); Reich et al. (2022); Skerath et al. (2023)**：证明可从眼动模式解码 L1，本文在此基础上进一步探究 L1 语言距离如何系统性地“污染”熟练度评分（即 L1 bias），而非仅仅分类 L1。
- **Shubi et al. (2024, 2025b)**：近期关于从眼动预测阅读理解及 EyeBench 基准的工作，本文聚焦于“外部熟练度”而非文本内部理解正确率，并在信度和偏差层面推进了该领域的实证基础。

## 局限性与未来方向
- 眼动方法仅能评估阅读理解相关语言能力，口语和书面表达等产出能力目前不在覆盖范围内。
- L1 去偏方法依赖外部参考测试的准确性假设（假设标准测试本身无 L1 偏差），若该假设不成立则会低估需去除的偏差量；同时去偏流程假设参与者只有单一 L1，难以直接应用于双语/多语者。
- 现有数据局限于英语目标语、两类文本域（信息类和新西兰新闻）、特定母语背景及成人年龄范围，需扩展至更多 L2 目标语、儿童群体及其他文本类型。
- 数据均采集自实验室高标准眼动仪（EyeLink 1000/Plus，1000Hz），在实际部署中需验证在低成本眼动设备和非实验室环境（如在线测试）下的性能稳定性。
- 尚未系统研究测试防作弊性（如参与者是否可通过 skim/top-down 策略人为改变眼动模式以获得更“像 L1”的眼动轨迹），以及跨多次测试会话的测试-重测信度。

## 研究启发与可借鉴点
- **偏差诊断与校正的通用范式**：通过拟合 `Score ~ ExternalReference` 获得残差，再检验残差与敏感属性（如 L1 距离）的关系来量化偏差，并直接从原始分数中减去预测偏差量，这一流程简单、可解释且易于迁移到其他可能存在群体偏差的预测性测量任务。
- **双层级泛化评估（Seen L1 / New L1）**：在去偏方法评估中区分“训练/测试均包含目标 L1”与“测试包含全新 L1”，为模型在未见群体上的公平性提供了更严格的评估视角，值得在其他公平性研究中借鉴。
- **固定文本 vs. 任意文本范式的区分**：明确界定 Fixed Text（测试文本有其他参与者眼动数据）和 Any Text（无测试文本眼动数据）两种部署场景，并分别报告性能，这对评估方法的实际应用价值至关重要，其他眼动研究可借鉴此划分。
- **信度优先报告**：将 Cronbach’s $\alpha$ 和分半信度作为核心报告指标，并与主流标准测试直接对比，为眼动测量方法的心理计量学质量提供了强有力的论证框架。
- **轻量级模型与复杂模型的对比**：同时评测 Ridge（简单可解释）、LightGBM（树模型）和 TabPFN-3（表格专用神经网络），发现简单模型在样本有限时表现稳健，而 TabPFN-3 在数据充足时提供更优性能，为后续研究在选择模型复杂度时提供了实证依据。

## 关键术语表
**EyeScore**：一种基于眼动的独立熟练度评分，通过计算 L2 参与者归一化眼动特征向量与 L1 原型向量的余弦相似度得出。
**L1 Bias（母语偏差）**：EyeScore 系统性偏向 L1 与英语语言学距离较近的 L2 学习者的现象，会损害测试的跨群体效度。
**EyeScore$^{db}$（去偏 EyeScore）**：从原始 EyeScore 中减去由 L1-英语距离预测的系统偏差量后得到的校正分数。
**Fixed Text Regime**：测试参与者所读文本的其他参与者眼动数据可用于构建模型的特征设置。
**Any Text Regime**：测试参与者所读文本无任何其他参与者眼动数据，模型只能依赖其他文本上的眼动特征进行泛化的更严格设置。
**WP-Coefs（Word Property Coefficients）**：一组 24 维特征，编码个体眼动指标对词汇长度、词频和 surprisal 的敏感度回归系数。
**Cronbach’s $\alpha$**：衡量测验内部一致性的经典心理测量指标，取值 0–1，越接近 1 表示各题目/项目间一致性越高。
**Split-Half Reliability（分半信度）**：将测验项目随机分成两半，计算两部分得分间的相关系数，作为测验稳定性的代理指标。

## 可复现要素
- **数据集**：MECO L2 (v2.1)、OneStopL2 (v0.1)、OneStop (v1.0.3) 均为公开可用语料库（论文未提供具体下载链接，需参考原发表期刊/会议资源）。
- **代码/权重**：论文未明确声明代码或模型权重的开源情况。
- **关键超参**：Ridge 正则化在 $10^{-3}$ 至 $10^4$ 的 10 个对数间距值中 CV 选择；LightGBM 最大叶子 {15,31,63}、学习率 {0.03,0.1}、特征采样 {0.3,1.0}；TabPFN-3 使用预训练权重不做调参。surprisal 来自 GPT-2，词频来自 Wordfreq，使用 psycholing-metrics v1.1.9 提取。
