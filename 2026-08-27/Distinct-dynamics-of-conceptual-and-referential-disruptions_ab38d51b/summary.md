---
title: "Distinct-dynamics-of-conceptual-and-referential-disruptions"
source: https://arxiv.org/pdf/2608.25999v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:46:42"
field: "计算心理语言学与话语指称加工"
keywords: ["conceptual distortion", "referential distortion", "self-paced reading", "contextual surprisal", "representational distance", "propagation profile", "sentence boundary", "language model"]
innovations: ["提出概念/指称干扰在人类阅读与LLM中的平行扰动对比范式", "揭示CD呈log-normal快速集中衰减而RD呈线性慢衰减且受句边界放大", "在LLM预测与表征两层面分离概念/指称扰动的下游动力学差异"]
benchmarks: ["Qwen3-4B-Base", "Llama-3.2-3B", "Mendeley Data (PB8MVz4tby)", "Aesop Fables改编语料"]
---

# 论文速读：Distinct-dynamics-of-conceptual-and-referential-disruptions

## 一句话总结
本文通过系统性扰动短篇叙事中的概念性信息或指称性关系，对比追踪了两种语义干扰在人类自我步速阅读和大语言模型（Qwen3-4B、Llama-3.2-3B）中的下游传播动态，发现概念性干扰产生强但局部集中的加工成本，而指称性干扰产生弱但逐渐分布且受句边界调节的效应，为概念与指称两种意义维度的可区分处理动力学提供了行为与计算层面的收敛证据。

## 研究问题与动机
1. **核心问题**：概念性（lexical-conceptual）与指称性（discourse referent）语义干扰在人类阅读和LLM处理中是否表现出不同的下游传播动力学（magnitude、persistence、structure sensitivity）？
2. **理论缺口**：以往心理语言学多将词汇语义异常与指称/语法异常分开研究，缺乏在同一扰动框架下直接比较两者传播轨迹的工作；尽管Soberats等（2025）发现词汇语义异常效应更局部、语法异常更弥散，但该对比未直接映射到concept–reference维度。
3. **结构调节机制不明**：指称关系依赖语法配置跨话语维持，其扰动成本可能不仅随线性距离衰减，还会受到句边界（grammatical cycle completion）的结构性调节；现有研究多将downstream effects视为预设的spillover区域，未刻画完整的propagation profile。
4. **LLM侧验证不足**：尽管LLM hidden states可预测神经响应，但概念 vs. 指称扰动在模型预测（surprisal）与表征（representation）两个层面是否呈现类似人类的行为分化，尚缺乏系统检验。

## 核心贡献（创新点）
1. **提出概念性/指称性干扰的平行扰动范式**：在相同叙事结构中分别替换概念不兼容名词（CD）与冲突代词/限定词（RD），保持整体叙事连贯性，使两种干扰在宏观语境可比的前提下产生差异化的局部压力。
2. **揭示人类阅读中两种干扰的差异化时间轨迹**：CD在阅读时间上呈log-normal峰形轨迹（K+1~K+2达峰后快速衰减），RD呈线性缓慢衰减轨迹，且RD效应在句边界处显著放大而CD不受边界调节。
3. **在LLM预测层面复现行为分化**：Contextual surprisal显示CD为log-Cauchy快速下降轨迹（K处峰最大），RD为指数衰减轨迹（初始较小但下游更持久），两者均在句边界呈现差异化调制（RD显著增强）。
4. **揭示LLM表征层面的不同动力学**：Output-layer cosine distance对两种干扰均呈幂律衰减，但RD引起更大的初始位移；句边界对两种干扰均降低表示距离，未出现RD特异性模式，表明预测与表征加工存在分离。
5. **提供跨架构与多指标的收敛证据**：以Qwen3-4B为主模型、Llama-3.2-3B独立复制，行为（RT）、预测（surprisal）、表征（distance）三维度共同支持concept–reference处理动力学的可区分性。

## 方法详解
1. **刺激材料**：改编21篇伊索寓言（原长度约60–120词，中位数95词），每篇生成三个版本：Original（基准）、CD（将语境适宜名词替换为语义不相容词，如rushing current→rushing mountain）、RD（替换代词/限定词使其与话语指称冲突，如he→she）。为避免CD落在句末产生wrap-up混淆，在必要处添加adjunct。
2. **刺激可比性控制**：使用Qwen3-4B-Base计算整篇perplexity，用wordfreq包获取Zipf词频均值，统计词长；通过Friedman检验与Siegel–Castellan两两比较+FDR校正验证三版本在宏观扰乱程度上可比。
3. **人类自我步速阅读（SPR）实验**：99名英语母语者（Prolific招募，年龄20–50岁）在线完成非累积Word-by-word SPR；每位被试读21故事各一版本（7 Original、7 CD、7 RD，随机顺序），每故事后跟4选1理解题（剔除<60%正确者）。
4. **阅读时间预处理**：剔除100–3000 ms外异常值并用参与者-故事中位数插补（0.32%）；log变换后，按Vasishth（2006）方法以混合效应模型对lag-1 RT残差化，消除连续按键的spill-over；最终以residualized log RT为因变量。
5. **局部效应分析**：在K-3至K+3及K_END位置，以条件（CD/RD vs. Original）为预测变量，控制词位置、lexical surprisal、词长，拟合含被试与项目随机截距的线性混合模型；FDR校正。
6. **下游轨迹分析**：以K为锚点，追踪K+1至K+10（或句末）的残差log RT；拟合含categorical word distance × condition、condition × sentence-boundary的混合模型；仅保留双方均有数据且≥20观测、≥2项目的距离点。
7. **函数形式拟合**：对每个干扰类型，将距离特异估计值（保留符号）与协方差矩阵一起，用广义最小二乘（GLS）拟合11种候选函数（constant/linear/quadratic/exponential/power-law/log-normal/log-Cauchy/Zipf-Alekseev/broken-stick/double-exponential/exponential-plus-power），以AICc选优；通过1000次参数Bootstrap构建95% CI。
8. **直接对比模型**：在单混合模型中比较CD与RD的距离×干扰类型交互，并对比早期窗口（K+1–K+3）与晚期窗口（K+8–K+10）的early-to-late变化差异。
9. **LLM分析**：在Qwen3-4B-Base（主）与Llama-3.2-3B（复制）上评估；**surprisal**：$\Delta S_t = S_t^{\mathrm{distorted}} - S_t^{\mathrm{Original}}$（子词token surprisal求和得词级）；**representational distance**：$D_t = 1 - \cos(h_t^{\mathrm{distorted}}, h_t^{\mathrm{Original}})$，取最终层final subword token的hidden state。
10. **模型侧轨迹与函数拟合**：轨迹从K延伸至K+10；使用与RT相同的11候选函数与AICc选择流程；统计推断采用cluster-robust SE（扰动故事-句子层级或轨迹聚类）。

## 实验与结果
1. **刺激可比性**：CD与RD均显著提高整篇perplexity（vs. Original），二者之间无显著差异；RD在词频上略异（功能词替换所致），词长无差异（Figure 1）。
2. **人类阅读局部效应**：K处无显著效应；CD从K+1起上升，K+2附近达峰后下降；RD效应较小且在各位置较均匀分布，延伸至句末（Figure 2A）。
3. **LLM局部效应**：Surprisal与representation均在K处即刻响应；CD在surprisal上峰值更大（Figure 2B），RD在representation上初始位移更大（Figure 2C）；人类行为与模型在 onset 上存在错位（行为延迟至K+1）。
4. **阅读时间轨迹**：距离×干扰类型交互显著（$\chi^2(9)=66.58, p<0.001$）；CD最佳拟合为log-normal（早峰晚衰），RD最佳拟合为linear（斜率-0.0038 log-RT/词）；RD在句边界显著放大（$\beta=0.030, p=0.005$，约3.06%额外成本），CD边界效应不显著（$p=0.191$）；early-to-late窗口差异显著（$\beta=0.022, p=0.042$）（Figure 3）。
5. **Surprisal轨迹**：距离×干扰类型交互显著（$F(10,122)=4.73, p<0.001$）；CD最佳拟合为log-Cauchy（K处大峰后陡降+小残留），RD为指数衰减；RD在句边界显著增强（$p=0.032$），CD不显著（$p=0.330$），交互显著（$p=0.019$）；early-to-late差异显著（$p=0.017$）（Figure 4）。
6. **Representational distance轨迹**：距离×干扰类型交互显著（$F(10,122)=7.70, p<0.001$）；两者均最佳拟合为power-law decay；RD初始位移显著大于CD，但早-晚窗口变化无差异（$p=0.443$）；句边界对两者均降低距离（CD $p<0.001$，RD $p=0.009$），边界×干扰交互不显著（$p=0.142$）（Figure 5）。
7. **Llama-3.2复制**：主要distance-dependent分化稳定复现；surprisal的CD集中/RD持久模式一致；representation的power-law衰减与初始RD更大位移得以保留；边界调节方向一致但交互未达显著，统计效力较弱（Figure S1–S3）。

## 相关工作脉络
1. **Soberats et al. (2025)**：发现词汇语义异常（类似CD）产生锐利局部效应，语法异常（部分涉及指称）产生更广分布效应；本文在此基础上将扰动直接锚定concept vs. reference维度，并以propagation profile（含句边界调节）进行定量刻画。
2. **Davis & van Schijndel (2020)**：表明LLM的指称偏好受话语结构调节，且在后续代词处理中对 intervening referential cues 保持敏感；本文扩展至概念/指称扰动在surprisal与representation双重指标下的传播差异。
3. **Skrill & Norman-Haignere (2023)**：揭示LLM表示从position-yoked指数窗口向structure-yoked幂律窗口的过渡；本文在其幂律衰减发现基础上，进一步区分概念/指称扰动的初始幅度与边界敏感性差异。
4. **Nieuwland et al. (2007, 2008); Venhuizen & Brouwer (2025)**：ERP研究区分概念匹配（N400相关）与指称解析（早期正波/P600相关）的加工时序；本文以行为阅读时间与模型预测/表征为指标，在自然叙事尺度上提供平行证据。
5. **Brothers et al. (2023); Chelvarajah et al. (2024)**：讨论多预测来源与皮层时间尺度的共享性；本文强调概念与指称扰动虽共享语言系统的向前推进机制，但在下游传播形态上呈现功能性分化。
6. **Mandelkern & Linzen (2024); Piantadosi & Hill (2022)**：争论LLM是否具有真正“指称”；本文结果表明指称扰动在模型内部状态留下独特签名（更大的初始表示位移），但不直接解决指称本体论问题。

## 局限性与未来方向
1. **扰动非过程纯净**：CD与RD在词类、局部显著性、语法特征、词频、扰动密度上存在系统性差异，部分轨迹分化可能混入形式属性效应；未来需更好地匹配local surprisal、词类、频率与扰动率。
2. **多扰动累积干扰**：每篇故事含多个干扰，后期关键位置的效应可能残留早期扰动影响（表现为K前非零表示距离）；需在单扰动材料上验证各效应的独立性。
3. **材料泛化受限**：仅使用21篇手动改编寓言，结果可能受体裁与手工插入策略限制；需在更多体裁、更长篇章与自动化扰动 pipeline 上检验。
4. **测量分辨率限制**：SPR无法区分在K处是否已检测到干扰但仅在K+1后才表达为行为成本；未来可结合eye-tracking或EEG精确定位检测与整合时相。
5. **表征指标单一**：仅使用最终层cosine distance，无法刻画中间层或不同距离度量下的模式；需跨模型架构、规模、层数与距离指标的系统比较。
6. **函数拟合的描述性**：11候选函数拟合约11个距离点，所选曲线族为描述性总结，未必对应特定认知机制；需理论驱动的约束模型与机制检验。

## 研究启发与可借鉴点
1. **Propagation profile量化范式**：以扰动点为锚、沿词汇距离拟合函数族并比较形状/边界敏感性的方法，可直接迁移至其他语义/语用扰动（如隐喻违背、预设失败、跨句连贯断裂）的比较研究。
2. **多指标并行验证策略**：同时考察行为RT、模型surprisal与模型representation，可在不同观测层级间交叉验证；团队可在后续工作中将其用于检验模型-人类对齐的局部-全局分离。
3. **句边界作为结构调节因子**：将sentence boundary纳入轨迹模型的交互项，能捕捉指称维持对语法周期的敏感性；该设计可扩展至段落边界、话题切换点等更高层级结构。
4. **双模型独立复制**：以Qwen3-4B为主、Llama-3.2-3B独立复制，可有效排除单一架构偏倚；团队在方法学中可沿用“主模型+不同规模/架构复制”的标准流程。
5. **曲线拟合的可重复 pipeline**：作者开源了包含GLS拟合、AICc选择、Bootstrap置信区间与11候选函数实现的脚本；团队可直接复用该fitting pipeline用于自身的时间序列/轨迹分析任务。

## 关键术语表
**Conceptual Distortion (CD)**：将语境中语义相容的名词替换为概念上不兼容的词，破坏局部词汇-概念延续性。
**Referential Distortion (RD)**：将代词或限定词替换为与话语指称冲突的形式，破坏表达式与既定 discourse referent 的语法锚定关系。
**Self-Paced Reading (SPR)**：被试逐词控制呈现速率的眼动/按键阅读范式，记录词级反应时为加工成本提供行为指标。
**Contextual Surprisal**：基于自回归模型下一词概率分布的信息论度量，$\Delta S_t$ 反映扰动对后续词汇预期的重塑程度。
**Representational Distance (Cosine Distance)**：扰动与原始序列在模型最终层hidden state上的余弦距离，刻画上下文表征的几何位移。
**Propagation Profile**：以扰动点为锚、沿下游词距离刻画效应大小随距离与结构边界变化的动态曲线。
**Sentence Boundary Modulation**：指扰动效应在句末位置受到的结构性放大或缩减，反映语法周期对指称维持的调节作用。
**Power-Law / Log-Normal / Log-Cauchy Decay**：用于拟合轨迹衰减形态的候选函数族，分别刻画幂律慢衰减、钟形峰形与长尾峰形模式。

## 可复现要素
- **行为数据**：去标识阅读时间数据公开于Mendeley Data（https://doi.org/10.17632/pb8mvz4tby.1）。
- **代码**：数据预处理与分析脚本公开于GitHub（https://github.com/RuiHe1999/SPR_distortion）。
- **刺激材料**：21篇改编自公共Aesop寓言集合（https://aesopfables.com/aesopsel.html），现代英语改写由GPT-5.3-mini完成；刺激模板与扭曲映射随代码提供。
- **模型**：主模型Qwen/Qwen3-4B-Base；复制模型meta-llama/Llama-3.2-3B；均不开参评估、单pass前向。
- **关键超参**：RT异常阈值100–3000 ms；log变换+lag-1残差化；曲线拟合11候选函数、AICc选择、1000次Bootstrap；统计推断采用FDR校正与cluster-robust SE。
- **样本量**：最终99名被试、1972有效participant-story trials（94.85%）；每位被试每故事一版本、7 Original/7 CD/7 RD平衡。
