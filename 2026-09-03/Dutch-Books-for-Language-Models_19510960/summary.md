---
title: "Dutch-Books-for-Language-Models"
source: https://arxiv.org/pdf/2609.02797v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:59:08"
field: "语言模型评估与概率推理"
keywords: ["probabilistic coherence", "Dutch book", "language model evaluation", "arbitrage profit", "elicitation protocol", "stock return prediction"]
innovations: ["基于de Finetti定理的无标签全局相干性度量框架", "揭示elicitation协议与逻辑依赖对不完美性的决定性影响", "发现无关上下文可使不完美性增加一个数量级"]
benchmarks: ["CRSP stock returns", "Refinitiv news headlines", "15 LLMs across open/closed-weight families"]
---

# 论文速读：Dutch-Books-for-Language-Models

## 一句话总结
论文基于 de Finetti 定理，提出一种无标签的"荷兰赌注"套利利润度量方法，系统评估 15 个语言模型在股票收益率预测任务上的概率相干性；发现模型预测普遍存在显著不完美性，且逻辑关系复杂度与无关上下文信息会大幅增加不完美程度。

## 研究问题与动机
- 语言模型越来越多被用于支持涉及概率预测的人生决策（如自然灾害、经济结果），用户通常隐式信任其预测来自一个相干的世界模型，但现实中模型的预测常违反概率公理（如事件与其补事件的概率之和不为1）。
- 现有评估方法（校准、Brier 分数、proper scoring rules）均依赖真实结果标签，无法在结果尚未观测或无法观测的场景下评估相干性。
- 缺乏对语言模型概率预测的全局性、集合层面相干性评估工具，现有工作仅检查部分跨预测一致性属性。
- 预测市场场景中二元期权合约广泛存在，模型预测的可相干性具有实际应用价值。

## 核心贡献（创新点）
- **无标签相干性度量框架**：将 de Finetti 定理引入 LLM 评估，通过线性规划求解最大套利利润 $\ell_\infty(p)$ 作为全局相干性度量，无需任何 outcome label。与 Paleka et al. [2025] 的局部一致性检查相比，本文覆盖事件集合的完整逻辑蕴含。
- **大规模实证发现普遍不完美性**：在 100 个 stock-days、15 个模型、365,100 次 elicitation 上系统评估，发现不完美性普遍存在且差异巨大（最大套利利润从 0.002 到 0.199，跨度约 100 倍），远超随机舍入误差（上限 $5 \times 10^{-5}$）。
- **揭示逻辑依赖与上下文敏感性**：证明事件间逻辑关系越丰富（多资产、多日期联合事件）、不完美性越高；无关上下文信息（如情绪表达、数字锚定）可使不完美性增加一个数量级。
- **揭示 elicitation 协议的关键作用**：单次单事件查询产生高不完美性，而一次性查询完整事件面板可使不完美性接近零，为 post-training 策略提供新方向。

## 方法详解
- **理论基础**：de Finetti [1937, 1974] 定理指出，有限事件集合的概率预测相干当且仅当不存在荷兰赌注（即套利者无法在所有结果下保证正收益）。
- **事件代数构造**：给定 $n$ 个事件，其代数生成 $m$ 个原子（atoms）$\omega_1, \ldots, \omega_m$， incidence matrix $M \in \{0, 1\}^{n \times m}$ 记录原子到事件的映射。
- **套利利润公式**：
  $$
  \ell_\infty(p) = \max_{\|b\|_1 \leq 1} \min_{j=1,\ldots,m} \sum_{i=1}^n b_i(M_{ij} - p_i)
  $$
  其中 $b_i$ 为在事件 $E_i$ 上的投注额（负值表示做空），约束 $\|b\|_1 = \sum |b_i| \leq 1$ 限制总投注规模。
- **对偶形式**：$\ell_\infty(p) = \min_{\pi \in \Delta^{m-1}} \|M\pi - p\|_\infty$，即陈述概率向量 $p$ 到相干预测集合的 sup-norm 距离；$\ell_\infty(p) = 0$ 当且仅当预测相干。
- **实验设计**：预测方差归一化的股票收益率落在预设 bin（边缘为 -1, 0, 1）的概率；horizon $h=1$ 时查询 14 个非平凡事件，更长 horizon 加入跨日/跨资产的联合事件。
- **评估统计量**：每个 stock-day prompt 运行 5 次独立 elicitation，分别求解 LP，取平均 LP 值作为该 prompt 的套利利润度量。

## 实验与结果
- **数据集**：CRSP 每日股票数据 + Refinitiv 新闻头条，锚定日期 2025 年 8 月 15 日至 12 月 23 日（避开 GPT-OSS-120B 发布时间后数据以减少 lookahead bias），抽样 100 个 stock-days、98 只股票。
- **模型**：15 个开放/封闭权重模型，包括 GPT-OSS-120B（主力）、GPT-OSS-20B、Qwen3-235B-A22B、DeepSeek V4 Flash、Mistral Small 3.2、Claude 系列、Gemini 3.5 Flash、Llama-4 Maverick、GPT-5.6 系列等。
- **主要结果**：
  - 基线 elicitation（单次单事件）：GPT-OSS-120B 平均套利利润 0.002067/单位投注（95% CI [0.001471, 0.002737]），48/100 stock-days 利润 ≥ 0.1%，15/100 ≥ 0.5%，6/100 ≥ 1%。
  - 模型间差异：套利利润跨度约 100 倍（GPT-OSS-120B: 0.002067 vs Mistral Small 3.2: 0.1994）；Brier 分数范围较窄（0.1966–0.2157）；两者 Spearman 秩相关 0.9143。
  - 多资产查询：双股票联合事件使不完美性平均增加 0.00244（CI [0.001971, 0.002961]）。
  - 多日期查询：horizon 越长、联合事件越多，不完美性越高。
  - Elicitation 策略影响：
    - "statistics" arm（附加历史 bin 频率）：显著降低不完美性。
    - "instructions" arm（指定预测程序）：大幅增加不完美性。
    - "headline" arm（附加新闻头条）：大幅增加不完美性。
    - "full panel" arm（单次查询全部 14 个事件）：不完美性接近零（与舍入误差同量级）。
    - "irrelevant context" arm（无关信息：咖啡、情绪、数字锚定等）：大多数导致显著增加，最大增幅约一个数量级。
- **最强模型**：GPT-OSS-120B，平均套利利润最低（0.002067）。
- **最弱模型**：Mistral Small 3.2，平均套利利润最高（0.1994）。

## 相关工作脉络
- **Paleka et al. [2025]**：检查 LLM 预测的局部一致性属性（如事件与其补事件的概率和），证明不完美预测可被修复；本文定位差异在于使用全局套利利润度量，穷尽事件集合的所有概率相干蕴含。
- **Betz & Richardson [2023], Fluri et al. [2024]**：测量 LLM 概率一致性和逻辑一致性；本文扩展至多事件集合的全局相干性评估，并提供无标签度量。
- **Zhu & Griffiths [2024], Chadwick et al. [2025], Matta et al. [2026]**：发现并修复 LLM 的荷兰赌注脆弱性；本文提供系统性评估框架而非修复方法。
- **Kadavath et al. [2022], Lin et al. [2022], Tian et al. [2023]**：eliciting model confidence；本文关注多事件联合 elicitation 对相干性的影响。
- **Kassner et al. [2021], Mitchell et al. [2022]**：改进 LLM 断言间的逻辑一致性；本文强调 elicitation 协议设计对相干性的决定性作用。
- **Vafa et al. [2024, 2025], Li et al. [2023]**：世界模型评估；本文将相干性作为隐式世界模型质量的可计算指标。

## 局限性与未来方向
- **领域局限性**：仅在方差归一化股票收益率预测场景验证，结论推广至其他领域需谨慎。
- **模型泛化未验证**：elicitation 变体实验仅针对 GPT-OSS-120B，未证明不同模型在不完美性排序上的一致性。
- **elicitation 策略的工程可行性**：full-panel elicitation 虽能消除不完美性，但不符合实际用户交互模式，难以直接部署。
- **未来方向**：开发鼓励跨事件推理的 post-training 策略（如 process reward models、group-level RL）；将 Dutch Book LP 作为训练奖励信号降低方差。

## 研究启发与可借鉴点
- **无标签评估范式**：Dutch Book 套利利润可作为通用的相干性评估指标，适用于任何需要多事件联合概率预测的场景（如风险评估、医疗诊断），无需等待 ground truth。
- **elicitation 协议设计意识**：实验揭示无关上下文（情绪、锚定）可使不完美性增加一个数量级，提示在实际部署中需严格控制 prompt 设计，避免 spurious features 干扰概率判断。
- **full-panel vs. single-event 权衡**：单次查询完整事件面板可消除不完美性，但牺牲了交互灵活性；可探索分阶段 elicitation（先查询关键约束事件，再推导其他事件）的折中方案。
- **训练信号创新**：Dutch Book LP 提供精确的集合层面 reward，方差低于 learned reward model，可作为 RLHF 或 process reward 的训练目标。
- **相干性-准确性解耦分析**：仲裁利润与 Brier 分数的相关性（0.9143）提示相干性是准确性的必要条件但非充分条件，可分别优化。

## 关键术语表
- **Dutch Book（荷兰赌注）**：一组投注策略，使套利者在所有可能结果下均获得正收益，用于检验概率预测的不完美性。
- **Probabilistic Coherence（概率相干性）**：一组预测满足概率公理（可扩展为定义在事件代数上的概率测度）的性质。
- **Arbitrage Profit（套利利润）**：$\ell_\infty(p)$，度量陈述概率向量到相干预测集合的 sup-norm 距离，为零当且仅当相干。
- **Atom（原子）**：事件代数中最细粒度的结果，由事件集合的交集/补集运算生成，无法进一步划分。
- **Elicitation（ elicitation 协议）**：向模型查询概率预测的交互方式，包括单次单事件、分组查询、完整面板等变体。
- **Brier Score（布里尔分数）**：预测概率与二值结果之间均方误差的均值，衡量预测准确性。
- **Lookahead Bias（前瞻偏差）**：模型训练数据中包含评估期后的信息导致的评估偏误，本文通过选择模型发布后日期规避。
- **Sup-norm Distance（上确界范数距离）**：$\|M\pi - p\|_\infty$，衡量离散概率分布与陈述概率向量的最大偏差。

## 可复现要素
- **数据集**：CRSP 每日股票数据 + Refinitiv 新闻头条；锚定日期窗口 2025 年 8 月 15 日至 12 月 23 日；抽样策略：10,000 个 stock-days 中均匀抽取 50 个交易日、每日 2 只股票，共 98 只股票。论文未明确声明公开，但数据源为公开商业数据。
- **代码/权重**：GPT-OSS-120B 为开源权重模型；API 调用使用 Groq、DeepInfra、Amazon Bedrock、Google Vertex、Azure 等平台。论文未声明代码开源。
- **关键超参**：temperature=1.0（provider 支持时）；completion limit=8,192 tokens；输出格式为 JSON，概率值保留四位小数；5 次独立 pass 取平均 LP 值；95% bootstrap CI，20,000 次重采样，按日期聚类。
