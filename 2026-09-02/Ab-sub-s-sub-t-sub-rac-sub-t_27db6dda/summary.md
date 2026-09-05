---
title: "Ab-sub-s-sub-t-sub-rac-sub-t"
source: https://arxiv.org/pdf/2609.01274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:53:05"
field: "大语言模型推理与强化学习后训练"
keywords: ["RLVR", "推理时搜索", "统一解码框架", "BOPTR", "行为可恢复性", "SimpleRL-Zoo", "推理预算"]
innovations: ["提出UDF统一框架将异构解码方法建模为预算化操作点并分离评估指标", "发现BOPTR低维转移规则N_Base≈α·N_RL^β且指数由任务regime决定", "证明RL后行为预测所需RL参考数据可压缩至单锚点甚至零RL先验"]
benchmarks: ["Math500", "AIME 2024/2025", "GPQA-Diamond", "IFEval", "AMC23", "Minerva", "OlympiadBench", "TinyMMLU"]
---

# 论文速读：From Base Rollouts to RL Reasoning: A Budgeted Search Perspective

## 一句话总结
本文从"预算化搜索"的行为视角出发，发现 SimpleRL-Zoo 训练得到的 RL 模型在四个基准上的默认策略表现曲线，往往可以由基座模型的推理时解码/搜索（UDF）在一条低维规则路径（BOPTR）上复现，从而将 RL 增益解释为"采样效率的提升"而非全新推理能力的涌现。

## 研究问题与动机
1. **RL 增益的来源不明**：RLVR（可验证奖励强化学习）能显著提升开源基座模型的推理能力，但近年研究表明基座模型在更大采样预算下可接近 RL 水平，RL 是否创造了基座不具备的新行为，还是仅提升了已有行为的采样效率，仍不清楚。
2. **同一策略比较的局限性**：若仅在相同解码策略和预算下比较 Base/RL，会遗漏基座模型在其更宽泛的解码/搜索空间中已能达到的操作点，从而低估两者关系。
3. **异构解码方法的建模难题**：token 级采样、beam-like 搜索、树搜索、序列级重采样等方法控制结构各异，缺乏统一的预算化操作空间使得跨方法比较难以形式化。
4. **过拟合式复现风险**：在大规模策略网格中可能出现逐预算的 post-hoc 匹配，但这类匹配未必构成低复杂度的可迁移路径，需要区分"结构化复现"与"孤立查找"。

## 核心贡献（创新点）
1. **UDF（SearchLens）统一框架**：将 token 级采样、beam-like 扩展、树式探索与序列级重采样统一为共享预算操作空间中的可执行策略，与评估指标（pass@k、SC、BoN、FFS）在概念上分离——这是将异构解码方法纳入统一度量空间的开创性尝试。
2. **二维恢复分析框架**：提出不仅在同一策略和预算下比较 Base/RL，而是询问 RL 默认策略曲线是否落在基座模型更宽策略-预算地形中的某个点上，从而定义"行为可恢复性"而非参数等价性。
3. **BOPTR-P1 低维转移规则**：发现 RL 的 pass@k 恢复路径遵循 $N_{\mathrm{Base}} \approx \alpha N_{\mathrm{RL}}^\beta$ 的预算转移规则，且指数 $\beta$ 由基准 regime 决定（数学 0.60、非数学 1.00、floor 0.00），本质区别在于将经验规律形式化为可迁移的路径选择器而非每预算查找表。
4. **水平/垂直转移的系统评估**：在同一 Hard RL 训练分布内直接复制路径误差低至 1.40–2.55 pp，跨 family（Llama、Mistral）退化为 3.68–10.16 pp；用一个锚点即可达到与七模型校准相近精度（4.08 vs 4.44 pp），揭示信息需求下界。
5. **RL 信息预算前沿**：证明预测 RL 后行为所需 RL 数据可压缩至最多一个锚点单元，甚至零 RL 的 base@rltrain 先验也能达到 5.13 pp，为后续"仅凭基座特征预测 RL 增益"提供了实证边界。

## 方法详解
1. **UDF（Unified Decoding Framework）操作点**：每个解码/搜索方法被表示为 $u = (\pi, n)$，其中 $\pi \in \Pi_{\mathrm{UDF}}$ 为五组件策略配置（本地策略/控制器/评估器/转移/调度器），$n \in \mathcal{B}=\{1,2,4,8,16\}$ 为推理预算。评估指标仅在生成后应用，不进入策略定义。
2. **操作点距离**：用组合距离 $d(u_i, u_j) = \lambda_{comp} d_{comp}(\pi_i, \pi_j) + \lambda_z \|z(u_i)-z(u_j)\|_1 + \lambda_b |\log n_i - \log n_j|$ 度量两个操作的差异，用于路径诊断而非机制识别。
3. **Same-policy 比较**：定义 $\Delta_{same}^c(b) = A_{RL}^c(\pi_0, b) - A_{Base}^c(\pi_0, b)$，隔离训练效应与推理策略增益；发现 RL 在低预算 pass@k/SC 上有优势，但基座在大预算时可追上或超越。
4. **Base+UDF 支撑区域**：对每指标每预算定义上包络 $E_{Base}^c(b) = \max_{\pi, n\leq b} A_{Base}^c(\pi, n)$，RL 目标在所有预算下均落在支撑区域内。
5. **Near-match 集与恢复路径**：$N_\epsilon^c(b) = \{(\pi, n): |A_{Base}^c(\pi, n) - A_{RL}^c(\pi_0, b)| \leq \epsilon\}$，从中选择构成单调预算路径的近匹配点，区分结构化恢复与孤立查找。
6. **BOPTR-P1 转移规则**：$\widehat{N}_{m,d}^{ch}(b) = \mathrm{round}_\mathcal{B}(\alpha_{m,d}^{ch} \cdot b^{\beta_{g(d)}^{ch}})$，其中 $\log \alpha_{m,d} = \mu_{g(d)} + \delta_m$，$\beta$ 由基准 regime 决定，$\delta_m$ 为模型级标量偏移；策略选择器最小化行为坐标损失 $\mathcal{L}(\pi) = \sum_b \|r(\pi, \widehat{N}(b)) - \bar{r}_{proto}\|_W$，$W=(1,1,1,0.5,0.2)$。
7. **Regime 发现**：Math500 为 sublinear（$\beta=0.60$）、GPQA/IFEval 为 near-linear OOD（$\beta=1.00$）、AIME 为 floor（$\beta=0.00$），三类 regime 在 R0/R1/R2 三种策略约束下稳定。

## 实验与结果
1. **数据集与基线**：使用 SimpleRL-Zoo 配对的 Base/RL checkpoint（Qwen2.5-0.5B/1.5B/7B/14B、Qwen2.5-Math-7B、Llama3.1-8B、Mistral-7B-v0.1），评估 Math500、AIME 2024/2025、GPQA-Diamond、IFEval 四个基准。
2. **水平转移（同模型不同基准）**：BOPTR-P1（H5）在 Qwen2.5-7B 上获得 3.41 pp 误差（95% CI [2.32, 5.53]），优于 anchor per-b copy（H3: 6.01 pp）和 strict BOPTR（H4: 6.56 pp）；三种子种子复现 $3.07 \pm 0.39$ pp。
3. **垂直转移（同 family 不同尺度）**：直接锚点路径在同 Hard RL 训练分布内误差 1.40–3.73 pp；V3（一维偏移校准）整体 4.44 pp，Llama 从 10.16 pp 降至 3.68 pp；V5b（纯基座预测偏移）达 4.19 pp。
4. **跨 family 压力测试**：Open-Reasoner-Zero-7B、OAT-7B、DeepSeek-Math-7B 分别得 4.87、4.48、3.28 pp；超出训练分布的场景误差升高。
5. **零基准转移**：冻结参数于四个从未拟合的基准（AMC23、Minerva、OlympiadBench、TinyMMLU），均值 5.03 pp vs 拟合时 4.44 pp，所有基准均值低于 8 pp 预注册标准。
6. **RL 信息前沿**：单锚点 Z1-N* 达 4.08 pp（匹配七模型 V3 的 4.47 pp）；零 RL 的 Z0-u 达 5.13 pp，仅差 ~1 pp。
7. **Ablation**：移除策略选择器（A1）误差 +2.76 pp；移除预算规则（A3）在数学 regime 上 +5.5 pp；移除 $\delta_m$（A2）对跨 family 的 Llama 降低 6.48 pp。

## 相关工作脉络
1. **RLVR 后训练**：SimpleRL-Zoo（Zeng et al., 2025）、DeepSeekMath（Shao et al., 2024）、DeepSeek-R1（Guo et al., 2025）——本文使用 SimpleRL-Zoo 配对 checkpoint 作为行为分析对象，而非提出新的 RL 方法。
2. **RL 增益的来源争论**：Yue et al.（2025）"RL 未创造新推理能力"、Zhao et al.（2025）"RL 放大预训练行为"、Karan & Du（2025）"训练时采样可匹敌 RL"——本文在明确预算轴之后重访此争论，提出 UDF+SearchPath 的行为学形式化。
3. **推理时计算缩放**：Snell et al.（2024）"test-time compute 可超过模型缩放"、Wu et al.（2025）"compute-optimal inference"——本文继承"预算作为显式轴"的思路，但聚焦于 Base/RL 在 UDF 空间中的关系而非纯计算预算优化。
4. **推理搜索与选择**：CoT（Wei et al., 2022）、Self-Consistency（Wang et al., 2023）、ToT（Yao et al., 2023）、GoT（Besta et al., 2024）、ReST-MCTS*（Hao et al., 2023）、Lightman et al.（2024）——UDF 将这些异构方法统一为预算化操作点，实现跨方法可比性。
5. **CoT 忠实性**：Turpin et al.（2023）、Lanham et al.（2023）——本文强调关注可观测输入输出行为而非参数级机制，与忠实性争议保持一致。
6. **长程 RL 训练效应**：Liu et al.（2025）"ProRL 扩展推理边界"——本文承认长程训练可能产生超出基座搜索可达范围的新行为，将结论限定在 tested recipe 和训练时长内。

## 局限性与未来方向
1. **行为等价≠参数等价**：BOPTR 仅证明 RL 增益在行为上可被 Base+UDF 路径近似，不 claim 参数层面的机制识别。
2. **策略池可比性**：部分 cell 仅含 AR-only 策略而另一些包含扩展控制器（MCMC、MCTS 等），跨池恢复比较需标记。
3. **度量语义异质性**：Math500/AIME 适合 pass@k/SC，GPQA 更像 concentration 任务，IFEval 需用 validity/FFS 分析，跨基准数字需考虑 regime 结构。
4. **模型覆盖有限**：仅覆盖一个 Qwen family 尺度扫描 + 少量 cross-family/stress 模型，regime map 不能直接 transfer 到其他 RL recipe。
5. **AIME floor regime 争议**：$\beta=0$ 的 AIME 映射与 $\beta=0.6$ 数学映射统计不可区分（95% CI 含零），"floor" 仅为描述性标签。
6. **大预算外推失效**：扩展到 $b>16$ 时 AIME 误差达 12.50 pp，反映 budget map 分配误差而非基座能力上限。
7. **未来方向**：v8-H 计划用 base-feature adapter 替换 $\delta_m$ 标量；引入更多模型 cohort 和 training-distribution calibration set；探索更长训练时长下的边界。

## 研究启发与可借鉴点
1. **UDF 统一框架可直接复用**：将异构解码方法建模为 $(\pi, n)$ 操作点的思路，可迁移到任何需要比较多种推理时策略的研究中（如对比 CoT vs ToT vs self-consistency 的效率边界）。
2. **二维恢复分析范式**："是否 RL 曲线落在 Base 的更宽策略-预算地形中"这一行为可恢复性检验框架，可作为评估任何后训练方法（DPO、SFT、RLAIF）与基座关系的标准诊断工具。
3. **Regime-conditioned 指数规律发现**：不同任务类型对应不同 $\beta$ 值的发现，提示未来可系统性地探索任务 regime 对 budget exponent 的影响，形成更普适的 transfer taxonomy。
4. **RL 信息预算前沿的压缩思想**：证明单锚点即可逼近多模型校准效果，为"如何用最少 RL 参考数据预测新模型 RL 增益"提供了实用范式，可直接迁移到模型选择与 checkpoint 排序场景。
5. **评估指标与策略分离设计**：UDF 将生成策略与 pass@k/SC/BoN/FFS 等评估指标解耦，避免了策略定义中的指标泄漏，这一设计原则适用于任何需要多指标评估推理行为的框架。

## 关键术语表
**RLVR（Reinforcement Learning with Verifiable Rewards）**：利用程序化或规则化正确性信号进行强化学习训练，以提升大模型推理能力的后训练方法。

**UDF（Unified Decoding Framework / SearchLens）**：将 token 级采样、beam-like 搜索、树式探索和序列级重采样统一表示为共享预算操作空间 $(\pi, n)$ 中可执行策略的分析框架。

**BOPTR（Budgeted Operating-Point Transition Rule）**：描述 RL 默认策略曲线可被基座模型预算化路径 $N_{Base} \approx \alpha N_{RL}^\beta$ 近似的低维转移规则。

**Regime（任务体制）**：基于 benchmark 性质划分的三类预算指数模式：math（sublinear, $\beta=0.6$）、nonmath/OOD（near-linear, $\beta=1.0$）、floor（budget-insensitive, $\beta=0$）。

**Same-policy 比较**：固定解码策略 $\pi_0$ 和预算 $b$，直接比较 Base 与 RL 的准确性，用于隔离后训练效应与推理策略增益。

**Base+UDF 支撑区域（Support Region）**：对每个预算 $b$，取所有 $\pi \in \Pi_{UDF}, n \leq b$ 对应的 Base 准确率的上包络，形成 RL 目标可被恢复的可行区域。

**Recovery Path（恢复路径）**：从各预算 near-match 集中选择构成预算单调、策略协调的连续轨迹，区别于逐预算孤立查找。

**RL 信息预算前沿（RL-Information Frontier）**：以使用的 RL 模型数量（0/1/7）为横轴，刻画预测 RL 后行为所需参考数据量的压缩下界。

## 可复现要素
- **代码**：https://github.com/HALIS-sh/Searchlens_boptr（开源，含 configs 和分析脚本）
- **数据集**：Math500（MATH 子集）、AIME 2024/2025、GPQA-Diamond、IFEval——均为公开 benchmark
- **模型权重**：SimpleRL-Zoo 提供的配对 Base/RL checkpoint（Qwen2.5 系列、Llama3.1-8B、Mistral-7B-v0.1）
- **关键超参**：预算集 $\mathcal{B}=\{1,2,4,8,16\}$；selector 权重 $W=(1,1,1,0.5,0.2)$；$\lambda_{cost}=0.01$；tolerance $\epsilon$ 用于 near-match（±3 pp）；典型性门控阈值 $\tau=0.02$
- **算力成本**：anchor cell（Qwen2.5-7B/Math500）约 461 H100 GPU-hours（230 wall-hours），含 56 个 deduped runs
