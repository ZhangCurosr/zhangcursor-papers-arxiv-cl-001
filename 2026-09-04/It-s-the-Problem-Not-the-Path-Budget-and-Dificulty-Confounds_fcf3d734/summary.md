---
title: "It-s-the-Problem-Not-the-Path-Budget-and-Dificulty-Confounds"
source: https://arxiv.org/pdf/2609.03436v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:05:02"
field: "LLM推理可解释性"
keywords: ["reasoning traces", "breakthrough detection", "process supervision", "interpretability", "test-time compute", "difficulty confound", "reproducibility"]
innovations: ["提出重启控制的截断探针，在匹配总生成token预算下分离预算拟合时间T_F与前缀价值时间T_V", "预注册难度控制预测协议，证明早期内部信号在question-only基线上无增量AUROC", "在公开多样本dump上拆解已发表early-window阳性，证明pooled AUROC主要来自跨问题难度信息"]
benchmarks: ["MATH", "OpenR1-Math-220k", "GPQA-diamond", "R1-Distill-Qwen-7B 256-per-problem dump"]
---

# 论文速读：It's the Problem, Not the Path: Budget and Difficulty Confounds in LLM Reasoning Trajectories

## 一句话总结
本文对 LLM 推理轨迹中两个流行信念（"突破时刻"和"早期信号可读命运"）施加了对照实验控制：引入**同预算重启曲线**与**纯问题基线**，发现大部分所谓"突破"只是预算充足的产物，早期内部信号无法在控制难度后预测推理结果。

## 研究问题与动机
- **信念一：推理轨迹含"突破时刻"**——强化训练 reasoning 模型的轨迹中，中途某位置似乎突然解锁正确答案，被称作"aha moment"（DeepSeek-R1 训练轶事）。但现有测量缺少反事实控制：截断轨迹后继续 vs 从空白重新开始，在总生成 token 数匹配的情况下，哪个更优？
- **信念二：轨迹早期命运可读**——多篇论文报道隐藏状态可从最初几个 token 预测对错（AUROC 0.79–0.95），甚至已应用于自适应推理系统。但这些评测未报告**问题难度基线**（question-only baseline），高 AUROC 可能只是在读题难度而非推理过程。
- **现有方法的共同缺陷**：两者都缺少对照控制——轨迹级论断需要在同一总预算下与 from-scratch 重启比较；信号预测论断需要在问题难度基线之上评估增量价值。

## 核心贡献（创新点）
1. **重启控制的截断探针（restart-controlled truncation probe）**：在匹配的总生成 token 预算下，将每个截断锚点的继续求解率 $\hat{p}(t; B)$ 与从空白重启曲线 $R(C)$ 对比，首次实现 per-anchor 继续率与 per-problem 重启曲线的联合测量。
2. **区分预算拟合时间 $T_F$ 与前缀价值时间 $T_V$**：将朴素"突破时间"分解为预算适配时间（solution first fits continuation budget）和前缀累积价值时间（prefix carries value beyond restart），这是之前文献中未报告的分解。
3. **预注册的难度控制预测协议**：以 question-only 基线（仅含问题文本特征 + 模型身份）为对照，预注册 endpoints、特征集、交叉验证方案与成功标准，发现早期窗口内部动力学摘要在测试集上不带来可检测的 AUROC 增量。
4. **在公开数据上复现并拆解已发表早期探针阳性**：利用 R1-Distill-Qwen-7B 的 128 问题 × 256 样本 dump，在相同问题内（within-problem）评估同一探针，发现 pooled AUROC 0.849 坍缩为 0.496（与随机无显著差异），证明已发表阳性几乎完全是跨问题难度信息。

## 方法详解
**截断探针设计（Section 2.2）**：
- 在 log-spaced 锚点 $t \in \{16, \ldots, 8192\}$ 截断模型自身轨迹，从截断前缀采样 $m=4$ 条续段，每条续段 $B=1024$ reasoning tokens + 512 answer reserve，固定分支 seed。
- 锚点求解率 $\hat{p}(t; B)$ 估计部分状态的**预算拟合值**。
- 突破定义为第一个 $\hat{p} \geq \tau=0.75$ 且在下一锚点保持稳定的点，记录为区间（bisection refinement），永远不交叉的轨迹右删截。
- **预算拟合时间** $T_F(B)$：解决方案首次 fit 到续段预算的点——即无对照探针报告的"突破"。

**重启控制与前缀价值时间（Section 2.3）**：
- 对每个问题测量重启曲线 $R(C)$：空前缀（同样 prompt）在 $C \in \{1024, 2048, 4096, 8192\}$ 的求解率，每预算 4 次尝试，$\hat{R}$ 在对数线性插值间估算。
- **前缀优势** $\text{adv}(t) = \hat{p}(t; B) - \hat{R}(t + B)$：在匹配总生成 token 预算下的增量价值。
- **前缀价值时间** $T_V(\delta)$：最早稳定满足 $\hat{p} \geq \tau$ 且 $\text{adv} \geq \delta$ 的锚点（主 margin $\delta=0.5$；灵敏度 $\delta=0.25, 0.75$）。
- 单元格分类体系（frozen precedence）：instant（事件区间上限 ≤16 tokens）、budget-limited（$\hat{R}(4096) \geq \tau$）、prefix-limited（存在 $\delta=0.5$ 交叉）、no-crossing、terminal、unsolved。

**噪声硬化（Section 2.4）**：
- 阈值删截轨迹（final-anchor rate ≥τ 但无稳定锚点）额外采 4 条分支，terminal 事件需 pooled rate ≥6/8。
- 所有模糊单元格（1–3/4 成功）在推导标签前扩展至 8 次尝试（共 148 cells）。
- 双重规则双向修剪：确认 13/16 候选，拒绝 1 个；pooling 移除 6 个事件，新增 1 个。

**预注册纪律（Section 2.5）**：
- 所有决策规则在观察任何结果前写入并提交 amendment，gate 失败（2 个 cohort gate + 1 个 pilot gate）记录为失败并通过 amendment 解决，绝不重新标注。
- Section 4 的确认性预测协议（endpoints、特征集、模型类、power gates、成功标准）在单次 test-split 评估前提交。

**早期信号预测协议（Section 4.1）**：
- 端点：primary = 16K 轨迹最终成功；secondary = 4096-token scratch-solvability。
- 特征集：question-only 基线（人类难度等级、主题类别、5 个文本表面统计 + 模型身份）vs 基线 + 15 个早期窗口动力学摘要（熵、surprisal、连续 KL/JS 散度、hidden-state 几何、谱摘要，前 512 tokens 内定义）。
- 模型：$\ell_2$ logistic regression，problem-grouped cross-validation on train+validation。
- 成功标准：paired $\Delta$AUROC 的 problem-clustered 95% CI 必须排除零。

## 实验与结果
**数据集与规模**：
- MATH benchmark（89 问题 × 2 模型 = 178 problem–model cells）：Gemma-4E4B + Ministral-3B（explicitly reasoning-tuned），4-bit MLX 量化，thinking mode，16K-token instrumented traces。
- 公开数据验证：192,315 条 DeepSeek-R1 generations over 91,573 问题（OpenR1-Math-220k）；16,384 条 GPQA-diamond 样本（64 问题 × 256 尝试）。
- 拆解分析：128 MATH 问题 × 256 R1-Distill-Qwen-7B samples（Nishad Singhi 公开 dump）。

**主要结果**：
- **突破绝大多数是预算假象**：178 cells 中恰好 **1 个** survive 为 prefix-limited（$T_V = 896$，advantage=0.75）；3 个跨 advantage margin，但 frozen precedence 将另外 2 个归为 budget-limited（4096-token 重启已解）。98 个 expansion cells（规则冻结后采集）贡献 0 个新 case。
- **重启剂量响应分离故障模式**：Ministral-3 从 scratch 求解率随预算攀升：0% → 19% → 45% → 79%（1024 → 8192 tokens），属**compute-starved**；Gemma-4 停在 10–12%，属**capability-limited**。两模型仅在 42/89 问题上共识。
- **累积推理主要是 compute compression**：13 个 threshold-censored 单元格中，9 个 matched budget 落在重启网格内，prefix 全部胜出（median adv +0.31）；largest measured restart 在 11/13 达到 $\tau$，prefix 自身率在 9/13 达到——长推理主要以更少总 token 达到相同成功率，而非拓展可达域。
- **中间可解性带持久存在**：148 模糊单元格扩展至 8 次尝试后，55 个仍停留在 3–5/8（success 0.3–0.6），独立后半段复现 58% 落在 1–3/4 中间带。
- **早期内部信号无检测到的结果信息**：预注册 test-split 评估，primary endpoint $\Delta$AUROC = +0.026 [−0.054, +0.167]（CI 跨越零）；secondary $\Delta = -0.090$ [−0.213, +0.033]（点估计为负）。post-hoc 窗口 sweep（$t \in \{128, 256, 1024, 2048\}$）10 个点中 8 个点估计为负，所有 CI 跨越零。
- **公开数据难度天花板**：trace-blind LOO pass-rate 在 192K DeepSeek-R1 generations 达到 AUROC **0.873** [0.870, 0.876]——完全落在已发表探针 0.79–0.95 范围内；GPQA-diamond 同样 estimator 达到 0.917 [0.875, 0.938]。
- **拆解已发表阳性**：重建 David [9] 的 last-four-token hidden-state probe at t=4，pooled AUROC **0.849** [0.735, 0.919]（与发表 0.84 一致），但在 22 个 mid-band 问题上 **within-problem AUROC = 0.496** [0.466, 0.527]（failure-count weighting）/ 0.515 [0.481, 0.562]（pair-weighting），与随机无显著差异；ten anchors 全部如此。pooled 阳性几乎完全由跨问题难度信息驱动。

## 相关工作脉络
1. **Lanham et al. [23]** 引入 early-answering truncation curves 衡量推理链的 faithfulness；本文在其基础上增加**匹配预算的重启基线**，将 continue-vs-restart 从定性描述升级为定量价值测量。
2. **Bogdan et al. [5]** resample sentence-level alternatives 归因单步贡献；**Bigelow et al. [3,4]** 定位 "forking tokens"；**Merrill & Srivastava [28]** 固定前缀 resample 定位 deceptive commitment；**Ballon et al. [2]** 读截断前缀答案分布；**Wang et al. [40]** 比较 continue truncated incorrect traces vs re-solve from scratch——所有工作均**缺少匹配总生成 token 预算的重启对照**，本文填补此缺口。
3. **Math-Shepherd [39]**、**OmegaPRM [26]**、**Phi-4 pivotal token search [1]**、**Setlur et al. [33]** 将续段求解率作为 process supervision 训练信号；本文将其**重purposed 为测量仪器**，并发现约 1/3 模糊状态在 8 次尝试后仍停留中间（ undermining single-crossing 单调值假设）。
4. **David [9]** 报道 first-four-token 内部状态 AUROC 0.79–0.84；本文在相同模型/数据 regime 上重建该阳性并拆解，发现 pooled 0.849 但 within-problem 0.496（与随机无差异），证明已发表结果本质是**难度测量**。
5. **Yuan et al. [44]** 报道 hidden error awareness，但其 early-window 阳性无 question-only 控制，within-problem effect sizes 低至 $d=0.13$；本文补充难度控制维度，与因果失效 [44]、coherence confound [22] 构成对 probe信号的第三个独立批评支柱。
6. **Lugoloobi et al. [25]** 在 generation 前 probing，自身结论为 activations encode model-specific difficulty；本文将其结论推广至 generation 窗口内，证明推理轨迹内的 pooled AUROC 同样主要是难度信息。

## 局限性与未来方向
- **模型规模受限**：仅两个小 open-weight 模型（Gemma-4E4B、Ministral-3B），在 4-bit 量化下运行；大 RL 训练 reasoning 模型（如 DeepSeek-R1 本身）的结论尚待验证。
- **预算匹配仅等化生成 token 数**，未测量 FLOPs、延迟、KV-cache 复用（继续存储前缀可摊销 KV cache，重启需重新计算）。
- **确认性基线为 text-level**，pre-generation activation probe 会是更强的 baseline（已引用 [25]）。
- **公开 dump 拆解继承未知采样 temperature**；teacher-forcing 提取 top-1 fidelity 0.906。
- **数学领域专属**：非数学领域的拆解等待 multi-sample dump from locally runnable model。
- **within-trajectory timing 问题统计功效不足**：真实 interior events 极少（test split 仅 4 个），这本身也是一个发现。
- **未预注册 equivalence margin**，结果为"未检测到增益"而非"证明等价"；primary CI 上限 +0.167 不排除中等效应。
- **未来方向一**：在同一冻结仪器上测试大型 RL-trained reasoner（aha-moment 声称的来源）， interior events 若存在则重燃 timing 问题，若不存在则扩展 budget-artifact 解释至动机该声称的 regime。
- **未来方向二**：将价值曲线 $\hat{p}(t)$ 的形状（gradual vs stepped）作为研究对象，当前数据已足够。

## 研究启发与可借鉴点
1. **对照设计的范式价值**：任何轨迹级论断（breakthrough、commitment、forking token）都需要**同预算反事实对照**——从空白重启在匹配总成本下能否达到相同效果。这提供了一个普适的测量框架，可迁移至其他 trajectory interpretation 工作。
2. **pre-registration 作为方法论标杆**：从 cohort selection 到 decision rules 到 confirmatory endpoints 全链条 pre-register，gate 失败公开记录并通过 amendment 解决而非 relabeling，为可重复性研究树立高标准。
3. **噪声硬化策略**：threshold-censored terminal replication（6/8 pooled rate 门槛）+ ambiguous cell enlargement（4→8 attempts）双规则，双向修剪 labels，为小规模 probing 实验提供可复用的可靠性方案。
4. **within-problem evaluation 的拆解脱耦方法**：利用 per-problem 多样本 dump（256 attempts/problem）天然实现 difficulty held fixed，将 pooled AUROC 分解为 between-problem（难度）和 within-problem（过程）两个正交分量——这一设计模式可复用于任何多样本推理 dump 的重新分析。
5. **与小团队方向的结合机会**：若本团队研究 process supervision 或 test-time compute scaling，本文的重启控制框架可直接用于评估"长推理轨迹是否比多次短重启更有价值"；中间可解性带的发现也提示 binary process label scheme 需要概率化改造。

## 关键术语表
**Budget-fit time ($T_F$)**：解决方案首次 fit 到续段预算的锚点位置，即无对照探针报告的"突破时间"，混同了前缀价值与预算充足两件事。

**Prefix-value time ($T_V$)**：在匹配总生成 token 预算下，前缀带来的求解率优势超过 margin $\delta$ 的最早稳定锚点，分离出"前缀本身有价值"的事件。

**Restart curve $R(C)$**：从空白（空前缀、同样 prompt）在预算 $C$ 下的求解率曲线，作为 per-problem 反事实对照基线。

**Question-only baseline**：仅含问题文本特征（难度等级、主题类别、5 个表面统计）+ 模型身份的分类器，不含任何推理轨迹内部信号，衡量难度信息的上界。

**LOO pass-rate（leave-one-out 通过率）**：对问题 $c$ 的第 $j$ 次尝试，用其余 $k_c - 1$ 次尝试的成功率估算 $p_c$，是完全不读 trace 的 trace-blind 难度代理。

**Compute-starved vs capability-limited**：前者求解率随预算增长显著提升（如 Ministral-3，0%→79%），后者几乎不变（如 Gemma-4，10%→12%），反映不同故障模式。

**Intermediate solvability band**：约 1/3 模糊中间状态在 8 次尝试后仍停留在 success 0.3–0.6 区间，既非确定正确也非确定错误， undermining binary process labeling。

**Pooled vs within-problem AUROC**：pooled AUROC 混合了跨问题（难度）和 intra-problem（过程）信息；within-problem AUROC 固定难度后仅测量过程信息，neutral reference 为 0.5。

## 可复现要素
- **数据集**：MATH benchmark [18]（公开）；OpenR1-Math-220k [19]（Hugging Face 公开，per-generation correctness annotations）；GPQA-diamond [31]（公开）；R1-Distill-Qwen-7B 256-per-problem dump [35]（Hugging Face 公开，由 Nishad Singhi 发布）。
- **代码与协议**：GitHub https://github.com/bulutyigit/problem-not-path，包含 frozen amendments、所有 result artifacts、generation/verification 脚本。
- **模型**：Gemma-4E4B、Ministral-3B，mlx-community 4-bit 量化 checkpoint（revision pinned in readiness manifests）。
- **关键超参**：temperature=0.6，top-p=0.95，top-k=20，thinking mode，16,384-token base trajectory cap；probe 分支 deterministic seeds (sha256)；重启每预算 4 attempts；anchor 网格 log-spaced 16–8192 tokens；续段预算 $B=1024+512$；$\tau=0.75$，$\delta=0.5$（主）；logistic regression C=0.1（instrumented）/ C=1（dump）；PCA ≤128 components。
- **验证**：final answer 提取 → numeric equivalence → symbolic equivalence；extraction failure 计为 incorrect；GPQA 比较 option letter。
