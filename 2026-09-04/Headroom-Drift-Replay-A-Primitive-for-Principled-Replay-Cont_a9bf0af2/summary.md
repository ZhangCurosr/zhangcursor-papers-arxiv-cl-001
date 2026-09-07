---
title: "Headroom-Drift-Replay-A-Primitive-for-Principled-Replay-Cont"
source: https://arxiv.org/pdf/2609.03941v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:03:01"
field: "大语言模型强化学习后训练"
keywords: ["Reinforcement Learning", "Replay Control", "GRPO", "Reasoning Models", "Policy Optimization", "Sample Efficiency"]
innovations: ["将重放控制解耦为学习价值排序(Headroom)和策略兼容性门控(Policy Drift)两个正交维度", "仅通过teacher-forced reevaluation实现无额外生成开销的原则性重放选择", "在数学推理/Agentic Search/多模态推理三域均超越简单重放并匹敌更复杂重放方法"]
benchmarks: ["AIME24", "MATH500", "Search-R1", "Geometry3K", "MathVista"]
---

# 论文速读：Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO

## 一句话总结
本文提出了Headroom-Drift Replay，一个面向GRPO训练的重放控制原语，通过将重放决策分解为"学习价值优先排序"（Headroom）和"当前策略兼容性门控"（Policy Drift）两个独立维度，在不增加额外生成或训练组件的前提下，实现了对历史轨迹的选择性复用，在数学推理、多模态推理和Agentic Search三个基准上均取得了优于简单重放、匹配或超越更复杂重放方法的效果。

---

## 研究问题与动机

1. **重放成本与价值的矛盾**：基于RL的推理模型后训练中，重复生成新鲜rollout消耗大量计算资源，尤其在Agentic场景中环境交互占主导。重放可复用历史轨迹降低成本，但简单重放（如FIFO或最近邻）往往导致训练不稳定或策略退化。

2. **重放贡献难以隔离**：现有重放方法（如RePO、EFRame、ExGRPO、BAPO）通常将重放嵌入更大的训练流水线中（如混合策略优化、探索-过滤、经验重构），使得重放自身的贡献难以单独评估。

3. **两类失败模式需要分别控制**：
   - 已存储组可能仍有学习价值但已过时（策略漂移过大）
   - 已存储组可能接近当前策略但已无进一步学习空间

4. **核心问题**：如果只把重放侧的控制做到极致，而不依赖其他辅助机制，能走多远？

---

## 核心贡献（创新点）

1. **将重放控制解耦为两个正交维度**：提出Headroom（学习价值）和Policy Drift（策略兼容性）两个独立判别器，而非将重放视为单一包含决策。

2. **构建角色对齐的基线体系**：设计了对比性问题——匹配重放预算的on-policy基线、更大新鲜数据缩放、简单重放量级、强非重放替代方案、更广重放方法——使本文方法的效果可被精确定位。

3. **系统性训练动态分析**：从同buffer反事实对比、重放年龄兼容性、多奖励进入动态、熵坍缩延迟等多个角度深入分析了原则性重放控制的运作机制。

4. **无额外训练开销**：不引入新的生成阶段、不修改主梯度流程、不添加辅助损失，仅对GRPO的混合batch选择性插入历史组。

---

## 方法详解

### 核心思想
将重放选择分解为两个独立决策：**排序**（哪些组值得重用）和**门控**（哪些组仍可安全重用）。

### 关键设计

**1. GRPO基础设定**
- 对提示Q，当前策略π生成n个响应$G_1,...,G_n$组成组$g = (Q, G_1,...,G_n)$
- GRPO计算response-level优势$A_i$，并应用到该response的所有token位置
- 重放组与新鲜组的区别：每个重放组携带生成时的参考策略$\pi_{\mathrm{gen}(g)}$及其对应的log-probabilities和优势$A_i$

**2. Headroom：学习价值优先排序**
- 对token位置$(i,j)$，定义token级Headroom贡献：
$$
h_{i,j}(\pi; g) = \begin{cases} 1 - \pi(a_{i,j}|s_{i,j}), & A_i > 0 \\ \pi(a_{i,j}|s_{i,j}), & A_i < 0 \\ 0, & A_i = 0 \end{cases}
$$
- 组级Headroom为所有token的均值：
$$
\mathrm{Headroom}(g;\pi) = \frac{1}{|\mathcal{T}(g)|}\sum_{(i,j)\in\mathcal{T}(g)} h_{i,j}(\pi; g)
$$
- 直觉：正向优势的action还有概率提升空间，负向优势的action还有概率下降空间；Headroom越大，该组越"值得"被重新训练

**3. Policy Drift：当前策略兼容性门控**
- 定义token级log-probability漂移：
$$
\Delta_{i,j}(g;\theta_t) = \log\pi_{\theta_t}(a_{i,j}|s_{i,j}) - \log\pi_{\mathrm{gen}(g)}(a_{i,j}|s_{i,j})
$$
- 组级Policy Drift为平方均值（L2风格，非L1）：
$$
\mathrm{Drift}(g;\theta_t) = \frac{1}{|\mathcal{T}(g)|}\sum_{(i,j)\in\mathcal{T}(g)} \Delta_{i,j}^2(g;\theta_t)
$$
- **为何用L2而非L1**：L1允许正负漂移相互抵消，掩盖整体序列的实际偏移；L2对集中式per-token mismatch更敏感，惩罚更大
- 重放准入条件：$\mathrm{Drift}(g;\theta_t) \leq \tau$

**4. 单个GRPO step的流程（Phase A-E）**
- **Phase A**：当前策略生成新鲜on-policy组，计算reward和优势，识别重放候选组$\mathcal{T}_t$
- **Phase B**：从buffer中检索已存组，按Headroom降序排序
- **Phase C**：对Headroom排序后的候选组逐个用当前策略重新计算log-probability（teacher-forced，单次前向），同时计算Drift和refresh Headroom；Drift≤τ则接受，否则拒绝；直到fill满重放预算$K_{\mathrm{rep}}$或无可接受组
- **Phase D**：接受的重放组与新鲜组合并为mixed actor batch $\mathcal{A}_t = \mathcal{G}_t^{\mathrm{on}} \uplus \mathcal{R}_t$，执行标准GRPO更新
- **Phase E**：将$\mathcal{T}_t$追加到FIFO buffer（防止同step重放），超过容量C时淘汰最老组

**5. 数学性质**
- **Proposition 2**（Policy-Drift控制Headroom不匹配）：对任何已存组$g$，$|H_t^{\mathrm{cur}}(g) - H^{\mathrm{ref}}(g)| \leq \sqrt{\mathrm{Drift}(g;\theta_t)}$
- 推论：通过Drift门控后，当前策略Headroom与参考Headroom的偏差不超过$\sqrt{\tau}$
- **Corollary 2.1**（排序保证）：若$H^{\mathrm{ref}}(g) - H^{\mathrm{ref}}(h) > \sqrt{\mathrm{Drift}(g)} + \sqrt{\mathrm{Drift}(h)}$，则$H_t^{\mathrm{cur}}(g) > H_t^{\mathrm{cur}}(h)$

---

## 实验与结果

### 实验设置
- **数学推理**：AIME24, AMC23, MATH500, Minerva, OlympiadBench（使用Qwen2.5-Math-1.5B）
- **Agentic Search**：NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, Musique, Bamboogle（Search-R1，使用Qwen2.5-3B/7B-Instruct）
- **多模态推理**：Geometry3K, MathVista, MathVision

### 主要指标
- **Mean@32**（headline metric）：每输入生成32个样本，取平均后跨benchmark macro平均
- **Best@32**、**Weighted Mean@32**（详见Appendix B）

### 关键结果

**数学推理（最全基线集）**
| Method | AIME24 Best@32/Mean@32 | AMC23 | MATH500 | Minerva | OlympiadBench | Macro Avg | Weighted Avg |
|--------|------------------------|-------|---------|---------|---------------|-----------|--------------|
| GRPO on-pol matched (b256) | 0.3083/0.0888 | 0.8000/0.4531 | 0.8760/0.6927 | 0.3529/0.1994 | 0.4095/0.2437 | 0.5494/0.3356 | 0.5473/0.3696 |
| GRPO on-pol larger (b384) | 0.2833/0.0833 | - | - | 0.3566/0.1957 | 0.4095/0.2403 | 0.5559/0.3344 | 0.5486/0.3664 |
| DAPO | 0.2750/0.0966 | 0.8500/0.4633 | 0.8800/0.6895 | 0.3824/0.1790 | 0.4347/0.2319 | 0.5704/0.3409 | 0.5722/0.3636 |
| ExGRPO | 0.3417/0.0758 | 0.8750/0.4469 | 0.9060/0.6701 | 0.3934/0.1545 | 0.4318/0.2220 | 0.5896/0.3139 | 0.5772/0.3448 |
| BAPO | 0.3083/0.0891 | 0.9000/0.5117 | 0.9060/0.7026 | 0.3971/0.1911 | 0.4362/0.2400 | 0.5895/0.3469 | 0.5778/0.3712 |
| GRPO+r matched (b256+r128) | 0.1417/0.0813 | 0.7000/0.5312 | 0.8200/0.7165 | 0.2721/0.1976 | 0.3323/0.2459 | 0.4166/0.3117 | 0.4525/0.3595 |
| GRPO+r larger (b256+r256) | 0.1333/0.0792 | 0.6250/0.4813 | 0.8160/0.7160 | 0.2721/0.1976 | 0.3234/0.2474 | 0.3922/0.3053 | 0.4421/0.3595 |
| **Headroom-Drift (b256+r128)** | **0.2750/0.0943** | **0.8750/0.5117** | **0.8820/0.7104** | **0.3456/0.2022** | **0.4050/0.2477** | **0.5565/0.3533** | **0.5455/0.3792** |

- Headroom-Drift在**Avg Mean@32上领先所有基线**（0.5565 vs. 次优BAPO 0.5895的Macro Avg，但注意DAPO/ExGRPO/BAPO Fresh Responses各不相同）
- 相对GRPO on-policy matched：同budget下提升显著
- 相对GRPO on-policy larger：用更少fresh response仍超越
- 相对简单重放（GRPO+r matched/larger）：差距进一步扩大，说明重放质量比数量更重要
- 超越DAPO、ExGRPO、BAPO等更复杂方法

**Agentic Search（成本敏感场景）**
| Method | NQ Mean@32 | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | Macro Avg | Weighted Avg | Step Time(s) |
|--------|-----------|----------|-------|----------|-------|---------|-----------|-----------|--------------|--------------|
| GRPO on-pol matched (b128) | 0.4908/0.4026 | 0.6569/0.5711 | 0.4885/0.4055 | 0.4013/0.2895 | 0.3913/0.2312 | 0.1528/0.0890 | 0.3370/0.2480 | 0.4169/0.3196 | 0.4733/0.3674 | 146.4 |
| GRPO+r matched (b128+r64) | 0.5319/0.4496 | 0.6785/0.5984 | 0.5114/0.4339 | 0.4539/0.3321 | 0.4862/0.2906 | 0.1986/0.1132 | 0.3760/0.2660 | 0.4623/0.3548 | 0.5201/0.4062 | 154.4 |
| **Headroom-Drift (b128+r64)** | **0.5449/0.4269** | **0.6940/0.5889** | **0.5301/0.4375** | **0.4721/0.3320** | **0.5115/0.3083** | **0.2305/0.1241** | **0.4320/0.2860** | **0.4879/0.3577** | **0.5399/0.4084** | 166.3 |
| GRPO on-pol larger (b192) | - | - | - | - | - | - | - | 0.4160/0.3212 | 0.4669/0.3719 | 197.2 |
| DAPO | 0.6420/0.5626 | 0.6533/0.5488 | 0.4907/0.4196 | 0.4050/0.3017 | 0.3702/0.2306 | 0.1546/0.0909 | 0.3477/0.2280 | 0.4038/0.2939 | 0.4646/0.3486 | 232.5 |

- **Pareto改进**：Headroom-Drift以166.3s/step的成本，达到Avg Mean@32=0.3577，而GRPO on-policy larger用197.2s/step且仅0.3212
- 相对简单重放GRPO+r matched：Avg Best@32差距更显著（0.4879 vs. 0.4623），说明原则性选择更关注质量上界
- 7B scale check（Qwen2.5-7B-Instruct，8-bit optimizer，Mean@4）：Headroom-Drift在7个benchmark中赢6个，Macro Avg Mean@4从0.3737→0.3955

**多模态推理（3-benchmark简化视图）**
| Method | Geo3K | MathVista | MathVision | Macro Avg | Weighted Avg |
|--------|-------|-----------|------------|-----------|--------------|
| GRPO on-pol matched | 0.5870/0.4927 | 0.7600/0.5695 | 0.5732/0.3986 | 0.4839/0.2740 | 0.4945/0.2788 |
| GRPO on-pol larger | 0.5753/0.4864 | 0.7740/0.5747 | 0.5786/0.4005 | 0.4945/0.2788 | - |
| DAPO | 0.5827/0.4952 | 0.7720/0.5860 | 0.6017/0.4137 | 0.4941/0.2793 | - |
| **Headroom-Drift** | **0.6245/0.5111** | **0.7720/0.5825** | **0.6017/0.4137** | **0.5148/0.2883** | - |

- Headroom-Drift在Avg Mean@32上领先（0.4137 vs. DAPO 0.4056、on-pol matched 0.3986）

---

## 相关工作脉络

1. **Prioritized Experience Replay (PER, Schaul et al., 2016)**：证明存储样本的学习价值差异很大，重放应按期望训练效用而非随机选择。Headroom本质上是PER思想在GRPO group-level上的扩展——但PER作用于transition/trajectory，Headroom作用于full GRPO组。

2. **Off-policy correction方法 (Munos et al., 2016; Espeholt et al., 2018)**：解决policy mismatch问题，通过importance weighting或truncation使过去经验在当前策略下仍可用。Policy Drift继承了这一思路，但将其作为hard gate而非soft权重。

3. **RePO (Li et al., 2025)**：将重放嵌入GRPO训练loop，提升sample efficiency。本文承认重放有价值，但指出RePO等将重放耦合到更大pipeline中，难以单独评估重放控制本身。

4. **EFRame (Wang et al., 2025)**：结合探索、过滤和重放。比本文更广，但本文的可组合性更强。

5. **ExGRPO (Zhan et al., 2026)**：按correctness和entropy信号组织推理轨迹后mixed-policy重用。同样更复杂。

6. **BAPO (Wan et al., 2026)**：buffer-centric off-policy设计，将历史hard sample与adaptive batch construction配对。

7. **DAPO (Yu et al., 2025)**：强非重放对比基线，open-source LLM RL系统。

8. **CISPO (Zheng et al., 2025)**：group sequence policy optimization，Appendix H初步验证Headroom-Drift可移植到CISPO objective。

---

## 局限性与未来方向

1. **单objective依赖**：当前主要在GRPO上验证，CISPO-style portability（Appendix H）仅为preliminary evidence，PPO/GSPO-family泛化仍需探索。

2. **Drift阈值τ的调参**：虽通过log-spaced sweep可快速定位可行区间，但τ对不同task-family（single-turn reasoning vs. agentic search）不同（$10^{-3}$ vs. $10^{-2}$），仍需人工选择。

3. **缓存Headroom的staleness**：未扫描组的cached Headroom会过时，仅靠Drift gate保护admission，可能在某些场景下丢失有价值组。

4. **FIFO buffer容量固定**：最老组无条件淘汰，不考虑其当前策略兼容性，可能造成历史信息浪费。

5. **Late-stage replay contraction**：Appendix E.4.2发现 collapse阶段replay mixture急剧收缩至age-1/2主导， corrective signal变窄，如何缓解仍是开放问题。

6. **多reward setting的ingress rule**：Appendix C揭示Geometry3K等multi-reward场景下，原$s_i^+=\mathbf{1}[r_i>0]$规则导致ingress干旱；需改为answer-based指示$a_i=\mathbf{1}[r_i\geq 0.9]$，这增加了工程复杂度。

---

## 研究启发与可借鉴点

1. **重放控制的两轴分解范式**：将重放问题分解为"学习价值"和"策略兼容性"两个正交维度，可作为通用设计模式迁移到其他RLHF/GRPO变体中。

2. **L2-style drift的方差敏感性**：L2聚合比L1更能捕捉集中式per-token mismatch，这对任何基于distribution distance的gate设计都有参考价值。

3. **无额外生成开销的原则性重放**：仅通过teacher-forced reevaluation（单次前向）即可同时计算Drift和refresh Headroom，工程实现轻量，适合部署在已有RL pipeline中。

4. **Training-score-matched entropy分析**：Appendix E展示了如何用训练分匹配而非绝对步数来评估entropy collapse delay，这是一种更严谨的训练动态分析方法。

5. **Answer-based ingress rule for multi-reward settings**：Appendix C揭示了单reward和multi-reward下ingress规则的语义差异，提示在复杂reward设计中需对齐目标语义信号而非仅看final reward positivity。

---

## 关键术语表

**Headroom**：衡量已存组在当前策略下remaining directional correction room的分数；正值优势response的概率提升空间和负值优势response的概率下降空间之和，Headroom越大该组越值得重放。

**Policy Drift**：已存组在stored action上的token级log-probability平方均值，衡量当前策略与生成时策略的mismatch magnitude，用作重放准入的hard gate。

**GRPO (Group Relative Policy Optimization)**：DeepSeekMath提出的group-relative RL训练方法，通过组内response reward比较计算advantage，再执行clipped PPO-style更新。

**Mixed-batch GRPO update**：将fresh on-policy组与selected replay组合并后的标准GRPO目标，形式上可分解为on-policy部分和replay部分的加权平均。

**Replay ingress**：新鲜生成组被判定为"值得进入replay buffer"的条件；本文使用mixed-outcome-only规则（组内reward有差异即准入）。

**Entropy collapse**：GRPO后期policy distribution过度集中的现象，表现为entropy骤降，与reward hacking和exploration failure相关。

**FIFO replay buffer**：固定容量先进先出队列，用于存储replay ingress candidates；超过容量时淘汰最老组。

**Cached vs. reference vs. runtime Headroom**：reference Headroom是生成时的不可变分数；cached Headroom是buffer中的排序优先级；runtime Headroom是当前策略重评估后的实际值。

---

## 可复现要素

- **数据集**：AIME24, AMC23, MATH500, Minerva, OlympiadBench, NQ, TriviaQA, PopQA, HotpotQA, 2WikiMulti HopQA, Musique, Bamboogle, Geometry3K, MathVista, MathVision——均为公开benchmark
- **模型**：Qwen2.5-Math-1.5B（数学推理）、Qwen2.5-3B-Instruct/7B-Instruct（Agentic Search）
- **代码**：论文未明确声明开源仓库，但附录提供了详细Algorithm pseudocode（Algorithm 1-2）和超参数（Appendix G.2 Table 13）
- **关键超参**：
  - 重放预算$K_{\mathrm{rep}}$：数学推理128，Agentic Search 64
  - Drift阈值τ：single-turn reasoning $10^{-3}$，agentic search $10^{-2}$
  - Buffer容量C：512 query groups（8,192 responses）
  - 新鲜on-policy batch size：b256（数学）、b128（Agentic Search）
  - Mini-batch size：64
  - GPU数：8×H100
- **开源关联**：DAPO为open-source system，本文可与DAPO框架对接；Search-R1为公开agentic benchmark suite

---
