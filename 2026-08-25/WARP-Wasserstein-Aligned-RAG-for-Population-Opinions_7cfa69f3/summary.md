---
title: "WARP-Wasserstein-Aligned-RAG-for-Population-Opinions"
source: https://arxiv.org/pdf/2608.22859v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:04:00"
---

# 论文速读：WARP-Wasserstein-Aligned-RAG-for-Population-Opinions

## 一句话总结
本文提出 WARP，一种后检索分布校准框架，利用序数 Wasserstein-1 距离将 RAG 检索到的证据集情感强度分布对齐至目标人群意见分布，解决标准 cosine 检索在意见查询中过度放大主流观点、掩盖少数意见的分布失真问题。

## 研究问题与动机
- **核心问题**：基于 cosine 相似度的 top-k 检索仅追求语义相关，不保证检索结果反映语料库的真实意见分布，导致生成的摘要系统性偏向主流观点，产生“看似有据但严重失真”的分布扭曲。
- **现有多样性重排不足**：MMR、DPP 等方法仅最大化成对离散度，缺乏明确目标分布；一旦每个情感区间已有一个代表（通常 5–7 个），信号即耗尽，后续仍退回相关性排序。
- **现有校准方法忽略序数结构**：基于 KL 或 JS 散度的校准（如 Steck 2018）虽优化目标分布，但将情感区间视为无序类别，混淆“极强正”与“极强负”的代价与混淆相邻区间相同，不符合意见的序数语义。
- **工程落地缺口**：Agrawal et al. (2026) 形式化了意见对齐的统计目标，但未提供可运行的检索策略；生产环境需在亚秒级延迟内完成校准，缺乏免微调、免改索引的部署方案。

## 核心贡献（创新点）
- **提出池扩展机制（Entity-Gated / Adaptive Expansion）**：通过针对性再检索恢复被 cosine 排序埋没的少数派意见极值；与已有工作相比，其在池已平衡时零开销自跳过，且不依赖任何上游检索器修改。
- **设计基于 W₁ 距离的重排器家族（$W_1$ Minimizer、$W_1$-MMR、WassRank OT）**：利用 7 档序数情感的 CDF 闭合形式实现 $O(m)$ 单次评估；与已有工作相比，首次将最优传输的序数代价作为推理时无训练的运行时选择目标，而非训练损失。
- **建立密度驱动的部署决策矩阵与端到端生成验证**：按实体匹配率（EM%）划分 Dense/Sparse/Variable 场景并映射至对应变体，证明分布校准增益可稳定传递至下游 LLM 生成阶段；与已有工作相比，填补了从检索分布对齐到生成质量评估的完整闭环。

## 方法详解
- **序数情感尺度与目标分布**：将每篇文档标注为 7 档情感强度区间 $\mathcal{S} = \{-30, -20, -10, 0, +10, +20, +30\}$；目标分布 $P_{\mathrm{pop}}$ 取自语料库中该实体的实际观测比例（或外部先验如星级、调查数据），作为校准锚点。
- **Stage 1.5 池扩展**：对比候选池的区间占比与 $P_{\mathrm{pop}}$，若某极值区间低于阈值则触发 pole-biased 再检索；若池已镜像目标则自绕过，不产生额外开销。
- **W₁ 重排目标函数**：最小化选中集合 $S$ 的经验分布与 $P_{\mathrm{pop}}$ 的 Wasserstein-1 距离：
  $W_1(P, Q) = \sum_{i=1}^{m-1} |\mathrm{CDF}_P(s_i) - \mathrm{CDF}_Q(s_i)| \cdot \Delta s$
  其中 $m=7$，$\Delta s=10$；相邻区间误差代价为 10，跨极值误差代价为 60，显式建模意见序数结构。
- **三变体设计**：
  - **$W_1$ Minimizer**：过滤实体匹配候选，贪心选取使 $W_1$ 下降最多的文档，平局时按相关性决胜；适用于密集池（EM≥85%）。
  - **$W_1$-MMR**：全池打分，融合相关性分数与归一化校准增益 $\text{score}(d) = \lambda \cdot \text{rel}(d) + (1-\lambda) \cdot \frac{\Delta W_1}{W_1^{\mathrm{cur}}}$；适用于稀疏池，避免实体过滤后候选不足。
  - **WassRank OT**：将矩形指派问题全局求解，按 $P_{\mathrm{pop}}$ 比例分配槽位并最小化融合代价；适用于密度未知场景，无需领域调参。

## 实验与结果
- **数据集**：Amazon Seller Forums（~8K）、Yelp Hotels（~14K）、OpinRank Cars（~13K）；共 156 查询、26 实体，默认 $N=200$ 候选池、$k=20$ 输出。
- **基线**：Top-k、MMR、DPP、OpinionMMR、KL(JS) Minimizer。
- **主要结果**：所有 $W_1$ 变体较 Top-k 降低分布误差至少 43%（Seller Forums $W_1$-MMR 降 43%，Yelp $W_1$ Minimizer 降 79%，OpinRank $W_1$ Minimizer 降 69%）；Yelp 实体匹配率从 40.1% 提升至 99.2%（EG+$W_1$ Min）。
- **生成评估**：5 模型裁判盲评（Claude Sonnet 4、Llama 3.3 70B、Mistral Large 3、Amazon Nova Pro、DeepSeek v3.2），k=5 时 $W_1$ Minimizer Fair% 达 86%，显著优于随机与语义多样性基线；指标隔离证明序数度量带来额外 8–10% 收益。
- **鲁棒性与延迟**：标签噪声 50% 下 Dense 池仍显著优于 Top-k；$P_{\mathrm{pop}}$ 用 50–100 样本即可收敛；重排 p99 < 330 ms（Intel Xeon 单线程 CPU），端到端含本地 FAISS 检索 < 365 ms。

## 相关工作脉络
- **MMR / DPP 等多样性重排**：最大化成对离散度但无目标分布约束，单文档覆盖各区间后信号耗尽；本文以序数 W₁ 替代无目标发散目标，保证比例保真。
- **KL/JS 校准推荐（Steck 2018, PM-2）**：优化分布保真但视标签为无序类别，无法惩罚极端误判；本文用 W₁ 保留序数代价结构，跨极值误差代价提升 6 倍。
- **最优传输在 IR（WassRank, Word Mover’s, OTExtSum）**：均作用于训练阶段或通用内容摘要；本文首次将 OT 作为推理时免训练、免调参的运行时分布校准目标。
- **意见感知检索（Agrawal et al. 2026, Nayeem & Rafiei 2025）**：前者形式化覆盖/保
