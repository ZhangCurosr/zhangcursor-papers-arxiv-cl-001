---
title: "It-Takes-Two-to-Match-Co-Evolving-Generative-Retriever-with"
source: https://arxiv.org/pdf/2609.00638v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:54:39"
field: "生成式检索与强化学习"
keywords: ["generative retrieval", "reinforcement learning", "co-evolving", "keyword matching", "inverted index", "query-item alignment", "GRPO", "search ranking"]
innovations: ["提出CoGR框架，用LLM同时在查询侧和商品侧生成关键词并通过倒排索引直接匹配", "设计两阶段训练：SFT建立对齐初始化后交替GRPO优化两侧，共享F1目标协同演化", "提出反事实边际奖励精确衡量单商品关键词对全局检索F1的增量贡献"]
benchmarks: ["WANDS", "Internal APP Marketplace"]
---

# 论文速读：It-Takes-Two-to-Match-Co-Evolving-Generative-Retriever-with

## 一句话总结
论文提出 CoGR（Co-Evolving Generative Retriever），一个通过强化学习交替优化查询侧与商品侧 LLM 生成关键词的检索框架，使两侧关键词空间逐步对齐并通过倒排索引直接匹配。在内部 APP Marketplace 数据集和公开 WANDS 基准上，CoGR 的 $F_1$ 分别比最强基线提升 10.9% 和 36.1%。

## 研究问题与动机
- **现有关键词/检索增强方法仅训练单侧生成器**：多数 LLM 增强检索工作（如 query expansion、query rewriting）只训练查询侧生成器，商品侧表示仍由传统 retriever 固定维护，两侧表示空间无法协同优化。
- **稀疏、稠密、生成式检索各自存在局限**：BM25 等稀疏方法语义能力有限；DPR/ANCE 等稠密检索依赖向量索引与 ANN 搜索，难以直接复用现有关键词基础设施；DSI/RIPOR 等生成式检索对 identifier 设计敏感、解码可扩展性与泛化存在挑战。
- **多对多相关性场景下的召回-精确权衡困难**：工业检索（广告、电商、应用商店）中每个 query 对应大量 relevant item（千级），需在保持召回覆盖的同时控制返回集规模，单一侧优化难以同时兼顾两者。
- **需解决两侧生成关键词空间对齐问题**：若仅独立训练 query/item 生成器，二者关键词空间可能语义不一致，导致匹配失败；需要一种机制使两侧在共享检索目标下协同演化。

## 核心贡献（创新点）
1. **提出 CoGR 框架：用 LLM 直接在查询侧和商品侧生成关键词作为检索表示**。与已有工作仅训练 query 侧改写/扩展、商品侧仍由下游 retriever 处理不同，CoGR 让两侧 LLM 分别生成 compact keyword set，并通过倒排索引直接完成匹配，兼容现有关键词检索基础设施。

2. **设计两阶段训练流水线：SFT 初始化 + 交替 GRPO 协同演进 RL**。与以往单阶段 RL 或仅 query 侧 RL（如 DeepRetrieval）不同，本文先通过 SFT 从 relevant item 关键词中构造 query-side 监督目标以建立对齐初始化，再以 frozen opposite index 为基准交替优化两侧，使两侧关键词空间在共享 $F_1$ 目标下逐步 co-evolve。

3. **提出基于反事实边际的 item-side 奖励设计**。不同于简单对称的 item-centric $F_1$ 或使用独立 item 检索目标的 baselines，本文 item 侧奖励定义为替换单个 item 关键词后整体 query-to-item $F_1$ 的增量（$\sum_q F_1^{cand} - \sum_q F_1^{ref}$），通过缓存与差分计算高效实现，精确隔离该 item 对检索质量的边际贡献。

4. **在工业 APP 市场和 WANDS 产品搜索基准上取得 SOTA**。在 10 种稀疏、稠密、生成式基线上全面领先，验证了 co-evolving 设计与反事实奖励的有效性；消融证实单独优化 query 侧、去掉 SFT、改用 transposed $F_1$ 均导致性能下降。

## 方法详解
**总体架构**：给定 query $q$ 和 item $i$，训练两个独立 LLM 生成器 $G^q$ 和 $G^i$，分别输出紧凑关键词集合 $S_q = G^q(q)$ 和 $S_i = G^i(i)$。检索时取两侧原始文本与生成关键词的并集做交集匹配：$I_{ret}(q) = \{i : (S_q \cup \{q\}) \cap (S_i \cup \{i\}) \neq \emptyset\}$，排序沿用 BM25 on generated keywords。

**Phase 1 — SFT 初始化（Algorithm 1）**：
- 对每个 item $i$，用初始 LLM $G_0$ 采样生成 $M$ 个 item-side 关键词 $S_i$。
- 对每个 query $q$，收集其所有 relevant items 的 $S_i$ 构成多重集 $B_q$，取 top-$N$ 高频词作为 query-side 监督目标 $S_q$。
- 分别对 $\{(q, S_q)\}_q$ 和 $\{(i, S_i)\}_i$ 做 SFT，得到 $G^q_{SFT}$ 与 $G^i_{SFT}$，建立初步对齐的关键词空间并提供足够初始召回以产生有信息的 reward 信号。

**Phase 2 — 交替 GRPO 协同演进 RL（Algorithm 2）**：
- **Query-side RL**：固定 item 侧倒排索引，对每个 query $q$ 采样多个 keyword set $S_q$，以检索 $F_1$ 为终端 reward：
  - $P(q) = |rel(q) \cap I_{ret}(q)| / |I_{ret}(q)|$，$R(q) = |rel(q) \cap I_{ret}(q)| / |rel(q)|$，$F_1 = 2PR/(P+R)$。
  - 加入关键词预算约束：$|S_q| > K_{max}$ 时 reward 为 0。
  - 使用 GRPO 对 rollout 组内 reward 做归一化得到 advantage，更新 $G^q$。
- **Item-side RL（反事实边际奖励）**：固定 query 侧索引，对每个 item $i$ 的新候选关键词 $S_i$，构造 counterfactual 索引（仅替换 item $i$ 的参考关键词），奖励为整体 $F_1$ 增量：
  - $\mathcal{R}_i(S_i) = \sum_{q \in \mathcal{Q}} F_1(I_{ret}^{cand}(q; S_i), rel(q)) - \sum_{q \in \mathcal{Q}} F_1(I_{ret}^{ref}(q), rel(q))$，超限 $K_{max}$ 则 reward 为 0。
  - 实现效率：缓存每个 query 的 $n_q^{ret}, n_q^{tp}, n_q^{rel}$ 以及被每个 item $i$ 命中的 query 集 $\mathcal{Q}_i^{ref}$；候选 rollout 仅做单次对 query 侧倒排索引的 lookup 得 $\mathcal{Q}_i^{cand}$，受影响 query 集 $\mathcal{Q}_i^{\Delta} = \mathcal{Q}_i^{cand} \triangle \mathcal{Q}_i^{ref}$，对该子集做 $O(1)$ 更新即可，避免全量重算。
- **交替循环**：每轮先用 $G^q$ 构建 query 索引更新 $G^i$，再用新 $G^i$ 构建 item 索引更新 $G^q$，共 5 轮交替（query 侧 10 个 GRPO epoch，item 侧 5 个 epoch）。

**关键超参**：base model 用 Qwen3-4B-Instruct（也有 1.7B 版本）；$K_{max}=30$；SFT 阶段 query top-N=15、item M=10；GRPO rollout.n=8，lr=1e-6，training batch 512/256，max_response_length=512；解码 RL rollout 用 temperature=1.0、top-p=1.0 鼓励探索，初始化/评估用 temperature=0.7、top-p=0.8、top-k=20。

## 实验与结果
**数据集**：
- 内部 APP Marketplace：13,500 train / 1,500 eval queries，39,600 applications，per query ≈1,000 relevant items。
- WANDS（Wayfair 产品搜索）：430 train / 50 eval queries，42,994 products，per query ≈200 relevant items。
- 相关性标注：Internal 用 LLM-as-judge 五档（excellent/good/acceptable/poor/bad）二值化为 acceptable 及以上为相关；WANDS 为人评三档（Exact/Partial/Irrelevant）二值化。

**基线（10 种）**：
- 稀疏：BM25、SPLADE-v2
- 稠密：DPR（bert-base-multilingual-uncased）、ANCE（roberta-base）、Qwen3-Embedding-4B zero-shot / ANCE-finetuned
- 生成式：DSI、DSI-QG、RIPOR（均基于 mT5-base）
- LLM+RL：DeepRetrieval 4B（Qwen3-4B-Instruct）
- 自身变体：CoGR*（item 侧冻结）1.7B / 4B，CoGR 1.7B / 4B

**主要结果（Table 2）**：
- **Internal 数据集**：CoGR 4B 达到 $F_1=0.3963$（P=0.3976，R=0.4569），超越最强基线 ANCE-Qwen4B（$F_1=0.3575$）约 10.9%；@100 cutoff 下 $F_1@100=0.1349$ 同样最高。
- **WANDS 数据集**：CoGR 4B 达到 $F_1=0.6819$（P=0.6885，R=0.6819），超越最强基线 SPLADE-v2（$F_1=0.4903$）约 36.1%；@100 cutoff 下 $F_1@100=0.4778$ 最高。
- 1.7B 版本同样在两类数据集上优于对应规模基线，显示方法规模可扩展性。

**关键分析**：
- **Co-evolving dynamics（Fig. 3）**：$F_1$ 从 SFT 后约 0.16 逐步提升至第五轮约 0.40，第一轮增益最大，后续稳步改善。
- **Ablation（Table 3，Internal，4B）**：Transposed $F_1$（0.3743）、Shared generator（0.3798）、No SFT（0.3751）均低于 full CoGR（0.3963），验证三项设计必要性；但各 variant 均稳定，说明框架鲁棒。
- **Keyword evolution（Fig. 4）**：RL 后 unigram 比例从 37% 降至 13%，三词以上 phrase 从 12% 升至 31%；item 侧词表先收缩后与 query 侧趋于一致，词汇空间逐渐对齐、关键词趋于具体化。
- **额外信息消融（Table 4）**：移除 item description 使 $F_1$ 降至 0.3759；在 query prompt 中加入现有搜索引擎返回结果可使 $F_1$ 提升至 0.4379，表明框架可自然吸收外部语义线索。

## 相关工作脉络
- **稀疏检索（BM25 / SPLADE-v2）**：经典与学习型稀疏方法依赖显式 term 匹配，语义泛化弱；本文 CoGR 利用 LLM 生成 richer lexical representations 并通过 RL 直接优化检索指标，超越两者。
- **稠密检索（DPR / ANCE / Qwen3-Embedding）**：two-tower 向量相似度检索精度较高但需 ANN 索引与额外存储；CoGR 以关键词倒排索引替代向量索引，保留工业界现有 infra 兼容性。
- **生成式检索（DSI / DSI-QG / RIPOR）**：自回归生成 semantic ID 面临 identifier 设计依赖与解码扩展性挑战；CoGR 生成的是自然语言关键词而非离散 ID，直接对应倒排索引接口，规避此类问题。
- **LLM 增强检索（query expansion / rewriting 类工作）**：Gao et al.、Wang et al.、Ma et al. 等仅利用 LLM 做 query 侧扩充，最终仍交予下游 retriever 匹配；本文让 LLM 同时控制两侧表示。
- **DeepRetrieval（Jiang et al., 2025a）**：用 RLVR 训练 query rewriting 模型、item 侧冻结，是本文最接近的对照；CoGR 进一步引入 item 侧 co-evolve 与反事实边际奖励，形成完整的双侧协同框架。
- **自我演进 LLM 系统（G-zero / R-zero，Huang et al., 2026）**：multi-agent self-play 思路；CoGR 与之精神相通——不同角色（query/item generator）在与对方 evolving behavior 的交互中共同提升。

## 局限性与未来方向
- **排序阶段仍依赖 BM25**：检索召回后按生成关键词做 BM25 排序，尚未引入学习排序或 reranker，存在进一步提升空间。
- **奖励仅限相关性指标**：当前 reward 为 $F_1$，未直接优化业务目标如无关广告占比、收入增量、用户体验指标等。
- **需高质量 relevance 标注**：SFT 与 RL 均依赖 (query, item) 相关对标注；在标注稀缺或噪声较大的场景下表现未知。
- **交替训练的收敛性与稳定性依赖 task 规模**：当前在数千 query / 数万 item 规模验证有效，更大规模下内存与计算开销、交替频率选择仍需研究。
- **作者展望方向**：（1）扩展 reward 至下游业务目标（如 irrelevant ads %、revenue）；（2）设计更强的 post-retrieval ranking 模块替代朴素 BM25。

## 研究启发与可借鉴点
1. **Co-evolving 交替 RL 范式可迁移**：任何需要两侧表示对齐的匹配任务（如推荐系统中的 user/item 表征、双塔重排序、对话系统中的 utterance 配对）均可借鉴 "固定一侧、优化另一侧、循环交替" 的策略，避免两侧同步更新带来的非平稳问题。
2. **反事实边际奖励构造思路**：以 "替换单个元素后整体目标的增量" 作为该元素的奖励，既精确隔离贡献又可通过缓存/差分高效计算；该思路可用于大规模索引中单条记录的特征/表示优化（如广告创意、商品属性、文档片段）。
3. **SFT 提供有信息的 RL 初始化**：从零开始 RL 易陷入低召回导致 reward 信号稀疏；通过 label 构造监督目标先建立基本对齐，再进入 RL  fine-tune，是一条稳健的训练路线。
4. **关键词作为中间表示兼顾 LLM 语义能力与倒排索引效率**：直接生成关键词而非向量/ID，可在享受 LLM 语义理解优势的同时复用成熟倒排索引 infra，适合有既有关键词检索系统的团队平滑升级。
5. **与团队方向的结合机会**：若团队关注生成式检索、广告检索、或 LLM-based 检索增强，可将 CoGR 的 alternating RL + marginal reward 接入现有 dense/sparse 混合管线，或在 RAG 场景中对 document chunk 生成检索关键词进行 co-evolve 优化。

## 关键术语表
- **CoGR（Co-Evolving Generative Retriever）**：本文提出的检索框架，训练查询侧与商品侧两个独立 LLM 生成关键词表示，并通过交替 RL 使两侧关键词空间协同演化对齐。
- **GRPO（Group Relative Policy Optimization）**：一种强化学习策略优化算法，对同一 prompt 采样的多个 rollout 的 reward 做组内归一化以估计 advantage，本文用于更新两侧生成器。
- **反事实边际奖励（Counterfactual Marginal Reward）**：item 侧 reward，衡量将某 item 关键词替换为新候选后，全量 query-to-item 检索 $F_1$ 总和的增量，用于精确隔离单个 item 对检索质量的贡献。
- **倒排索引（Inverted Index）**：将文档/项的关键词映射到包含该词的文档 ID 列表的索引结构，支持高效的关键词交集检索；CoGR 直接复用此类基础设施完成匹配。
- **SFT（Supervised Fine-Tuning）**：用人工或规则构造的 (input, output) 对微调 LLM；本文 Phase 1 用 relevant item 词频构造 query-side 目标、原始生成构造 item-side 目标做对齐初始化。
- **$F_1$ 检索目标**：精确率 P 与召回率 R 的调和平均，同时兼顾返回集质量与覆盖度；本文将其同时作为 query 侧直接 reward 与 item 侧边际 reward 的基础度量。
- **Co-evolving（协同演化）**：指 query 与 item 两侧生成器在交替训练中逐步适应对方关键词空间、最终形成语义对齐的联合表征体系的过程。
- **DeepRetrieval**： Jiang et al. (2025) 提出的用 RLVR 训练 query rewriting LLM 的工作；仅优化 query 侧、item 侧冻结，是本文最重要的对照基线之一。

## 可复现要素
- **数据集**：内部 APP Marketplace（未公开）；WANDS（公开，Wayfair 产品搜索基准）。
- **代码/权重**：论文未明确声明开源；使用了 verl 框架与 Qwen3 系列模型。
- **关键超参**：base model Qwen3-4B-Instruct / Qwen3-1.7B；$K_{max}=30$；SFT top-N=15、M=10；GRPO rollout.n=8、lr=1e-6、train_batch_size 512（query）/256（item）、epochs 10（query）/5（item）、max_response_length=512；5 轮交替；B200 GPU×8。
