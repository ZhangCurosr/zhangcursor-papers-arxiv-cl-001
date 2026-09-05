---
title: "Annotated-Surrogate-Retrieval-for-Polish-Statutory-Law"
source: https://arxiv.org/pdf/2608.30929v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:31:53"
field: "法律信息检索"
keywords: ["legal retrieval", "statutory law", "document surrogate", "Polish law", "reciprocal rank fusion", "reranking", "RAG evaluation"]
innovations: ["提出三种基于语言模型生成文档代理的成文法检索设计（ASCR/ASCR-H/DTF），覆盖九倍延迟范围", "揭示排名精度与下游引用准确性的解耦：DTF以rank-one落后20点匹配oracle引用准确性", "系统性报告三项负结果：词形还原、伪相关反馈、查询重写均无益于波兰成文法检索"]
benchmarks: ["Polish statutory law examination questions (2024-2025)", "82,508 articles from 1,133 acts"]
---

# 论文速读：Annotated-Surrogate-Retrieval-for-Polish-Statutory-Law

## 一句话总结
本文提出三种基于语言模型生成的文档代理（document surrogates）的波兰成文法检索方法，在 300 道司法考试题目上实现了显著优于传统 BM25 和密集检索的排名精度；其中 ASCR-H 以 rank-one 精度 72.3% 领先，而 DTF 以九分之一延迟和极低成本在深度覆盖和引用准确性上与 oracle 持平。

## 研究问题与动机
1. **目标极其狭窄**：识别管辖特定法律问题的条文，需在数万件条文中精确定位唯一正确的单一 article，rank-one 精度是下游生成器最敏感的指标。
2. **波兰语形态复杂性**：波兰语名词和形容词跨七种格变化，问题和法条常共享意义而不共享表面形式，纯词汇匹配受限。
3. **现有法律基准的不足**：现有法律检索基准主要集中于英语；波兰成文法条文从未作为独立检索目标；Smywinski-Pohl 等人的 LQuAD-PL 使用来自答案键的 3,654 条条文构建候选集（仅为本文候选集大小的 1/22.6），且报告几乎完美的 nDCG@5，"不构成实际挑战"。
4. **现有文献缺失的评估维度**：BSARD 等多标签数据集以 Recall@100+ 为主，无法使用 rank-one 指标；本文以国家考试委员会提供的单条参考条文为 ground truth，使 rank-one 精度具有实际意义。

## 核心贡献（创新点）
1. **三种文档代理检索设计覆盖不同成本-质量前沿**：ASCR（代理级联+重排序）、ASCR-H（融合密集列表的级联）、DTF（三路确定性融合，零预生成模型调用），延迟跨度达九倍。
2. **将索引端与查询端生成分离应用于法定条文检索并系统评估**：Nogueira 等人（2019）同时探索了四个组合，本文仅报告替代（substitution）而非拼接（concatenation），揭示生成表征作为互补信号有效但不可替代原文本。
3. **深度依赖收敛现象的系统揭示**：重排序在 rank-one 上贡献 27.6 个百分点，但在 cutoff=10 时两组设计统计不可区分，DTF 在 cutoff≥20 时以一点优势领先，证明精度与覆盖率由不同机制保障。
4. **关键负结果：重排序不可被更廉价方法替代，而词形还原、伪相关反馈、查询重写均无收益**。
5. **引用准确性与排名精度的解耦发现**：DTF 在 rank-one 落后 20 点，却以 70.3% 引用准确性匹配 oracle 上限，证明下游生成器对"是否存在于上下文"比"排在第几位"更敏感。

## 方法详解
**文档代理（Document Surrogate）**：对每条条文生成语言模型注释，包含四类字段——摘要 $\sigma_a$、主题 $\theta_a$、概念集 $C_a$、假设性问题集 $Q_a$（该条文会回答的问题）。覆盖 22,241 条（占语料库 82,508 条的 27.0%），但基准中全部 264 条可检索参考条文均有代理。

**ASCR（代理级联检索）**：两阶段检索后由语言模型重排序：
- 文档阶段评分法案级别（ pooling 同法案下文条的概念集和摘要）：$s_{doc}(d) = 3h(C_d, \tau(q)) + 2r(\text{title}_d) + 2r(\sigma_d) + \mathbb{1}[\phi_d = f_q] + \text{tier1Bonus}$，保留 top 40 法案。
- 文章阶段评分：$s_{art}(a) = 3h(\theta_a, \tau(q)) + 2h(C_a, \tau(q)) + 20r(Q_a, \tilde{q}) + 2r(\sigma_a) + 1000\mathbb{1}[a \in R_q]$，其中 $R_q$ 为题干中显式引用的条文集合，系数 1000 为硬性覆盖。
- 语言模型对 top 40 文章进行 listwise 重排序。

**ASCR-H（融合密集列表的级联）**：在 ASCR 基础上，将密集检索候选列表通过加权倒数排名融合（RRF）合并后送入重排序器，密集分支与查询分析并行，不增加壁钟时间。

**DTF（确定性三路信号融合）**：三路检索器互补融合，零预生成模型调用：
- 密集检索 $R_{den}$：$\arg\max^D_c \cos(E(q), E(c))$，弥补形态变化和 paraphrase。
- 原始文本 BM25 $R_{txt}$：$\arg\max^D \text{BM25}(q, \xi(a))$，捕捉字面引用。
- 覆盖检索 $R_{cov}$：$\arg\max^D \text{BM25}(q, Q_a)$，弥合问题表述与法条写作的差距。
- RRF 融合：$(w_{den}, w_{txt}, w_{cov}) = (1.0, 0.7, 1.0)$，$k_0 = 20$。
- 确定性重评分：$s(a) = s_{RRF}(a) + \beta\mathbb{1}[d(a) \in \Sigma(q)] + \tau_0\mathbb{1}[\theta_a \cap q] + \gamma\min(3, |C_a \cap q|) - \rho\mathbb{1}[\text{repealed}] + \pi\mathbb{1}[a \in R_q]$，其中 $(\beta, \tau_0, \gamma, \rho, \pi) = (0.35, 0.04, 0.02, 0.15, 1.0)$。
- **法案作用域机制**：题干显式指明管辖法案，提取法案短语的五字符前缀集合 $P(q)$，与法案标题前缀集合 $P(d)$ 匹配：$\Sigma(q) = \{d : |P(q) \cap P(d)| / |P(q)| \geq 0.6\}$，匹配到的法案享有决定性加分 $\beta$。

## 实验与结果
**数据集**：2024–2025 年波兰司法考试（egzamin wstępny na aplikację adwokacką i radcowską）共 300 道单选题，排除 9 道无法解析和 27 道参考条文不在语料库中的题目后，可检索子集 $n^* = 264$（2024 年 133 题，2025 年 131 题）。语料库 1,133 部法案、82,508 条条文、159,434 个嵌入 chunk。

**评估基线**：14 个 lexical/dense/fused 基线（bm25-raw、bm25-lemma、bm25-surrogate、bm25-expanded、dense、dense-rephrased、rrf、dense-surro-rrf、dense-surro-rescore、dense-prf 等）+ 4 个控制组（closed-book、random、oracle、oracle-doc）。

**主要结果（Hit@1）**：
| 配置 | Hit@1* | MRR* | Citation Acc. | G |
|------|---------|------|---------------|---|
| **ASCR-H** | **72.3%** | 0.764 | 67.3% | 0.87 |
| BM25 (best single-call) | 61.7% | 0.677 | 66.0% | 0.81 |
| Dense retrieval | 52.3% | 0.590 | 61.7% | 0.63 |
| **DTF** | 51.9% | 0.618 | **70.3%** (match oracle) | 1.00 |
| Oracle | 100.0% | 1.000 | 70.3% | 1.00 |

- ASCR-H 在 20 组配对 McNemar 检验中以 $p < 0.005$ 显著优于 18 组（唯一不显著的是自身消融 no-concepts，$p = 0.082$）。
- **深度收敛**：ASCR-H 领先至 cutoff=10（83.7% vs 82.2%），cutoff=20 时 DTF 以 86.0% 点估计反超（不显著，$b=6, c=10, p=0.45$）。
- **消融**：移除重排序器代价最大——Hit@1 下降 27.6 个百分点（72.3%→44.7%），超过其他组件贡献之和。
- **负结果**：词形还原（-1.5pp）、伪相关反馈（-6.1pp at H@1）、查询重写（-2.3pp at H@1）均无收益。
- **oracle-doc 危害**：正确法案+随机条文，Hit@1=0.4%，引用准确性 37.7%，显著低于 closed-book（47.0%，$p < 0.001$），证明"正确但错误位置的上下文"有害。
- **成本/延迟**：ASCR-H 中位延迟 7,784ms/$0.0023/问；DTF 中位延迟 820ms/$0.00108/问（九分之一延迟，超 2 倍低成本）。

## 相关工作脉络
1. **BSARD（Louis & Spanakis, 2022）**：比利时法语成文法检索基准，多标签（每问最多 100 条相关条文），评估 Recall@100+；本文是其波兰对应物，规模 3.6 倍，单标签，以 rank-one 为 metric。
2. **PIRB（Dadas et al., 2024）**：波兰多任务检索基准，发现大多数重排序器泛化不佳；但本文重排序器（gemini-3.1-flash-lite 小模型）显著提升性能，原因是本文任务Lexical-over-Dense 的逆转（61.7% vs 52.3%）提供了更强的头部排序信号。
3. **LQuAD-PL（Smywinski-Pohl et al., 2025）**：使用相同考试来源，但候选集仅 3,654 条来自答案键的条文；本文使用完整 82,508 条语料库，难度高 22.6 倍，绝对分数天然更低。
4. **HyDE（Gao et al., 2023）**：从查询端生成假设性文档用于密集检索；本文从文档端生成多字段代理（总结+主题+概念+假设问题），两者位于不同轴，互补而非替代。
5. **Document Expansion（Nogueira et al., 2019）**：在文档端拼接预测查询；本文 bm25-surrogate 以代理字段替换原文（而非拼接），揭示了替代（-15.9pp）与拼接可能存在的性能差异，后者未做实验。
6. **Polish Dense Retriever（Rybak & Ogrodniczuk, 2024）**：在 26,000 段落（非条文粒度）上评估，与本文在 corpora size、检索单元、参考标签来源、评估指标四个维度均不同。

## 局限性与未来方向
1. **代理覆盖率与基准的相关性偏置**：代理覆盖 27.0% 语料但 100% 参考条文， annotation 优先级偏向主要法典和宪法，两者非独立；任何基于代理的配置的 rank-one 优势中，部分可能源于此非对称性而非检索机制本身，需在覆盖匹配子集上重测。
2. **DTF 法案作用域先验未隔离**：$\beta$ 的分值高于融合得分上限，实际上将候选集划分为两个区域而非倾斜排序；未在 $\beta=0$ 条件下做消融，无法分离融合与作用域的各自贡献。
3. **题目格式过友好**：多选题、直接引用法条语言、题干明确标注管辖法案，三项特性使任务比 practitioner 查询容易得多，报道的绝对数字应视为上界。
4. **潜在训练数据污染**：closed-book 准确率 92.7% 远高于 Karp 等人报告的 60s-70s，无法区分模型更强、考试更容易还是数据污染。
5. **缺乏新编码器和更大样本测试**：multilingual-e5-large 已被更新编码器超越；深度比较的 discordant 对极少（cutoff=20 仅 16 对），统计功效不足。

## 研究启发与可借鉴点
1. **文档端多字段代理是低成本的语义增强手段**：用轻量 LLM 在索引时生成总结/主题/概念集/假设问题，可作为密集检索的强补充信号，尤其适用于领域术语固定、形态复杂的语言（如波兰语七格）。
2. **确定性先验重评分可替代昂贵的模型调用**：DTF 的法案作用域加分（$\beta$）和废止条文惩罚（$\rho$）以确定性规则操作，实现了"硬约束"级别的排序控制，为 cost-quality frontier 的最廉价格点提供了范式。
3. **排名精度 ≠ 下游答案质量**：DTF 以 rank-one 落后 20 点的代价换取 citation accuracy 匹配 oracle，提示团队在评估检索系统时应同时报告 G（achievable band recovery）而非仅 Hit@k，避免被 rank-one 指标误导部署决策。
4. **"正确但错误的上下文"比无上下文更有害**：oracle-doc 的引用准确性（37.7%）显著低于 closed-book（47.0%），证明检索系统需保证"粒度的精确性"，仅匹配到正确大类（法案）而放错具体条文反而会挤出模型的参数知识。
5. **负结果的系统性报告具有方法论价值**：词形还原无效（题目直接引用法条语言）、PRF 损害头部精度（扩大的候选池破坏排序）、查询重写无收益——这些结果共同划定了"何时简单方法足够好"的边界，避免后续研究重复无效尝试。

## 关键术语表
**Document Surrogate（文档代理）**：附加在条文上的语言模型生成注释，包含摘要、主题、概念集和假设性问题，作为原文本的衍生表示参与检索匹配。

**ASCR / ASCR-H / DTF**：本文提出的三种代理检索设计——ASCR 为两阶段代理级联+重排序；ASCR-H 在此基础上融合密集列表；DTF 以三路检索器+确定性重评分完全替代语言模型调用。

**Hit@k***：在可检索子集 $n^*=264$ 上，参考条文出现在 top-k 中的问题比例（带星号指标）；Citation Accuracy 和 Answer Accuracy 在全部 $n=300$ 题目上计算。

**Achievable Band Recovery G(m)**：$G(m) = \frac{\text{cit}(m) - \text{cit}(\text{closed-book})}{\text{cit}(\text{oracle}) - \text{cit}(\text{closed-book})}$，衡量模型回收了多少可达到的引用准确性提升空间（floor=47.0%，ceiling=70.3%）。

**McNemar 配对检验**：在 per-question 二元结果上进行配对显著性检验，两个考试年份的 discordant 单元格分别计算后汇总，报告 discordant 总数 $b, c$ 及 $p$ 值。

**Act Scoping（法案作用域）**：从题干提取管辖法案短语，以五字符前缀匹配法案标题集合 $\Sigma(q)$，匹配到的法案文章享有决定性加分 $\beta$，将候选集划分为作用域内/外两个区域。

**RRF（Reciprocal Rank Fusion）**：加权倒数排名融合，公式 $s_{RRF}(a) = \sum_i \frac{w_i}{k_0 + \text{rank}_{R_i}(a)}$，对不可比分数 scale 不敏感，本文 $k_0=20$，权重手工设定。

## 可复现要素
- **数据集**：波兰司法部公开的 2024–2025 年司法考试题目及官方答案键；语料库为波兰成文法（1,133 部法案，82,508 条条文）。论文声明 benchmark、per-question 输出和配对显著性检验已公开。
- **代码/权重**：论文未明确声明代码开源仓库，但提到 "the benchmark, per-question outputs and paired significance tests are publicly available"。
- **关键超参**：BM25 $k_1=1.2, b=0.75$；RRF $k_0=20$，权重 $(w_{den}, w_{txt}, w_{cov})=(1.0, 0.7, 1.0)$；DTF 重评分系数 $(\beta, \tau_0, \gamma, \rho, \pi)=(0.35, 0.04, 0.02, 0.15, 1.0)$；法案作用域前缀阈值 0.6；ASCR 文档阶段保留 top 40 法案；生成器 temperature=0，context size=10，token budget=6,000。所有超参均为手工设定，未在开发集上调优。
- **模型**：surrogate 生成/查询分析/重排序/答案生成均使用 gemini-3.1-flash-lite；密集编码使用 multilingual-e5-large。
