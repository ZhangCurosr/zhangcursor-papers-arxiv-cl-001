---
title: "Inspicio-Open-Vocabulary-LLM-Based-Sense-Retrieval-for-Histo"
source: https://arxiv.org/pdf/2609.00998v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:19:20"
field: "历史语言计算语义学与词义消歧"
keywords: ["开放词汇词义消歧", "历史语言", "LLM检索", "WordNet", "混合检索", "MMR重排"]
innovations: ["零样本开放词汇检索将历史语言token直接链接到OEWN", "翻译驱动的定义与lemma联合生成支持密集-稀疏混合检索", "定义排行衰减与MMR重排提升Top-K多样性与可解释性"]
benchmarks: ["Perception-Verb 150-token bilingual set", "PREMOVE", "Diachronic Italian MIDIA sample"]
---

# 论文速读：Inspicio-Open-Vocabulary-LLM-Based-Sense-Retrieval-for-Historical-Languages

## 一句话总结
本文提出 INSPICIO，一种基于 LLM 驱动的开放词汇词义消歧检索管道，无需源语言词义清单即可将拉丁语、古希腊语等历史语言 token 链接至 Open English WordNet (OEWN) synset，在感知动词测试集上达到 96% Recall@50，并在跨域与跨语言场景下保持竞争力。

## 研究问题与动机
- 经典 WSD 高度依赖源语言 sense inventory 与 word-to-sense mapping，而历史/低资源语言普遍缺乏完备 WordNet 或映射资源。
- 手动构建词网成本高昂，自动迁移方法常继承源语言覆盖缺口与噪声。
- 现有检索式词义链接系统在主映射缺失时会显著退化，难以直接用于历史语言。
- 需要一种零样本、语言无关的开放词汇方案：既利用英语成熟词汇语义资源，又不在源语言侧引入先验清单。

## 核心贡献（创新点）
- **零样本开放词汇检索框架**：不依赖源语言 sense 清单或词到义的映射，直接以 OEWN 作为目标检索空间。
- **翻译驱动的密集-稀疏混合检索**：由 LLM 生成字面/自然双译句、候选定义与候选 lemma，分别驱动密集定义相似度与稀疏 lemma 倒排检索。
- **定义排行衰减加权融合**：结合 LLM 生成的定义顺序加权聚合相似度，避免高置信定义被低置信结果稀释。
- **MMR 多样化重排**：缓解 OEWN 细粒度导致的近义重复聚集，提升 Top-K 多样性与可解释性。
- **多语言多数据集验证**：在自建拉丁/古希腊感知动词集、PREMOVE 及跨语言历时意大利语上评估，并给出成分消融。

## 方法详解
- **输入与流程**：针对单个 token，给定目标词、字典 lemma、所在句子与语言标签，返回排序后的 OEWN synset 列表及中间产物。
- **翻译与假设生成（零样本）**：
  - 第一次调用 LLM 生成两版英译：literal（贴近原文结构）与 natural（地道现代英语），温度 T=1.0，用于获取风格与词汇变体。
  - 第二次调用提供原句、目标 token、lemma 与两版翻译，输出 1-3 条按置信度排序的英语词典式定义，以及 1-5 个候选英语 lemma 或多词表达；温度 T=0.8，输出 JSON。
  - 提示词明确要求区分动词自身语义与论元语义、处理否定语境、同时覆盖字面与隐喻用法。
- **密集检索（定义到 synset）**：
  - 按词类构建 OEWN 四个索引，实验主要使用动词分区；每条 synset 文档拼接 lemmas、gloss、examples、hypernyms、lexname。
  - 各定义嵌入后在 ChromaDB 中检索 Top N=50 nearest synset。
- **定义排行衰减得分**：
  - 公式：
    - $S_{base}(s) = \sum_{d=1}^{D} w_d \cdot \cos(e_d, e_s)$
    - 权重 $w = (1.0, 0.75, 0.5)$ 反映定义顺序置信度衰减。
- **稀疏 lemma 检索**：
  - 候选 lemma 查询预计算倒排索引，返回包含该 lemma 的 synset 集合；用于捕获嵌入空间中语义较远但词汇对齐明确的 sense。
- **融合与最终打分**：
  - 公式：
    - 若 $S_{base}(s) > 0$：$S_{final}(s) = S_{base}(s) \cdot (1 + \gamma \cdot 1[s \in \mathcal{L}])$
    - 若 $S_{base}(s) = 0$ 且 $s \in \mathcal{L}$：$S_{final}(s) = \beta$
    - $\gamma = 0.8$，$\beta = 0.65$
  - 合并后截取 Top 500 进入下一步。
- **MMR 多样化重排**：
  - 公式：
    - $MMR(s) = \lambda \cdot S_{final}(s) - (1-\lambda) \cdot \max_{s' \in S} \cos(e_s, e_{s'})$
    - $\lambda = 0.8$，最终输出 Top K=50。
- **可审计输出**：每 token 输出 JSON，包含翻译、定义、候选 lemma、top-K synset 得分与贡献来源，便于字典编者回溯与修正。

## 实验与结果
- **数据集**：
  - 自建双语感知动词测试集：150 token（拉丁 72、古希腊 78），标注一致性 $\kappa = 0.895$。
  - PREMOVE：约 2800 token 的历时拉丁/古希腊前缀运动动词，sense 编码为 OEWN synset。
  - 历时意大利语：100 token，来自 MIDIA，标注一致性 $\kappa = 0.914$。
- **模型设置**：6 种指令微调 LLM × 6 种 sentence embedding；以 Recall@k 评估，headline 指标为 Recall@50。
- **主结果（感知动词）**：
  - 最优配置 DeepSeek V4 Pro + KaLM-Embedding-Gemma3-12B-2511 达到 **96.00% Recall@50**、Recall@20 = 90.67%。
  - KaLM 在 6 个 LLM 中 5 个最优；DeepSeek V4 Pro 在 6 个 embedding 中 4 个最优。
- **PREMOVE**：Recall@50 = **81.65%**、Recall@20 = 74.91%，较感知动词下降约 15 点，反映更广语义分布与历时隐喻漂移。
- **历时意大利语**：Recall@50 = **91.00%**、Recall@20 = 84.00%，显示英语枢纽策略在源语受训练数据充分覆盖时仍有效。
- **消融（最佳配置，感知动词）**：
  - 去除翻译阶段降至 92.00%，说明双译提供额外信息；但单步可节省推理成本。
  - 去除 lemma boost 降至 94.00%，证明密集与稀疏互补。
  - 去除定义排行衰减与去除 MMR 对聚合 recall 影响不明显；MMR 保留以提升 Top-K 多样性。

## 相关工作脉络
- **Bejgu et al., 2024 (Word Sense Linking)**：在开放词汇设定下将 span 链接到参考词表；但主映射缺失时性能显著退化，本文以英语枢纽消除源语言映射依赖。
- **Bevilacqua et al., 2020 / Meconi et al., 2025**：生成式/自由释义路线表明 LLM 在开放定义任务上更强；本文借鉴生成假设思路，但以 OEWN synset 作为可检索的稳定目标空间。
- **Bamman & Burns, 2020 / Lendvai & Wick, 2022**：历史拉丁语上下文化的监督/微调路线，依赖本地词典与标注；本文完全零样本、不依赖源语言标注。
- **Ghinassi et al., 2024 / Ghizzota et al., 2025**：自动传播标注与 fine-tuned 中等规模 LLM 优于大规模 zero-shot；本文定位为无标注资源时的候选池生成器。
- **Farina & Ciletti, 2025 / Farina et al., 2026**：拉丁/古希腊前缀运动动词与地理名词的 LLM 标注研究；本文与 PREMOVE 同源目标资源并扩展至感知动词与意大利语。
- **Marchesi et al., 2025 / Santoro et al., 2025**：利用 LLM 辅助填充古代希腊语/拉丁语 WordNet；本文可作为银标生成与词网启动资源的基础检索管线。

## 局限性与未来方向
- 当前评估集中在动词；虽然架构支持多词类且有四类索引，但仍需名词、形容词、副词的实证检验。
- 两阶段 LLM 生成带来随机性与误差传播风险，翻译阶段的高温度设计进一步增加运行间波动。
- 英语枢纽策略使 OEWN 覆盖受当代英语词汇化模式限制，部分源语言 sense 无完美对应，只能返回最接近 synset。
- 未来可引入 LLM 裁决式 reranker，由候选池收敛到最终 1-2 个 sense；同时可探索银标传播、词网自举与 agentic 检索场景。
- 在具备标注数据的语言上可探索 SFT 以提升理解与稳定性。

## 研究启发与可借鉴点
- **双译句+定义/lemma联合生成**可同时捕获字面与习惯用法，适合作为开放词汇检索的标准前置；其产出可直接用于构建可解释候选池。
- **密集-稀疏混合融合策略**（定义相似度 + lemma 倒排 + 加权 boost）在跨语言/历史文本中具有较强鲁棒性，可迁移到多语言词汇语义任务。
- **MMR 多样化重排**对细粒度词表（如 WordNet）非常实用，有助于降低 top-K 重复聚集并提升人工审核效率。
- **审计日志机制**可复用到下游词网构建与银标生成：翻译、定义、lemma 与得分均可追溯，便于错误定位与提示迭代。
- **以英语枢纽衔接历史语言**的思路可用于资源匮乏语言的快速评测与原型系统搭建；后续可向本地词网自举演进。

## 关键术语表
- **Open-Vocabulary WSD**：不依赖源语言预定义 sense 清单的词义消歧，强调开放目标词表或开放假设生成。
- **OEWN**：Open English WordNet，本文用作跨语言检索的目标词汇语义网络。
- **Synset**：同义集合，表示一组具有相同词义的词语单元及其 gloss/examples 等元数据。
- **Recall@K**：在 Top-K 候选中包含金标准 synset 的比例，本文 headline 指标取 K=50。
- **MMR**：Maximal Marginal Relevance，兼顾相关性并引入多样性的重排策略。
- **PREMOVE**：覆盖公元前 8 世纪至公元 2 世纪拉丁与古希腊前缀运动动词的历时标注数据集。
- **定义排行衰减**：按 LLM 生成定义的置信顺序分配递减权重，避免低置信定义稀释高分 synset。
- **银标 (silver-standard) 标注**：由模型自动生成、经人工抽检或下游流程校对的标注数据。

## 可复现要素
- 数据集：自建拉丁/古希腊感知动词测试集、PREMOVE、历时意大利语样本。
- 代码/权重：论文声明代码、提示词与评估数据已发布在 GitHub。
- LLM 清单：DeepSeek V3.2、DeepSeek V4 Pro、Kimi K2.6、GLM 5.1、Qwen 3.5 397B A17B、Mistral Medium 3.5。
- Embedding 清单：text-embedding-3-large、KaLM-Embedding-Gemma3-12B-2511、Qwen3-Embedding-8B、Cohere Embed v4、Harrier-OSS-27B、jina-embeddings-v5-text-small。
- 关键超参：翻译温度 T=1.0；定义/lemma 温度 T=0.8；N=50；$\gamma=0.8$；$\beta=0.65$；MMR $\lambda=0.8$；Top-K=50。
- 其他：ChromaDB 存储 L2 归一化嵌入；OEWN 按词类分索引；prompt 见附录 A。
