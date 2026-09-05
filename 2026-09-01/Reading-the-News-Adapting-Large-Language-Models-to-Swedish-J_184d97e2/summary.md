---
title: "Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J"
source: https://arxiv.org/pdf/2608.30609v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:28:57"
field: "低资源语言领域适配"
keywords: ["持续预训练", "低资源语言", "新闻领域适配", "参数高效微调", "灾难性遗忘", "指令向量", "瑞典语NLP"]
innovations: ["构建首个瑞典新闻领域专用CPT方案与BonEval评测基准", "揭示instruct vector仅与LoRA兼容的适配方法依赖现象", "验证经验回放混合策略（代码+英语+瑞典语）对领域适配的关键作用"]
benchmarks: ["BonEval", "EuroEval瑞典语", "SwedishFacts", "SvD SEO Headline", "Aftonbladet Summary"]
---

# 论文速读：Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J

## 一句话总结
本文通过将大规模瑞典新闻语料进行持续预训练（CPT），适配大型语言模型到瑞典新闻编辑领域，并构建专用评测基准 BonEval 验证适配效果；研究发现 CPT 仅在配合 experience replay 时能有效提升生成质量和事实知识，且 LoRA + instruct vector 组合效果最佳。

## 研究问题与动机
- **目标领域稀缺**：新闻领域极少作为 LLM 适配的目标领域，仅有一项先前研究（Yao et al., 2024）；现有新闻 LLM 工作多聚焦高资源语言从训练，对低资源语言如瑞典语不适用。
- **评测资源匮乏**：瑞典语新闻领域仅有 3 个评测任务（lead paragraph、摘要、SEO 标题生成），通用基准（EuroEval）无法捕捉领域内适配增益。
- **灾难性遗忘风险**：纯粹在单一领域语料上 CPT 会导致通用能力退化，需探索有效混合策略。
- **PEFT 方法兼容性待探究**：不同适配方法（FFT vs. LoRA vs. LLaMA Pro）与 instruct vector 的组合效果尚不明确。

## 核心贡献（创新点）
1. **构建 BonCorpus 新闻语料库**：从 Bonnier News 的 21.7M 篇文章中经清洗、去重得到 14.6M 篇高质量瑞典新闻，总计 8.5B tokens，覆盖 1991–2026 年 35 年内容。
2. **提出 BonEval 评测基准**：构建涵盖生成、判别、知识密集型三类共 6 个编辑任务的专用基准，包括标题生成、导语生成、摘要、话题分类、实体识别和问答测验。
3. **系统性 CPT 适配研究**：首次在瑞典新闻领域全面对比不同数据混合策略（经验回放比例）、PEFT 方法（LoRA/LLaMA Pro/FFT）及 instruct vector 的影响。
4. **揭示 IV 兼容性的方法依赖性**：发现 instruct vector 仅与 LoRA 兼容（+3.55 分提升），对 FFT 反而有害，解释了参数空间中模型漂移的差异。
5. **论证针对性评测的必要性**：证明 EuroEval 等通用基准无法反映 CPT 在领域内的真实增益，强调领域专用评测的重要性。

## 方法详解

**数据构建流程（BonCorpus）**：
- 数据来源：Bonnier News 旗下 21.7M 篇瑞典文章（1991–2026），涵盖日报、晚报、财经、地方、生活方式、体育等刊物。
- 预处理：Unicode 规范化、正则表达式去除链接/广告/作者信息；基于文本统计（长度≥20词、去重单词≥17、换行符2–2000个等）、fastText 语言识别（>0.9 阈值）和元数据过滤低质量内容。
- 去重：MinHash LSH（112 哈希函数，相似度阈值 0.25）识别近似重复，Levenshtein 距离<0.2 判定为重复，保留每组连通分量中最长文章。
- 最终规模：14.6M 篇，8.5B tokens（基于 Ministral 分词器）。

**经验回放的混合策略**：
- 从 Common Corpus 采样代码（Code）、英语（English）、瑞典语（Swedish）各占 10% token，形成 8 种混合（B, BC, BE, BS, BCE, BCS, BES, BCES），BCES 为最优混合。
- 混合设计动机：防止纯领域语料导致灾难性遗忘，同时验证通用域文本对编辑任务的影响。

**模型适配方案**：
- 基座模型：Mistral Ministral 3B 和 8B base。
- PEFT 方法：LoRA（rank=256, α=512, dropout=0）与 LLaMA Pro（每 6 层插入 1 扩展层）参数量接近。
- Instruct Vector（IV）：从 Ministral instruct 版本计算参数差值，直接加到 CPT 模型上。
- 训练配置：AdamW，cosine lr schedule，max lr=5e-5，序列长度 16,384，global batch size=64，bfloat16 精度，使用 DeepSpeed ZeRO-2 和 FlashAttention-3。

**BonEval 任务设计**：
- 生成类：Headline（根据正文生成标题）、Lead（生成导语）、Summary（要点式摘要，3–5 条）。
- 判别类：Topic（19 类 IPTC 话题分类）、Entity（中心人物/组织/地点 NER）。
- 知识类：Quiz（15 类多元选择问答，涵盖瑞典文化到全球常识）。
- 评估指标：CHRF3++（生成任务）、MCC（分类/问答）、micro-F₁（NER）。

## 实验与结果

**数据集与基线**：
- 自构基准：BonEval（6 任务，Headline/Lead/Topic/Entity 各 8,192 样本，Summary 100 样本，Quiz 11,960 样本）。
- 补充基准：SvD SEO 标题、Aftonbladet 摘要、SwedishFacts 知识问答。
- 通用基准：EuroEval 瑞典语 7 任务（SweReC, SUC 3.0, ScaLA, MultiWikiQA, SweDN, MMLU, HellaSwag）。
- 对比基线：GPT-SW3 6.7B、Llama SW3 8B、Apertus 8B、Ministral 3 8B（Base/Instruct/Reasoning）。

**主要结果**：

| 模型 | BonEval 平均分 | 关键提升 |
|------|---------------|---------|
| Ministral 3 8B Base | 41.59 | — |
| BonLM FFT | 44.14 | +2.55 |
| BonLM LoRA | 44.47 | +2.88 |
| BonLM LoRA + IV | **46.02** | **+4.43（最佳）** |
| BonLM FFT + IV | 38.58 | -3.01（退化） |

- 3B 实验中 BCES 混合最优（41.48），纯 BonCorpus（B）退化至 36.98，证明经验回放必要性。
- LoRA 在 8B 上全面优于 FFT，且 IV 仅与 LoRA 兼容（LoRA+IV 平均 46.02 vs. FFT+IV 38.58）。
- CPT 显著提升生成任务（Headline +11.41, Lead +5.22, Summary +7.39）和知识任务（Quiz +11.24），但轻微下降判别任务（Topic -3.91, Entity -1.63）。
- EuroEval 结果：BonLM 整体低于 Ministral Base，主要因 MMLU（58.10→30.62）和 HellaSwag（43.68→17.66）严重退化，体现灾难性遗忘。
- 关键发现：CPT 在 native 瑞典语任务（SUC 3.0, SweReC）上保持竞争力，但在翻译/合成任务上显著退步。

## 相关工作脉络

1. **BloombergGPT (Wu et al., 2023) / MediaGPT (Wang et al., 2023)**：面向金融/中文新闻的从零训练，计算成本极高；本文采用 CPT 路线适配低资源场景。
2. **News GPT (Yao et al., 2024)**：唯一此前研究新闻领域适配的工作；本文首次系统研究瑞典语+新闻双维度适配。
3. **GPT-SW3 (Ekgren et al., 2024) / Llama SW3 (AI Sweden, 2024)**：瑞典语基础模型；前者从零训练，后者通过 CPT 适配，本文方法与之形成对比基准。
4. **领域 CPT 研究（金融 Li et al. 2023, 法律 Colombo et al. 2024, 生物 Labrak et al. 2024）**：本文验证 CPT 方法迁移到新闻领域的有效性及特殊挑战。
5. **经验回放方法（Chaudhry et al. 2019; Scialom et al. 2022）**：本文扩展发现代码+英语+瑞典语三元混合最优，超越仅用代码/英语的先前结论。
6. **Instruct Vector (Huang et al. 2024; Ilharco et al. 2023)**：本文揭示 IV 与 FFT 不兼容的新现象，拓展了模型合并方法的应用边界认知。
7. **EuroEval (Nielsen 2023)**：本文指出其无法捕捉领域适配增益，呼吁开发语言/领域专用评测。

## 局限性与未来方向

- **数据/代码/模型不可公开**：因商业和法律原因，无法复现；降低研究可验证性。
- **单一模型家族**：仅用 Ministral 系列，结论外推到其他架构（Llama/Qwen）存在不确定性。
- **基座模型已有较强瑞典语能力**：Ministral 本身瑞典语表现优异，其他模型可能呈现不同趋势。
- **原始预训练数据未公开**：无法评估与 BonCorpus/BonEval 的数据重叠及其影响。
- **缺乏下游应用微调**：仅做 CPT，未探索 headline generation、style correction 等具体编辑场景的微调效果。
- **纯自动评测**：无人工评估，无法断言真实新闻编辑室中的实际效用。
- **未来方向**：下游任务微调（标题生成、风格校正）、与记者合作设计人工评测协议。

## 研究启发与可借鉴点

1. **经验回放的混合策略设计**：除代码/英语外，加入目标语言的通用文本（Swedish from Common Corpus）对知识密集型任务（Quiz）提升显著，可借鉴到低资源语言+垂直领域的 CPT 研究中。
2. **PEFT 与 IV 的兼容性诊断**：LoRA 因参数空间漂移小（相对欧氏距离 0.08 vs. FFT 0.18）而兼容 IV，为其他领域适配中的方法选择提供决策依据。
3. **专用评测基准的重要性**：EuroEval 在 MMLU/HellaSwag 上的退化掩盖了 BonEval 上的显著增益，提醒我们在领域适配中必须构建针对性评测，避免误判。
4. **数据清洗pipeline的可复用设计**：基于文本统计+fastText+MinHash LSH+Levenshtein 的多级清洗去重流程，适用于其他语言/领域的新闻语料构建。
5. **四节点实验设计**：先 3B 快速探索数据混合/PEFT 组合，再 8B 验证最优配方，可在资源受限下高效定位最佳实验配置。

## 关键术语表

**Continued Pre-Training (CPT)**：在预训练模型基础上继续使用目标域语料进行训练，以适配特定领域或语言。

**Experience Replay**：在 CPT 过程中混入一定比例的原始预训练数据（如代码、通用文本），以缓解灾难性遗忘。

**Low-Rank Adaptation (LoRA)**：参数高效微调方法，通过低秩矩阵近似参数更新，限制参数漂移幅度。

**Instruct Vector (IV)**：从指令微调模型与基础模型的参数差值中计算的向量，可无训练地赋予 CPT 模型指令遵循能力。

**BonCorpus**：本文构建的瑞典新闻预训练语料，包含 14.6M 篇文章、8.5B tokens，覆盖 1991–2026 年。

**BonEval**：本文提出的瑞典新闻领域专用评测基准，含 6 个任务（标题生成、导语生成、摘要、话题分类、实体识别、知识问答）。

**Catastrophic Forgetting**：模型在新领域训练过程中遗忘原有通用能力的现象。

**MinHash LSH**：基于 MinHash 签名和局部敏感哈希的高效近似去重算法。

## 可复现要素

- **数据集**：BonCorpus 和 BonEval **不可公开**（论文明确声明因商业和法律原因）；Common Corpus 为开源可用。
- **代码**：**未开源**（论文声明不提供）。
- **模型权重**：**未开源**（仅限 Bonnier News 内部使用）。
- **关键超参**：LoRA rank=256, α=512, dropout=0；max learning rate=5e-5, global batch size=64, sequence length=16,384, cosine schedule with 10% warmup, weight decay=0.1, gradient clipping=1.0；训练 8,000 steps（3B 约 1 epoch）/ 4 epochs（8B）。
