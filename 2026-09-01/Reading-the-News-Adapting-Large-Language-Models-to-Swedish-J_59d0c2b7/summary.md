---
title: "Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J"
source: https://arxiv.org/pdf/2608.30609v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:28:57"
field: "低资源语言大模型适配"
keywords: ["持续预训练", "低资源语言", "瑞典语", "新闻领域适配", "经验回放", "LoRA", "指令向量", "领域评估基准"]
innovations: ["构建首个瑞典新闻CPT适配研究，提出BonCorpus语料库与BonEval评估基准", "发现经验回放混合策略对缓解灾难性遗忘的关键作用，BCES组合最优", "揭示LoRA与指令向量的兼容性优势，LoRA+IV在新闻生成与知识任务上取得最佳性能"]
benchmarks: ["BonEval", "EuroEval (Swedish)", "Svenska Dagbladet SEO Headline", "Aftonbladet Summary", "SwedishFacts"]
---

# 论文速读：Reading-the-News-Adapting-Large-Language-Models-to-Swedish-J

## 一句话总结
本文通过持续预训练（CPT）将大语言模型适配到瑞典新闻领域，构建了高质量新闻语料库 BonCorpus 与专项评估基准 BonEval，证明结合经验回放（experience replay）的数据混合策略可有效提升生成质量与事实知识，但全参数微调会损害判别性任务性能，而 LoRA + 指令向量（IV）组合效果最佳。

## 研究问题与动机
- **核心问题**：现有 LLM 在低资源语言（如瑞典语）及特定专业领域（新闻业）的能力较弱，而从零预训练门槛过高。
- **现有方法不足**：新闻领域极少作为 LLM 适配目标域；瑞典语仅有少量小模型 CPT 研究，缺乏系统性方案与针对性评估资源。
- **评估缺失**：现有瑞典基准（EuroEval 等）无法充分反映领域内生成能力与事实知识的提升，需构建专项评测。
- **技术空白**：CPT 对生成、判别、知识三类任务的影响差异，以及 PEFT 方法与指令向量的兼容性尚未在瑞典新闻场景中厘清。

## 核心贡献（创新点）
1. **首个面向瑞典新闻的 CPT 适配研究**：构建了 35 年跨度、1460 万篇高质量瑞典新闻语料库 BonCorpus（85 亿 tokens），填补了新闻领域 + 瑞典语双重空白。
2. **提出专项评估基准 BonEval**：覆盖标题生成、导语生成、摘要、主题分类、实体识别、问答共 6 个编辑任务，揭示现有 EuroEval 基准对领域增益的低敏感度。
3. **揭示数据混合的关键作用**：证明纯领域语料（BonCorpus only）会导致灾难性遗忘，加入代码、英文、瑞典通用文本的混合能显著提升性能，其中 BCES 组合最优。
4. **发现 PEFT 与指令向量的兼容性规律**：LoRA 相比全参数微调（FFT）更好地保留判别性任务能力，且 LoRA + IV 在 BonEval 上取得最高分（46.02），而 FFT + IV 反而导致性能下降。
5. **提供可复用的领域适配蓝图**：为低资源语言 + 垂直领域的 LLM 适配提供了从数据清洗、评估构建到训练策略的完整方法论。

## 方法详解
- **数据清洗管线**：对 2170 万篇原始文章进行 Unicode 标准化、正则过滤（去除链接、广告、作者署名）、fastText 语言识别（阈值 >0.9）、基于文本统计的质量过滤（字数、换行符、类型-词元比等）、MinHash LSH 去重（相似度阈值 0.25，Levenshtein 距离阈值 0.2），最终获得 1460 万篇文章。
- **经验回放混合策略**：从 Common Corpus 采样代码、英文、瑞典语通用文本，每种占 10% tokens，构建 8 种混合比例（B/BC/BE/BS/BCE/BCS/BES/BCES），基线为纯 BonCorpus。
- **持续预训练配置**：基于 Ministral 3 系列（3B/8B），AdamW 优化器，余弦学习率调度（最大 5e-5， warmup 10%），序列长度 16384，全局 batch size 64，bfloat16 精度，使用 DeepSpeed ZeRO-2 + FlashAttention-3。
- **PEFT 方法对比**：LoRA（rank=256，α=512）与 LLaMA Pro（每 6 层插入额外层）在 3B 模型上训练参数量相近（395M vs 466M）。
- **指令向量（IV）注入**：计算预训练版与指令微调版ministral 8B 的参数差向量，直接加到 CPT 后模型权重上，无需额外训练。

## 实验与结果
- **数据集**：BonCorpus（14.6M 篇瑞典新闻，8.5B tokens）；BonEval（6 任务，Headline/Lead/Topic/Entity 各 8192 样本，Summary 100 样本，Quiz 11960 样本）；补充任务包括 Svenska Dagbladet SEO 标题、Aftonbladet 摘要、SwedishFacts 问答。
- **评估基线**：GPT-SW3 6.7B、Llama SW3 8B、Apertus 8B、Ministral 3 8B（Base/Instruct/Reasoning）。
- **核心结果**：
  - 3B 实验中 BCES 混合表现最优（41.48），纯 BonCorpus（B）退化最严重（36.98）。
  - 8B 模型中 BonLM LoRA + IV 在 BonEval 上取得最高平均分 46.02，较 Ministral 8B Base（41.59）提升 4.43；在生成任务（Headline/Lead/Summary）和 Quiz 上均显著提升。
  - FFT + IV 在 BonEval 上仅 38.58，低于 LoRA + IV。
  - 在 EuroEval 瑞典任务上，CPT 导致 MMLU、HellaSwag、MultiWikiQA 等泛化能力下降，但原生瑞典语任务（SweReC、SUC 3.0）保持或略有提升。
- **关键结论**：CPT 显著提升生成质量与领域知识，但会损害判别性任务与跨语言泛化；LoRA 相比 FFT 更利于保留判别能力，且与 IV 兼容；专用基准对评估领域适配效果至关重要。

## 相关工作脉络
- **BloombergGPT / MediaGPT**：从零预训练金融/中文新闻 LLM，计算成本高，适用于高资源语言；本文聚焦 CPT 适配路径。
- **Yao et al. (2024) News GPT**：仅有一项工作研究新闻领域 LLM 适配，但针对中文且基于检索增强生成；本文是瑞典语新闻的首个系统性 CPT 研究。
- **GPT-SW3 / Llama SW3 / Apertus**：瑞典语/北欧语基础模型，多为从头预训练或通用 CPT；本文针对新闻垂直领域进行深度适配并构建专项评估。
- **Experience Replay 相关工作**（Chaudhry et al., 2019; Scialom et al., 2022）：本文验证在新闻领域 CPT 中经验回放的必要性，纯领域语料会导致灾难性遗忘。
- **LoRA / LLaMA Pro**：PEFT 方法；本文发现 LoRA 在保留判别能力方面优于 LLaMA Pro，且与 IV 兼容，而 LLaMA Pro 不适合 IV 注入。
- **EuroEval / Superlim**：现有瑞典语评估基准；本文指出这些基准对领域内生成与知识增益不敏感，凸显构建专项基准的必要性。

## 局限性与未来方向
- **数据与代码未公开**：因商业与法律原因，无法开源语料、代码与模型权重，降低可复现性。
- **仅测试单一模型族**：计算资源限制使实验仅基于 Ministral 3，其他模型族可能得出不同结论。
- **基座模型已较强**：Ministral 已有较好瑞典语能力，在其他基座上 CPT 增益可能不同。
- **预训练数据未知**：无法评估 BonCorpus/BonEval 与基座模型预训练数据的重叠及其影响。
- **缺乏下游任务微调**：未将 BonLM 微调至具体编辑应用（如标题生成、文体校正），实际应用价值待验证。
- **纯自动评估**：无人工评估，无法断言真实新闻室效用；计划未来与记者合作设计人类评估协议。

## 研究启发与可借鉴点
1. **经验回放在领域 CPT 中不可或缺**：纯领域语料训练易导致灾难性遗忘，混合 10% 通用文本（代码/英文/母语）可显著提升领域性能，适用于其他垂直领域适配。
2. **PEFT 方法与指令向量的兼容性需单独验证**：LoRA + IV 效果优异，但 FFT + IV 反而有害；在引入指令向量前应先测试其与适配方法的兼容性。
3. **专项评估基准的价值**：EuroEval 未能反映 CPT 在新闻生成与知识上的提升；未来研究应构建领域内评估以准确捕捉适配效果。
4. **新闻语料清洗的可复用管线**：多层过滤（格式标准化 → 文本统计 → 语言识别 → 去重）策略可迁移至其他语言/领域的新闻数据构建。
5. **生成 vs 判别任务的差异化影响**：CPT 一致提升生成与知识任务但损害判别任务；可借鉴此观察指导后续研究的任务选择与指标设计。

## 关键术语表
- **Continued Pre-Training (CPT)**：在预训练模型基础上继续使用新领域语料进行训练，以适配特定领域或语言。
- **Experience Replay**：在 CPT 数据中混入少量原始训练分布样本，以缓解灾难性遗忘。
- **Parameter-Efficient Fine-Tuning (PEFT)**：仅更新模型部分参数（如 LoRA 的低秩矩阵）以实现高效适配。
- **Instruct Vector (IV)**：通过指令微调模型与预训练模型的参数差向量，以零训练代价恢复或增强指令遵循能力。
- **BonCorpus**：本文构建的瑞典新闻语料库，含 1460 万篇文章、85 亿 tokens，时间跨度 1991–2026。
- **BonEval**：本文提出的瑞典新闻专项评估基准，包含标题生成、导语生成、摘要、主题分类、实体识别、问答 6 个任务。
- **LoRA**：Low-Rank Adaptation，通过在 Transformer 模块中注入低秩分解矩阵实现高效参数更新。
- **LLaMA Pro**：一种模型扩层方法，在冻结的预训练骨干网络中插入新层以扩大模型容量。

## 可复现要素
- **数据集**：BonCorpus 因商业与法律原因**未公开**；BonEval 也未公开；补充任务数据来源为公开瑞典新闻网站（SvD、AB 等）。
- **代码/权重**：**未开源**。
- **关键超参**：AdamW（β1=0.9/0.95，β2=1e-8，weight decay=0.1，cosine schedule，max lr=5e-5，warmup=10%，min lr=5e-6，gradient clip=1.0），序列长度 16384，batch size 64，bfloat16，DeepSpeed ZeRO-2，LoRA rank=256，α=512，dropout=0.0，训练步数约 1 epoch（3B）至 4 epoch（8B）。
