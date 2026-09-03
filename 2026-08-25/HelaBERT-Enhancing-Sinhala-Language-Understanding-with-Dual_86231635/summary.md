---
title: "HelaBERT-Enhancing-Sinhala-Language-Understanding-with-Dual"
source: https://arxiv.org/pdf/2608.22922v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:20:41"
field: "低资源语言预训练模型"
keywords: ["僧伽罗语 NLP", "BERT 预训练", "单语语言模型", "双池化分类头", "低资源语言", "Sentiment Analysis", "Text Classification"]
innovations: ["从零预训练僧伽罗语 BERT 模型 HelaBERT-Small/Large，使用约 1B tokens 单语文料与定制 Unigram 分词器", "提出双池化（dual pooling）分类头，通过 [CLS] 与 token 序列的双向注意力交互提升分类性能", "系统分析 co-attention 分类头在不同序列长度、模型规模和任务难度下的适用边界"]
benchmarks: ["News Category Classification", "News Source Classification", "Sentiment Analysis (3-class)", "Writing Style Classification"]
---

# 论文速读：HelaBERT-Enhancing-Sinhala-Language-Understanding-with-Dual

## 一句话总结
本文从零预训练了两个 BERT 系单语模型 **HelaBERT-Small**（~23.3M）和 **HelaBERT-Large**（~110M），使用约 10 亿词元的僧伽罗语语料和专为黏着形态设计的 SentencePiece Unigram 分词器；同时在标准 [CLS]-linear 分类头之外，提出了**双池化（dual pooling）分类头**，在情感分析上带来 +3.9–5.6 macro-F₁ 的稳定提升。

## 研究问题与动机
- **僧伽罗语 NLP 资源极度匮乏**：使用该语言的约 1700 万人，但预训练模型、标注语料和基准benchmark都稀缺。
- **多语模型的容量稀释**：mBERT、XLM-R 虽覆盖僧伽罗语，但其在低资源语言上的表征能力有限；专用单语模型一贯优于多语模型（SinBERT 等已验证该结论）。
- **现有僧伽罗语模型 Backbone/分词器局限**：SinBERT 基于 RoBERTa + BPE，而僧伽罗语具有复杂的abugida书写系统和黏着形态，需要更贴合的分词策略。
- **分类头设计未充分利用序列表示**：标准 [CLS]-linear 仅依赖单一token的汇总表示，忽略了 [CLS] 与完整 token 序列之间的双向交互。

## 核心贡献（创新点）
1. **两个从零预训练的僧伽罗语 BERT 模型**：HelaBERT-Small/Large，使用约 10 亿词元单语语料 + 定制的 SentencePiece Unigram 分词器；与 SinBERT 的区别在于 backbone（BERT vs RoBERTa）、分词器（Unigram vs BPE）和预训练语料规模（~1B vs ~192M tokens）。
2. **系统化的下游评估基准**：在四个分类任务（新闻类别、新闻源、情感、写作风格）上使用 5 次独立 seed 运行的 stratified 80/20 划分，与 SinBERT 等基线可直接比较。
3. **双池化（dual pooling）分类头**：让 [CLS] token 与完整 token 序列在同一输入内双向注意力交互；本质区别在于超越了标准 [CLS]-linear 的"单一汇总表示"假设，显式建模全局-局部信息互补。
4. **细致的任务-架构匹配分析**：证明 co-attention 在情感分析（+3.9–5.6 pp）和较小模型的新闻类别（+3.1 pp）上增益显著，而在短输入（新闻源，平均 8 tokens）或天花板任务（写作风格，>95%）上标准头仍具竞争力。

## 方法详解
**预训练**
- 语料：MADLAD-400（僧伽罗子集）、CulturaX（僧伽罗子集）、自定义语料（僧伽罗 Wikipedia + 新闻 + 网络爬取）。
- 预处理：NFC 归一化、去除不可见 Unicode 字符（保留 ZWJ/ZWNJ 用于连字渲染）、丢弃不含僧伽罗脚本或少于 5 字符的行、非僧伽罗字符剥离（保留 ASCII 数字/标点）、重复标点/多余空白/不匹配括号/日期数字模式归一化。
- 分词器：SentencePiece Unigram，词表 32,000，字符覆盖率 99.95%，从零训练（非从多语词表迁移）。
- 架构：标准 BERT encoder-only，MLM 目标（15% token 被 mask：80% [MASK]、10% 随机替换、10% 保持不变），GELU 激活，绝对位置编码，dropout 0.1。
- Small（6 层，hidden 384，heads 6）训练 2 epoch / ~52K 步；Large（12 层，hidden 768，heads 12）训练 6 epoch / ~90.7K 步。

**微调**
- 4 个下游任务：新闻类别（5类，平均 23.8 词）、新闻源（9类，平均 8.3 词）、情感（3类 POS/NEG/NEU，平均 16.8 词）、写作风格（4类 NEWS/ACADEMIC/CREATIVE/BLOG，平均 182.6 词）。
- 5 次独立 seed（42/123/456/789/1024），stratified 80/20 划分，macro-F₁ 为主指标，取最好一次结果。
- AdamW + linear LR schedule（6% warmup）+ weight decay 0.01 + batch 16 + FP16。

**双池化分类头（Dual Pooling / Co-attention Head）**
- 输入：编码器输出中 [CLS] 向量 $\mathbf{c} \in \mathbb{R}^H$，以及其余 token 表示 $\mathbf{T} \in \mathbb{R}^{T \times H}$，及 padding mask $\mathbf{m} \in \{0,1\}^T$。
- 共享亲和分数（additive attention）：
  $$a_i = \frac{1}{\sqrt{H}} \mathbf{v}^\top \tanh(\mathbf{W}_c \mathbf{c} + \mathbf{W}_t \mathbf{h}_i)$$
- 方向 1（[CLS] attend tokens）：softmax 后加权求和得到 $\tilde{\mathbf{c}} = \boldsymbol{\alpha} \mathbf{T}$，padding 位置填 $-10^4$ 防 FP16 溢出。
- 方向 2（tokens attend back to [CLS]）：sigmoid 门控 $\boldsymbol{\beta} = \sigma(\mathbf{a}) \odot \mathbf{m}$，归一化后得到 $\tilde{\mathbf{T}} = \hat{\boldsymbol{\beta}} \mathbf{T}$。
- 分类层：$[\mathrm{LN}(\tilde{\mathbf{c}}); \mathrm{LN}(\tilde{\mathbf{T}})] \in \mathbb{R}^{2H} \xrightarrow{\mathrm{Linear}} \xrightarrow{\mathrm{Dropout+GELU}} \xrightarrow{\mathrm{Linear}_{H \to C}} \hat{y}$。
- 额外参数约 $4H^2$：Small ≈ 590K，Large ≈ 2.4M。

## 实验与结果
| 任务 | 最强结果（模型） | 关键提升 |
|---|---|---|
| 新闻类别 | HelaBERT-Large = **90.38%** macro-F₁ | 超 XLM-R-large（89.54%）0.8 pp；超 SinBERT-Large（85.19%）5.2 pp |
| 新闻源 | HelaBERT-Large = **63.65%** | 超 SinBERT-Large（60.51%）3.1 pp；最挑战性任务（风格重叠） |
| 写作风格 | XLM-R-large = 98.41%（最高）；HelaBERT-Large = **97.73%** | 超 SinBERT-Large（95.49%）2.2 pp；接近天花板 |
| 情感（3类） | HelaBERT-Large（标准头）= **64.60%** | —（注：基线用不同 4 类数据集，不可直接比较） |

**双池化头的增量效果**（相较于同模型标准 [CLS]-linear 头）：
- 情感：**Small +3.93 pp（65.34→69.27）**；Large **+5.62 pp（64.60→70.22）**，最大受益任务。
- 新闻类别：Small **+3.05 pp（85.97→89.02）**；Large +0.12 pp（提升有限，因 H=768 时 [CLS] 已具足够表征力）。
- 新闻源：Small −0.26 pp；Large −0.34 pp（平均 8.3 词太短，key set 稀疏，且 3 epoch 不足以收敛额外 2.4M 参数）。
- 写作风格：Small +0.50 pp；Large +0.02 pp（已在 >95% 天花板，几乎无提升空间）。

## 相关工作脉络
1. **mBERT / XLM-R**：多语预训练模型，覆盖广但低资源语言容量稀释；HelaBERT 通过单语密集预训练 + 更大语料（~1B vs XLM-R 中僧伽罗语仅 ~0.15%）实现超越。
2. **SinBERT**（Dhananjaya et al., 2022）：首个僧伽罗语 RoBERTa 单语模型，使用 ~192M tokens + BPE；HelaBERT 在其同一基准上竞争，使用 BERT backbone + Unigram tokenizer + ~5× 更大语料，全面超越 SinBERT。
3. **SinLlama**（Aravinda et al., 2025）：基于 Llama-3-8B 的生成式僧伽罗语 LLM；HelaBERT 定位互补——轻量级编码器用于分类/序列标注 vs 生成式模型用于指令跟随。
4. **AraBERT / CamemBERT / PhoBERT / IndicBERT**：其他语言的单语 BERT 类模型，印证"单语专用 > 多语通用"的普遍规律；HelaBERT 沿此路线为僧伽罗语建立同等基线。
5. **早期僧伽罗语分类工作**（Naïve Bayes、SVM、Word2Vec/fastText、LDA 主题分层等）：建立在传统特征工程之上；HelaBERT 代表 Transformer 时代的端到端预训练范式升级。

## 局限性与未来方向
- **仅支持单语**：无法做跨语言迁移。
- **分词器兼容性**：SentencePiece 需手动加载 `sentencepiece` 库，不能直接用 HuggingFace `AutoTokenizer`。
- **长序列未充分测试**：Small 仅在 256 token 窗口内预训练，>256 token 性能未知。
- **语料偏差**：网页爬取语料可能残留噪声，且偏向正式书面僧伽罗语，方言和口语变体覆盖不足。
- **下游任务局限**：仅评估文本分类；序列标注（NER、POS）、问答、生成任务尚未覆盖。
- **情感数据集不一致**：HelaBERT 的情感评估使用公开 3 类数据集，与前人 4 类（含 CONFLICT 标签）数据集不统一，难以直接对比。

## 研究启发与可借鉴点
1. **双池化分类头可作为通用增强模块**：在情感分析、细粒度分类等依赖少量判别性 token 的任务上，替换标准 [CLS]-linear 可带来显著增益（+3.9–5.6 pp），尤其对中等规模模型（H=384~768）效果最佳。
2. **"模型规模 × 分类头复杂度"的匹配原则**：当 base model 的 [CLS] 已足够强（Large、长序列、天花板任务）时，标准头即可；容量有限或序列中等长度时，co-attention 才能发挥补充作用——这一判断框架可迁移至其他语言的模型选型。
3. **句子级任务 vs 短文本任务的头设计差异**：对平均 <10 token 的输入（如新闻标题分类），额外参数的收敛成本可能超过收益，此时标准头更经济。
4. **从_scratch_ 训练语言定制分词器**：Unigram vs BPE 在黏着语上的优势值得在低资源语言复现；character coverage 99.95% 可作为分词器质量的一个量化参考。
5. **环境足迹透明报告**：论文详细报告了 Small（0.29 kg CO₂eq）和 Large（6.09 kg CO₂eq）的训练碳排放，可作为绿色 AI 研究的示范范式。

## 关键术语表
- **HelaBERT**：本文提出的两个从零预训练的僧伽罗语 BERT 系语言模型家族（Small/Large）。
- **SentencePiece Unigram tokenizer**：一种无词边界假设的子词分词器，适合黏着语形态；本文词表 32K，字符覆盖率 99.95%。
- **Dual Pooling / Co-attention Classification Head**：双向注意力分类头，同时让 [CLS] attend tokens 和 tokens attend back to [CLS]，拼接后经 MLP 输出类别。
- **Macro-F₁**：每个类别的 F₁ 取算术平均，对类别不平衡鲁棒，是本文主评估指标。
- **MADLAD-400 / CulturaX**：两个大规模多语文档/网页语料库的僧伽罗语子集，构成 HelaBERT 预训练数据的主要来源。
- **SinBERT**：Dhananjaya et al. (2022) 提出的僧伽罗语 RoBERTa 单语模型，是本文的主要对照基线。
- **MLM（Masked Language Modeling）**：BERT 预训练目标，随机 mask 15% token 进行预测。
- **Stratified 80/20 split**：按类别比例划分的训练/测试集，确保每次 seed 运行的数据分布一致。

## 可复现要素
- **数据集**：四个下游分类数据集均来自 Dhananjaya et al. (2022) 公开数据集；情感 3 类数据集论文注明为 "publicly available"；具体链接在论文脚注中引用（如 [2][3][4][5]），未在当前 PDF 文本中给出直接 URL。
- **代码**：预训练代码开源（匿名链接：https://anonymous.4open.science/r/HelaBERT-Train），使用 Claude Code（Anthropic）辅助开发。
- **模型权重/分词器**：论文声明 "Both models and the SentencePiece Unigram tokenizer are released publicly"（已公开释放）。
- **关键超参**：见原文 Table 4（微调超参）和 Appendix Table 9（预训练超参）。预训练有效 batch size 均为 256，LR = 1e−4，Cosine LR schedule；微调 LR 因任务和模型大小而异（1e-5 至 5e-5）。
