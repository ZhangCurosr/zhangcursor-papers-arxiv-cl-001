---
title: "Institutional-Books-sub-sub-Enriched-Text-A-customizable-mul"
source: https://arxiv.org/pdf/2608.19026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:41:07"
field: "多语言语料预处理"
keywords: ["数据清洗", "多语言处理", "OCR后处理", "预训练数据", "富文本标注", "去重", "语义分块", "bits-per-byte"]
innovations: ["提出Enriched Text范式，以HTML注解替代暴力删除以保留元数据与结构", "结合合成数据与Model2Vec静态嵌入实现多语种封底分类（F1=0.97）", "引入BPB作为tokenizer无关的多语言文本质量信号"]
benchmarks: ["IB-HL-ET", "Qwen3-0.6B-Base", "BAAI/BGE-M3", "Nupunkt", "SaT(sat-3l-sm)"]
---

# 论文速读：Institutional-Books-sub-sub-Enriched-Text-A-customizable-mul

## 一句话总结
本文提出了 **Enriched Text（富文本）** 处理范式，在大规模 OCR 清洗任务中**以注解替代删除**，通过 HTML 标签保留元数据与结构信息，生成可灵活配置的 IB-HL-ET 数据集（217B tokens），平衡了 LLM 训练需求与文化文献保真度。

## 研究问题与动机
1. **现有清洗范式与文化文献特性的冲突**：主流 LLM 数据预处理管线（如 CCNet、The Pile）针对网络文本设计，倾向于激进过滤、语言限制和段落去重，这与哈佛图书馆机构藏书的多元性、历史性和完整性目标相悖。
2. **多语言多世纪 OCR 噪声的清洗挑战**：IB-HL 包含约 250 种语言、跨越数个世纪的排版风格，标准互联网预处理难以应对不同语言的标点、分词和文本结构差异。
3. **元数据丢弃与信息保真度的矛盾**：传统管线往往扁平化文本并丢弃 paratext（如页眉页脚、目录、参考文献），损失了重要文献学信息；研究人员重复进行类似处理造成资源浪费。
4. **缺乏用户级配置灵活性**：不同下游任务（长上下文训练、跨语言检索、历史语言学分析）对数据处理的需求差异巨大，统一清洗结果无法满足多样化需求。

## 核心贡献（创新点）
1. **"Enriched Text" 范式**：以 HTML 层级注解标记结构信息（章节、段落、语言、重复关系、BPB 质量分）而非直接删除或扁平化，使下游用户可根据自身需求灵活过滤。
2. **全自动多语言处理管线**：涵盖 Unicode 软/硬归一化、simhash 页面级去重、合成数据微调的封面/正文分类器（F1=0.97）、n-gram 语言模型驱动的断词还原、MinHash LSH 页眉页脚检测、TextTiling 语义分块等 12 个核心步骤，覆盖 ≈250 种语言。
3. **bits-per-byte (BPB) 质量评分机制**：使用 Qwen3-0.6B-Base 模型计算每段落的 BPB 值，提供语言无关、tokenizer 无关的质量信号，支持相对过滤（按卷级百分位数）而非绝对阈值截断。
4. **大规模公开数据集与工具链**：发布 IB-HL-ET（217B o200k_base tokens，1.39B 标注段落，983K 卷）及开源 Python 管线和轻量解析库，支持流式处理与 HuggingFace Datasets 集成。

## 方法详解
### 管线架构
输入 IB-HL（242B tokens）经 12 步串联变换输出 HTML 标注文本，主要步骤如下：

1. **分片与基模型构建**：将 983,004 卷按语言分为 Nupunkt 兼容（138 种）与非兼容两类分片；从各语种子集中选最多 30 本书训练 n-gram 语言模型、Nupunkt 分句基模型、BAAI/BGE-M3 的 Model2Vec 静态嵌入模型及端到端/子分类器。

2. **Unicode 归一化**：
   - **软归一化**（永久保留）：NFC 组合 + 移除 U+200B 零宽空格 + 保留可见字符。
   - **硬归一化**（瞬态）：NFKC + ASCII 引号/空格映射，用于去重和断词判断。

3. **重复页面去除**：对 ≥50 字符的页面计算基于 9-gram 的 128-bit simhash（MurmurHash3），Hamming 距离 ≤6 视为重复，采用 union-find 聚类（保守阈值 6 bit）。

4. **封底/正文分离**：使用 Model2Vec 蒸馏的静态嵌入分类器（F1=0.97）检测 frontmatter/backmatter，subclassifier 细分 TOC_INDEX / BIBLIO / OTHERENDMATTER；合成数据训练（150K 示例，经 gpt-oss-20b 生成）。

5. **断词还原（Dehyphenation）**：对每行末尾连字符，基于字符 n-gram（1-5 元）语言模型评分三种选项（合并/保留连字符换行/替换为空格），结合卷内特定 n-gram 模型与基础语言模型等权融合，使用 Stupid Backoff（回退概率 0.4）处理未登录词。

6. **页眉页脚去除**：提取每页上下各 5 行，经 MinHash（128 排列，字符 5-gram）+ LSH（Jaccard≥0.85）检测近重复，5 页窗口内 ≥3 次重复则移除。

7. **页码去除**：检测上下 5 行内 ≤8 字符且数字占比高的行；后处理阶段对单行短数字"句子"再次过滤。

8. **句子切分**：138 种 Nupunkt 兼容语言采用 per-book 自适应 Nupunkt 模型（≈10s/卷）；其余 112 种语言使用 SaT（sat-3l-sm 三阶层模型，≈1.18s/卷 @ A100）。

9. **语义分块（Topical Chunking）**：借鉴 TextTiling，使用 BAAI/BGE-M3 静态嵌入计算相邻 5 句均值余弦相似度，定义 valley_depth = (left_peak − gap) + (right_peak − gap)，以 `mean − 0.5·std` 为阈值贪婪选边界，最小段长 3 句；段落级与章节级分块采用相同流程。

10. **重复段落识别**：对 1.39B 段落计算 128-bit simhash（硬归一化文本，≥4 不同字符的 9-gram），按 6 个 band 分区 LSH，Hamming 距离 ≤5 视为重复，union-find 聚类并标注 representative/non-representative。

11. **BPB 计算**：使用 Qwen3-0.6B-Base 模型按段落计算：
    $$\text{BPB} = \frac{1}{\ln 2} \cdot \frac{-\sum_t \ln p_\theta(x_t | x_{<t})}{\text{UTF-8 bytes}}$$
    ≤4 字符或 >14,134 token 段落赋哨兵值 −1。

12. **最终标注与格式化**：输出 frontmatter/middlematter/backmatter 三段 HTML-like 字符串，标注字段包括 `<section data-bpb>`、`<p data-bpb data-language data-representative data-clusterid>`、`<aside data-cluster>`，并附 21 项元数据。

## 实验与结果
### 数据集规模
- **IB-HL-ET**：217B o200k_base tokens，983,003 卷，1.39B 段落，297M 子节，7B 句子
- 覆盖 ≈250 种语言，Top-5 语种占比：英语 52.52%、德语 15.88%、法语 12.50%、拉丁语 3.70%、意大利语 3.03%

### 关键步骤性能
| 步骤 | 耗时 | 核心指标 |
|------|------|----------|
| 重复页面去除 | 164h | 1,981,296 页重复，180,521 卷受影响 |
| 断词还原 | 640h | 移除 ≈89% EOL 连字符，保留 ≈7% |
| 页眉页脚去除 | 927h | 移除 272,583,638 行，77.3% 卷受影响 |
| 句子切分 | 2,855h | Nupunkt 99% vs SaT 1% 卷 |
| 语义分块 | 1,568h | 平均 4.69 段落/节，≈5 句/段 |
| 重复段落识别 | 129h | 40.1M 重复类，96.2% 跨卷重复 |
| BPB 计算 | ≈2000 GPU 小时 | 平均 BPB=1.666，SaT 语言平均 2.179 |

### 质量对比（IB-HL vs IB-HL-ET）
| 指标 | IB-HL | IB-HL-ET | 变化 |
|------|-------|----------|------|
| Tokens | 242B | 217B | −25B（主要为封底分离） |
| Tokenizability | 80.43 | 86.57 | **+6.14** |
| Bigram unique % | 46.40% | 38.79% | −7.61pp |
| Trigram unique % | 74.46% | 69.23% | −5.23pp |

### 语言复用分析
- 跨语种重复段落仅占 5.3%，其中拉丁语/希腊语显著高于其他语言（拉丁语 61% ≥200字符段落存在跨语种重复），反映学术引用模式。
- BPB 随年代递减：17 世纪中位 BPB ≈ 2.5 → 19 世纪中 ≈ 1.4 → 20 世纪中 ≈ 1.0，反映语言变化与现代 LM 训练数据覆盖度的影响。

## 相关工作脉络
1. **The Pile / C4 / DCLM**：传统大规模预训练语料管线采用激进过滤与精确/模糊去重策略，倾向于扁平化文本；本文主张"注解而非删除"的可配置范式形成鲜明对比。
2. **TextTiling（Hearst 1997）**：本文的语义分块直接继承自 TextTiling，但以静态嵌入替代词频重叠特征，实现跨语言语义连贯性检测。
3. **CCNet（Wenzek et al. 2020）/ DataComp-LM**：网络文本清洗管线依赖启发式规则和语言过滤；本文面向机构多语种历史文献，采用保留元数据的分级处理策略。
4. **SemDeDup（Abbas et al. 2023）**：语义去重工作主张激进删除重复段落；本文选择标记重复并保留原文，供下游按需过滤。
5. **ICDAR OCR 校正竞赛（2017/2019）**：已有 BERT-based 等方法进行 OCR 后处理；本文聚焦于结构化清洗与标注，不重做 OCR，与上游处理形成互补。
6. **Nupunkt（Bommarito et al. 2025）**：本文采用 Nupunkt 作为英语类语言的句子切分引擎，体现对现代法律/历史文本分割技术的复用。

## 局限性与未来方向
1. **技术文本断词还原误差**：数学公式中的连字符（减号/分数线）可能被误判为行末连字符而错误合并，导致语义改变。
2. **引用注释页脚误删**：高度重复的学术引用格式（如"Ebendort."、"Do. Do."）可能被当作运行页脚移除，损失文献学信息。
3. **表格与边缘注释处理不足**：当前管线对表格区域识别不佳，大量密集重复段落（如表格省略号、"Dollars. Dollars."）产生异常长段落，部分因 BPB 超 32 位索引限制而标注为 −1。
4. **OCR 质量与年代偏差**：BPB 趋势受 OCR 噪声与年代选择性偏差（1930 年后公版书籍有限）双重影响，需谨慎解读历史语言变化结论。
5. **多语种低资源语言处理受限**：168 种语言仅有少量书籍参与基模型训练，部分语言（如丹麦语、葡萄牙语）在短段落上出现语言识别不稳定现象。
6. **未来方向**：改进表格/边缘注释识别、支持更多语言的高质量分段标注、建立标准化富文本注解格式以促进跨库互操作性、通过社区协作形成"标注—模型改进—数据提升"的正向循环。

## 研究启发与可借鉴点
1. **"注解优于删除"的范式转换**：将结构信息（章节边界、重复关系、质量分数、语言标签）以 HTML 属性形式叠加于原文之上，而非暴力截断，为数据工程提供了可审计、可回溯的处理哲学。
2. **合成数据驱动多语种分类器**：在缺乏多语种人工标注资源的约束下，使用 gpt-oss-20b 生成 150K 示例并经过严格清洗（44.5K 有效样本），成功训练出 F1=0.97 的封底分类器，验证了合成数据在低资源场景的实用性。
3. **双模型融合的断词还原策略**：结合卷内特定 n-gram 模型（捕捉专有名/技术词汇）与基础语言模型（提供统计稳健性），用简单的联合概率估计替代昂贵的 KenLM，在效率与效果间取得平衡。
4. **BPB 作为 tokenizer 无关质量信号**：相比 perplexity 受 tokenizer 和语言影响，BPB 以 UTF-8 字节归一化，提供跨语言可比的质量估计；按卷内百分位数过滤而非全局阈值的设计增强了适用性。
5. **Frugal Computing 的工程实践**：目标可在单卡 DGX Spark 一个月内复现，绝大部分步骤使用 CPU 或小型 ML 模型，仅在句子切分和 BPB 计算时少量使用 GPU，为大规模数据处理提供了资源友好的参考模板。

## 关键术语表
- **Enriched Text（富文本）**：以 HTML 层级注解标记结构、质量和元数据信息的文本表示形式，支持下游灵活过滤而非一次性扁平化处理。
- **IB-HL-ET**：Institutional Books Enriched Text，基于 IB-HL 的富文本版本，包含 217B tokens、1.39B 标注段落，已开源。
- **Bits-Per-Byte（BPB）**：每个 UTF-8 字节所需的编码比特数，由因果语言模型计算，作为语言无关的文本质量/可预测性度量。
- **Soft vs Hard Normalization**：软归一化（NFC + 移除零宽空格，保留可见字符）永久应用；硬归一化（NFKC + ASCII 映射）瞬态用于去重和断词判断。
- **Nupunkt / SaT**：两种句子切分引擎；Nupunkt 为基于 Punkt 的无监督算法，适用于 138 种类英语语言；SaT（Segment-any-Text）为多语言神经分割器，用于其余语言。
- **TextTiling 语义分块**：基于嵌入余弦相似度的段落划分算法，检测语义连贯性低谷作为段落边界，本文将其适配为句子级嵌入输入。
- **Simhash + LSH 去重**：基于 9-gram 的 128-bit simhash 近重复检测；页面级用 Hamming≤6，段落级拆分为 6 个 band 的局部敏感哈希加速。
- **Paratext（副文本）**：Genette 概念，指围绕正文的辅助文本元素（封面、目录、页眉页脚、注释等）；本文主张标注保留而非删除。

## 可复现要素
- **数据集**：IB-HL-ET 已公开于 HuggingFace（https://huggingface.co/datasets/institutional/institutional-books-hl-enriched-text）
- **代码**：Python 处理管线开源（https://github.com/institutional/institutional-books-enriched-text-pipeline）；轻量解析库开源（https://github.com/institutional/institutional-books-enriched-text-parser）
- **训练数据**：封底分类器合成训练数据随 GitHub Release 提供
- **关键超参**：simhash 128-bit、9-gram、Hamming 阈值 5（段落）/6（页面）；Nupunkt per-book 自适应；SaT sat-3l-sm；BPB 模型 Qwen3-0.6B-Base，batch_size=16，max_length=14,134 token
