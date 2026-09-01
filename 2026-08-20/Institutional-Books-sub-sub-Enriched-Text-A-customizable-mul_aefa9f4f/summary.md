---
title: "Institutional-Books-sub-sub-Enriched-Text-A-customizable-mul"
source: https://arxiv.org/pdf/2608.19026v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:41:01"
field: "多语言大规模语料预处理"
keywords: ["大规模语料清洗", "OCR后处理", "多语言数据管线", "去重与标注", "BPB质量度量", "Enriched Text", "史料数字化"]
innovations: ["提出'标注而非删除'的Enriched Text范式，以HTML式标签保留元数据供下游按需解析", "跨约250种语言的端到端OCR清洗管线（Unicode归一化/端页分离/去连字符/主题分块/BPB打分）", "BPB（每字节位数）作为跨语言tokenizer无关质量信号的全量计算"]
benchmarks: ["IB-HL-ET", "Qwen3-0.6B-Base BPB评估"]
---

# 论文速读：Institutional Books — Enriched Text

## 一句话总结
本文提出了 **Enriched Text（富文本）** 方法，将哈佛图书馆 IB-HL 数据集（983,004 卷/242B 令牌）构建为带结构化标注的 IB-HL-ET 数据集（217B 令牌/1.39B 段），通过 HTML 式标签保留元数据而非粗暴过滤，使下游用户可按需自定义处理。

## 研究问题与动机
- **大模型训练数据预处理与史料保存之间的张力**：现有 LLM 训练管线偏好激进的语言过滤（仅保留英语）和去重，会丢弃 IB-HL 中一半以上的多语言内容并破坏书籍完整性。
- **现有工作重复加工**：TypewriterLM、talkie-1930、K2-V2 等项目已基于 IB-HL 各自做预处理，缺乏统一标准，造成重复劳动与结果不可比。
- **paratext（副文本）被盲目丢弃**：前言、目录、脚注、索引等承载重要上下文的信息常被管线抹除，研究者无法再按需使用。
- **多语言/多世纪 OCR 清洗缺乏可扩展方案**：IB-HL 覆盖约 250 种语言，历史排版差异大，需要在保持原始文本保真度的同时实现大规模可复现清洗。

## 核心贡献（创新点）
1. **"富文本"范式**：以 HTML 式标注（`<section>`, `<p>`, `<aside>` 等）叠加元数据而非直接删除或重写文本，允许下游按需解析过滤；相较于传统"干净令牌流"思路，本质区别在于"标记而非丢弃"。
2. **端到端多语言 OCR 清洗管线**：覆盖 Unicode 归一化、重复页去除、正文/非正文分离、去连字符、页眉页脚与页码去除、句子切分、主题分块、段落级重复识别、BPB 打分共 12 步，适用于约 250 种语言；与单语管线（如只处理英语的 C4/Pile 风格）相比，其本质差异是跨脚本/跨语种的统一处理。
3. **BPB（每字节位数）作为跨语言质量信号**：用 Qwen3-0.6B-Base 计算段级 BPB，替代困惑度（受 tokenizer 影响），为不依赖具体分词器的语言无关质量度量；相比传统 perplexity 过滤，本质差异在于以字节而非 token 归一化。
4. **合成数据训练多语言 endmatter 分类器**：用 GPT-OSS-20B 生成 150k 样本、精选 71k 条训练 Model2Vec 蒸馏的 BGE-M3 静态模型微调分类器与子类分类器，F1=0.97；与人工标注路线相比，本质差异是用合成数据解决低资源语种标注难题。
5. **可复现的廉价计算设计**：管线整体目标是在单台 NVIDIA DGX Spark 上一个月内复现；与依赖 GPU 集群的高成本管线相比，本质差异是将多数步骤设计为 CPU 友好（包括使用 Model2Vec 蒸馏实现纯 CPU 嵌入推理）。

## 方法详解
**管线流程（按顺序）：**
1. **预处理与切块**：将 983,004 卷按主语言分为 Nupunkt 兼容（138 语种）与非兼容两组，各组成 shard（≤200 卷），并行处理。
2. **Unicode 归一化**：Soft 归一化（NFC + 去除零宽空格，保留可见字符）用于输出；Hard 归一化（NFKC + ASCII 化引号/空格/标点）仅内部用于去重比对。
3. **重复页去除**：每页经 hard 归一化后生成 128-bit SimHash（MurmurHash3 + 9-gram，要求 9-gram 至少有 4 个不同字符），同卷内 pairwise 比较，汉明距离≤6 且经 Union-Find 聚类的标记为重复。
4. **Endmatter 分离**：用 Model2Vec 蒸馏的 BGE-M3 静态模型（512 维）微调的分类器对每页分类为 frontmatter/middlematter/backmatter（F1=0.97）；子类分类器将 endmatter 细分为 TOC_INDEX / BIBLIO / OTHERENDMATTER。
5. **去连字符**：对行末连字符用 1-gram 至 5-gram 统计语言模型（语言级 base model + 卷内 book-specific model）打分，从三选一（合并/保留连字符+换行/保留连字符+空格）中选最高概率者，概率公式采用 add-k 平滑（k=0.001）与 Stupid Backoff（backoff=0.4）。
6. **页眉页脚去除**：取每页上下 5 行，作 MinHash 签名（128 排列 + 5-gram），LSH Jaccard 阈值 0.85，在 5 页窗口内出现≥3 次近重复的行删除。
7. **页码去除**：位于上下 5 行内、≤8 字符、含至少 1 个数字且仅 1 个非标点字符的行删除；句切分后再做一轮短数字句子删除。
8. **句子切分**：138 个 Nupunkt 兼容语种进行 per-book 自适应（≈10s/卷）+ 推理（≈0.6s/卷）；其余 112 语种使用 SaT（sat-3l-sm）GPU 推理（≈1.18s/卷，A100）。
9. **主题分块**：基于 BAAI/BGE-M3 蒸馏的静态嵌入对句子做 Cosine 相似度计算，用改进的 TextTiling 方法（谷深 = 左峰−谷值 + 右峰−谷值，阈值 = mean−0.5std，最小段长 3 句）切分为 subtopic paragraph（1.39B）和 subtopic section（297M）。
10. **段落级重复识别**：对 1.39B 段做 hard-normalized 128-bit SimHash（≤5 位汉明距离视为重复），6-band LSH 分组（22+21×4+22），全局 union-find 聚类；代表性段落标 `data-representative`，其余标 `<aside data-cluster>`。
11. **BPB 计算**：`BPB = (1/ln 2) × (-Σ ln p_θ(x_t|x_<t)) / UTF-8 字节数`，使用 Qwen3-0.6B-Base，按段长排序后 16 段一批；≤4 字符或>14,134 token 赋哨兵值 −1。
12. **最终格式化**：输出 frontmatter/middlematter/backmatter 三个 HTML-like 字符串，段级标注含 `data-bpb`、`data-language`（ISO-639-3 via polyglot）、`data-representative`/`data-clusterid`；附带 21 项元数据；另提供已过滤的 `processed_middlematter`（去重 + BPB [p10, p90]）。

## 实验与结果
- **数据集规模**：IB-HL-ET 含 217B o200k_base 令牌，983,003 卷，70 亿句子，1.39B 主题段，297M 主题节。
- **Tokenizability 提升**：IB-HL（原始 OCR）80.43 → IB-HL-ET 86.57（+6.14）；Top-5 语种从 88.63 → 89.24；SaT 语种（非 Top-5）从 69.84 → 74.72（+4.88，增幅最大）。
- **去重效果**：移除 1,981,296 张重复页；识别 51.2M 重复代表性段落 + 72.9M 重复非代表性段落（共 1.93M 同类书内聚类 + 49.3M 跨书聚类，96.2% 跨书）。
- **BPB 分布**：全量均值 1.666，中位数 1.510；SaT 语种 BPB 均值 2.179（高于 Nupunkt 的 1.661）。最低 BPB=0.0297（重复专利记录），极高 BPB 多为 OCR 乱码。
- **跨语种重复文本**：1.24 亿重复段中仅 5.3% 出现在不同语种卷中；英语/德语/法语/俄语 ≥200 字符重复段的同语种保留率 95–99%，拉丁语/希腊语保留率仅 55–61%（反映经典引用传播模式）。
- **BPB 随时间变化**：17 世纪中位数 BPB≈2.5 → 19 世纪中≈1.4 → 20 世纪中≈1.0，反映现代 LM 对当代散文可预测性更高，但也混杂 OCR 退化与样本选择偏差。
- **计算成本**：预处理 9.5h（CPU 4核）；句切分 2,855h（CPU/GPU 混合）；去连字符 640h；页眉页脚去除 927h；主题分块 1,568h；BPB 计算≈2,000 GPU 小时（≈83 GPU 天）；重复段落识别 129h。

## 相关工作脉络
- **C4 / The Pile**（Raffel et al. 2020; Gao et al. 2020）：网络爬取数据清洗管线，偏向激进英语过滤与 chunk 级去重；本文定位为在史料收藏场景下提供"保留元数据而非删除"的替代方案。
- **SemDeDup**（Abbas et al. 2023）：语义近重复删除；本文与之相反，选择标注重复段而非删除，以支持跨语引用分析。
- **Textbooks are all you need**（Gunasekar et al. 2023）与 **Scaling data-constrained LMs**（Muennighoff et al. 2023）：强调训练数据质量与去重对 LLM 的重要性；本文与之互补，提供一套可直接用于训练的高质量多语言史料数据。
- **DCLM / Datacomp-LM**（Li et al. 2024）：建立数据整理基准与综述；本文是其"高质量多语种历史文本处理"方向的具体实现。
- **TextTiling**（Hearst 1997）：经典主题分段算法，基于词频重叠检测主题切换；本文将其现代化为基于静态句子嵌入余弦相似度的谷深检测。
- **OCR 后处理研究**（ICDAR 2017/2019 竞赛、Nguyen et al. 2021 综述）：本文延续这一方向但聚焦于多语种超大规模批处理，而非单本精确修正。

## 局限性与未来方向
- **表格/技术文本处理不佳**：连字符去除在数学公式文本中会错误合并符号（如将 `x - 1 - \n 4x-7` 错误拼接），导致语义改变；附录 D 有具体案例。
- **注释与页边注未被结构化处理**：当前管线将所有 marginalia 并入正文，部分引用页脚被误删（如某军事史书中 137 次引用减至 64 次），无法恢复书目溯源信息。
- **极长段落 BPB 缺失**：>14,134 token 的段因 PyTorch CUDA kernel int32 溢出被赋哨兵 −1，约 12,391 段受影响；未来可用 fused cross-entropy 或分块推理修复。
- **OCR 质量差异未消除**：BPB 混合了模型困惑与 OCR 噪声，1930 年后样本存在选择偏差（仅限公版书），使 BPB 时间趋势解读需谨慎。
- **分类器准确率有限**：子类分类器 F1=0.90，endmatter 常误判为 middlematter；未来需高质量多语种人工标注或更优合成数据策略。
- **部分语种处理依赖 SaT GPU**：2.1% 的卷需 GPU 推理，限制无 GPU 环境的完全 CPU 复现。

## 研究启发与可借鉴点
1. **"标注而非删除"的数据哲学**：将去重、语言、BPB、endmatter 等作为 HTML 属性标注保留，让下游按需解析——这一范式可迁移到任何大规模语料管线，避免"一刀切"清洗丢失研究价值。
2. **Model2Vec 蒸馏 BGE-M3 实现 CPU 友好嵌入**：用 Model2Vec 将 BAAI/BGE-M3 蒸馏为静态嵌入模型，使得多语言 chunking/classification 可纯 CPU 运行，对资源受限团队极具参考价值。
3. **Per-book 自适应 + 基础模型结合的去连字符策略**：同时使用语言级 base n-gram model 和卷内 book-specific model（后者仅在有连字符时构建），在精度与效率间取得平衡；其联合打分思路可迁移至其他文本规范化任务。
4. **BPB 作为跨语言质量信号的工程化实现**：用 Qwen3-0.6B-Base 跑全量 1.4B 段 BPB（≈2,000 GPU 小时），并配套提供卷级百分位数统计——这一模式可作为大规模语料质量预计算的参考方案。
5. **与现有管线（IB-HL）的衔接设计**：本管线不重新 OCR，而是作为下游可插拔层处理已有 OCR，适合各机构在其既有数字化成果上扩展，无需从头重建 pipeline。

## 关键术语表
- **Enriched Text**：一种数据格式理念，将 OCR 文本与结构化元数据以 HTML 式标签叠加而非抹除，允许下游按需解析。
- **BPB（Bits-Per-Byte）**：用语言模型负对数似然除以 UTF-8 字节数得到的跨语言质量度量，替代受分词器影响的 perplexity。
- **SimHash**：基于 n-gram 哈希的局部敏感哈希，用于近似重复页面/段落的检测，汉明距离≤阈值视为重复。
- **Endmatter**：书籍的非正文部分（封面、目录、致谢、参考文献、索引等），与 middlematter（正文）相对。
- **TextTiling（现代化版本）**：基于静态句子嵌入余弦相似度曲线谷深检测主题切换点的分段算法，替代传统的词频重叠方法。
- **Model2Vec**：将 Transformer 嵌入模型蒸馏为静态（无上下文）向量表示的工具，使推理仅需 CPU。
- **Nupunkt**：基于 Punkt 算法的现代句子切分库，对 138 种类英语标点语种高效，支持 per-book 自适应。
- **SaT（Segment-any-Text）**：多语言神经句子切分模型（sat-3l-sm），适用于标点系统与英语差异较大的 112 种语种。

## 可复现要素
- **数据集**：IB-HL-ET 公开于 HuggingFace（https://huggingface.co/datasets/institutional/institutional-books-hl-enriched-text），983,003 卷，217B o200k_base 令牌。
- **代码**：管线开源於 GitHub（https://github.com/institutional/institutional-books-enriched-text-pipeline）；解析器库开源於 GitHub（https://github.com/institutional/institutional-books-enriched-text-parser）。
- **合成训练数据**：endmatter 分类器与子类分类器的合成训练数据随 GitHub release 分发。
- **关键超参**：SimHash 汉明距离阈值（页级≤6，段级≤5）；n-gram 去连字符 k=0.001、Stupid Backoff=0.4；MinHash LSH Jaccard 阈值 0.85；TextTiling 段最小长度 3 句、阈值 mean−0.5std；BPB 模型 Qwen3-0.6B-Base、batch=16、最大段 14,134 token。
- **计算环境**：论文使用 Harvard FASRC Cannon 集群（heterogeneous nodes，含 Intel Cascade Lake/Sapphire Rapids CPU + NVIDIA A100 80GB GPU）；目标可复现在单台 NVIDIA DGX Spark。
