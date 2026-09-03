---
title: "One-Form-to-Transfer-Them-All-Pretraining-Multilingual-Langu"
source: https://arxiv.org/pdf/2608.25904v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:43:32"
field: "多语言语言模型预训练"
keywords: ["multilingual language models", "romanization", "script equalization", "cross-lingual transfer", "autoregressive pretraining", "IPA transcription"]
innovations: ["系统对比罗马化/IPA/原始文本在三规模自回归多语言LM预训练中的效果", "发现罗马化预训练全面优于其他表示且优势随规模扩大", "揭示文本预训练+罗马化微调对已有脚本覆盖语言的性能退化现象"]
benchmarks: ["XStoryCloze", "XCOPA", "XNLI", "MASSIVE", "XL-Sum"]
---

# 论文速读：One-Form-to-Transfer-Them-All: Pretraining Multilingual Language Models Beyond Native Orthography

## 一句话总结
本文系统比较了自回归多语言语言模型在预训练阶段使用原始正交文本、IPA音标和罗马化三种输入表示的效果，发现罗马化预训练在所有规模和任务上均产生最强的跨语言迁移能力，而传统的"文本预训练+罗马化微调"方案反而会在已有脚本覆盖的语言上造成性能下降。

## 研究问题与动机
1. **脚本障碍导致跨语言迁移失效**：多语言LM依赖共享子词词表实现跨语言知识迁移，但当语言使用不同书写系统时（如西里尔字母vs拉丁字母），表层文本形式掩盖了语言间的相似性，导致词表重叠几乎为零。
2. **现有工作缺乏统一对比**：此前研究主要通过罗马化或IPA转录进行脚本规范化，但多数聚焦于encoder-only模型在英语上的微调适配，且两种机制从未在现代autoregressive LM中进行过受控对比。
3. **罗马化微调的真实效果存疑**：已有工作报告对文本预训练模型进行罗马化微调可提升跨语言迁移，但未检验该方法在多语言场景下对已有脚本覆盖语言的影响。

## 核心贡献（创新点）
1. **首次系统对比三种输入表示在自回归多语言预训练中的效果**：在467M–1.03B三种规模、8种语言（4对类型学配对）上控制架构/数据/词表大小/训练流程一致，隔离输入表示变量的影响。
2. **揭示罗马化预训练的跨尺度优势**：罗马化预训练在所有评估设置（zero-shot/few-shot prompting、fine-tuning、seen/unseen语言迁移）中均表现最强，且优势随模型规模扩大而增大。
3. **发现罗马化微调的双刃剑效应**：对文本预训练模型进行罗马化微调在已有脚本覆盖的语言上造成显著性能退化（如Large模型XNLI宏平均从82.94%降至64.01%），仅在基础模型缺乏目标脚本覆盖时才有边际帮助。
4. **提供词表重叠与序列长度的量化分析**：证明罗马化可将印地语/泰米尔语等字符集膨胀语言的token长度压缩至与拉丁/西里尔语同等水平，解决多语言模型中的不公平计算负担。

## 方法详解
**实验设置**：
- **语言对**：English–Spanish（同拉丁字母）、Russian–Polish（西里尔vs拉丁）、Hindi–Urdu（天城文vs阿拉伯文，口语互通）、Tamil–Malayalam（德拉威语系，高语音相似）。
- **数据**：从FineWeb-2抽取单语文档，4个语料对总规模约21.7B词，词匹配后得到50B Text tokens vs 33B IPA/Romanized tokens。
- **词表**：Byte-Level BPE，统一100K词表大小，确保仅文本表层形式不同。
- **模型架构**：基于modded-nanogpt的causal LM，含value embeddings残差连接，三规模配置见原文Table 1。
- **训练优化**：双优化器（AdamW for embeddings/head，Muon for hidden layers），每步524,288 tokens，8×H100，3 epochs。

**四种输入表示**：
1. **Orthographic text**：原始脚本文本。
2. **IPA transcription**：使用Phonemizer库转换，去除重音/长度标记等diacritics以粗粒度化，保留纯音位片段。
3. **Romanization**：使用Uroman库统一转为拉丁字符。
4. **Text→Rom（微调方案）**：文本预训练后，在罗马化下游数据上fine-tune。

**评估**：
- **Prompting**：XStoryCloze（故事续写）、XCOPA（因果常识推理），zero/few-shot。
- **Fine-tuning**：XNLI/MASSIVE（分类）、XL-Sum（摘要生成），macro-F1/ROUGE-L。
- **未见语言迁移**：Arabic、French（同脚本）、Bengali、Greek（不同脚本）。

## 实验与结果
**主要结果**（关键数字）：
- **罗马化预训练全面最优**：Large模型XNLI fine-tuning宏平均达84.93%（Romanized）vs 82.94%（Text）vs 84.45%（IPA），对未见语言MASSIVE宏平均70.42% vs 61.42%（Text）。
- **IPA改善有限**：在Hindi–Urdu对（语音高度对齐）上与罗马化相当，但在其他对及未见语言迁移上落后。
- **罗马化微调严重退化**：Large模型XNLI宏平均从82.94%降至64.01%，MASSIVE从82.94%降至64.01%；仅在未见语言Bengali/Greek上有小幅改善。
- **词表重叠模式**：Romanization将Russian–Polish（0.067→0.200）、Tamil–Malayalam（0.004→0.195）的对重叠大幅提升；IPA在Hindi–Urdu上达到0.243最高重叠。
- **序列长度压缩**：Text表示下Hindi/Tamil/Malayalam的token数膨胀约3倍，Romanization/IPA将其压缩至与Latin/Cyrillic语同等水平。

**结论排序**：Romanized pretraining > IPA > Text（大多数场景）；Text→Rom仅适用于基础模型无目标脚本覆盖的情况。

## 相关工作脉络
1. **Conneau et al. (2020) mBERT**：多语言BERT证明共享词表可实现跨语言迁移，但未处理脚本差异场景，本文针对此局限提出输入表示改造。
2. **Purkayastha et al. (2023) / Husain et al. (2024) RomanSetu**：探索罗马化用于encoder/decoder模型微调适配，本文揭示其仅在模型缺乏脚本覆盖时有效，并在预训练阶段验证罗马化的更优性。
3. **Jung et al. (2024) / Goriely et al. (2024)**：研究IPAphonemic表示在encoder和GPT-2上的效果，本文首次在autoregressive多语言LM中对比IPA与Romanization，发现二者不可互换。
4. **Miletic et al. (2026) 同期工作**：对比Text vs IPA tokenizer在24语言240M模型上的压缩效果，本文扩展至Romanization并报告下游任务性能提升（与其compression-only结论相反）。
5. **Xhelili et al. (2024) / Liu et al. (2025b)**：发现transliteration收益取决于是否暴露lexical overlap，本文实验结果与此一致，并指出操作本身不产生增益。

## 局限性与未来方向
1. **规模限制**：最大1.03B参数，远低于当前7B–100B级多语言模型，需在更大规模验证结论。
2. **书写系统覆盖不足**：仅限alphabetic/abugida，未包含logographic系统（如中文）。
3. **音素化工具质量参差**：Phonemizer对低资源语言支持不完整，diacritic剥离策略较启发式，需探索更鲁棒的phoneme编码。
4. **不可逆性**：Romanization和IPA均为有损转换，缺乏从输出形式还原正交文本的评估（P2G转换未检验）。
5. **未来方向**：扩展至更大规模、更多书写系统；优化音素化Pipeline；分析模型内部表征如何吸收romanization带来的lexical overlap。

## 研究启发与可借鉴点
1. **输入表示应作为预训练核心设计变量**：本文证明romanization不是post-hoc fix而是pretraining-time design choice，团队在多语言模型构建时应将输入表示纳入与数据配比、tokenizer同等重要的决策。
2. **控制变量实验设计**：固定架构/数据/词表大小/训练流程，仅改变输入表层形式，可干净地隔离表示效果，该controlled setup值得复用。
3. **罗马化微调的适用边界**：发现该方法仅在基础模型缺乏目标脚本覆盖时有效，对已有覆盖语言有害，团队在适配多语言模型时应先检验base model的脚本覆盖范围。
4. **词表重叠量化指标**：使用corpus-size-normalized、frequency-weighted Jaccard overlap衡量潜在跨语言迁移能力，可作为tokenizer设计的辅助评估工具。
5. **序列长度公平性分析**：通过FLORES平行语料测量各语言token膨胀倍数，揭示text表示下非拉丁/西里尔语的计算不公平，为多语言资源分配提供依据。

## 关键术语表
**Script equalization**：将不同书写系统的文本转换为共享表示形式（如罗马化或IPA），以消除脚本障碍、促进跨语言词表重叠。
**Romanization**：使用转写工具（如Uroman）将非拉丁脚本映射到拉丁字符集的输入表示方法。
**IPA (International Phonetic Alphabet)**：国际音标，用标准化符号记录语音音位的转录表示。
**Cross-lingual transfer**：模型在一个语言上学到的知识迁移到另一语言的能力，依赖于共享词表/表征。
**Subword overlap**：不同语言间共享子词token的比例，衡量词表层面的跨语言重叠潜力。
**Text→Rom finetuning**：在文本预训练模型上，使用罗马化下游数据进行微调的适配方案。
**Value embedding**：modded-nanogpt中引入的额外词汇表大小嵌入，通过learned scalar混合到attention value projection。
**FLORES parallel corpus**：跨8语言平行语料，用于公平比较不同表示下各语言的序列长度。

## 可复现要素
- **数据集**：FineWeb-2（Penedo et al., 2025）单语文档子采样；XStoryCloze、XCOPA、XNLI、MASSIVE、XL-Sum、FLORES均公开；中文/日文等未覆盖。
- **代码/权重**：论文声明"Code and datasets will be released upon acceptance"，目前未公开。
- **关键超参**：词表大小100K；AdamW(β1=0.8, β2=0.95) + Muon(momentum=0.95)；峰值Muon学习率0.025；每步524,288 tokens；3 epochs；batch size {128, 256}微调搜索。
