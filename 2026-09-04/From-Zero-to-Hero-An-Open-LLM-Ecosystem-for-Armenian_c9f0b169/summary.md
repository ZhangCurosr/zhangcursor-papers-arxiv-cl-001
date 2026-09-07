---
title: "From-Zero-to-Hero-An-Open-LLM-Ecosystem-for-Armenian"
source: https://arxiv.org/pdf/2609.03350v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:02:12"
field: "低资源语言大模型"
keywords: ["低资源语言模型", "持续预训练", "数据去污染", "翻译验证", "亚美尼亚语LLM", "灾难性遗忘"]
innovations: ["提出答案保持型盲解验证范式，将翻译数学数据用于持续预训练以逆转灾难性遗忘", "发布首个完整开源且可复现的亚美尼亚语LLM生态系统（语料+模型+配方+评测）", "首次系统量化公共语料库对低资源语言评测集的污染率并提出去污染标准流程"]
benchmarks: ["Belebele-hye", "m-MMLU-hy", "INCLUDE-Armenian", "ARC-hy", "HellaSwag-hy", "MultiBLiMP-hye", "ArmBench-LLM", "FineWeb-2-hy"]
---

# 论文速读：From-Zero-to-Hero-An-Open-LLM-Ecosystem-for-Armenian

## 一句话总结
本文构建了首个开源且完全可复现的高卢语（亚美尼亚语）LLM生态系统，发布了经过严格验证的新闻语料库 ArmWeb（437万文档、3.3B Gemma tokens）和首个大规模英文–亚美尼亚语并行 STEM 数据集 ArmSTEM（37.3万条带逐步解答的数学/科学题目），基于 Gemma-4-E4B 通过持续预训练得到 arm-gemma-e4b，性能超越所有现有开源亚美尼亚语模型。

## 研究问题与动机
- **亚美尼亚语 LLM 缺乏透明可复现的训练体系**：现有开放模型（HyGPT-10b、tweety-7barmenian、ArmenianGPT-1.0-3B）仅发布权重，不公开训练数据与配方，无法审计或复现，与巴斯克语（Latxa）、哈萨克语（Sherkala）、波兰语（Pllvm）等已建立语言专属生态系的语种形成鲜明对比。
- **公开语料与评测基准严重重叠，导致评测失真**：对三大公共亚美尼亚语网络爬取切片（CulturaX-hy、HPLT-v2-hy、FineWeb-2-hy）的系统扫描发现，各语料库与评测集存在 7.9%–17.4% 的文档级重叠，其中 FineWeb-2 自身的亚美尼亚语测试集甚至泄漏到其训练集，困惑度类评测系统性高估了模型性能。
- **亚美尼亚语缺乏带逐步解答的大规模 STEM 训练文本**：现有资源仅限于小规模评估集，无可用于训练的带解答过程的数学/科学文本。
- **低资源语言适应的"流畅度–知识"权衡未得到机制性研究**：简单新闻持续预训练会导致灾难性遗忘（Belebele 下降达 −21.2pp），但如何在保持语言流畅度提升的同时逆转知识流失，尚无系统性答案。

## 核心贡献（创新点）
1. **发布 ArmWeb：首个作者运营15年爬取的亚美尼亚语新闻语料库（4.37M文档、3.3B Gemma tokens）**，含完整清洗–去重–防泄漏管线与每文档溯源元数据，经 410M 消融网格与 Scaling Ladder 验证，预测 1.3B 模型损失误差仅 0.47%。→ 区别于 multilingual crawl slices（CulturaX/HPLT/FineWeb），ArmWeb 为亚美尼亚语专属精心策划，且评测数据经过了严格的 13-gram 去污染处理。

2. **发布 ArmSTEM：首个训练规模亚美尼亚语 STEM 语料库（373K EN–HY 平行题对，32.4万含逐步解答）**，采用占位符遮罩翻译（placeholder-masked translation）防止数字/公式篡改，并通过盲解校验（blind re-solving with exact-match）与双人母语者人工评估（299/300 有效，κ=1.0）双重验证。→ 区别于此前仅依赖 LLM 翻译质量评估的做法，本文以"答案保持"为核心功能验证标准。

3. **发布 arm-gemma-e4b：首个带完整训练数据与配方的开源亚美尼亚语 LLM**，在六任务 Likelihood Suite 上均值 0.50，超过所有已有开源模型（0.35–0.47）及未适配的 Gemma-4-E4B 基础模型（0.48）。→ 区别在于本文提供了可复现的端到端 pipeline，而既有模型均为黑盒发布。

4. **首次系统揭示并机制性地缓解低资源语言适应中的"流畅度换知识"现象**：验证温和学习率（3×10⁻⁵）可恢复约 2/3 的知识损失，而经验证的 STEM 翻译数据可逆转遗忘，使均值超过基础模型 +2.2pp。→ 提出"经验证翻译知识数据"作为第三种遗忘缓解杠杆，区别于经验回放的延缓作用。

5. **发布亚美尼亚语评测污染审计报告**：量化三大公共语料库对十个亚美尼亚语基准的文档级污染率（7.9%–17.4%），并开源去污染工具与报告。→ 将英语语境下的污染审计方法论首次系统化应用于低资源语言生态。

## 方法详解
- **ArmWeb 语料管线**：
  - 来源：作者在 2011–2026 年间运营爬取的 19 个亚美尼亚语新闻网站结构化数据库（非原始 HTML），天然规避网页模板噪声。
  - 清洗：极简策略——仅做标题+正文拼接、最低长度 100 字符、空白符标准化与 NFC 归一化；不进行传统网页爬虫常用的 boilerplate 过滤/黑名单/质量分类器（这些往往误伤合法文本）。
  - 语言识别：GlotLID 保留 98.3% 文档为 hye/hyw。
  - 去重：先 xxh128 哈希精确去重，再 MinHash LSH（word 5-gram shingles，112 permutations，14 bands×8 rows，Jaccard 阈值 0.72，针对新闻 syndication 调低阈值）全局去重（必须在 split 前完成，避免跨 split 泄漏）。
  - 划分与泄漏检查：预留 3×20K 文档（验证集、同分布测试集、时间尾部测试集），零精确碰撞，近重复率 ≤0.045%；13-gram 去污染覆盖十个亚美尼亚语基准。

- **ArmSTEM 翻译与验证管线**：
  - 源数据：GSM8K（7.4K）、AceReason-Math（49.6K）、OpenScience（271.4K）、OpenScienceReasoning-2（57.6K）。
  - 占位符遮罩翻译：将数字、LaTeX 公式、问题/解答分隔符替换为 indexed placeholder token 后再翻译，从结构上杜绝数字/符号损坏；使用 Gemini-3.1-flash-lite（阈值法选定，较 GPT-5.5 成本低且差异不显著）。
  - 三级验证门：G0（占位符完整性）→ G1（GlotLID 语言检查）→ G2（盲解：o4-mini 独立求解亚美尼亚语题目，精确匹配标准答案），自由形式回答（<5%）使用三模型 judge panel 2/3 多数决。
  - 双向去污染：英文源against英文基准（MMLU-Pro test set），已接受译文against亚美尼亚语基准。

- **持续预训练配方（arm-gemma-e4b）**：
  - 基础模型：Gemma-4-E4B（词表对亚美尼亚语 fertility 最高，4.15 tokens/word）。
  - 五流混合（总计 10B tokens）：69% ArmWeb + 4% ArmSTEM-HY + 2% ArmSTEM-EN（并行数据双向有益）+ 20% English web replay（FineWeb-Edu，缓解分布偏移遗忘）+ 5% 代码（Stacksmol）。
  - 学习率：cosine schedule，3×10⁻⁵（经消融实验确定为最优，比初版 10⁻⁴ 温和三倍）。
  - Tokenizer 决策：经消融验证，mean-initialized 扩展词表在 2B token 预算下 bpb 恶化 40–45%，故保留原 Gemma 词表。
  - 数据轮次：ArmWeb 约读 2 遍（near-free），ArmSTEM 每个 token 读 7–9 遍（处于 Table 4 所示高效区间）。

- **Scaling Ladder 与外推验证**：
  - 在 70M → 1B 四个 Chinchilla-optimal 预算点训练，拟合三参数幂律 bpb(N) = 4.9×10⁵ N⁻⁰·⁷⁹⁵ + 0.439。
  - 外推预测 1.3B 模型 bpb = 0.4675，实测 0.4697，误差仅 0.47%。

## 实验与结果
- **评测设置**：
  - Likelihood Suite（六任务，zero-shot log-likelihood）：Belebele-hye、m-MMLU-hy、INCLUDE-Armenian、ARC-hy、HellaSwag-hy、MultiBLiMP-hye。
  - ArmBench-LLM（24任务生成式评测）：Scientific MCQA、Belebele(gen.)、SynDARin、MMLU-Pro-Hy、Exam history/literature/math 等。
  - Baselines：Gemma-4-E4B（base）、HyGPT-10b、ArmenianGPT-1.0-3B、tweety-7barmenian、Gemma-2-9B。

- **主要结果（Likelihood Suite 均值 accuracy）**：
  - **arm-gemma-e4b：0.50**（最强）> Gemma-4-E4B base：0.48 > ArmenianGPT-1.0-3B：0.47 > HyGPT-10b：0.44 > tweety：0.35。
  - Belebele 子项：arm-gemma-e4b 达 0.716，较 base（0.619）**提升 +9.7pp**，较 News-CPT（0.550）提升 +16.6pp。
  - INCLUDE 子项：arm-gemma-e4b 达 0.456，较 base（0.416）+4.0pp。
  - m-MMLU-hy：arm-gemma-e4b 0.337，恢复到 base（0.343）置信区间内。

- **生成式评测（ArmBench，12个 0–1 任务均值）**：
  - arm-gemma-e4b：0.62，优于 instruction-tuned ArmenianGPT-1.0-3B 的 0.57（且 arm-gemma-e4b 未经指令微调）。
  - Scientific MCQA：1.000（满分，经 n-gram 审计排除泄漏）。
  - SynDARin：0.92（较 base 0.04 提升 +88pp）。
  - Exam history：2.50（较 base 1.00 提升 +1.50 分）。

- **消融关键发现**：
  - News-CPT（LR=10⁻⁴）：Belebele −21.2pp，m-MMLU-hy −7.1pp，呈典型灾难性遗忘。
  - News-CPT（LR=3×10⁻⁵）：恢复约 2/3 Belebele 损失。
  - STEM-CPT（3×10⁻⁵）：均值 +2.2pp over base，Belebele 达 0.716（+9.7pp over base）。
  - STEM-CPT-full（使用全 373K STEM 数据）：Likelihood Suite 与 arm-gemma-e4b 无显著差异（0.494 vs 0.500），但生成式 exam math 达 2.75 高于 arm-gemma-e4b 的 1.75；POS tagging 回归至 0.01（base 为 0.18）。

- **污染审计**：FineWeb-2-hy 对评测集文档级污染率最高达 17.4%，HPLT-v2-hy 为 10.9%，CulturaX-hy 为 7.9%；手建 MCQA 基准（Belebele、INCLUDE、SynDARin）各语料库污染率 ≤0.06%。

## 相关工作脉络
1. **Latxa（巴斯克语，Etxaniz et al., 2024）**：确立了"语料库+模型+评测+训练配方"一体化开源范式，本文沿此设计但进一步引入翻译知识数据作为遗忘缓解机制。
2. **Sherkala（哈萨克语，Koto et al., 2025）与 Pllvm（波兰语，Kocon et al., 2025）**：同为低资源语言专属生态，但均缺少带解答的 STEM 数据与翻译验证管线。
3. **FineWeb-2（Penedo et al., 2025）**：多语言通用爬取管线；本文发现其亚美尼亚语切片污染率达 17.4%，论证了领域策划语料库的必要互补性。
4. **Catastrophic Forgetting 缓解文献（Rolnick et al., 2019; Ibrahim et al., 2024）**：经验回放与温和学习率为主流方案；本文证明仅回放不足以防遗忘（20% replay 组仍出现遗忘），需叠加经验证的知识数据。
5. **翻译推理数据研究（Chen et al., 2024; Shi et al., 2023）**：证明 CoT 跨语言迁移与翻译数学数据的效用；本文将其应用于持续预训练而非指令微调，并首次以"盲解答案保持"作为翻译验证标准。
6. **污染审计（Dodge et al., 2021; Sainz et al., 2023）**：在英语大模型语境下建立 n-gram 审计范式；本文将其系统引入低资源语言，发现 FineWeb-2 自身 train/test 泄漏问题。

## 局限性与未来方向
- **ArmWeb 领域单一**：集中于新闻语域，register 多样性有限；规模约 3.3B Gemma tokens，约为 HyGPT-10b 未公开语料（~10B tokens）的三分之一。
- **STEM 数据贡献不可分割**：6% 混合替换中，STEM 内容增益、QA 格式训练增益与翻译验证增益尚无法分离，需格式匹配对照实验。
- **评测以 MCQA 为主**：缺少生成质量评估与安全评估；Scientific MCQA 满分（50/50）的集子较小（50题），paraphrase-level 审计留待未来工作。
- **部分任务出现负向结果**：POS tagging 从 0.18 降至 0.01；考试数学题分数在 released model 上持平（1.75），需在 STEM-CPT-full 上获得提升（2.75），暗示数据多样性而非难度可能是瓶颈。
- **未来方向**：竞争级数学子集、指令微调版 arm-gemma-e4b、可扩展至其他低资源语言。

## 研究启发与可借鉴点
1. **"答案保持"验证范式可迁移至其他低资源语言**：将盲解（blind re-solving）+ 精确匹配作为翻译数学/科学数据的验证标准，比单纯 LLM fluency judgment 更严格、更具功能性保证，适用于任何具有可验证答案的领域数据构建。
2. **温和学习率 + 翻译知识数据的双重遗忘缓解策略**：仅回放或仅降 LR 均不足以完全逆转灾难性遗忘；"20% replay + 5–6% 经验证翻译知识数据"的组合实现了知识净增长，可作为低资源语言 CPT 的通用配方参考。
3. **评测污染审计应成为低资源语言生态的标准流程**：本文首次系统量化了三大方公共语料库对亚美尼亚语评测集的污染率（7.9%–17.4%），并提出 13-gram 去污染管线；对于任何新语种生态建设，污染审计应前置而非事后报告。
4. **Scaling Ladder 外推验证确保小尺度消融结论可靠**：在 70M→1B 四档 Chinchilla-optimal 预算点训练并拟合幂律，外推至独立训练的 1.3B 模型误差仅 0.47%，这一方法论可为资源受限团队提供可靠的小规模实验–大规模结论映射方案。
5. **占位符遮罩翻译（placeholder-masked translation）保护数值/符号完整性**：在 STEM 类多公式文本翻译中，结构性防止数字/LaTeX 损坏比事后检测更可靠，此技巧可直接迁移至任何包含符号系统的翻译数据构建场景。

## 关键术语表
- **Catastrophic Forgetting（灾难性遗忘）**：持续预训练中模型在新领域/新语言数据上学习时，原有知识能力急剧退化的现象，本文证实其在亚美尼亚语新闻 CPT 中可造成 Belebele −21.2pp 的损失。
- **Placeholder-Masked Translation（占位符遮罩翻译）**：翻译前将数字、LaTeX 公式等敏感元素替换为索引占位符 token，翻译完成后再还原，从结构上杜绝翻译过程中的数值/符号损坏。
- **Blind Re-solving（盲解验证）**：用独立模型对翻译后的题目求解，要求精确匹配标准答案，作为翻译"意义保持"的功能性验证门。
- **Tokens-per-word Fertility（词粒/token 生育率）**：衡量 tokenizer 将单字切分为多少 token 的指标；越高表示该语言在此 tokenizer 下编码效率越低，本文 Gemma-4 以 4.15 tokens/word 胜出。
- **Decontamination（去污染）**：通过 n-gram 重叠检测并移除训练数据中与评测集重合的文档，本文采用 13-gram 对十个亚美尼亚语基准进行去污染，移除 3.3% 文档。
- **Epoch-capped Mixture（轮次封顶混合）**：对各数据流设定最大训练轮次（如 ArmWeb ≈2 遍、ArmSTEM 7–9 遍），避免在有限数据上过拟合，依据 data-constrained scaling laws 设定。
- **Scaling Ladder（扩展梯度）**：在同一配方下于多个模型规模（70M→1B）训练，通过幂律拟合验证小尺度消融结论是否可外推至更大模型。

## 可复现要素
- **数据集**：ArmWeb 与 ArmSTEM 均已开源（HuggingFace / 作者仓库）；ArmWeb 含每文档 URL、来源、发布时间与爬取时间元数据；ArmSTEM 含每条目溯源与子集许可证。
- **代码**：全部 pipeline 代码、per-stage 报告、训练 job 脚本与完整配置均已开源。
- **模型权重**：arm-gemma-e4b 权重已开源。
- **关键超参**：10B tokens 总训练量，序列 packed 至 4096，per-device batch=2，gradient accumulation=2，global batch≈2.1M tokens/step，约 4770 步；cosine LR schedule，初始 LR=3×10⁻⁵，warmup=100 steps；AdamW（β₂=0.95，weight decay=0.1），bf16 精度；16 节点 × 8 H100（共 128 H100），约 10 小时 wallclock / 1250 H100-hours。
- **翻译模型**：Gemini-3.1-flash-lite（主翻译）、GPT-5.5（escalation）、o4-mini（盲解）；accessed June–August 2026。
- **训练环境**：HuggingFace Trainer + torchrun；Megatron-LM 用于 410M/1.3B 消融实验。
