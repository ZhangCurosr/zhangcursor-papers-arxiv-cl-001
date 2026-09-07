---
title: "Knowledge-Acquisition-During-Pre-training-Large-Language-Mod"
source: https://arxiv.org/pdf/2609.04180v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 11:05:59"
field: "大语言模型预训练与知识获取"
keywords: ["auxiliary views", "knowledge acquisition", "continued pre-training", "data diversity", "LLM pre-training", "knowledge representation"]
innovations: ["在固定token预算下，将重复文档的token分配给辅助视图（教科书/博客/问答）可同时提升事实回忆与推理能力，且效果随模型规模单调增长", "揭示了辅助视图的机制性签名——FFN层中上层中部变化减少、中层与末层变化增加的层wise偏置，以及以更少参数扰动实现更高学习效果的压缩效应", "澄清paraphrasing仅在较小batch size下有益，调和了Prior work的矛盾结论，并证明辅助视图质量不依赖强教师模型"]
benchmarks: ["LAMA-style factual probes", "LAMA-style inference probes", "OLMo-2-32B/13B/7B/1B on domain-adapted MCQA"]
---

# 论文速读：Knowledge-Acquisition-During-Pre-training-Large-Language-Models-Learn-Better-With-Auxiliary-Views

## 一句话总结
本文通过受控实验证明，在预训练阶段为知识提供"辅助视图"（如教科书、博客、Stack Exchange问答等不同形式的表述）比单纯重复原文或仅做改写更有助于大语言模型的知识获取，且这种优势随模型规模扩大而增强，并能通过更高效的参数编码（中层与末层聚焦、更少权重扰动）实现事实记忆与推理能力的双重提升。

## 研究问题与动机
- **核心问题**：在预训练中，知识应如何表征才能在文本中呈现，才能最大化模型学习效果？现有研究多关注语料特征（去重、过滤、质量、多样性），却忽视了"同一知识应如何被表达"这一更根本的问题。
- **重复与改写之争未解**：Prior work对paraphrasing（改写）是否有益存在矛盾结论（Chang et al. 2024 vs. Allen-Zhu & Li 2024），本文发现这种分歧源于batch size差异，旨在澄清这一争议。
- **领域适应中知识注入效率低**：现有continued pre-training在专业领域（生物医学、法律）的知识注入效果有限（如70B模型仅能回忆62.7%的事实），亟需更可靠的方法。
- **复杂知识的表征需求**：原子事实（如传记属性）的表征较简单，但生物医学、法律等复杂领域知识是多面向且相互依赖的，需要多视角呈现。

## 核心贡献（创新点）
- **首次系统隔离"辅助视图"效应**：在固定token预算下，将重复文档的token重新分配给辅助视图（而非仅重复或改写原文），能够同时提升事实回忆与推理能力——这与直觉相反，因为目标token在原文中出现频率更高。
- **揭示辅助视图的机制性签名**：辅助视图训练使模型在FFN层呈现独特的层-wise重分布模式（中层和末层变化更多，上层中部16-24层变化更少），并以更少的参数扰动实现更高学习效果（"压缩"效应）。
- **澄清paraphrasing的收益边界条件**：paraphrasing仅在较小batch size下有益（防止过拟合），在大批量（如256+）下其与Source无显著差异，从而调和了Prior work的矛盾结论。
- **证明辅助视图质量不依赖强教师模型**：辅助视图的有效性不取决于生成模型的强度（Pearson r = -0.14，与模型规模无关；r = -0.24，与生成器自身准确率无关），最强收益来自"数据增强"而非"知识蒸馏"。
- **区分上下文知识与先修知识的互补作用**：上下文知识（被引文献）对事实回忆增益更大，先修知识（基础概念教材）对推理能力增益更大，二者各有所长。

## 方法详解
- **实验框架**：基于OLMo-2系列模型（1B/7B/13B/32B），采用单批次知识注入（single-batch knowledge injection）：每个batch中填充目标文档，其余填充通用数据（DCLM子集），共N=100次注入。
- **条件设计（token匹配）**：
  - **Source**：每batch重复注入原文档。
  - **Para. M**：在原文档与M个改写版本间循环（本文用M=9或M=49）。
  - **Para. M + Aux.**：在改写文档基础上额外注入辅助视图（博客、教科书章节、Stack Exchange式问答各一份）。
  - 所有条件通过upsampling Source和Para. M实现token-level对齐，确保知识相关token总数一致。
- **数据来源**：36篇文档（arXiv计算机论文12篇、美国联邦上诉法院法律意见12篇、PubMed Central医学病例报告12篇），均经过Infini-gram API验证未泄露自OLMo-2预训练语料。
- **辅助视图生成**：使用GPT-5生成大纲、GPT-5-mini生成正文内容；每种文档生成博客、教科书、Stack Exchange问答三种形式；先修知识以教科书形式生成。
- **Probe设计**：
  - **Factual probe**：从知识句中抽取事实，转换为自包含的cloze语句（6,435条事实probe + 4,515条MCQ变体）。
  - **Inference probe**：整合多个support sentence推断未明确陈述的信息（430条推理probe + 322条MCQ变体）。
  - 评估指标：log-probability、target rank（越低越好）、MCQA准确率（5-shot prompt）。
- **机制分析**：比较训练后FFN权重与base模型的cosine distance、relative delta norm、Gini系数，逐层/逐channel追踪知识编码变化。

## 实验与结果
- **主要结果（OLMo-2-32B，Figure 1/Table 5）**：
  - 事实回忆（Factual MCQA）：Source < Para. 9 < Para. 9 + Aux.，辅助视图条件获得最大增益。
  - 推理能力（Inference MCQA）：同样呈Source < Para. 9 < Para. 9 + Aux.趋势。
  - Pre-training-faithful设置（Table 5，OLMo-2-7B从step 925,000恢复训练，batch size=1024）：辅助视图条件下事实log prob. = -9.56（最优）、MCQA = 0.403；推理log prob. = -10.82、MCQA = 0.492，显著优于其他条件。
- **规模效应（Figure 2）**：1B模型从辅助视图中获益极少；7B/13B/32B差距稳步扩大，辅助视图优势随模型规模单调增长。
- **Qwen-2.5-7B泛化（Table 11）**：辅助视图优势在Qwen模型上同样成立，说明结论跨模型架构有效。
- **人类撰写辅助视图（Figure 3）**：从开放网络收集的真实人类写作辅助视图同样带来规模递增的收益，排除纯合成文本artifact的可能。
- **Token-match upsampling消融（Table 4）**：即使将upsampling减半或完全移除，辅助视图仍保持优势，Source在降低重复度后从未改善。
- **Paraphrasing的batch size依赖（Figure 5）**：batch size=64时Para. 9显著优于Source（防止过拟合）；batch size=256时两者基本持平；batch size=1024时（Table 5）paraphrasing无稳定优势。
- **教师模型强度无关性（Table 13）**：使用11种不同生成器配置（gpt-5-mini、gpt-oss-20B/120B、Gemma-4 12B/31B、GLM-5系列等），下游事实准确率稳定在0.405-0.422，与生成器规模（r=-0.14）和生成器自身准确率（r=-0.24）均无显著相关，仅与生成文本量中度正相关（r=+0.62）。
- **上下文vs.先修知识（Table 9）**：加入先修知识对推理probe增益更大（arXiv: +4.629 vs. +4.574 log prob.）；加入上下文知识对事实probe增益略大（arXiv: +6.893 vs. +6.690），但token-matched条件下差异不大。
- **最强结果**：OLMo-2-32B在Para. 9 + Aux.条件下，事实MCQA达约0.40+，推理MCQA达约0.49+（具体数值因模型大小而异，32B最大）；相比Source基线，推理MCQA提升约8-10个百分点，事实MCQA提升约3-5个百分点。

## 相关工作脉络
- **Allen-Zhu & Li (2024)**：发现paraphrase augmentation可将传记事实记忆率从9.7%提升至96.6%，但未研究复杂领域知识；本文与其结论的差异源于batch size设置不同（本文澄清paraphrasing仅在batch size较小时有效）。
- **Chang et al. (2024)**：在预训练中间歇注入虚构事实，发现知识是增量式获取且会衰减；但其使用2048-token chunk与batch size=2048的设置导致paraphrasing表现不佳，本文在其相同设置下复现并解释了这一现象。
- **Jiang et al. (2024) / Hoffbauer et al. (2024)**：指出continued pre-training在专业领域（Reddit AskHistorians）的知识注入效果有限，本文提供了更具操作性的改进路径（辅助视图合成）。
- **Gunasekar et al. (2023) "Textbooks Are All You Need"**：强调教科书风格文本对训练的价值；本文将其思想延伸至多样化的辅助视图（博客、问答、教科书混合），并量化了各类视图的效果差异。
- **Ovadia et al. (2024)**：比较fine-tuning与retrieval的知识注入方式；本文聚焦pre-training阶段的数据表征策略，提供了不同层面的干预方案。
- **Zhang et al. (2025)**：研究数据多样性对预训练的影响；本文进一步将"多样性"操作化为围绕单点知识的概念多样性（auxiliary views），而非仅语料级多样性。

## 局限性与未来方向
- **领域覆盖有限**：仅测试计算机科学、法律、医学三个领域，次要发现（上下文/先修知识差异）在不同领域间不一致；医学领域因病例报告缺乏引用结构而无法构建上下文知识。
- **预训练忠实性受限**：pre-training-faithful控制实验仅在checkpoint step 925,000处进行100步训练，而非从头预训练，结论向完整预训练推广需谨慎。
- **模型规模上限**：实验最大模型为32B，对更大规模模型（如175B+）的结论外推存在不确定性。
- **低资源/高度专业化领域的教师能力边界**：对于LLM自身可能缺乏理解能力的极冷门领域，生成辅助视图的教师模型强度可能变得关键，本文结论在此场景下未经验证。
- **辅助视图类型的细粒度分析不足**：教科书、博客、Stack Exchange问答三类视图整体效果相近，但混合使用的轻微优势及其原因仍有待深入研究。

## 研究启发与可借鉴点
- **数据构造层面**：在领域适应或低资源继续预训练中，可优先合成辅助视图（而非单纯重复原文），以概念多样性替代语言多样性，从而以更高效的参数编码实现知识内化。
- **实验设计借鉴**：token-matched控制（而非仅控制文档数量）是分离重复与多样化效应的重要手段；本文的LAMA-style probe构造流程（自动抽取→cloze转换→人工校验）可作为知识获取评估的标准范式。
- **机制分析工具**：逐层FFN cosine distance与Gini系数的组合分析，为"知识在模型中如何被编码"提供了可操作的诊断工具，可用于后续工作验证不同数据策略的表征效率差异。
- **batch size敏感性意识**：任何涉及paraphrasing或数据增强的实验设计都必须将batch size纳入控制变量，否则结论可能因过拟合程度不同而截然不同。
- **合成数据质量假设重构**：辅助视图有效性不依赖强教师模型这一发现，为低资源场景下的低成本数据增强提供了理论支撑——即使使用小型模型生成辅助视图，也能获得显著收益。

## 关键术语表
- **Auxiliary Views（辅助视图）**：对同一知识点的不同形式表述（如教科书、博客、问答），区别于仅做语言层面改写的paraphrase，强调概念层面的多元化呈现。
- **Factual Probe（事实探针）**：测量模型对文本中明确陈述信息的回忆能力，目标span是原文的verbatim短语。
- **Inference Probe（推理探针）**：要求模型整合多个support sentence推断出原文未直接陈述的信息，测试知识的泛化与组合能力。
- **Single-batch Knowledge Injection（单批次知识注入）**：每个训练batch中仅包含少量目标文档，其余填充通用数据，用于在大规模预训练语料背景下隔离特定知识的获取过程。
- **Token-matching（token级匹配）**：通过upsampling控制各实验条件的知识相关token总数一致，确保比较公平。
- **Layer-wise Bias（层 Wise偏置）**：辅助视图训练导致参数变化在Transformer不同层间呈现非均匀分布模式（中层/末层变化增多，上层中部变化减少）。
- **Compression效应**：辅助视图使模型以"更少的参数扰动"编码"更多的知识"，体现为更小的FFN权重变化幅度和更高的Gini集中度。

## 可复现要素
- **数据集**：36篇领域文档（arXiv/法律/医学）+ 合成的辅助视图与先修知识文本；作者声明数据集与代码已公开（论文中标注为dataset¹和code²）。
- **代码/权重**：代码已开源；使用OLMo-2开源模型（1B/7B/13B/32B）。
- **关键超参**：学习率4e-5（ablation中测试2e-5/4e-5/8e-5）、context size 4096、batch size 256（main）/ 1024（pre-training-faithful）、weight decay 0.1、cosine decay scheduler（warmup ratio 0.1）、AdamW（β₁=0.9, β₂=0.999）、BF16精度、seed 42、max gradient norm 1、N=100次注入。
- **辅助视图生成模型**：GPT-5生成大纲、GPT-5-mini生成正文；paraphrase由GPT-4.1生成。
- **数据泄漏检查**：通过Infini-gram API验证36篇文档未出现在OLMo-2预训练语料中。
