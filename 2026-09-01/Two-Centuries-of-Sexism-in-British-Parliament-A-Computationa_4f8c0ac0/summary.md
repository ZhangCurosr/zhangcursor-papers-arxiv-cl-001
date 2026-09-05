---
title: "Two-Centuries-of-Sexism-in-British-Parliament-A-Computationa"
source: https://arxiv.org/pdf/2608.30485v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:46:53"
field: "计算社会科学与历史NLP"
keywords: ["矛盾性别歧视", "Hansard语料库", "立场分类", "大型语言模型", "英国议会", "女性选举权", "计算社会科学"]
innovations: ["将矛盾性别歧视量表引入两百年历史政治话语分析", "发布89%性别匹配的670万篇Hansard数据集", "揭示支持方与反对方使用截然不同的性别修辞策略"]
benchmarks: ["ParlVote", "SST-2情感分类", "Cohen's kappa人工标注验证"]
---

# 论文速读：Two-Centuries-of-Sexism-in-British-Parliament-A-Computational-Analysis

## 一句话总结
本文利用大型语言模型对1803-2005年英国下议院6,531篇女性选举权相关演讲进行立场与矛盾性别歧视（敌意性/benevolent）分类，揭示反对与 supporting 方使用了截然不同的性别修辞策略，并发布含670万篇演讲、89%实现性别匹配的Hansard数据集。

## 研究问题与动机
- 英国议会两百年来关于女性政治代表权的辩论，其语言中是否编码了系统性的性别歧视模式？
- 支持与反对女性选举权的阵营，是仅结论不同，还是在性别推理的**类型**上存在根本差异？
- 矛盾性别歧视理论（Ambivalent Sexism Theory）如何在真实的制度性话语中得到验证？
- 现有计算社会科学工作多关注立场/情感，缺乏对性别歧视**类型**的系统分类。

## 核心贡献（创新点）
1. **发布大规模性别匹配Hansard数据集**：包含670万篇演讲、120万场辩论，下议院89.3%实现演讲者性别匹配，为计算社会科学研究提供基础设施。
2. **将矛盾性别歧视量表引入历史政治话语分析**：首次在制度性语料上验证Glick & Fiske的敌意性/ benevolent性双重维度，证明LLM作为"judge"可捕捉超越表层情感的社会心理学构念。
3. **揭示立场与性别歧视类型的系统性耦合**：反对派54%含性别歧视（敌意+ benevolent混合），支持派21%含性别歧视（几乎全是benevolent），证明"支持女性权利"与"性别本质主义推理"可并存。
4. **证明benevolent sexism的持久性与隐蔽性**：敌意性性别歧视从1870s的60%降至1929年后的27-30%，而benevolent sexism稳定在74-83%，提示伪装成"赞扬"的父权修辞更难识别与挑战。

## 方法详解
- **数据提取**：两级关键词检索——Tier 1精确匹配（如"women's suffrage""votes for women"），Tier 2宽泛匹配（women/female与vote/franchise等词25词内共现），共获6,531篇演讲。
- **性别匹配**：级联策略——议员头衔（Mr/Ms/Duke等）→ WikiData查询 → Wikipedia议员列表模糊匹配（Levenshtein距离）→ Gender Guesser库（97%准确率），优先保证精度避免误标性别。
- **立场分类**：使用Claude Sonnet 4.6，每篇目标演讲附前后各5篇上下文，分类为For/Against/Both/Irrelevant四类，强制JSON输出。
- **性别歧视分类**：基于Glick & Fiske (1996)的Ambivalent Sexism Inventory，分两道并行标记：Hostile Sexism（支配性父权、竞争性性别差异、异性恋敌意）与Benevolent Sexism（保护性父权、互补性性别差异、异性恋亲密），每类需引用原文佐证。
- **人工验证**：300篇随机抽样构建验证集，两名具备历史话语经验的标注员独立标注，Cohen's κ=0.644；LLM与人类共识标签κ=0.711。
- **控制分析**：情感混淆检验（DistilBERT on SST-2）确认敌意与benevolentSpeech均 predominantly negative sentiment（87%/77%），排除纯情感驱动。

## 实验与结果
- **数据集规模**：Hansard 1803-2005，总计6,783,015篇演讲、52,661名发言者；下议院5,575,783篇，89.3%性别匹配（男性议员4,829,844篇，女性150,867篇）。
- **立场分布**：6,531篇中55%被判定为Irrelevant，剩余2,942篇中74% For、19% Against、7% Both；对立派随时间递减，1950年后几乎消失。
- **性别差异**：女性议员93%支持 vs 男性70%支持（χ²=86.75, p<0.001）；logistic回归控制年代后性别仍显著（OR=2.01, p=0.002）。
- **性别歧视率**：2,942篇相关演讲中886篇（30%）含至少一种性别歧视；其中敌意性392篇、benevolent性706篇、两者兼具212篇。
- **核心发现**：
  - For演讲：21%含性别歧视，其中81%仅为benevolent性，仅2%为Hostile-only。
  - Against演讲：54%含性别歧视，37% Hostile-only、11% Benevolent-only、44%两者兼具。
  - 时间趋势：敌意性从1870-1899的60%降至1929年后的27-30%，benevolent性稳定在74-83%。
- **跨模型验证**：Claude Sonnet 4.6 (κ=0.711) > GPT-5 (κ=0.692) > DeepSeek V3 (κ=0.658) > Gemini 2.5 Flash (κ=0.533)，结论不因单一模型偏差产生。

## 相关工作脉络
- **议会语料NLP分析**：Abercrombie & Batista-Navarro (2018, 2020) 构建ParlVote情感标注语料，本文扩展至性别立场与矛盾性别歧视分类；Slapin & Proksch (2008) Wordfish意识形态缩放方法仅基于词频，无法捕捉修辞类型。
- **性别与政治话语**：Hargrave & Blumenau (2022) 发现女性议员风格趋近"男性化"，本文补充证明即便风格趋同，benevolent sexism仍潜伏于支持者论述中。
- **立场检测**：Duthie et al. (2016) 检测ethotic appeals极性，本文引入时间序列与性别类型维度；Card et al. (2022) 分析140年美国移民话语，本文将其方法迁移至性别议题。
- **性别歧视NLP**：Guest et al. (2021)、Zeinert et al. (2021) 构建在线内容性别歧视标注数据集，本文首次将ASI框架应用于制度性历史语料；Kirk et al. (2023) SemEval任务聚焦线上sexism检测，本文强调历史文本的archaic language增加了标注难度。
- **LLM偏见评估**：Kotek et al. (2023)、Ding et al. (2025)、Mirza et al. (2025) 评估LLM自身性别刻板印象，本文反向使用LLM作为分析工具检测历史话语中的偏见，方法论互补。

## 局限性与未来方向
- **单一主要LLM judge**：虽经多模型交叉验证，但Claude Sonnet 4.6仍是主力分类器，可能存在模型特定的判断偏好。
- **关键词检索遗漏**：Tier 2关键词可能遗漏未使用明确选举权术语但实质讨论女性政治权利的历史演讲（如"franchise"指男性选举权时的false positive）。
- **仅分析下议院**：上议院性别匹配覆盖率仅1.2%，故排除；但上议院的贵族话语可能呈现不同模式。
- **benevolent sexism检测难度更高**：敌意性更外显易识别，benevolent性嵌入"正面赞扬"中更难捕捉，导致实际benevolent sexism比例可能被低估。
- **二元性别框架局限**：历史记录的性别分类为男性/女性二元，未覆盖non-binary身份；研究承认性别是光谱。
- **制度性话语的特殊性**：议会演讲受正式性、说服策略与观众预期塑造，可能不同于私人信念表达。

## 研究启发与可借鉴点
1. **矛盾性别歧视框架的计算化应用**：将社会心理学量表（ASI）转化为LLM可执行的分类schema，为其他偏见类型（如种族主义、年龄歧视）的历史话语分析提供模板。
2. **上下文增强的LLM-as-judge策略**：每篇目标演讲附前后各5篇上下文，模拟人类标注员的理解情境，显著提升立场分类质量（κ=0.711接近人类水平）。
3. **噪声鲁棒性检验**：通过1,000次随机翻转3%性别标签的噪声诱导实验，证明性别-立场关联结论对标注误差稳健，该方法值得在其他社会敏感维度研究中复用。
4. **情感混淆控制设计**：用独立情感分类器检验目标构念（敌意vs benevolent）是否只是情感的proxy，排除了"支持方态度更正面的"替代解释，这一控制逻辑可迁移至其他多维分类任务。
5. **数据集增强范式**：从公开档案（Hansard）出发，通过多级信息源（EveryPolitician、WikiData、Wikipedia）级联匹配元数据（性别），以高精度优先原则实现89%覆盖率，为历史语料结构化提供可复用流程。

## 关键术语表
- **Ambivalent Sexism Inventory (ASI)**：Glick & Fiske (1996)提出的性别歧视二分框架，区分敌意性性别歧视（ overtly negative）与benevolent性别歧视（ superficially positive but restrictive）。
- **Hostile Sexism (HS)**：显式贬低女性、认为女性低劣或具有威胁性的态度，包括支配性父权、竞争性性别差异、异性恋敌意三种子类型。
- **Benevolent Sexism (BS)**：以"保护""赞美"为名的性别歧视，将女性本质化为纯洁、柔弱、需要男性庇护，进而合理化其角色限制。
- **Complementary Gender Differentiation**：主张男女具有互补而非竞争的特质（如女性更有道德感、温柔），是benevolent sexism的核心机制。
- **Protective Paternalism**：benevolent sexism子类型，主张男性应为女性做决定、提供保护，隐含女性缺乏自主能力。
- **Hansard Corpus**：英国议会自1803年以来的官方辩论记录数据库，本文提取670万篇演讲并进行性别元数据增强。
- **LLM-as-a-Judge**：使用大型语言模型作为自动分类器/评估器，替代或部分替代人工标注，本文通过交叉模型验证与人类标注对比评估其可靠性。
- **Suffrage**：选举权运动，本文特指19-20世纪英国女性争取投票权与担任公职的政治斗争。

## 可复现要素
- **数据集**：Hansard Parliamentary Corpus (UK Parliament, 2018) 公开可用；本文发布的增强版本含性别元数据，论文声明将开源（见脚注2）。
- **代码/权重**：论文未提供GitHub链接或模型权重；分类使用Claude Sonnet 4.6 API，无开源代码。
- **关键超参**：LLM使用Anthropic API；上下文窗口为前后各5篇演讲；关键词匹配Tier 1/2阈值见Appendix A；验证集300篇均匀采样；噪声鲁棒性检验1,000次迭代、3%标签翻转率。
- **标注协议**：两名标注员独立标注，Cohen's κ=0.644，分歧通过讨论解决；强制JSON输出schema见Appendix C。
