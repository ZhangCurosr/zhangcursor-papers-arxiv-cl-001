---
title: "Unseen-Harm-Measuring-Cross-Script-Safety-Inconsistency-with"
source: https://arxiv.org/pdf/2608.24191v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:37:58"
field: "多语言LLM安全评估"
keywords: ["hate speech detection", "cross-script inconsistency", "Urdu NLP", "LLM safety evaluation", "Missed-in-Urdu", "low-resource languages"]
innovations: ["提出Missed-in-Urdu指标量化跨脚本安全不一致性", "完整审计WOAH文献证实乌尔都语十年零研究", "六数据集四条件对照实验设计隔离脚本/转写/混码效应"]
benchmarks: ["HateInsights", "Cyberbullying", "HS-RU-20", "Abusive Tweets", "RU-EN Emotion", "HateXplain"]
---

# 论文速读：Unseen-Harm-Measuring-Cross-Script-Safety-Inconsistency-with

## 一句话总结
本文首次系统评估了五大主流LLM在乌尔都语（Urdu）多脚本条件下的仇恨言论检测一致性，提出"Missed-in-Urdu"指标揭示模型对同一内容在不同脚本下安全判断的显著偏差，并证实WOAH/WOAH十年文献中乌尔都语相关研究为零。

## 研究问题与动机
- 乌尔都语是全球第十大语言（2.46亿使用者），但在WOAH系列研讨会过去九年的205篇论文中完全没有专门研究，形成严重的"安全评估盲区"。
- 乌尔都语存在多种书写变体（Nastaliq阿拉伯 script、Roman Urdu拉丁转写、Urdu-English混合代码），不同脚本下模型行为可能存在系统性差异，但现有工作未对此进行量化。
- 现有LLM安全评估主要依赖英语或翻译后文本，忽略原始脚本检测，导致对非英语用户的"虚假安全保证"——模型可能正确识别英文中的有害内容，却在原始乌尔都语脚本中"沉默放行"。
- 开源小模型（如Qwen-2.5、Llama-3.1）与前沿闭源模型在乌尔都语安全性能上是否存在可量化的差距，以及这种差距是否具有统计显著性，尚未有系统研究。

## 核心贡献（创新点）
- **完整WOAH文献审计**：通过ACL Anthology API穷举205篇ALW/WOAH论文，量化证实乌尔都语在过去十年中零代表性，而德语、意大利语等使用人口更少却有专门研究。
- **首创"Missed-in-Urdu"指标**：定义并测量"在英文翻译中被识别为有害、但在原始Nastaliq乌尔都语中被标记为正常"的比例，揭示跨脚本安全不一致性的隐蔽危害。
- **六数据集跨脚本对照实验设计**：通过C1（原始Nastaliq）、C2（英文翻译）、C3（Roman Urdu转写）、C4（混合代码代理）四个条件隔离脚本/转写/混码的影响，HateXplain作为零不稳定性控制组排除pipeline误差。
- **统计验证框架**：结合McNemar检验、卡方检验和Stuart-Maxwell检验，证明模型间的不稳定性差异具有统计显著性（p < 0.0005），排除随机噪声解释。

## 方法详解
**实验条件设计**：
- C1：原始Nastaliq乌尔都语文本直接分类
- C2：由同一模型翻译为英文后分类，隔离脚本效应而非翻译质量差异
- C3：使用HS-RU-20数据集的Roman Urdu转写，隔离转写效应
- C4：用RU-EN Emotion作为Urdu-English混合代码的代理（因缺乏专属数据集）
- 控制组：HateXplain（纯英文）验证pipeline稳定性

**模型与评估**：
- 五个LLM零样本分类：GPT-4o、Claude Sonnet 4.5、Gemini 2.5 Flash、Qwen-2.5-7B-Instruct、Llama-3.1-8B-Instruct
- 统一提示模板要求输出HATE/OFFENSIVE/NORMAL三类标签之一
- 核心指标：Label Instability（C1 vs C2标签分歧率）和Missed-in-Urdu（C2标记为有害但C1标记为正常的内容比例）

**统计检验**：
- McNemar检验用于配对二元不稳定性比较（关注不一致对）
- 卡方检验用于Missed-in-Urdu率的双边比较
- Stuart-Maxwell检验扩展McNemar至三分类条件差异
- Bonferroni校正（α = 0.005，10次比较）控制族系误差率

## 实验与结果
**数据集与规模**：
- 六个数据集：HateInsights (N=1000)、Cyberbullying (N=916)、Abusive Tweets (N=656)、HS-RU-20 (N=835)、RU-EN Emotion (N=989)、HateXplain控制组 (N=1000)
- 总有效样本：约22,732条（每模型4,531-4,553条）
- 所有数据集于2021-2024年发布，模型于2024年中-2025年初推出，确保评估反映当代SOTA性能

**主要结果**（Table 3）：
| 模型 | Label Instability | Missed-in-Urdu |
|------|-------------------|----------------|
| GPT-4o | 18.0% | 2.4% |
| Claude Sonnet 4.5 | 16.3% | 3.6% |
| Gemini 2.5 Flash | 15.9% | 4.3% |
| Qwen-2.5-7B | **31.6%** | **9.9%** |
| Llama-3.1-8B | **27.3%** | **6.5%** |
| 中位数(SD) | 18.0% (7.1%) | 4.3% (2.9%) |

- 所有模型均在Nastaliq与英文翻译之间产生分歧，中位不稳定性18.0%，Open模型约为Frontier模型的**两倍**
- Qwen-2.5的Missed-in-Urdu率最高（9.9%），意味着近十分之一在英文中被判为有害的内容在原始乌尔都语中被"默默放行"
- 统计检验证实：三大前沿模型在不稳定性上无显著差异；前沿vs开源模型的对比全部极显著（p < 0.0005）
- Stuart-Maxwell检验显示每个模型的C1→C2标签分布均发生显著偏移（p < 0.01）

**RQ1结论**：205篇WOAH论文中乌尔都语零代表，Bengali（全球第七大语言）同样为零。

**RQ3结论**：缺乏专门的Nastaliq Urdu-English代码混合仇恨数据集本身即为结构性资源缺口；同时发现1,216条原始标注为Hate的内容被所有模型判为Normal，提示潜在标注偏见。

## 相关工作脉络
- **WOAH文献生态**：该会议自2017年ALW起聚焦在线辱骂检测，但语言覆盖高度偏向英语（100+篇）、德语（5-6篇）和印地语（6-8篇），乌尔都语、孟加拉语为零，反映NLP研究的地域性偏差。
- **LLM零样本仇恨检测**：Ghorbanpour et al. (2025) 评估八种非英语语言，覆盖印地语、西班牙语、法语、阿拉伯语、葡萄牙语，但遗漏汉语普通话、孟加拉语、俄语和乌尔都语——后者与本文发现的空白完全吻合。
- **翻译失效研究**：Chan et al. (2024) 直接证明翻译对代码混合内容无效，检测性能显著下降，支持本文"必须评估原始脚本"的方法论立场。
- **低资源跨语言转移失败**：Nozza (2021) 证明零样本跨语言仇恨检测对低资源语言失效，为本文的跨脚本评估设计提供理论基础。
- **Urdu仇恨检测单一工作**：Ahmad et al. (2025) 使用LLM在单一数据集上检测乌尔都语仇恨，超越BERT但未涉及跨脚本一致性；本文首次系统化这一维度。
- **平行低资源案例**：Francielle Vargas (2024) 对豪萨语（Hausa）的研究与本文形成结构性类比，表明低资源语言仇恨研究空白是系统性而非孤立问题。

## 局限性与未来方向
- **样本量限制**：部分数据集规模较小（Abusive Tweets仅N=700目标，HS-RU-20实际85%达成），可能影响统计检验效力。
- **翻译-分类耦合**：C2翻译与分类由同一模型完成，无法分离翻译质量与分类一致性的贡献；使用外部翻译器可解决此混淆。
- **标注偏见未被穷究**：1,216条模型与黄金标注分歧的样本未经过人工复核，可能是标注偏差而非模型失败，需定性审查。
- **代码混合代理的局限性**：RU-EN Emotion使用Roman Urdu而非Nastaliq，且情感标签映射为危害标签，仅为近似代理，非理想数据集。
- **API依赖与可复现性**：商业API访问受版本更新影响，长期可复现性受限；Perspective API虽支持18种语言但不包括乌尔都语，进一步凸显生态空白。
- **审计方法局限**：ACL Anthology检索仅基于标题和摘要，未在全文中搜索，可能遗漏未提及乌尔都语但实际使用该数据的论文。

## 研究启发与可借鉴点
- **跨脚本对照实验范式**：C1→C2的"同内容不同脚本"设计可有效隔离脚本效应，适用于其他多脚本语言（如阿拉伯语Naskh/Nastaliq、藏文等）的安全评估迁移。
- **"Missed-in-X"指标的可迁移性**：定义"翻译中被识别但原文中被放过"的隐蔽危害率，可推广至阿拉伯语、希伯来语、梵文等其他多脚本语言的安全审计。
- **开源vs前沿模型的差距量化**：本文证明开源小模型在低资源语言安全性能上显著劣于闭源SOTA，为资源分配决策提供实证依据——开源模型不应假设在低资源语言上"够用"。
- **控制组排除pipeline误差**：使用同语言数据集（HateXplain英文）验证实验流程零不稳定，为后续跨语言评估提供了方法论基准。
- **系统性文献审计的价值**：通过API穷举会议论文确认语言覆盖空白，是一种可复用的"研究生态诊断"工具，可应用于其他 marginalized language 的评估。

## 关键术语表
- **Missed-in-Urdu**：在英文翻译中被模型判定为有害（Hate/Offensive）、但在原始Nastaliq乌尔都语中被判定为正常的比例，衡量脚本依赖型安全盲点。
- **Label Instability**：同一内容在C1（原始Nastaliq）与C2（英文翻译）条件下获得不同分类标签的比例，量化跨脚本一致性。
- **Nastaliq Urdu**：乌尔都语的传统阿拉伯 script 变体，具有独特的书法风格，与Roman Urdu（拉丁转写）在视觉上显著不同。
- **Roman Urdu**：使用拉丁字母拼写的乌尔都语，广泛见于社交媒体，因键盘输入便利性而普及。
- **Code-switching（代码混合）**：在同一话语中交替使用乌尔都语和英语，巴基斯坦社交媒体中的常见现象。
- **Stuart-Maxwell test**：McNemar检验对多分类（任意大小方阵）的扩展，用于检验配对条件下边际标签分布是否相同。
- **McNemar test**：针对配对二元结果的统计检验，关注不一致对（discordant pairs），对方向性错误模式敏感。
- **Frontier vs Open-weight models**：本文区分"前沿闭源模型"（GPT-4o/Claude/Gemini）与"开源权重模型"（Llama/Qwen），前者在乌尔都语安全检测上表现显著更优。

## 可复现要素
- **数据集**：HateInsights、Cyberbullying、Abusive Tweets、HS-RU-20、RU-EN Emotion、HateXplain均为公开数据集或注册后可获取；论文代码匿名发布于 https://anonymous.4open.science/r/WOAHJul26-047C，接受后替换为去匿名链接。
- **模型访问**：通过OpenAI、Anthropic、Google、Together.ai、Groq API访问，依赖商业API版本，长期可复现性受限。
- **关键超参**：零样本设置，无in-context示例；统一prompt模板（见Appendix A）；Bonferroni校正α=0.005（10次比较）。
- **环境**：Windows + Anaconda虚拟环境，Python，API速率限制为主要瓶颈（尤其Groq免费tier）。
- **统计工具**：McNemar检验、卡方检验、Stuart-Maxwell检验，Bonferroni校正。
