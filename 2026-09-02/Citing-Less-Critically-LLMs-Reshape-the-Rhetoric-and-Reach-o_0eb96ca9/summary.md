---
title: "Citing-Less-Critically-LLMs-Reshape-the-Rhetoric-and-Reach-o"
source: https://arxiv.org/pdf/2609.01432v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:23:18"
field: "大语言模型与科学传播"
keywords: ["scientific citation", "LLM citation behavior", "citation intent", "rhetorical analysis", "coauthorship network", "citation bias", "masked-citation task"]
innovations: ["提出位置对齐的遮蔽引用槽位级基准，实现人类与LLM引用的反事实对照", "首次从修辞意图维度揭示LLM引用系统性语气变暖及偏向放大效应", "基于2000万边合著网络测量作者社交距离，发现LLM放弃人类近邻引用习惯"]
benchmarks: ["ACL/EMNLP/NAACL 2025", "Dimensions bibliometric database", "20.3M-edge coauthorship network 2015-2024"]
---

# 论文速读：Citing-Less-Critically-LLMs-Reshape-the-Rhetoric-and-Reach-o

## 一句话总结
本文构建了遮蔽引用任务（masked-citation task），对比人类与六大主流LLM在相同语境下的引用行为，发现LLM生成的引用存在系统性"语气变暖"现象——显著减少对比性引用、放大流行度/时间偏向，并放弃人类学者依赖近邻社交网络的习惯，从而重塑科学引用的修辞策略与覆盖范围。

## 研究问题与动机
- 科学引用承载修辞意图（支持/对比/提及），LLM日益参与学术写作，但其引用行为是否复现人类的修辞意图分布尚不清楚。
- 现有LLM引用审计工作多关注可靠性（幻觉/误引）、引用选择或作者人口统计差异，缺乏从**修辞意图**维度系统比较LLM与人类引用的工作。
- 引用选择受多重社会因素塑造（累积优势、Matthew效应、合著同群性），这些偏向在LLM生成引用中如何体现未知。
- 前人工作将引用视为同质检索输出，未保留每个遮蔽引用槽位的局部修辞语境，也未对引用意图进行条件化比较。

## 核心贡献（创新点）
- **提出位置对齐的槽位级基准**：每个LLM引用都是在同一修辞语境下对人工选择的反事实替代，实现句级别的人类-模型对照。
- **首次意图条件化的LLM/人类引用比较**：揭示LLM少产对比性引用，且流行度与时间偏向在修辞意图维度上系统性放大。
- **作者层面社会结构分析**：基于2030万边的合著网络测量引文-被引作者距离，发现人类倾向引用近邻社交圈（尤其支持性引用），而LLM引用更社交疏远的作者。
- **揭示LLM引用的双刃效应**：虽可拓展至学者社交圈之外的文献，但语气更温和、偏向经典老作，可能侵蚀学术批判性。

## 方法详解
- **遮蔽引用重建任务**：从1746篇ACL/EMNLP/NAACL 2025主会论文中提取63944个引用上下文，将引用句替换为占位符，LLM根据上下文、章节标题、引文数量等约束生成反事实引用句，温度设为0。
- **LLM-as-judge意图标注**：使用Gemini-3-Flash-Preview作为独立评委，仅输入引用句及±1句语境（不呈现被引论文标题），将每句标注为supporting/contrasting/mentioning，并与DeepSeek-V4-Flash交叉验证。
- **引用匹配与属性提取**：通过DOI（缺失则用标题）将引用链接到Dimensions数据库，获取团队规模、发表时间、引用计数等元数据。
- **合著网络构建与距离计算**：基于2015-2024年间104k+匹配论文构建210万节点、2030万边的无向合著图，用BFS计算引文作者与被引作者四组角色对的 shortest-path distance，汇总为每上下文平均距离⟨d⟩。

## 实验与结果
- **数据集**：1746篇NLP顶会论文（ACL 668篇、EMNLP 940篇、NAACL 138篇），共63944个引用上下文、132913个引用槽位；六款LLM参与测试。
- **RQ1结果**：人类引用中supporting占21%、contrasting占19%、mentioning占60%；所有LLM的contrasting比例降至10.7%-17.2%（DeepSeek最高17%，Claude-3.5/Llama-4仅11%），五款LLM的supporting比例超人类（GPT-5.1达41%）。人类contrasting语境被LLM保留的仅为34.6%-50.6%，约一半被改写为supporting。
- **RQ2结果**：LLM的引用偏向受意图调节。supporting时LLM引用高影响力论文最多（DeepSeek达人类4.45倍）；contrasting时LLM引用论文比人类老1.6-3.3年（Gemini-2.0达+3.27年）；mentioning时LLM引用小团队论文最多（Qwen-72B仅人类团队的0.30倍）。
- **RQ3结果**：人类平均社交距离⟨d⟩=3.40，六款LLM均在3.65-3.89之间。人类supporting引用社交距离显著更近（⟨d⟩=3.31 vs. contrasting 3.45），而LLM各意图间距离差异不超过±0.05跳。人类in-network引用率（d≤1）为7-10%（supporting达9.8%），LLM均低于1.6%。
- **最强结果**：对比GPT-5.1开启live web search后contrasting仍仅10.2%，证实"语气变暖"源于生成阶段而非检索局限。

## 相关工作脉络
- **引用可靠性审计**（Press et al., 2024; Linardon et al., 2025）：聚焦LLM幻觉与误引，本文进一步分析引用修辞意图的系统性偏差。
- **引用选择研究**（Algaba et al., 2025, 2026）：发现LLM偏向高影响力小团队作品，本文揭示该偏向受修辞意图调节且随意图放大。
- **作者差异研究**（Tian et al., 2024; He, 2025）：关注性别/种族/国家偏差，本文转向引用修辞角色与社会距离维度。
- **引用意图分类**（Teufel et al., 2006; Jurgens et al., 2018; Cohan et al., 2019）：传统NLP工作分类引用功能，本文将其用于审计LLM引用行为。
- **科学计量与社会网络**（Merton, 1968; Newman, 2001; Wallace et al., 2012）：刻画人类引用的累积优势、合著同群性，本文检验LLM是否复现这些社会偏向。
- **scite.ai intent分类**（Nicholson et al., 2021）：本文意图定义借鉴其supporting/contrasting/mentioning三分法。

## 局限性与未来方向
- LLM-as-judge的意图标签依赖模型判断，虽经人工验证（κ=0.60）且双评委交叉验证，但仍可能与人类标注存在系统性差异。
- 合著网络覆盖率低（人类26.5%、LLM 14.5-25.5%可达），早期研究者、非西方机构、产业界学者被系统性低估。
- 语料仅限英语NLP顶会论文，结论在其他学科/语言/地理区域的外推性待验证。
- 未深入解释LLM为何在不同意图下呈现差异化偏向，需结合训练数据统计特征进一步分析。
- 未来可通过agent式检索增强、提示工程干预或微调来缓解上述偏差，并拓展至更多学科领域审计。

## 研究启发与可借鉴点
- **位置对齐的遮蔽重建设计**：通过在相同语境下强制模型生成反事实引用，实现对齐比较，避免传统基准中常见的位置/长度混淆，可迁移至其他生成质量审计场景。
- **意图条件化的偏差分析框架**：将引用属性偏差按修辞意图分层考察，揭示人类-模型差异的结构性根源，该方法可推广至其他文体生成任务（如法律、医疗引用）。
- **合著网络社交距离测量**：将作者关系量化为图距离，揭示LLM放弃人类"就近引用"习惯的现象，此指标可用于评估推荐系统的多样性与社会公平性。
- **检索增强条件下偏差持续性验证**：通过加入live web search条件排除检索局限的解释，强化因果推断，此设计思路适用于任何声称"检索可解决生成偏差"的工具审计。

## 关键术语表
**Masked-citation task**：将原文引用句遮蔽为占位符，让LLM在相同语境下生成反事实引用，用于对比人类与模型引用行为的实验范式。
**Citation intent**：引用的修辞意图，分为supporting（支持）、contrasting（对比/反驳）、mentioning（提及）三类。
**LLM-as-judge**：使用独立LLM作为评判器，仅基于引用句及局部语境标注意图，不呈现被引论文标题以避免内容偏好干扰。
**Social distance (coauthorship network)**：引文作者与被引作者在合著网络中的最短路径距离，d=0为自引，d=1为直接合作者。
**In-network citation rate**：引文作者与被引作者处于同一连通分量且距离≤1的比例，衡量"圈内引用"倾向。
**Intent-conditioned bias**：引用选择偏向（如流行度、时间、团队规模）因修辞意图不同而呈现系统性差异的现象。
**Citation warming**：LLM引用语气比人类更温和的系统性偏差，表现为contrasting减少、supporting增加。
**Matthew effect in citations**：科学引用中的马太效应，高影响力论文获得更多引用，形成累积优势。

## 可复现要素
- **数据集**：1746篇ACL/EMNLP/NAACL 2025主会论文（arXiv提供.tex/.bib），DOI匹配使用Dimensions数据库。
- **代码/权重**：论文声明将开源代码及去标识化的聚合输出，但未提供具体仓库链接。
- **关键超参**：生成温度=0；judge模型使用Gemini-3-Flash-Preview及DeepSeek-V4-Flash交叉验证。
- **合著网络**：基于2015-2024年104k+匹配论文构建，团队规模上限25人，无向边连接共著作者。
- **附录资源**：Prompt模板（Figure 6-8）、意图示例（Table 4）、稳健性分析（Appendix C-H）均提供细节。
