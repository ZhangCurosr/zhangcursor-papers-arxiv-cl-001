---
title: "Who-is-the-Agent-to-Blame-Localizing-Faithfulness-and-Citati"
source: https://arxiv.org/pdf/2608.24306v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 23:40:47"
field: "多智能体系统评估与可信生成"
keywords: ["Deep Research", "citation recall", "faithfulness evaluation", "multi-agent systems", "hallucination detection", "attribution", "agentic AI"]
innovations: ["提出逐agent本地评估+递归溯源的方法，将最终报告引用错误归因到引入错误的特定agent", "定义四分类错误本体（幻觉/依赖未引用输入/无引用输出/引用不充分）用于DR系统故障分析", "基于诊断实施两项简单干预，将AI-Q引用召回率提升5%且不降低输出质量"]
benchmarks: ["DeepResearch Bench", "RACE quality metric", "Citation Recall", "Citation Precision"]
---

# 论文速读：Who-is-the-Agent-to-Blame-Localizing-Faithfulness-and-Citation-Mistakes-in-Agentic-Deep-Research

## 一句话总结
本文提出了一个**逐 agent 归因的评估框架**，用于定位多智能体 Deep Research (DR) 系统中引用召回错误（citation recall errors）的来源与类型，并通过该诊断在 AI-Q 上实施了两项简单干预，将引用召回率提升 5%、引用精确率提升 3-7 个百分点，且不降低输出质量。

## 研究问题与动机
1. **DR 系统引用质量差**：当前 Deep Research 系统（如 AI-Q、MS-Agent）在生成带引用的长报告时，引用召回率普遍偏低（AI-Q 58.7%、MS-Agent 28.5%、TrajectoryKit 7.1%），但缺乏可操作的诊断手段。
2. **信息在多 agent 间传递如同"传话游戏"**：内容经过搜索器→研究员→指挥者等多层压缩与合成，引用与事实可能在跨 agent 边界时失真，现有全局评估无法定位错误源头。
3. **错误类型难以区分**：单纯统计引用召回率无法区分幻觉、依赖未引用输入、缺少引用、引用不充分等不同失效模式，阻碍针对性优化。
4. **已有评估工作的局限**：既往工作（如 LongCite、DEER）聚焦于最终报告的全局引用召回测试，未考虑多 agent 架构下"哪个 agent 引入了该错误"这一归因问题。

## 核心贡献（创新点）
1. **本地化单 agent 调用评估方法**：通过逐句子测试输出对输入的忠实性（faithfulness）与可验证性（verifiability），实现错误归因到具体 agent，而非仅评估最终报告。
2. **四分类错误本体**：区分 hallucination（幻觉）、uncited input reliance（依赖未引用输入）、uncited output（输出无引用）、insufficient citations（引用不完整），为错误分析提供统一术语。
3. **系统性实证诊断**：首次在三个顶级开源 DR 系统（AI-Q、MS-Agent、TrajectoryKit）上同时执行 agent 级统计与 system-level 递归溯源，揭示 orchestrator 主导错误的普遍规律。
4. **基于诊断的干预实验**：证明两条简单干预（替换为原始 snippet、添加"不使用无引用信息"指令）即可显著提升 AI-Q 的引用质量，验证了诊断框架的实践价值。

## 方法详解
- **评估单位**：单个 agent 调用（一次 agent invocation），提取其所有输入（可能来自多次 LLM 调用，如多次搜索）和所有输出。
- **两类输出分别处理**：
  - **合成信息（synthesized information）**：使用五步 LLM-as-a-judge 算法（基于 gpt-5-mini-2025-08-07 作为评判器）：
    - Step 1：忠实性测试——从输入中提取支持目标句子的语料跨度；若无法完全支持则判定为 **hallucination**。
    - Steps 2-3：验证性测试一（依赖未引用输入）——先识别每个输入跨度携带的引用，移除无引用的跨度后检验剩余跨度是否仍蕴含目标句；若不蕴含则判定为 **uncited input reliance**。
    - Steps 4-5：验证性测试二（引用对齐）——检查目标句自身引用是否完整：若无任何引用则为 **uncited output**；若有引用则筛选"引用是目标句引用的子集"的输入跨度，再次检验蕴含关系；若不蕴含则为 **insufficient citations**。
  - **提取片段（extracted snippets）**：直接做蕴含测试，检验源文档是否支持该片段，否则判定为 **snippet missing context**。
- **System-level 递归溯源**：对最终报告中每一句存在引用召回错误的句子，从其所在 agent 开始递归调用上述本地测试：若本地测试通过，则找到支持该句子的输入跨度，沿信息流回溯到生产该跨度的上游 agent，继续递归，直至发现错误或到达原始文档。
- **基座提示工程**：Span 提取基于 LAQuer（Hirsch et al., 2025）；引用分类基于 DEER（Han et al., 2026）；蕴含测试基于 LongCite（Zhang et al., 2025）；全部为已有提示的复用与组合。
- **人工评估**：50 个样本上两两标注（Cohen's κ=0.71），与算法一致率 76%（κ=0.62）；支撑跨度定位一致率 75%，验证了 system-level 追溯的可靠性。

## 实验与结果
- **数据集**：DeepResearch Bench（英文示例），诊断分析用 20 个样本（#51-70），干预实验扩展至 50 个（#51-100）。
- **评估系统**：Nvidia AI-Q（#1）、MS-Agent（#2）、TrajectoryKit（#3），均基于 DeepResearch Bench 开源排行榜（截至 2026 年 5 月）。
- **Agent 级结果（Table 1）**：
  - 单文档搜索器的 snippet 几乎不出错：AI-Q searcher snippet 错误率 3.8%，MS-Agent 为 6.4% / 0.9%；TrajectoryKit 为 14.9%。
  - AI-Q researcher 错误率高达 70.8%，MS-Agent reporter 为 61.6%，TrajectoryKit 所有 agent 均为 50%+（因其统一使用小模型 gpt-oss-20b）。
  - 深层 agent（orchestrator）幻觉更少：AI-Q orchestrator 仅 31% 错误为幻觉（researcher 为 79%）；MS-Agent orchestrator 零幻觉（reporter 为 78%，researcher 为 92%）。
- **System-level 错误来源（Table 2）**：
  - AI-Q：**84.7%** 的最终报告错误归因于 orchestrator；其中 70% 为引用相关错误，31% 为幻觉。
  - MS-Agent：orchestrator（52.6%）与 reporter（47.4%）各占约一半。
  - TrajectoryKit：100% 归因于 orchestrator，且 95% 为幻觉（小模型能力不足导致各阶段均幻觉）。
- **干预实验（Table 3）**：在 AI-Q 上两项干预均将 **引用召回率从 64.5% 提升至约 69.6%（+5%）**，引用精确率提升 3-7pp，RACE 质量指标无显著下降。
  - 干预一（替换为原始 snippet）：C. Recall = 69.7±4.1，C. Precision = 94.1±2.0。
  - 干预二（添加"不使用无引用信息"指令）：C. Recall = 69.6±3.2，C. Precision = 91.0±2.3。
- **干预后溯源（Table 4）**：添加引用引导指令后，orchestrator 错误占比从 84.7% 降至 77.0%，其引用相关错误从 70% 降至 60%，研究者错误占比相应上升至 23.0%。

## 相关工作脉络
1. **Attributed text generation / citation generation**：Gao et al. (2023) LongCite 提出细粒度引用生成评估；本文在其蕴含测试提示基础上扩展，首次应用于多 agent DR 系统的局部归因。
2. **DEER (Han et al., 2026)**：提出引用分类方案（识别哪些引用与句子语义相关）；本文复用了 DEER 的引用分类提示用于本地和系统级评估。
3. **LAQuer (Hirsch et al., 2025)**：局部归因查询，用于从多文档中提取支持某句子的语料跨度；本文将其作为 Step 1 span 提取的基础。
4. **DeepResearch Bench (Du et al., 2025)**：主流 DR 系统评测基准；本文在其英文子集上系统评估三个排名前列的开源 DR 系统。
5. **Webthinker / Webweaver 等 DR 系统**：Li et al. (2025a,b) 和 Team et al. (2025) 提出的 DR 架构；本文对比的 AI-Q、MS-Agent、TrajectoryKit 均共享类似的 searcher-researcher-orchestrator 逻辑角色。
6. **Factuality / Hallucination 评估**：He et al. (2025) 研究长文本生成的精确信息控制；本文关注的是多 agent 传递过程中的错误传播与归因，而非单次生成的幻觉检测。

## 局限性与未来方向
1. **LLM-as-a-judge 可靠性**：虽有人工验证（κ=0.62），但算法仍依赖 gpt-5-mini 判断，在边界案例上可能存在误判。
2. **模型能力的因果推断受限**：观察到"错误类型与模型能力相关"是相关性结论，未做控制实验（同一系统在相同模型下的跨角色对比），需未来工作验证。
3. **仅评估文本句子**：表格内容（含引用标记的数字单元格）被排除在外，未纳入引用召回统计，可能低估实际错误。
4. **仅针对 AI-Q 做了干预实验**：虽然发现了通用规律，但改进措施仅在 AI-Q 上验证，其他系统的泛化性待考察。
5. **未来方向**：将系统级诊断结果作为 critique signal 反馈到生成循环中（post-hoc mitigation），实现自动化纠错；扩展到多语言场景（当前仅英文）。

## 研究启发与可借鉴点
1. **Local vs. Global 评估分离思路**：将全局性能指标拆解为逐 agent 本地测试，再通过递归溯源建立系统级错误归因——该方法可迁移到其他多 agent 管道（如代码生成 pipeline、多 hop QA 系统）的故障定位。
2. **四分类错误本体的通用性**：幻觉/未引用输入依赖/无引用输出/引用不充分这四种错误模式具有清晰的语义边界，可作为 DR 类系统评估的通用分析框架。
3. **简单 prompt 干预即有效**：一条"不使用无引用信息"的指令即可获得 5% 的召回率提升，说明 orchestrator 的引用对齐问题有较大的优化空间，提示工程中显式约束引用完整性是高价值方向。
4. **Snippet 替换策略的架构启示**：用单文档 raw snippet 替代多文档合成笔记作为 orchestrator 输入，可减少跨文档聚合时的引用丢失——这一设计对多 agent RAG 系统有直接参考价值。
5. **Trace-based 适配器模式**：通过为不同系统编写轻量 trace 解析适配器来统一评估接口（Section I），使得评估框架可在不修改目标系统代码的前提下应用，为后续工作提供了可扩展的工程范式。

## 关键术语表
**Deep Research (DR)**：自主搜索网络并合成带引用长报告的 multi-agent 系统，代表工作包括 AI-Q、MS-Agent 等。
**Citation Recall**：被正确引用的陈述占总需引用陈述的比例，是 DR 系统事实性的核心指标。
**Faithfulness（忠实性）**：agent 输出是否忠实于其输入（即不包含幻觉），通过 span 提取与蕴含测试检验。
**Verifiability（可验证性）**：输出中每个引用标记是否与输入中对应的引用标记一致，涵盖引用完整性和对齐性。
**Orchestrator（指挥者 agent）**：DR 系统中负责最终报告合成的顶层 agent，本文为主要错误来源。
**Hallucination（幻觉）**：输出包含输入中不存在的信息，即 no-span-found 或 entailment 失败的错误类型。
**Uncited Input Reliance（依赖未引用输入）**：输出依赖了输入中未被任何引用支持的信息单元。
**Insufficient Citations（引用不充分）**：输出虽有引用，但未包含其依赖的输入跨度的全部引用。

## 可复现要素
- **数据集**：DeepResearch Bench（英文 subset），Apache License 2.0，公开可用；具体使用 #51-70（20 例）用于诊断分析，#51-100（50 例）用于干预实验（论文未明确提供独立下载链接，需从原项目获取）。
- **代码**：论文未提供独立开源代码仓库；评估框架依赖三个系统的 trace 解析适配器（见 Section I 描述），需自行实现或联系作者获取。
- **权重/模型**：评估判官使用 gpt-5-mini-2025-08-07（闭源）；DR 系统模型见 Table 7（AI-Q：GPT-5.2 + Nemotron Nano 30B；MS-Agent：GPT-5/GPT-5 mini/GPT-5 nano；TrajectoryKit：gpt-oss-20b）。
- **关键超参**：span 提取重试上限 K=5；每 agent 最多采样 10 个句子；人工评估 50 个样本，2 名标注员+仲裁机制。
- **提示词**：完整提示见 Appendix B（Figure 6-11），基于 LAQuer、DEER、LongCite 已有提示适配，可复现。
