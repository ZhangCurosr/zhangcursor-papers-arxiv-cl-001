---
title: "Beyond-Verdicts-A-Graph-Based-Analysis-of-Human-and-LLM-Reas"
source: https://arxiv.org/pdf/2608.23047v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 15:39:40"
field: "可解释 AI / 科学事实核查"
keywords: ["scientific fact-checking", "reasoning graph", "LLM explanation", "fallacy analysis", "human-LLM alignment", "grounding validation"]
innovations: ["有类型推理图实现人类与LLM在谬误子图层面的过程级对齐", "非对齐路径的grounding+relevancy+sufficiency三阶段验证流水线", "揭示裁决准确率与推理路径质量脱钩的实证结论"]
benchmarks: ["MISSCIPLUS", "SciFact", "HealthFC"]
---

# 论文速读：Beyond Verdicts: A Graph-Based Analysis of Human and LLM Reasoning in Scientific Fact-Checking

## 一句话总结
本文提出一种**基于有类型推理图的框架**，将科学事实核查中的人类专家解释与 LLM 解释转化为结构化推理图，在"谬误特异性子图"层面进行一一比对，并对未与人类对齐的 LLM 推理路径进行 grounding、相关性、充分性验证；实验表明，最终裁决准确率与推理路径质量**存在显著脱钩**——错误召回最高的模型并非人类对齐率最高的模型。

## 研究问题与动机
- **现有评估仅看结果，不看过程**：已有自动事实核查系统主要评测裁决正确率、证据检索、谬误标签或自由文本解释，却不考察 LLM 是否沿着与人类专家相同的路径抵达结论。
- **"不同路径未必是错误路径"**：LLM 可能通过不同于专家但同样有效（ grounded、relevant、sufficient）的推理链得出相同 Incorrect 裁决，单纯以裁决准确率评判会高估推理质量。
- **科学谣言的"曲解式"危害**：引用真实研究却歪曲其结论的谣言最具迷惑性，必须同时考察"证据如何被解读"及"从研究结果到裁决的推理链如何构建"。
- **前序工作（MISSCI/MISSCIPLUS）聚焦谬误重建与分类，尚未进入"过程级比较"**：这些工作识别并标注谬误，但未在推理图层面对齐人类与 LLM 的解释，也未对非对齐路径做有效性检验。

## 核心贡献（创新点）
1. **有类型推理图（Typed Reasoning Graph）框架**：把每条解释分解为 Claim → Study Context → Study Findings → Fallacy-Supporting Premises → Fallacies 五类节点的有向图，实现人类与 LLM 在谬误特异性子图层面的逐一比对。*本质区别：从终值评估跃迁至过程级图对齐。*
2. **非对齐子图有效性验证流水线**：对未与人类对齐的 LLM 子图进行两阶段验证——QA-based grounding（逐组件与原文对照）+ 三 LLM 裁判的 relevance & sufficiency 投票。*区别于现有 faithfulness 评估：不仅看文本是否与原文一致，更要求子图对 Incorrect 裁决构成充分支撑。*
3. **过程-结果解耦的实证结论**：在 84 条 MISSCIPLUS 假说上，GPT-5 人类对齐率最高（19.0%），Qwen3-32B 错误召回最高（76.98%）但仅 10.2% 人类对齐；说明"裁决最准 ≠ 推理最像人"。*这一发现直接挑战了以 Incorrect 率为核心指标的评测范式。*
4. **Graph Constructor + 人工复检的双重校准**：用 GPT-5 做自动图构造器，并通过与人类标注员的一致性验证其可靠性；后续对 QA 层过检的未 grounded 组件再经人工二次复检，提升 Grounded (final) 比例 5–19 个百分点。

## 方法详解
- **任务形式化**：实例为 (claim c, cited study s)，人类参考为 (v, x)，LLM 输出 (v', x')；分析时 verdict 取值 {Correct, Incorrect, Not Enough Information, No Response}，期望均为 Incorrect。
- **推理图定义**：$G=(V,E)$，节点类型包含 Claim / Study Context / Study Findings / Fallacy-Supporting Premises / Fallacies；边遵循"证据→前提→谬误"的链式顺序。
- **谬误特异性子图（Fallacy-Specific Sub-graph）**：把单条解释按每个谬误类型切分为子图，每个子图含对应 Study Context、Study Findings 和一个 Fallacy-Supporting Premise，作为对齐与验证的基本单位。
- **图构造器**：用 GPT-5 从自由文本解释中**逐字抽取**（verbatim，禁止改写/摘要/ paraphrase）各组件，并依据 MISSCI 九类谬误词表分配 Fallacy 标签（见附录 Figure 10 提示词）。
- **子图对齐判据**：用三名裁判（GPT-5 / Qwen3-32B / Claude Opus 4.7）对每对文本组件打分 STS(0–5)，多数投票 + 无平局时取中位数；总分 ≥3 视为语义匹配，同谬误类型 + 各组件均匹配则判定为 Human-Aligned。
- **Grounding 验证**：对非对齐子图的每个组件生成 5 个事实问题，分别用"组件自身"和"原文全文"作答，人工对比答案一致性；任何一对不一致则初判为 ungrounded，再经同一作者盲复检二次确认。
- **Relevance & Sufficiency 验证**：对通过 grounding 的子图，三名裁判按 0/1/2 三级分别打分，多数投票聚合；仅当两者均得 2 时才给出 combined score = 2，进而决定 Accepted。
- **裁决分布统计**：每张输入重复 3 次，按 $p_{i,\ell} = n_{i,\ell}/3$ 计算每条 claim 的 label 比例，再跨 N 条取均值 $\bar{p}_\ell$；同时报告 consistency（3 次全同的比例）。

## 实验与结果
- **数据集**：MISSCIPLUS 测试集，84 条英语医学/健康假说及对应原始研究；人类参考由一名作者从 HealthFeedback 专家评阅中手工提取未改写原文 span 构成。
- **基线与模型**：GPT-5、Claude Opus 4.7、Qwen3-32B；两套提示词（简版 vs 详细 fact-checking 提示词，Figure 7）；两档证据（内部知识 IK、选定段落 selected passages、全文 full study）。
- **裁决预测（Table 1）**：详细提示显著提升 GPT-5 / Qwen3-32B 的 Incorrect 率（GPT-5 全研究+详细提示 74.60%，Qwen3-32B 选定段落+详细提示 76.98%）；Claude Opus 4.7 最低（全研究+详细 44.58%）但一致性 100%。
- **推理路径对齐（Table 2，全研究 + 详细提示主结果）**：
  - GPT-5：19.0% 人类对齐，整体 Accepted 81.7%；
  - Qwen3-32B：10.2% 人类对齐，Accepted 80.3%；
  - Claude Opus 4.7：11.9% 人类对齐，Accepted 86.9%（最高接受率）。
- **谬误标签交叉一致性（Table 3）**：Cohen's κ 0.43–0.51，属中等一致，表明谬误分类本身模型敏感。
- **污染检查（Table 4，10 条 2025–2026 新假说）**：三模型 Incorrect 率 70–80%，但人类对齐率差异大（GPT-5 64.7% vs Qwen3-32B 11.8%）。
- **核心结论**：错误召回最高的 Qwen3-32B 人类对齐仅 10.2%；GPT-5 虽然 Overall Incorrect 率非最高，但产生的**人类对齐子图最多、accepted 占比高**；大量非对齐路径经 grounding & sufficiency 检验后仍被接受为"合理替代推理"。

## 相关工作脉络
- **SciFact / SciFact-Open / SciVer / CLAIM-BENCH**：侧重证据检索与裁决预测，不评估推理路径对齐；本文在其基础上引入"过程级图比对"。
- **MISSCI (2024) / MISSCIPLUS (2025)**：将科学谣言建模为谬误论证并定位支持前提；本文延续其九类谬误词表，但把关注点从"谬误分类"推进到"人类 vs LLM 推理链是否重合"。
- **HealthFC / PUBHEALTH / HEALTHVER**：公共卫生/生物医学事实核查基准，仅评测最终标签和证据 span；本文扩展为子图级 grounded/sufficient 评判。
- **ERASER / e-SNLI / FActScore**：解释忠实度与原子事实支撑评估；本文与之相承但更强调"推理链结构"而非单句忠实度。
- **WorldTree / EntailmentBank**：多跳推理解释图；本文引入有类型图与谬误约束，并面向"替代路径有效性"而非纯粹多跳推断。
- **Faithfulness 研究（Turpin et al., Atanasova et al.）**：揭示流畅解释可能不忠实于决策过程；本文用图对齐 + 验证流水线进一步区分"有效替代路径"与"伪装合理的错误路径"。

## 局限性与未来方向
- **样本量有限**：84 条假说，未报告置信区间 / bootstrap / 显著性检验，差异几个百分点仅为趋势性结论。
- **单一标注员**：人类参考提取与 grounding 复检均由一名作者完成，未测 inter-annotator agreement，亦无双盲复检。
- **证据集不对称**：全部 84 条为假说，Incorrect 率实质是 recall；缺少 Correct 支持的真claim  Held-out 集，无法计算 precision 与 label 偏置。
- **数据污染可能**：MISSCIPLUS 及 HealthFeedback 为公开网页，Proprietary 模型可能过拟合；Qwen3-32B 作为开源模型的"人类对齐率偏低"可能部分源于记忆差异而非推理能力差异。
- **领域局限**：仅限英语生物医学/健康 claims，跨语言与其他学科的可迁移性待验证。

## 研究启发与可借鉴点
- **"终值-过程解耦"评测范式**可迁移到其他生成任务（如法律推理、临床决策），即把最终答案正确性与推理路径质量分开评估，更能暴露模型真实能力边界。
- **谬误特异性子图对齐 + 三裁判 STS 投票**的组合值得复用：既保留多模态一致性的鲁棒性，又避免了单一裁判的主观偏差。
- **verbatim 抽取 + QA grounding 的双层验证**可作为通用"解释可信度"检测管线，适用于任意需溯源到原文的生成场景。
- **详细 prompt  vs 简短 prompt 对 Different Models 效果异质**的发现提示：未来 prompt 工程需与模型特性配对设计，不能一概而论。

## 关键术语表
- **Typed Reasoning Graph**：将解释显式分解为 Claim / Study Context / Study Findings / Fallacy-Supporting Premises / Fallacies 五类节点的有向图，刻画"证据→前提→谬误→裁决"的推理链。
- **Fallacy-Specific Sub-graph**：以单一谬误类型为中心的子图单元，含对应的 Study Context、Study Findings 与一个支持性前提，是图对齐的最小可比单位。
- **Grounding Validation**：通过 QA-based 逐组件对照 + 人工复检，检验子图组件是否在原文中有支撑，区分"有根推理"与"凭空编造"。
- **Relevance & Sufficiency Validation**：三名 LLM 裁判分别评估子图与 claim 的相关性及对 Incorrect 裁决的充分性，combined score = 2 方被接受为"有效替代路径"。
- **MISSCIPLUS**：Glockner 等人 (2025) 构建的基准，在 MISSCI 谬误词表基础上将每条谬误锚定到被歪曲研究的原文 passage。
- **Verdict Consistency**：同一输入三次重复推理输出完全一致的比例，反映模型在该设置下的稳定性。
- **Human-Aligned Sub-graph**：谬误类型相同且所有文本组件 STS 聚合分 ≥3 的 LLM 子图，视作与人类专家走同一路径。

## 可复现要素
- **数据集**：MISSCIPLUS test set（84 条），公开可从 arXiv 论文附属获取链接获取；原始 HealthFeedback 评阅亦为公开资源。论文未提及自建额外数据集。
- **代码**：论文中未明确声明开源仓库，附录仅给出提示词（Figures 7–11）；需要向作者索取代码或自行实现图构造/对齐/验证管线。
- **关键超参**：temperature = 0.1；重复次数 = 3；STS 判定阈值 = 3；relevance/sufficiency 三级评分（0/1/2）；Cohen's κ 用于谬误标签一致性衡量。
- **模型访问**：GPT-5 via Academic-AI gateway；Claude Opus 4.7 via Anthropic API；Qwen3-32B 本地 vLLM 部署；Claude Sonnet 4.6 仅 Seed 1 结果（Table 1 备注）。
