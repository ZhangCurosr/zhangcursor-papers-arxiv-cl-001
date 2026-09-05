---
title: "ScienceArena-Benchmarking-LLMs-on-Latest-Scientific-Olympiad"
source: https://arxiv.org/pdf/2608.30517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:29:48"
field: "大模型科学推理评测"
keywords: ["LLM evaluation", "scientific reasoning", "olympiad benchmark", "LLM-as-judge", "multimodal reasoning", "process-credit scoring", "interleaved prompting"]
innovations: ["专家审计的奥林匹克科学基准+过程计分rubric", "奖牌得主校准的LLM-as-judge系统误差≤1分", "interleaved prompting在16组对比中全面优于one-shot"]
benchmarks: ["ScienceArena", "IPhO 2025/2026", "IChO 2025/2026", "IBO 2023", "USAPhO 2026", "CPhO 2025", "CChO 2025"]
---

# 论文速读：ScienceArena-Benchmarking-LLMs-on-Latest-Scientific-Olympiad

## 一句话总结
本文提出ScienceArena，一个涵盖2023-2026年13个国际/国家级物理、化学、生物奥林匹克竞赛的基准测试，通过奥运奖牌得主专家审计的数字化管道和LLM-as-judge评分校准系统，评估前沿大模型在开放型、多步骤、过程计分科学推理任务上的能力。

## 研究问题与动机
- **基准饱和与污染问题**：现有流行基准日益饱和且易受训练数据污染（data contamination），难以真实衡量前沿LLMs的科学推理能力；近年提出的实时/前沿基准虽采用时间敏感来源，但科学问题的长答案、视觉元素和部分正确性使评分困难。
- **现有科学评测的局限性**：多数科学评估局限于短答案或选择题格式，无法测试真实科学问题解决所需的能力——长推导、图表与结构解读、多步骤一致性、显式评分规则下的部分得分（partial credit）。
- **手工评分成本瓶颈**：奥运专家评分最可靠，但反复要求专家对每个新模型评分不切实际；需要可扩展的自动化评分方案。
- **解题协议选择问题**：单步提示（one-shot）与分段交互提示（interleaved）在长竞赛题上的表现差异尚未系统研究。

## 核心贡献（创新点）
- **构建专家审计的奥林匹克科学基准**：首次将13个2023-2026年物理/化学/生物公开竞赛以结构化JSON形式标准化，保留原始层级、官方答案、详细解析和评分细则，并经奥运奖牌得主审计。
- **可复用的数字化管道**：开发五阶段pipeline（收集→OCR/多模态解析→字段标准化→专家审计→导出），将含公式、图表、分子结构的PDF源材料转换为LLM可读的benchmark items。
- **专家校准的LLM-as-judge系统**：以奥运奖牌得主为ground truth校准LLM评分器，Gemini 3.5 Flash和Gemini 3.1 Pro在所有模型-考试对上误差≤1分，实现可扩展自动化评分。
- **解题协议对比与诊断分析**：首次系统比较one-shot与interleaved提示（interleaved全16组对比胜出），并提炼奖牌得主评分注释为子领域能力边界分析（可视化接地、结构保真度、全局问题控制）。
- **多模态输入消融实验**：控制模态消融显示原生图像比无视觉信息高22.90个百分点（95% CI [20.02, 25.77]），化学领域原生图像比忠实描述高5.01分，支持"captioner-solver pipeline"实用方案。

## 方法详解
**基准构建管道**：
- 第一阶段：策展人收集官方公开问题、图表、答案键、解析和评分方案
- 第二阶段：页面渲染+OCR/多模态模型解析
- 第三阶段：内容归一化为字段（question_text, official_answer, official_detailed_solution, official_rubric, max_score, release_date等）
- 第四阶段：学科奖牌得主审计完整性、图表对齐、答案保真度、分数合计
- 第五阶段：导出为统一interleaved数据集，每子问题一行

**LLM-as-judge系统**：
- 对非客观题，judge接收静态化问题、官方答案、详细解析、评分细则、满分和候选回复
- 扮演严格奥运 examiner，分配部分学分，返回JSON（total_score, max_score, sub_scores, evidence, deductions, confidence）
- 后处理验证JSON、修复LaTeX转义错误、钳制不可能分数、检查满分、聚合子题
- 2026年四场竞赛使用更严格密封实现（固定Gemini 3.5 Flash judge路由，支持重试）
- 校准基于IPhO/IChO 2025的5个模型存档答案：两个judge均与奖牌得主总分误差≤1分

**解题协议**：
- One-shot：模型一次性接收完整问题写出完整解
- Interleaved：按子部分顺序回答，保留先前上下文
- 空/不完整/非具体响应重试最多5次，错误前向计分仅在响应含具体中间结果时适用

**评分校准验证**：
- 专家重放一致性：525条专家评分行中，1050条结构化重放得Interjudge归一化MAE 0.093，Pearson相关0.907
- 跨judge稳定性：90.11%-96.70%严格重放项目一致性

## 实验与结果
**数据集**：13个竞赛，涵盖物理（IPhO、APhO、EuPhO、NBPhO、USAPhO、CPhO）、化学（IChO、INChO、CChO）、生物（IBO），2023-2026年发布

**评估模型**：14个前沿LLM（Gemini 3/3.5系列、GPT-5.4/5.5、Claude Opus 4.7、Doubao Seed 2.0 Pro、Qwen3.6/3.7、DeepSeek V4、MiniMax M2.7、GLM-5.1、Xiaomi MiMo）

**主要结果**：
- **IPhO 2025**（满分30）：Gemini 3.1 Pro获29.45分，超过人类金牌线23.4分，接近冠军29.2分；多个模型达到金牌等效水平
- **IChO 2025**（满分60）：Gemini 3.1 Pro获48.95分，超金牌线36.6分，但低于人类冠军57.1分
- **IBO 2023**（满分453）：Gemini 3.1 Pro获431.00分，多个模型超过人类冠军参考395.4分
- **综合加权平均**：Gemini 3.1 Pro第一（88.7），Gemini 3.5 Flash第二（86.1）
- **2026四场密封测试**：GPT-5.5以93.31%最高，Gemini 3.5 Flash 89.54%，Gemini 3.1 Pro 88.47%
- **稳定性**：GPT-5.5在IPhO上SD=0（零 spread），Claude Opus 4.7变异性最大但仍在操作可重复范围

**子领域诊断**：
- 化学：立体化学最弱（34.7%），分析光谱较强（86.6%）；需结构图题64.5% vs 不需74.1%
- 物理：热力学74.1%、天体物理76.8%最弱；力学81.3%、电动力学78.2%、现代物理83.7%较强
- 生物：生化91.4%、分子遗传学85.8%最强；实验数据分析64.3%最弱

**模态消融**：原生图像72.75% > 忠实文本描述71.73% > 无视觉信息49.85%

## 相关工作脉络
- **Chatbot Arena/Arena-Hard**：侧重人类/judge偏好而非可验证科学正确性，ScienceArena聚焦科学推理的可验证性
- **OlympicArena/OlympiadBench**：跨学科难推理，但ScienceArena保留最新科学考试、发布日期、官方评分规则和奖牌得主审计
- **MathArena**：新鲜数学答案/证明，ScienceArena扩展至物理/化学/生物，增加图表、结构和科学过程计分
- **HiPhO/PhyArena/PhysicsArena**：仅物理奥林匹克，ScienceArena统一三科学赛道
- **SUPERChem/BABE**：领域研究或专家任务，ScienceArena用官方竞赛+可比评分尺度+奖牌参考
- **SciArena**：研究者偏好，ScienceArena对抗官方答案和评分规则评分，分析奖牌得主评分注释
- **LLM-as-judge实践**（Zheng et al., 2023; Rao & Callison-Burch, 2026）：本文扩展至过程计分rubric场景，强调专家校准和证据引用

## 局限性与未来方向
- **范围局限**：仅覆盖13个竞赛，不泛化至所有科学课程；仅评估理论部分（实验部分需物理设备和监考）
- **专家评分成本**：目前仅在IPhO/IChO上覆盖专家校准；其他竞赛仅做spot check
- **重复调用测量混杂**：三次重复调用共同测量solver生成和固定judge变异，未能完全分离
- **模态-能力耦合**：视觉/文本路由比较同时反映能力和模态，需更多解耦实验
- **子领域分类依赖classifier先验**：Gemini 3.5 Flash分类器可能引入偏差，替代分解可能产生不同诊断切片
- **发布时间与污染**：大部分结果在公开发布之后，仅Gemini 2.5 Pro（2025年6月17日发布，早于2025年7月IPhO/IChO）可作为有限的时间锁定holdout

## 研究启发与可借鉴点
- **过程计分rubric的价值**：将评分规则结构化（sub_scores + evidence字段）而非仅最终答案匹配，能揭示"流利但物理 grounding 脆弱"与"正确问题模型"的区别，适用于任何需要部分信用的评测场景
- **专家校准+自动化评分的混合范式**：以少量专家评分为锚点校准LLM judge，后续规模化评估成本骤降；该范式可迁移至法律、医学等专业领域评测
- **interleaved prompting作为标准基线**：多步问题分subpart逐步求解并保留上下文，在全部16组对比中胜出；建议作为长推导任务的默认评测协议
- **子领域诊断雷达图的实用性**：将专家注释抽象为可复用能力轴（符号推导、数值+单位、近似极限、全局一致性、图表阅读、视觉接地、结构+立体化学等），为模型能力剖面提供细粒度诊断
- **模态消融的实验设计**：控制原生图像 vs 忠实描述 vs 无视觉信息的三臂设计，证明"高质量captioner-solver pipeline"在缺少原生视觉输入时的实用价值

## 关键术语表
- **ScienceArena**：涵盖2023-2026年13个物理/化学/生物奥林匹克竞赛的基准测试，含官方答案、详细解析和评分细则
- **LLM-as-judge**：使用大语言模型作为评分器，按评分细则分配部分学分的自动化评测方法
- **Interleaved prompting**：将多部分问题拆分为顺序子部分，每步回答保留先前上下文的提示协议
- **Process-credit rubrics**：过程计分评分规则，允许错误最终值因正确推导获得部分学分
- **Visual grounding**：模型将非文本证据（图表、分子结构、实验装置图）与符号/声明正确绑定的能力
- **Structure fidelity**：化学问题中精确呈现分子连接性、立体化学关系和结构图的能力
- **Error-forward credit**：当候选中间结果具体且内部一致时，后续推理可继承部分学分的评分规则
- **Staticization**：将含视觉元素的问题页面冻结为纯文本表示，供文本-only模型使用

## 可复现要素
- **数据集**：13个竞赛的公共材料（IPhO/IChO/IBO官方发布），结构化JSON已随论文发布
- **代码/权重**：论文提供交互式demo链接，pipeline细节在附录D提供prompt模板
- **关键超参**：重试次数上限5次；judge固定Gemini 3.5 Flash；2026密封实现固定single judge route
- **专家审计**：物理/化学/生物各2名金牌得主（详见Table 3）
- **计算资源**：未明确提及
