---
title: "ScienceArena-Benchmarking-LLMs-on-Latest-Scientific-Olympiad"
source: https://arxiv.org/pdf/2608.30517v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:29:49"
field: "科学推理与大模型评测"
keywords: ["科学推理评测", "奥林匹克竞赛", "LLM-as-judge", "多步过程评分", "多模态benchmark", "interleaved prompting"]
innovations: ["专家审计的奥林匹克科学竞赛结构化流水线", "基于奖牌得主锚定的LLM-as-judge±1分校准", "rubric gates与error-forward的显式承诺评分规则"]
benchmarks: ["ScienceArena", "IPhO 2025/2026", "IChO 2025/2026", "IBO 2023", "CPhO 2025", "USNCO 2025"]
---

# 论文速读：ScienceArena-Benchmarking-LLMs-on-Latest-Scientific-Olympiad

## 一句话总结
本文构建了ScienceArena，一个涵盖13项国际/国家奥林匹克竞赛（物理、化学、生物）的结构化基准，配套专家审计的数据流水线与校准的LLM-as-judge评分系统，用于可靠评估前沿LLM的科学推理能力。

## 研究问题与动机
- 现有科学推理评测基准日益饱和且易受数据污染，难以反映LLM的真实前沿能力
- 科学问题答案长、含多模态要素（图表、分子结构），且评分需过程性学分（process-credit rubrics），传统自动匹配或简单多选无法测量
- 人工奥运奖牌级专家评分成本过高，难以持续支撑模型迭代评测
- 长程多步推导中模型容易出现全局一致性与视觉 grounding 失败，现有评测缺少可诊断的细粒度指标

## 核心贡献（创新点）
- 提出ScienceArena基准：覆盖2023–2026年十三项公开科学奥林匹克试题、官方答案、详细解答与计分rubric，保留题目层级与发布源
- 设计专家审计的数字化流水线：从PDF解析到JSON结构化，由物理/化学/生物奥运奖牌得主逐题审核图题对齐、答案保真与总分一致
- 构建专家校准的LLM-as-judge系统：以IPhO/IChO 2025的奖牌得主评分为锚，通过rubric封装、结构化子分、后处理校验实现±1分的专家一致性
- 对比one-shot与interleaved求解协议，提出基于固定PNG原图/答案盲静态化的双通道评测方案，并提供可复用的提示模板与审计日志
- 提供跨子领域的可诊断能力剖面：分离符号推导、图形读取、全局一致性、结构/立体化学、数据分析等六类得分瓶颈

## 方法详解
- **基准构建流程**：收藏官方公开题、图、答案、评分标准→OCR/多模态解析→字段规范化（question_text、official_rubric、max_score、release_date等）→奥运奖牌得主双盲审核→导出为interleaved子题JSON
- **LLM-as-judge管线**：输入冻结题面（含视觉证据文本化或原图）、候选作答、官方答案、rubric与max_score→扮演严格评卷人输出JSON（total_score、sub_scores、evidence、deductions、confidence）→后处理校验JSON、修复LaTeX、夹紧无效分、累加子题；IBO客观题走确定性runner而非judge
- **评分校准**：以五模型在IPhO/IChO 2025的存档作答为校准集；Gemini 3.5 Flash与Gemini 3.1 Pro在两考卷均与奖牌得主总分差≤1分；item-level Pearson≈0.83–0.91
- **求解协议**：one-shot一次性生成多部分解答；interleaved逐子题问答并保留上下文；实验显示interleaved在全部16组对比胜出
- **路由与静态化**：2026新赛段采用密封方案，视觉路由直接下发原始question-page PNG，文本路由仅用Gemini 3.1 Pro生成答案盲的静态化文本，经逐题校对后冻结
- **提示模板库**：包含数字化提示、2026密封静态化提示、interleaved求解提示、IBO客观提示、rubric忠实judge提示、子领域/模态分类提示

## 实验与结果
- **数据集**：13项竞赛，含IPhO 2025/2026、IChO 2025/2026、IBO 2023、APhO/EuPhO/NBPhO/USAPhO/CPhO/CChO/INChO/USNCO等；国际赛给出奖牌阈值作人类参考
- **评测基线**：14个主流模型家族（Gemini 3/3.5、GPT-5.4/5.5、Claude Opus 4.7、Doubao Seed 2.0 Pro、Qwen3.6/3.7、DeepSeek V4、MiniMax M2.7、GLM-5.1、MiMo V2.5 Pro）
- **主要结果**：Gemini 3.1 Pro在IPhO 2025拿到29.45/30，超过人类金牌参考23.4；在IChO 2025拿到48.95/60（人类金牌57.1）；IBO 2023最佳431/453，超人类冠军参考395.4
- **加权平均**：Gemini 3.1 Pro全套装均分最高（物理89.5%、化学86.3%、生物95.1%）；GPT-5.5在四赛段2026密封宏分达93.31%
- **稳定性**：三次重跑SD在1.0–2.6之间；同卷跨judge item-level MAE 0.066–0.537，Pearson 0.829–0.909
- **模态消融**：原生图像均分72.75%，忠实文本描述71.73%（差异不显著），无视觉信息49.85%；化学原生优势约5.01分
- **子领域诊断**：化学立体化学34.7%最低，物理热力学74.1%、天体物理76.8%偏弱；生物生化91.4%、分子遗传85.8%最强，实验数据解释仅64.3%

## 相关工作脉络
- **Chatbot Arena/Arena-Hard**：基于偏好排序，ScienceArena改用可验证科学正确性与官方rubric计分
- **OlympicArena/OlympiadBench/MathArena**：侧重跨学科难题，ScienceArena保留最新科学竞赛、发布日期、官方rubric与奖牌得主审核
- **HiPhO/PhyArena**：单一物理赛道，ScienceArena将物理/化学/生物统一到一套流程与评分标准
- **SUPERChem/BABE/SciArena（ prior）**：前者偏科研任务或偏好，本文使用正式竞赛+可比量尺+专家校准与可追溯审计
- **LiveCodeBench/FrontierMath/Hand-E-Exam**：强调时间锁定去污染，本文以透明发布窗口与sealed路由为主，公开宣称不宣称严格无污染
- **AutoRubric**：统一rubric化LLM评测，本文在其框架思想上实现奥运场景的过程性打分与结构保真gate

## 局限性与未来方向
- 仅覆盖理论题，实验部分因需要器材与监考无法复现，结论对全课程泛化性受限
- 专家校准目前仅在IPhO/IChO上完成，其他竞赛依赖自动化judge，高风险场景仍需专家抽检
- 多赛段2026列存在发布窗口重叠，不能严格证明无训练数据污染，仅为能力测量
- 子领域分类由模型辅助完成，可能引入分类器先验，替代划分会得到不同诊断切片
- 三重复跑合并了求解生成与固定judge的方差，未完全解耦二者；冻结答案跨judge实验仅覆盖一家solver
- 视觉/文本路由与模型能力绑定，表9给出的模态估计仍伴随路由差异

## 研究启发与可借鉴点
- **专家锚定judge校准**：以少量高质量人工评分为锚，构造结构化输出与后处理校验，可迁移到任何过程性评分领域
- **rubric gates与error-forward规则**：明确要求显式中间承诺、禁止推断缺失结构，可指导复杂学科（化学/材料/工程制图）的自动评分设计
- **interleaved vs one-shot对比范式**：以自然交互为基线、控制变量对比，可直接移植到代码、数学、规划等多步任务
- **sealed路由与PNG直传审计**：对每个请求记录SHA-256与payload清单，便于复现实验与反作弊，适合作为长期评测工程的默认规范
- **子领域能力雷达与专家注释沉淀**：把人工评语抽象为得分/扣分轴，再反哺judge提示，形成可迭代的诊断闭环

## 关键术语表
- **Process-credit rubrics**：按解题步骤与明确承诺给分的评分标准，区别于仅看最终答案
- **LLM-as-judge**：用语言模型扮演评卷人，依据rubric输出结构化分数与证据引文的评测方法
- **Interleaved prompting**：将多子题问题拆分为顺序交互、保留上下文逐题求解的提示协议
- **Staticization**：把多模态题面冻结为答案盲的文本表示，用于文本路由且防止题泄
- **Error-forward credit**：当中间结构/数值明确且自洽时，后续合理推导可继承部分分数
- **Sealed evaluation**：固定输入、路由、judge配置与审计账本的封闭评测，保证可复现与反篡改
- **Normalized macro score**：将各赛题归一化后取平均的聚合指标，用于跨体量竞赛横向比较
- **Visual grounding**：将图表、结构、曲线等非文本证据正确绑定到符号与结论的能力

## 可复现要素
- **数据集**：13项竞赛来源均为官方公开材料；论文提供交互式demo与结构化导出；具体开源链接论文中未给出明确仓库URL
- **代码/权重**：评测提示模板、rubric judge管线与审计协议在附录给出，但未声明独立开源仓库
- **关键超参**：judge使用Gemini 3.5 Flash/3.1 Pro；重试上限5次；2026密封路由最多6次重试；item-level重评MAE与Pearson见附录表
- **统计方法**：配对bootstrap 95% CI、sequence-cluster区间、跨judge重复MAE与相关系数；样本量N为model-turn标注实例数，非独立题数
