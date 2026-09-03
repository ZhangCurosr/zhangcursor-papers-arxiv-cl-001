---
title: "What-Proves-You-Wrong-Benchmarking-Language-Models-on-Falsif"
source: https://arxiv.org/pdf/2608.22948v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:22:23"
field: "LLM科学发现与评测基准"
keywords: ["LLM评测", "研究构思", "可证伪性", "成对判官", "Bradley-Terry", "控制实验"]
innovations: ["以六字段最小可证伪测试契约作为唯一评估单元，同时要求支持与拒绝双向结果", "双顺序盲审折叠 + Bootstrap 稳健检验，显式分离并报告 order-sensitive 案例", "构建包含 hidden control、sham-adjusted preference、单字段破坏与 same-source 渲染的诊断控制族"]
benchmarks: ["Lit2Test"]
---

# 论文速读：What Proves You Wrong: Benchmarking Language Models on Falsifiable Research Ideation

## 一句话总结
论文提出了 **Lit2Test** 基准，通过六字段契约强制模型给出最小可证伪测试方案（含支持结果与拒绝结果），以盲审双顺序对比+Bootstrap稳健检验的方式评估大模型在可证伪研究构思方面的能力。GPT-5.2 > Claude Sonnet 4.6 > GLM-5 > DeepSeek-V3.2 的严格排序在全部 10,000 次 Bootstrap 中复现，且与人类标注在 87.2% 稳定案例上一致。

## 研究问题与动机
- **核心问题**：现有 LLM 研究想法评测范式缺乏"谁会被证伪"的共享决策规则——自由格式评分被文风和呈现顺序绑架；以未来论文为答案钥匙则奖励对齐单一实现路径、惩罚合法替代方向，且易受知识截止日期污染。
- **动机 1（范式缺陷）**：自由评判 (free-form judging) 在同一个 ICLR 长上下文 QA 邻里上，GPT-5.2 与 GLM-5 的方案在两种呈现顺序下均获相同偏好（Figure 1a），但与"未来论文对齐"打分（Figure 1b）结论相反，且换答案钥匙后反转——暴露评估不稳定。
- **动机 2（设计目标）**：需要一种"携带研究构想从文献走到测试"的可审计协议：固定邻里上下文、无答案钥匙、六字段契约绑定支持/拒绝双向结果、双顺序盲审折叠聚合、Bootstrap 不确定性、人类校准。
- **动机 3（研究问题）**：RQ1 模型排序及其对呈现顺序与重采样的稳健性；RQ2 排序由哪些字段驱动、评判是否响应实质而非文风；RQ3 抽样人类是否能 corroborate 聚合结论、在哪类边界失效。

## 核心贡献（创新点）
1. **六字段联合评估单元**：以"最小可证伪测试"为中心，要求 literature_gap / hypothesis / minimal_test / decisive_metric / supporting_result / falsifying_result 同时出现并由同一判官 adjudicate，区别于仅评新颖性/可行性的既有基准（Appendix F 对 27 个系统的系统综述支撑）。
2. **前瞻性构建管线**：200 个来自 OpenReview/ICLR 邻域的真实邻里（800 篇唯一文献，5 批次各 40），无 future-paper 答案钥匙，配合 provenance audit 关闭污染通道；四种前沿模型各出一版原生六字段提案，共 1,200 规范对。
3. **可靠性审计协议**：双顺序盲判 → 折叠出 950 稳定/250 敏感案例 → Bradley–Terry + Condorcet 聚合 + 10,000 次 case-level Bootstrap；插入 8 个 naive 隐藏控制、same-source 渲染控制、单字段明显破坏控制、自然化微妙破坏审计（sham-adjusted preference）。
4. **经验发现三项**：严格三档排序在 Bootstrap 全覆盖；falsifiability 充当 admissibility floor、minimality/feasibility 与 decisive metric 提供主要区分；20 邻里/90 对的三人人类抽样在稳定案例上达成 87.2% 一致，并复现同阶层结构（Krippendorf's α=0.238 界定边界）。

## 方法详解
- **任务定义**：实例 = 真实四篇论文的邻里 c，模型输入 c 返回单条提案 P = (literature_gap, hypothesis, minimal_test, decisive_metric, supporting_result, falsifying_result)；评测比较同一邻里下的提案对 (P_i, P_j | c)。
- **六字段契约**：
  - literature_gap：邻里内可见张力/未解比较；
  - hypothesis：带条件、机制、预期差的方向性断言；
  - minimal_test：能判别 hypothesis 的最小实验（数据集/基线/程序/资源预算），anti-grandiosity 条款优先小而可执行；
  - decisive_metric：裁决机制的单一测量，禁止"改善即进步"的便利聚合；
  - supporting_result / falsifying_result：双向决策规则，使可证伪性可文本核查。
- **生成设置**：temperature=1.0，max_tokens=1600，schema 校验失败最多重试 2 次；四种参与者模型（GPT-5.2、Claude Sonnet 4.6、GLM-5、DeepSeek-V3.2）共用同一 prompt。
- **判官设置**：主判官 Gemini 3.1 Pro (Preview)，temperature=1.0，max_tokens=1200；不参与者不生成，去 self-preference。
- **折叠规则**：1,200 规范对 × 双顺序 = 2,400 有序判词；同时胜/同时负 → order-stable；胜负反转或至少一序 tie → order-sensitive（250 例）并单列报告。
- **聚合与不确定性**：稳定例进 Bradley–Terry 估计 + Condorcet head-to-head；case-level Bootstrap 10,000 次（非行级重复）；遇到完全分离时只报告序数结构，BT 强度仅作诊断。
- **诊断控制族**：
  - 维度审计：90 对的五维结构化评分（grounding / hypothesis specificity / minimality & feasibility / decisive metric / falsifiability）与全判一致率；
  - 隐藏 real-vs-naive：8 个模板/关键词 naive 方案混入；
  - same-source 渲染：同内容 schema vs prose 对照；
  - 明显单字段破坏：每维度 20 例，干净方案 40/40 胜出；
  - 微妙破坏审计：自然化缺陷 vs 风格匹配 sham，sham-equivalence gate 全过，clean 的 adjusted preference 95% CI 均不跨零（grounding 0.550、metric 0.775、falsifiability 0.537，pooled 0.621）。
- **人类校准**：20 邻里 / 90 对 / 3 名 CS 高年级本科生，双语协议+维度锚点+10 分钟/例限时；多数投票作校准证据而非 gold。

## 实验与结果
- **数据规模**：200 邻里、800 篇源文献、4 模型 × 200 = 800 提案、1,200 规范对、2,400 有序判词。
- **主排序（所有 10,000 Bootstrap 复现）**：
  GPT-5.2 > Claude Sonnet 4.6 > GLM-5 > DeepSeek-V3.2
  BT 中心化 log-ability：GPT-5.2 = 1.26 [1.14, 1.40]；Claude = 0.74 [0.62, 0.87]；GLM-5 = −0.73 [−0.86, −0.61]；DeepSeek = −1.28 [−1.42, −1.15]。
- **Condorcet**：每一档在其跨档稳定对决中全胜，跨 5 个批次一致；两个 tier 结构（GPT/Claude 上 / GLM/DeepSeek 下）即使在 adversarial 把 250 敏感全给低排者时也仅翻转 2 对相邻对。
- **第二判官（Doubao Seed 2.0 Pro，与参与者及主判官无关）**：独立复现同排序；与主判在稳定子集上 86.1% 一致（1,635/1,900），case-folded 一致 79.9%（759/950）。
- **维度诊断**：structured 总判与 canonical 一致 152/180（84.4%，CI 78.9–89.4%）；minimality/feasibility 信号最强，falsifiability 在自然输出中区分度低但经两重破坏控制证实是有效门槛；hidden real-vs-naive 8/8 均选真实。
- **人类校准**： naive 检测 11/12；60 稳定案例中 39 有决定性多数，34/39（87.2%，CI 76.9–97.4%）与 Gemini 稳定判一致；人类 BT 排序 88.3% Bootstrap 样本与主排序只差 ≤1 次翻转；tier 分割全保留。多人leave-one-out 稳健。
- **语境聚类 Bootstrap**：相比 pair-level，CI 增宽 11–27%，仍 10,000/10,000 复现模态排序。

## 相关工作脉络
- **开放思路质量**：Si et al. (2025) 大规模专家研究；LiveIdeaBench (Ruan et al., 2026)  divergent thinking；ScholarEval (Moussa et al., 2025) 基于检索文献；InnoEval (Qiao et al., 2026) 多视角推理框架——均不设 minimal test 与双向 falsifier，本文在"判据单元"上形成差异。
- **轨迹恢复类**：AI Idea Bench (Qiu et al., 2025)、IdeaBench (Guo et al., 2025)、ResearchBench (Liu et al., 2026) 以已发表后续论文为答案钥匙，存在知识泄漏与路径偏置；本文以"无答案钥匙的前瞻性邻里"定位自己，表 2 逐项对比。
- **假设生成类**：HypoBench (Liu et al., 2025) 用预测效用打分；MOOSE-Chem (Yang et al., 2025) 面向化学重发现——侧重解释已有数据，不涉及邻里张力驱动的假说+最小测试联合设计。
- **未来影响预测**：HindSight (Jiang, 2026)、Proof of Time (Ye et al., 2026)、Wen et al. (2025) 以后续引用/奖项/实验结果为标签；本文拒用答案钥匙，转向"决策规则是否可预先写下"。
- **具可证伪机器的研究智能体**：HARPA (Vasu et al., 2025) 在 execution 阶段评估；AI Co-Scientist (Gottweis et al., 2026) 加入 disproof review；AI Scientist (Lu et al., 2026) 端到端自动化——它们把 falsification 嵌入生成/执行循环，评测终点是运行日志或分类；本文把完整六字段作为被测单元，与 IdeaSpark (Zhao et al., 2026) 同时期独立工作形成互补：后者保留生成时 falsification prediction 作为 admission safeguard，但末态质量判分不含该字段，本文则将其作为直接度量对象。
- **评测可靠性原语**：MT-Bench/Chatbot Arena (Zheng et al., 2023)、G-Eval (Liu et al., 2023)、JudgeBench (Tan et al., 2025)、LiveBench (White et al., 2025) 等；本文不宣称任一原语为创新，而是围绕新单元"六字段契约 + 双顺序折叠 + 控制族 + 人类校准"的组合创新（表 2 汇总）。

## 局限性与未来方向
- **人类校准覆盖有限**：仅 20/200 邻里 × 3 标注员，支撑聚合结论但不足以 benchmark-wide 验证；扩展覆盖是最直接补救。
- **单一规范判官**：经第二判官交叉验证稳健，但每个版本仍只固定一个 LLM 作为 canonical judge，更换需版本化全量重跑。
- **微妙破坏审计的 Frozen-20 协议边界**：reserve-template 替换 8/20、语义校验器同意 65.6%、自然性泄漏 3.9%，仅能支撑局部敏感度结论。
- **不测执行结果**：可测试性只是下游成功的必要条件而非充分条件，无法预测实验成败。
- **领域局限**：200 邻里来自 ML 邻近文献；管线可复制，跨领域/时间切片的泛化是扩展工作。
- **未来方向**：针对 minimal-test formulation 与 metric–mechanism reasoning 微调；研究人机分歧案例的合作模式；把微妙破坏审计扩展出 frozen 协议；与前向可测试性对接执行式验证。

## 研究启发与可借鉴点
- **契约化输出倒逼可评审性**：把 "supporting + falsifying result" 作为必须字段纳入生成 prompt 与判官 rubric，可有效过滤"听起来合理但无法裁决"的空泛创意；该设计可迁移至任何需要"可被证据驳回"的输出评测（如临床方案、政策建议、算法设计）。
- **双顺序折叠 + 敏感性分离报告**：不强行抹平位置偏差，而是显式报告 order-sensitive 案例数量/比例，把"判官无法裁决"本身作为度量产物——适用于所有 LLM-as-judge 的成对评测协议。
- **sham-adjusted preference 控制风格干扰**：针对每个被破坏字段构造"风格匹配但不损内容"的 sham，用 clean-vs-sham 通过门控后再算 clean-vs-subtle 的净偏好；可通用到任何"文本细微降级检测"的诊断实验。
- **case-level vs cluster-level 双重 Bootstrap**：既做实例重采样也做邻里聚类重采样，前者反映判词噪声、后者反映邻里内相关性，二者对比可量化"配对间共变性"，适合小样本高相关结构的评测。
- **团队结合机会**：可借用 Lit2Test 的六字段 schema + 控制族，接入本团队在 [具体子方向] 的已有数据/模型，构造领域专用的可证伪评测；或将 falsifiability floor 作为生成侧的 admission filter，与现有 RLHF/ORPO 训练流程对接。

## 关键术语表
- **Lit2Test**：作者提出的基准，以"从文献到测试"的六字段契约为核心，评估 LLM 提出可证伪研究构思的能力。
- **六字段契约（Six-field contract）**：literature_gap / hypothesis / minimal_test / decisive_metric / supporting_result / falsifying_result 的组合，构成唯一可判定的最小评估单元。
- **Order-stable / Order-sensitive**：一对方案在 A/B 两种呈现顺序下是否给出相同胜负；后者单独列出，不入主排序聚合。
- **Folding（折叠）**：将同一规范对的两次有序判词合并为一个 case outcome（win/loss/tie），避免把两次有序判断误当独立样本。
- **Bradley–Terry (BT) 估计**：基于成对胜负拟合 each-model 的 log-ability，此处以 centered log-ability 报告并附 Bootstrap 置信区间。
- **Condorcet ordering**：若每一档都在 head-to-head 中胜过所有低档，则构成严格 Condorcet 序；本文四模型满足。
- **Sham-adjusted preference**：在微调破坏审计中，用 clean-vs-sham 通过门控后，衡量 clean-vs-subtle 的净偏好以剥离风格效应。
- **Falsifiability as admissibility floor**：falsifying_result 维度在自然输出上区分度低，但明显/微妙破坏控制表明它是"准入底线"而非"区分主因"。

## 可复现要素
- **数据集**：200 邻里、800 篇源文献、800 提案、1,200 规范对、2,400 有序判词；公开仓库含 construction pipeline 与 audit artifacts（论文附录 B/G 及 footnote 1、2 指代仓库）。
- **代码/权重**：构建管线、所有原始 API 响应归档、分析脚本；closed-source 模型不保证位精确，但依赖固定 seed 与冻结输出实现下游确定性。
- **关键超参**：
  - 生成：temperature=1.0，max_tokens=1600，schema 失败最多重试 2 次。
  - 判官：Gemini 3.1 Pro (Preview)，temperature=1.0，max_tokens=1200；第二判官 Doubao Seed 2.0 pro。
  - 统计：10,000 次 case-level Bootstrap；context-cluster bootstrap 对 200 邻里整簇重采样。
  - 随机种子：construction seed 20260620，bootstrap seed 20260721，cluster-bootstrap seed 20260729。
- **论文未提及**：显式训练数据/参数规模、推理成本预算、GPU 集群规格。
