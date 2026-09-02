---
title: "Most-of-the-LLM-routing-gap-is-task-type"
source: https://arxiv.org/pdf/2608.23023v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:11:32"
field: "大语言模型路由与组合"
keywords: ["LLM routing", "router plateau", "task type decomposition", "multilingual evaluation", "reproducibility floor", "static lookup baseline", "both-run scoring"]
innovations: ["以双重复测量化名义确定性配置的执行噪声（5.37% cell翻转率）作为路由收益判定下限", "证明Oracle与最佳单模型之间72.4%的缺口可由按任务类型的静态单模型预分配吸收", "提出无符号边际平局规则，抑制跨轮符号翻转的伪优势并优先按账单择低"]
benchmarks: ["MMLU-ProX", "MGSM", "IFEval", "BFCL v4", "LiveCodeBench"]
---

# 论文速读：Most-of-the-LLM-routing-gap-is-task-type

## 一句话总结
本文通过完整执行14个模型×294道问题（7种任务类型×3种语言×14道题/格）的全矩阵实验发现：现有路由方法难以逼近"最佳单模型vs.Oracle"之间的大部分差距，主要源于可观测的**任务类型（task type）**结构（占72.4%），而非路由设计本身不足；在相同矩阵上训练的静态任务类型→模型查找表，即可超越最强单模型，剩余不可解释差距（6/294项）小于执行噪声下限。

## 研究问题与动机
1. **路由性能 plateau（平台期）现象**：Lu等（RouterBench，21种路由方法/5个基准）和Li等（LLMRouterBench，33模型/21数据集）均报告不同设计的路由器收敛到极窄区间，且多数无法稳定超越"始终调用最强单模型"基线，距Oracle仍有10~30pp差距。
2. **缺口构成未明**：既有分解沿"题目难度"（不便于路由时观测）、"数据集身份"（粗粒度域结构）、"标签随机性"（非路由可选项）展开，但**缺少语言维度**的系统考察。
3. **可复现性基线偏低**：即便temperature=0、seed=42、同一网关，同一矩阵重跑仍有**5.37%的model-query cell**正确性翻转（221/4116），暗示小幅度路由收益可能仅是噪声波动。
4. **未建立"值得构建路由器"的门槛**：若预计算的静态映射已能吸收大部分缺口，则复杂学习路由的收益空间极其有限。

## 核心贡献（创新点）
1. **全矩阵穷举执行+双重复测**：首次在同一网关、同配置（temperature=0, seed=42）下完成14模型×294题×2次（共8232 keys）的完整矩阵，量化了名义确定性配置下的残余不可复现性（5.37% cell翻转率、−3.81%账单成本漂移）。
2. **以"双次运行一致正确"为主评分规则（both-run rule）**：只在两轮均判对才算正确，排除单次抽样的运气成分，并证明该规则将最佳单模型从Claude Fable 5/Claude Opus 5切换为Claude Opus 5（245/294），且使"谁是最佳单模型"在10种合理评分变体下分散于5个候选模型（最大差距≤2.72%）。
3. **任务类型分解揭示缺口主体结构**：将Oracle−最佳单模型的29项差距按task type×language分解，发现**72.4%（21/29）可由"按任务类型预分配单一模型"的静态查找表吸收**；再加语言维度仅多吸收2项，残差降至6/294（2.04%），低于噪声下限。
4. **提出静态查找表策略并实测**：在21个task×language单元格内以both-run准确率最大、同分按账单成本最小化选模型，并通过"无符号边际平局规则"消除跨轮符号翻转的伪优势，得到262/294 @ $3.33/run，相对Claude Opus 5（245/294 @ $7.69/run）准确率高+17项、成本低−56.7%，且两项差异均被噪声下限判定为"已分辨"。
5. **建立可复用噪声下限读取程序**：给出三档参考（9/294=3.06% per-item cap-change参考、221/4116=5.37% cell翻转率、−3.81%矩阵级账单漂移），规定差值需同时超过两个准确率参考方可判为"resolved"，并公开所有读数与判定分支，便于下游读者套用不同阈值。

## 方法详解
- **矩阵设计**：7个task type（knowledge/math/instruction/extraction/toolcall/abstention/coding）×3种语言（Korean/English/Hindi，98条item ID冻结、跨语言同ID）×14 items/cell=294 queries；14个模型来自OpenRouter 2026-08-21目录快照，均在temperature=0、seed=42下单网关执行两遍（Run A/B）。
- **评分规则（7种变体）**：
  - `single-run A/B × all items`
  - `both-run × all items`（主规则：一项仅当两轮均为correct才计分）
  - `single-run/both-run × cap_hit-excluded`（输出截断项计0但保留在分母）
  - `single-run/both-run × parse_failed-excluded`（解析失败项计0但保留在分母）
  - 各任务类型具体scorer：MCQ、exact match、IFEval约束检查、JSON key-value完全匹配、AST语法树匹配、工具调用有无（abstention视为无调用即正确）、代码执行单元测试。
- **噪声下限**：
  - A/B重跑cell翻转率：221/4116 = 5.37%（115个1→0、106个0→1）；
  - 单模型cap调整前后per-item参考：9/294 = 3.06%；
  - 矩阵账单变化：$51.557→$49.592（−3.81%）。
- **判定程序（§3.2）**：任一准确率差值需同时超过3.06%（item尺度和5.37%（cell尺度）；成本差值与−3.81%（矩阵）或−14.79%（策略集中成本）比较。未达阈值则报告"unresolved"并给出大小与方向。
- **静态查找表构建**：
  1. 对21个task×language cell，取both-run准确率最高的模型（ties按A/B账单均值最小化）→初版268/294；
  2. **Coding-Hindi override**：原选Claude Fable 5（+1 unsigned margin，$3.45/cell）→换为Grok 4.3（$0.09/cell），得中间表267/294；
  3. **无符号边际平局规则（unsigned-margin tie rule）**：凡top模型对最便宜one-behind候选存在跨轮符号翻转的unsigned margin且需支付溢价时，视作tie并按最低账单选取。应用于所有21 cell后得到最终表262/294 @ $3.332910/run。
- **损失/优化目标**：无学习路由，本质是每cell目标`max both-run accuracy`，次目标`min mean billed cost`，并显式惩罚unsigned margin溢价。

## 实验与结果
- **数据集与覆盖**：MMLU-ProX（knowledge）、MGSM（math）、IFEval（instruction）、in-house extraction（14文档/9模式）、BFCL v4（toolcall+abstention）、LiveCodeBench v6 2025-01~04（coding，无污染控制）。
- **主要结果（both-run 主规则）**：
  - Oracle = 274/294；最佳单模型 = Claude Opus 5 = 245/294；gap = 29项（9.86pp）。
  - **Task-type-only selector**：吸收21/29 = **72.4%** 缺口；残差8项（7项coding、1项instruction）。
  - **Task×language argmax**：多吸收2项→残差6/294 = 2.04%，**低于两个准确率噪声下限**（unresolved）。
  - **Adopted lookup**（含override+tie rule）：**262/294（89.12%）**，mean billed **$3.332910/run**；对比Claude Opus 5（245/294，$7.688898/run），准确+17项（5.78%，>5.37%，resolved）、成本−$4.356（−56.7%，<−3.81%噪声带，resolved）。
- **最佳单模型的不稳定性**：10种合理评分规则下5个不同模型称王，最大top-2差距=8/294=2.72%（<3.06% per-item参考，unresolved）。
- **语言维度增益**：仅按语言选最佳单模型在single-run A增5项、both-run增3项；全局加语言后增益仅+2/294（both-run）。
- **成本分布**：coding贡献$3.107/$3.333 = 93.2%，其中coding-ko（Claude Fable 5）单cell占87.8%账单；该cell跨轮账单漂移−17.0%，导致lookup的A/B账单带为−14.79%。
- ** cheaper symmetric alternative（§6.5）**：所有coding改Grok 4.3 + 其余按同款tie rule选 → 259/294 @ $0.508/run；lookup相对其准确高3项（1.02%）但贵$2.825/run。前者unresolved，后者cost resolved，结论：matrix无法支撑为这3项多付$2.825。

## 相关工作脉络
1. **Lu等（RouterBench, 2026, arXiv:2606.07587）**：21种路由方法在5基准上收敛于≤0.23pp窄带，距oracle仍差10~30pp。本文在其" plateau "结论基础上进一步定位缺口构成坐标（task type×language），并证明大部分已被静态映射吸收。
2. **Li等（LLMRouterBench, ACL 2026 Findings, arXiv:2503.10657）**：重评33模型/21数据集，发现包括商业路由器在内的若干方法不能稳定击败best-single-model基线。本文沿用"gap由router design之外的因素主导"这一判断，并补充此前缺失的语言轴。
3. **Hu等（RouterBench, ICML 2024 Workshop, arXiv:2403.12031）**：奠定多LLM路由基准与nearest-neighbour frozen-encoder路由思路。本文指出该基准未见按语言的breakout结果。
4. **Huang等（RouterEval, EMNLP 2025 Findings, arXiv:2503.10657）**：另一大规模路由基准。同样缺乏语言/任务类型细粒度分解，本文与其形成互补定位。
5. **Chen（arXiv:2607.03436, 2026）**：将gap分解为"可复现 specialist advantage"与"single-draw label noise"。本文沿用其label stochasticity视角，但以双重复测量化名义确定性配置的残余噪声（5.37% cell翻转）作为判定参照。
6. **Ong等（RouteLLM, arXiv:2406.18665）**：用偏好数据学习路由。本文并不否定学习路由价值，而是提出应先问"请求本身已可见多少距离"——在此矩阵上23/29已可见，从而降低对学习路由的必要性质疑。

## 局限性与未来方向
1. **任务类型与数据集源混淆（§7.2）**：7种task type各取自单一source（6个公开+1个自采），"按任务类型选择"在此矩阵中等价于"按数据集选择"，无法分离"任务种类效应"与"语料身份效应"。
2. **每cell仅14题、无独立holdout（§7.3）**：21个cell上的选择均拟合于同一294题，in-sample bias未校正；且单题变化即可翻转cell决策。
3. **Coding数据污染未知（§7.4）**：LiveCodeBench v6无污染控制，部分题目可能对pool中多数模型已暴露，既可能压缩model差异也可能放大已有差异。
4. **评分仅为规则化二值正确（§7.5）**：无judging/human rating，无法推广到开放式写作、摘要、多轮对话等"质量不可简单对错判定"的场景。
5. **单快照、单网关、14模型（§7.6）**：结果绑定该目录时点与网关路径，扩池/换提供商/跨时段均可能改变gap结构。
6. **成本为实报账单而非目录价（§7.7）**：8/14模型价格含限时/分层/网关差异等条件；lookup中16/21 cell依赖成本平局打破，目录价变动可能重写大部分选择。
7. **语言轴非原生流量（§7.8）**：韩/印由GPT-4.1 pipeline翻译英文题目而来，register/topical priors继承自英文；Hindi instruction样本经过三语管道存活筛选，可能偏乐观。
8. **噪声下限仅基于2次执行（§7.9）**：5.37%翻转率与3.06% per-item参考均为单次A/B realization；少数关键判定（如2.04%残差、17项差距）对参考值的微小变化敏感。
9. **未来方向（作者自述）**：① 单task type内引入≥2个source以解耦任务种类与语料身份；② 增加至≥3次独立重跑以给出噪声区间估计；③ 将结论外推至非规则化质量评估场景。

## 研究启发与可借鉴点
1. **"噪声下限先行"评估范式**：在建学习路由前，先以双重复测量化目标环境的执行噪声（cell翻转率、账单漂移、per-item cap参考），将任何宣称收益与噪声下限比较后再下结论；避免把不可复现的微小优势当作router能力。
2. **task type×language静态查找可作为强in-sample baseline**：即便忽略泛化性，"按任务类别预分配单模型+平局按成本打破"能在同一份矩阵上轻易超越best single model并在双向（准确/成本）占优。后续工作应以此为floor，并重点评估**跨分布迁移**。
3. **unsigned-margin平局规则**：对跨run符号不一致的accuracy edge显式视为噪声、按成本择低，能抑制在噪声带内"付费买假优势"。该规则易于嵌入任意静态查找构建流程。
4. **"best single model"本身依赖于评分规则**：文中10种合理规则下5个不同模型称王、最大差距≤2.72%（低于噪声下限）。因此路由增益Report（如Gain@B）必须明确标注底层评分规则，否则可比性存疑。
5. **成本集中在单一cell时的解读修正**：lookup成本87.8%来自coding-ko单cell，该cell跨轮账单漂移−17.0%带动整体A/B成本带至−14.79%。对高度集中成本的策略，应分开报告cell级与矩阵级噪声带，避免以矩阵级均值掩盖尾部风险。

## 关键术语表
- **both-run rule**：主评分规则，一项仅在两次独立运行中均判correct时才计为正确，用于剔除单次抽样运气。
- **single-run rule**：仅读取一次运行结果作为评分，存在canonical run选择歧义。
- **task type**：任务种类维度（knowledge/math/instruction/extraction/toolcall/abstention/coding），本文为主要分解轴。
- **model-query cell**：矩阵中一个模型对应一道题目的位置（共14×294=4116），与item（题目）和key（一次执行中的model-question对）区分。
- **unsigned-margin tie rule**：当某cell top模型对最便宜one-behind候选存在跨轮符号翻转的margin且需额外付费时，视作tie并按最低账单选择，避免为噪声溢价。
- **in-sample argmax**：在同一294题上按cell最大化both-run正确数、平局最小化账单所得的查找表（268/294），因post-hoc override与tie rule最终采用262/294。
- **oracle（per-item）**： hindsight oracle，每道题只要pool中任一模型答对即计正确（union over 14 models），代表该pool的理论上限。
- **resolution procedure（§3.2）**：将差值与3.06% per-item、5.37% cell翻转率（准确率）及−3.81%/−14.79%账单漂移（成本）比较；同时超过两档准确率参考判为resolved，否则unresolved。

## 可复现要素
- **数据集**：MMLU-ProX、MGSM、IFEval、BFCL v4、LiveCodeBench v6均为公开数据（含revision hash）；in-house extraction为自建未见公开。
- **代码/权重/工件**：论文未公开代码仓库或权重；提供frozen item ledger的SHA-256（903af87e...）作为承诺，但未随本版本附送ledger与label matrix。
- **关键超参**：temperature=0、seed=42；output cap（coding 32768 token，其余8192）；重试上限3次；单cell 14 items。
- **网关**：OpenRouter（单一网关），2026-08-21目录快照，2026-08-22在线验证model id存在与路由命中。
- **模型池**：14个模型（表1），含vendor/tier配对设计。
- **论文未提及**：开源代码链接、独立held-out set、训练数据/第三方权重下载方式。
