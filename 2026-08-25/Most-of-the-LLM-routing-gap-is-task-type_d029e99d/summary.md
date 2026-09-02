---
title: "Most-of-the-LLM-routing-gap-is-task-type"
source: https://arxiv.org/pdf/2608.23023v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 22:11:12"
field: "大语言模型路由与调度"
keywords: ["LLM routing", "model selection", "task type decomposition", "reproducibility", "cost-aware routing"]
innovations: ["首次系统分解路由gap至任务类型与语言结构，发现72.4%可通过静态查找恢复", "建立严格的执行噪声基底测量方法（5.37%翻转率），提出resolved/unresolved判定程序"]
benchmarks: ["RouterBench", "MMLU-ProX", "MGSM", "IFEval", "BFCL v4", "LiveCodeBench"]
---

# 论文速读：Most-of-the-LLM-routing-gap-is-task-type

## 一句话总结
本文通过全矩阵执行（14模型×294个问题，7种任务类型×3种语言）定位LLM路由性能天花板，发现**路由与oracle的准确率差距主要由任务类型和语言决定**，静态查找表即可恢复大部分增益，剩余未解释差距小于执行噪声。

## 研究问题与动机
1. **路由性能平台期**：Lu et al. [7] 报告21种路由方法在5个基准上收敛于极窄区间（前5名仅差0.22pp），均远低于oracle上限（10-30pp差距）
2. **"谁该被修复"未明确**：现有分解仅沿查询难度、数据集身份、标签随机性三个坐标展开，但难度不可路由观测，数据集身份与任务类型混淆，语言维度完全缺失
3. **多语言路由空白**：RouterBench、RouterEval等基准未报告按语言分解的路由结果，韩/印等语言的gap构成未知
4. **路由设计 vs 可观察信号**：现有工作将oracle gap归因于"模型召回失败"，但未量化路由时可观测特征（任务类型、语言）能解释多少差距

## 核心贡献（创新点）
1. **首次系统分解路由gap的任务类型与语言成分**：通过全矩阵执行定位，发现61.3%（单跑）至72.4%（双跑）的gap可由静态任务类型选择恢复，本质区别在于将gap从"路由设计缺陷"重新定位为"可观察结构信号"
2. **提出严格的噪声基底测量方法**：在温度0+固定seed下重跑同一配置，测量到5.37%的模型-问题对score翻转（221/4116）和-3.81%成本波动，建立可复现性floor判定程序
3. **揭示评分规则对"最佳模型"身份的操控性**：14模型中5种在不同评分规则下均可居首，top-2差距最大仅2.72%，证明"最佳单模型"并非矩阵固有属性
4. **构建并评估静态任务×语言查找表**：在无学习组件的查找表中达到262/294（89.12%）准确率，成本$3.33，vs Claude Opus 5的245/294@ $7.69

## 方法详解
1. **全矩阵设计**：14模型×7任务类型×3语言×14题/单元格=4116单元格/跑，执行两次（温度0，seed 42，单一OpenRouter网关）
2. **两种评分规则**：
   - 单跑规则（single-run）：仅读一次执行结果
   - 主规则（both-run）：仅当模型在两次运行中均答对才计为正确（筛选出稳定正确的29题可改进项）
3. **gap分解框架**：将oracle与最佳单模型的差距按任务类型划分为"between-type"（跨类型选择可恢复）和"within-type"（类型内部差异）
4. **静态查找表构建**：对21个任务×语言单元格，选取双跑准确率最高且A/B平均成本最低的模型；对跨跑符号不稳定的unsigned margin采用"无符号差距tie-breaking"规则
5. **噪声比较程序（§3.2）**：准确差距需同时超过3.06%/题参考值和5.37%/单元格翻转率才算"resolved"；成本差距对比-3.81%矩阵级波动

## 实验与结果
**数据集**：
- MMLU-ProX（knowledge，平行源）、MGSM（math）、IFEval（instruction）、in-house extraction（4类86题）、BFCL v4（toolcall/abstention）、LiveCodeBench（coding，2025-01至2025-04）
- 翻译管线：GPT-4.1温度0，仅翻译指定字段，数字/代码/标识符保持英文

**主要结果（both-run规则）**：
| 基线/策略 | 正确数/294 | 准确率 | 成本/跑 |
|-----------|-----------|--------|---------|
| Oracle（回看最优） | 274 | 93.20% | - |
| Best Single（Claude Opus 5） | 245 | 83.33% | $7.69 |
| 任务类型选择器 | +21题（从最佳单模型恢复） | 72.4% of gap | - |
| 任务类型+语言选择器 | +23题 | 79.3% of gap | - |
| **采纳查找表** | **262** | **89.12%** | **$3.33** |

**关键结论**：
- 静态查找表比最佳单模型准确+17题且成本低56.7%
- 剩余gap仅6题（2.04%），小于5.37%执行噪声floor
- 剩余8题within-type中7题为coding，语言仅恢复其中2题
- 按语言分解：Korean/English/Hindi各自最佳模型差距7-10/98题（7.14%-10.20%），均超过噪声参考

## 相关工作脉络
1. **Lu et al. [7] Routing plateau**：报告21种路由方法收敛于窄带，本文从"gap构成"侧面验证并定位来源
2. **Li et al. [6] LLMRouterBench**：重新评估33模型21数据集，发现商业router不可靠超越best-single，归因于model-recall failure；本文进一步分解recall失败的结构化成分
3. **Chen [1] Label stochasticity**：分离reproducible specialist advantage与single-draw noise，本文在其基础上引入任务类型和语言坐标
4. **Hu et al. [2] RouterBench**：多LLM路由基准，本文指其缺乏语言分解且将long-tail任务列为future work
5. **RouteLLM [8]**：基于偏好数据的学习路由框架，本文论证对可观察结构信号而言复杂学习可能不必要
6. **Janghoon Lee [4] Constrained decoding**：作者前作发现enum-constrained decoding改变abstention测量结果，与本文"评分规则影响排名"现象呼应

## 局限性与未来方向
1. **内样本拟合与评估**：查找表在294题上拟合并在同集上评分，无holdout；无法提供out-of-sample性能估计
2. **任务类型-数据源混淆**：每种任务类型仅来自单一数据集，无法区分"任务类型效应"与"语料库效应"
3. **每单元格仅14题**：n=14导致单元格级选择不稳定（1题变化即可翻转模型选择），无法支撑cell-level结论
4. **Coding任务污染风险**：LiveCodeBench无contamination control，部分模型可能记忆训练数据
5. **单网关快照**：14模型来自单一目录快照，通过单一网关路由；成本/准确率可能受provider-side行为影响
6. **二元评分限制**：所有scorer为rule-based binary check，未覆盖open-ended写作/摘要/多轮对话等quality-judged场景
7. **噪声floor估计薄弱**：5.37%翻转率基于单次A/B pair，非预先注册的阈值，存在估计不确定性
8. **两非英语语言实为翻译品**：Korean/Hindi由GPT-4.1从英文翻译，非native traffic分布

## 研究启发与可借鉴点
1. **可复用的噪声基底测量范式**：在宣称任何路由增益前，先通过重复执行测量配置确定性floor（温度0+固定seed），以区分"真实信号"与"执行噪声"
2. **评分规则的敏感性分析**：不应报告单一"最佳模型"排名，而应展示多评分规则（排除cap-hit、parse-failed、per-language strata）下的稳定性
3. **任务类型分解作为路由预检**：在构建复杂路由前，先评估任务类型/语言等粗粒度信号的修复潜力，避免过早投入学习成本
4. **in-sample vs out-of-sample的诚实声明**：明确区分"本矩阵上的性能"与"泛化性能"，不暗示upper bound为expected performance
5. **成本轴的波动测量**：不仅报告mean billed cost，还应报告run-to-run cost spread（本文coding-ko单元波动达-17.0%），评估成本稳定性

## 关键术语表
**Oracle router**：回看最优路由，为每题选择能答对且成本最低的模型，代表理论性能上限
**Routing plateau**：不同路由方法收敛于窄带准确率的现象，暗示gap主要由模型池限制而非路由设计决定
**Both-run rule**：主评分规则，要求模型在两次独立执行中均答对才计为正确，过滤单次随机正确
**Unsigned margin**：跨执行符号不稳定的准确率差距，本文视为噪声而非可靠信号
**In-sample argmax**：在相同294题上拟合的单元格最优模型选择，存在overfitting偏差
**Cost floor**：路由成本轴的最小可分辨波动，本文测得矩阵级-3.81%，单单元可达-14.79%
**Task×language cell**：21个任务类型与语言组合单元格，每单元格14题，是查找表的索引单元

## 可复现要素
- **数据集**：MMLU-ProX、MGSM、IFEval、BFCL v4、LiveCodeBench（开源，但有coding污染风险）；Extraction任务为in-house构建（Apache-2.0，非公开）
- **代码/权重**：论文声明 companion note [5] in preparation，当前版本不含label matrix和ledger（仅提供SHA-256 hash: 903af87e...）
- **关键超参**：温度0、seed 42、output cap（除coding为32768外均为8192）、翻译管线GPT-4.1温度0
- **网关**：OpenRouter，2026-08-21目录快照
